# In this file we are gonna try a new matching method: before we were performing the following operations:
# obtain the raw gaussian , cluster them decode partial image and then try to match the partial rendered image with refinement camera
# now we are trying to do the following:
# obtain the raw gaussians, cluster them, then consider the subset portion of the target camera image obtained considered a mask over the target camera using the partial rendered image 
# (the pixel all black in the partial rendered image are excluded from the target camera); now we consider the match between a target and its correspondent refinement cameras (using sift).
# for each subset we consider the matching point placed in that area taking them from the matching pre computed

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

# import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat 
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import torch.nn.functional as F
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from PIL import Image
from ..misc.utilities import save_gaussians, cluster_gaussians
from .types import Gaussians
from .model_wrapper import OptimizerCfg, TestCfg, TrainCfg, TrajectoryFn
import torch
import torch.nn.functional as F
# import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2
import matplotlib.pyplot as plt

def sift_match(target_image, refinement_image):
    """
    Match the cluster image with the refinement image using SIFT.
    """
    # Scale the target image from [0,1] to [0,255] and convert to numpy.
    target_np = (target_image.clone().detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    # Scale the refinement image similarly.
    refinement_np = (refinement_image[0].clone().detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Convert to grayscale.
    target_gray = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
    refinement_gray = cv2.cvtColor(refinement_np, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp_target, des_target = sift.detectAndCompute(target_gray, None)
    kp_refinement, des_refinement = sift.detectAndCompute(refinement_gray, None)

    # Use a Brute-Force matcher with L2 norm and cross-check enabled.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_target, des_refinement)
    
    # Sort matches in ascending order of distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return kp_target, kp_refinement, matches



# Function to visualize the keypoint matching
def visualize_keypoint_matching(cluster_image, refinement_image, kp_cluster, kp_refinement, matches):
    """
    Visualize the keypoint matching between the cluster image and refinement camera image.
    This will draw lines between corresponding keypoints using SIFT.
    """
    # Draw the matches between the two images
    matched_image = cv2.drawMatches(
        cluster_image.clone().detach().permute(1,2,0).cpu().numpy().astype(np.uint8), kp_cluster, 
        refinement_image[0].clone().detach().permute(1,2,0).cpu().numpy().astype(np.uint8), kp_refinement, 
        matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Convert the result to RGB for plotting with matplotlib
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

    # Save the matched image
    matched_image_path = Path("matched_keypoints.png")
    matched_image_bgr = cv2.cvtColor(matched_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(matched_image_path), matched_image_bgr)


class GaussianRefinementMLP(nn.Module):
    def __init__(self, input_dim=86, hidden_dim=128, output_dim=86):
        """
        MLP that maps raw Gaussian parameters to refined Gaussian parameters.
        - Input: Raw Gaussians (means, harmonics, opacities, scales, rotations)
        - Output: Refined Gaussians with valid covariance (positive semi-definite)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        if isinstance(x, Gaussians):
            x = self.prepare_gaussians(x)
        return self.mlp(x)
    
    def enforce_valid_covariance(self, scales, rotations):
        """
        Ensure positive semi-definite covariance matrix.
        - Scales must be positive -> Use softplus activation.
        - Rotations must be normalized quaternions -> Use normalization.
        """
        scales = F.softplus(scales)  # Ensures positive scales
        rotations = F.normalize(rotations, dim=-1)  # Normalize quaternion
        return scales, rotations
    
    def prepare_gaussians(self, gaussians):
        """
        Prepare Gaussians for the MLP by concatenating the components.
        """
        return torch.cat([
            gaussians.means, # bs,g,3
            rearrange(gaussians.harmonics,'b g c h -> b g (c h)',b=gaussians.harmonics.shape[0], g=gaussians.harmonics.shape[1], c=gaussians.harmonics.shape[2], h=gaussians.harmonics.shape[3] ), # bs,g,3,75
            gaussians.opacities.unsqueeze(-1), # bs,g,1
            gaussians.scales, # bs,g,3
            gaussians.rotations #bs,g,4
        ], dim=-1)
        
    def add_to_prediction(self, prediction : Gaussians, incremental: Tensor) -> Gaussians:
        """
        Add the incremental change to the prediction.
        """
        harmonics = rearrange(incremental[:, :, 3:78], 'b g (c h) -> b g c h', b=incremental.shape[0], g=incremental.shape[1], c=3, h=25)
        return Gaussians(
            means=prediction.means + incremental[:, :, :3],
            harmonics=prediction.harmonics + harmonics,
            opacities=prediction.opacities + incremental[:, :, 78:79].squeeze(-1),
            scales=prediction.scales + incremental[:, :, 79:82],
            rotations=prediction.rotations + incremental[:, :, 82:86],
            covariances=None
        )


class ModelWrapper_KD_MLP(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        refiner: Optional[nn.Module],
        refiner_cfg,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder 
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.refiner = refiner
        self.refiner_cfg = refiner_cfg
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            
        # Add MLP for Gaussian refinement mapping
        self.gaussian_mlp = GaussianRefinementMLP()
        
    # TODO: Create a fusion of clustered Gaussians and the refinement features
    def fuse_gaussians_with_refinement_features(self,cluster, matched_features):
        """
        Fuse Gaussian features with refinement features using matched keypoints.
        """
        gaussian_features = self.gaussian_mlp.prepare_gaussians(cluster)
        
        # Here we can use matched keypoints or their descriptors to fuse the refinement features
        matched_descriptors = torch.tensor([desc for _, desc in matched_features])  # Extract descriptors from the matches
        fused_features = torch.cat((gaussian_features, matched_descriptors), dim=-1)  # Concatenate along feature dimension
        return fused_features

    def on_train_start(self):
        self.encoder.eval()

    
    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, t, _ , h, w = batch["target"]["image"].shape

        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        
        # Get the target images predicted directly form Mv
        output_mv = self.decoder.forward(
            gaussians=gaussians, # use the gaussians from the encoder (keep the gradient)
            extrinsics=batch["target"]["extrinsics"],
            intrinsics=batch["target"]["intrinsics"],
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
            depth_mode=None,
            use_scale_and_rotation=True,
        ) # color has shape (b,t,c,h,w)
            
        metrics = {
            'psnr_refined': [], # There will be an list for each target, each element of the inner list will be the psnr for the incremental prediction
            'ssim_refined': [],
            'lpips_refined': [],    
            'psnr_mv': [], # psnr that obtained mv applied directly to the raw gaussian over the target views
            'ssim_mv': [],
            'lpips_mv': [],
            'improvement':{
                'psnr': [],
                'ssim': [],
                'lpips': [],
            },
            'loss_predicted_refined': []
        }
        
        # Metrics for the MV
        metrics['psnr_mv'] = compute_psnr(
            rearrange(batch['target']['image'], "b v c h w -> (b v) c h w"),
            rearrange(output_mv.color, "b v c h w -> (b v) c h w"),
        ).tolist()
        metrics['ssim_mv'] = compute_ssim(
            rearrange(batch['target']['image'], "b v c h w -> (b v) c h w"),
            rearrange(output_mv.color, "b v c h w -> (b v) c h w"),
        ).tolist()
        metrics['lpips_mv'] = compute_lpips(
            rearrange(batch['target']['image'], "b v c h w -> (b v) c h w"),
            rearrange(output_mv.color, "b v c h w -> (b v) c h w"),
        ).tolist()

        # List big as the batch size, each element is a list big as the number of cluster. Each element of the inner list is a Gaussian object
        clusters = cluster_gaussians(self.gaussian_mlp.prepare_gaussians(gaussians), exclude_z=False)

        # Decode an image for each cluster for a target view
        cluster_images_targets = []  # Outer list: one element per batch sample
        for b_i in range(b):
            batch_cluster_images = []  # For each batch, list of target views
            for t_i in range(t):
                target_cluster_images = []  # For each target view, list of cluster images
                for i, cluster in enumerate(clusters[batch_idx]):
                    output = self.decoder.forward(
                        gaussians=cluster,
                        extrinsics=batch["target"]["extrinsics"][b_i][t_i].unsqueeze(0).unsqueeze(0),
                        intrinsics=batch["target"]["intrinsics"][b_i][t_i].unsqueeze(0).unsqueeze(0),
                        near=batch["target"]["near"][b_i][t_i].unsqueeze(0).unsqueeze(0),
                        far=batch["target"]["far"][b_i][t_i].unsqueeze(0).unsqueeze(0),
                        image_shape=(h, w),
                        depth_mode=None,
                        use_scale_and_rotation=True,
                    )
                    target_cluster_images.append(output.color[0][0])
                # Stack clusters for this target view.
                batch_cluster_images.append(torch.stack(target_cluster_images))
            # Append the per-batch clusters (organized per target view).
            cluster_images_targets.append(batch_cluster_images)

        # We have a list [bs, targets, clusters]
        
        
        # Create the Comparison with number of column equal to the number of targets and the first row is cothe original target image while the other rows are the cluster images
        # for t_i in range(len(cluster_images_targets)):
        #     # Add the original target image to the given target cluster images
        #     cluster_images_targets[t_i] = torch.cat((batch['target']['image'][t_i].unsqueeze(0), cluster_images_targets[t_i]), dim=0)
        
        # comparison = hcat(
        #     add_label(vcat(*cluster_images_targets[0]), "Target 0"),
        #     add_label(vcat(*cluster_images_targets[1]), "Target 1"),
        #     add_label(vcat(*cluster_images_targets[2]), "Target 2"),
        #     add_label(vcat(*cluster_images_targets[3]), "Target 3"),
        # )
        
        # self.logger.log_image(
        #     'Cluster plot for targets',
        #     [prep_image(add_border(comparison))],
        #     step=self.global_step,
        #     caption=batch["scene"],
        # )
        
        # --- UPDATED MATCHING PROCESS ---
        # Step 1: Aggregate SIFT matches for the entire target image per target view.
        aggregated_matches = []  # one element per target view
        for b_i in range(b):
            for t_i in range(t):
                # Convert the entire target image to numpy (for SIFT input).
                target_img = batch['target']['image'][b_i][t_i]
                
                # Containers to collect keypoints and matches from all refinement images.
                target_keypoints_all = []
                refinement_keypoints_all = []
                all_matches = []
                
                # Iterate over all refinement images corresponding to this target.
                for r_img in batch['refinement']['image'][b_i][t_i]:
                    # r_img is (c, h, w): convert to batch format and numpy.
                    r_img_tensor = r_img.unsqueeze(0)  # shape: (1, c, h, w)
                    kp_target, kp_ref, matches = sift_match(target_img, r_img_tensor)
                    target_keypoints_all.extend(kp_target)
                    refinement_keypoints_all.extend(kp_ref)
                    all_matches.extend(matches)
                
                aggregated_matches.append((target_keypoints_all, refinement_keypoints_all, all_matches))

        
        # Step 2: For each target view, for each cluster (obtained from partial rendering),
        # filter the aggregated matches by checking if the target keypoint lies within the cluster mask.
        matched_features = []  # one list per target view
        for b_i, clusters_img in enumerate(cluster_images_targets):
            # clusters_img is a tensor stack where the first row is the original target image.
            # We process each cluster (skipping row 0 if it represents the full target).
            target_kp_all, refinement_kp_all, all_matches = aggregated_matches[t_i]
            matched_features_per_target = []
            for i in range(1, clusters_img.shape[0]):  # assuming row 0 is the full target image
                cluster_image = clusters_img[i]
                # Compute a binary mask from the cluster image (exclude all-black pixels).
                cluster_np = cluster_image.clone().detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                cluster_gray = cv2.cvtColor(cluster_np, cv2.COLOR_RGB2GRAY)
                cluster_mask = cv2.threshold(cluster_gray, 1, 255, cv2.THRESH_BINARY)[1]
                
                # Filter matches: check if the target keypoint lies within the cluster mask.
                filtered_matches = []
                filtered_target_points = []
                filtered_refinement_points = []
                for m in all_matches:
                    # m.queryIdx refers to the index of the keypoint in the target image.
                    x, y = target_kp_all[m.queryIdx].pt
                    # If the keypoint is inside the valid region defined by the cluster mask.
                    if cluster_mask[int(y), int(x)] > 0:
                        filtered_matches.append(m)
                        filtered_target_points.append(target_kp_all[m.queryIdx].pt)
                        filtered_refinement_points.append(refinement_kp_all[m.trainIdx].pt)
                        
                matched_features_per_target.append((filtered_target_points, filtered_refinement_points))

                # Optionally visualize the filtered matching.
                visualize_keypoint_matching(
                    cluster_image,
                    batch['target']['image'][t_i].unsqueeze(0),
                    target_kp_all,
                    refinement_kp_all,
                    filtered_matches
                )
            matched_features.append(matched_features_per_target)


            
                
                
        # TODO: Forward the fusion to the MLP and get the incremental prediction
        
        
  
        return None
                    

            

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"Validation step {self.global_step}; "
                f"Scene = {[a[:20] for a in batch['scene']]}; "
                f"Context = {batch['context']['index'].tolist()}"
            )

        # --- Get Original Gaussians (No Refinement) ---
        b, t, _, h, w = batch["target"]["image"].shape
        assert b == 1  # Single batch size
        
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        original_gaussians = Gaussians(
            means=gaussians.means.clone().detach(), 
            harmonics=gaussians.harmonics.clone().detach(), 
            opacities=gaussians.opacities.clone().detach(), 
            scales=gaussians.scales.clone().detach(), 
            rotations=gaussians.rotations.clone().detach(),
            covariances=gaussians.covariances.clone().detach()
        )

        
        batch['target']['extrinsics'] = rearrange(batch['target']['extrinsics'], "b t i j -> (b t) i j")
        batch['target']['intrinsics'] = rearrange(batch['target']['intrinsics'], "b t i j -> (b t) i j")
        batch['target']['image'] = rearrange(batch['target']['image'], "b t c h w -> (b t) c h w")
        batch['target']['near'] = rearrange(batch['target']['near'], "b t -> (b t)")
        batch['target']['far'] = rearrange(batch['target']['far'], "b t -> (b t)")
        
        
        # --- Get MLP-Mapped Gaussians ---
        starting = self.gaussian_mlp.prepare_gaussians(original_gaussians)
        for t_i in range(t):
            for i in range(5):
                incremental_prediction = self.gaussian_mlp(starting)
                predicted_gaussians = self.gaussian_mlp.add_to_prediction(original_gaussians, incremental_prediction)
                # decode the gaussians and compute the metrics
                output_mlp = self.decoder.forward(
                    gaussians=predicted_gaussians,
                    extrinsics=batch["target"]["extrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    intrinsics=batch["target"]["intrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    near=batch["target"]["near"][t_i].unsqueeze(0).unsqueeze(0),
                    far=batch["target"]["far"][t_i].unsqueeze(0).unsqueeze(0),
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,   
                )
                
                # Compute evaluation metrics
                psnr_mlp = compute_psnr(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    output_mlp.color.squeeze(0),
                ).mean().item()
                
                ssim_mlp = compute_ssim(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    output_mlp.color.squeeze(0),
                ).mean().item()
                lpips_mlp = compute_lpips(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    output_mlp.color.squeeze(0),
                ).mean().item()
                
                print(f'Target {t_i}', f'Iteration {i}', f'PSNR {psnr_mlp}', f'SSIM {ssim_mlp}', f'LPIPS {lpips_mlp}')
                
                # Update the starting point
                starting = predicted_gaussians
            
            
                
            
        # Compute evaluation metrics
        psnr_mlp = compute_psnr(
            rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
            rearrange(output_mlp.color, "b v c h w -> (b v) c h w"),
        ).mean().item()

        ssim_mlp = compute_ssim(
            rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
            rearrange(output_mlp.color, "b v c h w -> (b v) c h w"),
        ).mean().item()

        lpips_mlp = compute_lpips(
            rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
            rearrange(output_mlp.color, "b v c h w -> (b v) c h w"),
        ).mean().item()

        self.logger.log_metrics({
            "val/psnr_mlp": psnr_mlp,
            "val/ssim_mlp": ssim_mlp,
            "val/lpips_mlp": lpips_mlp
        }, step=self.global_step)

        
        self.log("val/psnr_mean", psnr_mlp, sync_dist=True)

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, t, _ , h, w = batch["target"]["image"].shape
        assert b == 1

        # Get Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
            original_gaussians = Gaussians(
                means=gaussians.means.clone().detach(), 
                harmonics=gaussians.harmonics.clone().detach(), 
                opacities=gaussians.opacities.clone().detach(), 
                scales=gaussians.scales.clone().detach(), 
                rotations=gaussians.rotations.clone().detach(),
                covariances=gaussians.covariances.clone().detach()
            )
            
        with self.benchmarker.time("decoder", num_calls=t):
            output_mv = self.decoder.forward(
                gaussians=original_gaussians,
                extrinsics=batch["target"]["extrinsics"],
                intrinsics=batch["target"]["intrinsics"],
                near=batch["target"]["near"],
                far=batch["target"]["far"],
                image_shape=(h, w),
                depth_mode=None,
                use_scale_and_rotation=True,
            )
            
        if self.refiner is not None:
            psnr_per_view, ssim_per_view, lpips_per_view = [], [], []
            psnr_per_view_refiner ,ssim_per_view_refiner ,lpips_per_view_refiner = [], [], []
            psnr_impr , ssim_impr, lpips_impr = [], [], []
            
            for t_i in range(t):  # Iterate over target views
                # Forward the data through the refiner for the given target view
                psnr_improvement,ssim_improvement, lpips_improvement = self.refiner.forward(
                    {   
                        "extrinsics": batch['refinement']['extrinsics'][:, t_i],
                        "intrinsics": batch['refinement']['intrinsics'][:, t_i],
                        "near": batch['refinement']['near'][:, t_i], #bs ,num_views
                        "far": batch['refinement']['far'][:, t_i],
                        "image": batch['refinement']['image'][:, t_i],
                    }, 
                    gaussians, 
                    self.global_step
                )
                psnr_impr.append(psnr_improvement)
                ssim_impr.append(ssim_improvement)
                lpips_impr.append(lpips_improvement)
                
                # get the refined gaussians for the target view
                refined_gaussians_i = Gaussians(
                    means=self.refiner.means.detach(), 
                    harmonics=self.refiner.harmonics.detach(), 
                    opacities=self.refiner.opacities.detach(), 
                    scales=self.refiner.scales.detach(), 
                    rotations=self.refiner.rotations.detach(),
                    covariances=None # We use scales and rotations
                )
                # Splat using the refined gaussians for the target view
                output_refiner_i = self.decoder.forward(
                    gaussians=refined_gaussians_i,
                    extrinsics=batch['target']['extrinsics'][:, t_i].unsqueeze(1),
                    intrinsics=batch['target']['intrinsics'][:, t_i].unsqueeze(1),
                    near=batch['target']['near'][:, t_i].unsqueeze(1),
                    far=batch['target']['far'][:, t_i].unsqueeze(1),
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )
                
                    
                (scene,) = batch["scene"]
                name = get_cfg()["wandb"]["name"]
                path = self.test_cfg.output_path / name
                rgb_mv = output_mv.color[0][t_i].unsqueeze(0) # t_i-th mv's output target view
                rgb_refiner = output_refiner_i.color[0][0].unsqueeze(0) # t_i-th refiner's output target view using the refined gaussians with obtained with its refinement set
                rgb_gt = batch["target"]["image"][0][t_i].unsqueeze(0) # t_i-th target view
                
                # ---- Compute scores per view ----
                if self.test_cfg.compute_scores:
                    psnr_per_view.append(compute_psnr(rgb_gt, rgb_mv).item())
                    ssim_per_view.append(compute_ssim(rgb_gt, rgb_mv).item())
                    lpips_per_view.append(compute_lpips(rgb_gt, rgb_mv).item())
                    psnr_per_view_refiner.append(compute_psnr(rgb_gt, rgb_refiner).item())
                    ssim_per_view_refiner.append(compute_ssim(rgb_gt, rgb_refiner).item())
                    lpips_per_view_refiner.append(compute_lpips(rgb_gt, rgb_refiner).item())
                # ------------------------------
            
            # ---- Compute Averages Over Target Views ----
            if self.test_cfg.compute_scores:
                # Initialize metric storage if not already done
                if "psnr" not in self.test_step_outputs:
                    self.test_step_outputs["psnr"] = []
                    if self.refiner is not None:
                        self.test_step_outputs["psnr_refiner"] = []
                if "ssim" not in self.test_step_outputs:
                    self.test_step_outputs["ssim"] = []
                    if self.refiner is not None:
                        self.test_step_outputs["ssim_refiner"] = []
                if "lpips" not in self.test_step_outputs:
                    self.test_step_outputs["lpips"] = []
                    if self.refiner is not None:
                        self.test_step_outputs["lpips_refiner"] = []

                # Compute and store averages
                self.test_step_outputs["psnr"].append(np.mean(psnr_per_view))
                self.test_step_outputs["ssim"].append(np.mean(ssim_per_view))
                self.test_step_outputs["lpips"].append(np.mean(lpips_per_view))

                if self.refiner is not None:
                    self.test_step_outputs["psnr_refiner"].append(np.mean(psnr_per_view_refiner))
                    self.test_step_outputs["ssim_refiner"].append(np.mean(ssim_per_view_refiner))
                    self.test_step_outputs["lpips_refiner"].append(np.mean(lpips_per_view_refiner))
                        
                # Save images.
                if self.test_cfg.save_image:
                    idx = 0
                    for index, color in zip(batch["target"]["index"][0], rgb_mv.squeeze(0)):
                        save_image(color, path / scene / f"color/{index:0>6}.png")
                        idx += 1
                        if self.refiner is not None: # TODO and save metrics
                            Image.fromarray((rgb_refiner[0] * 255).cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)).save(path / scene / f"color/refined_{index:0>6}.png")
            
            # Print the improvement metrics
            if self.refiner is not None:
                print(f"PSNR improvement batch {batch_idx}: {np.mean(psnr_impr)}")
                print(f"SSIM improvement batch {batch_idx}: {np.mean(ssim_impr)}")
                print(f"LPIPS improvement batch {batch_idx}: {np.mean(lpips_impr)}")
        else:
            if self.test_cfg.compute_scores:
                if "psnr" not in self.test_step_outputs:
                    self.test_step_outputs["psnr"] = []
                if "ssim" not in self.test_step_outputs:
                    self.test_step_outputs["ssim"] = []
                if "lpips" not in self.test_step_outputs:
                    self.test_step_outputs["lpips"] = []
                    
                psnr_mv = compute_psnr(
                    rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                    rearrange(output_mv.color, "b v c h w -> (b v) c h w"),
                ).mean().item()
                ssim_mv = compute_ssim(
                    rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
                    rearrange(output_mv.color,"b v c h w -> (b v) c h w"),
                ).mean().item()
                lpips_mv = compute_lpips(
                    rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
                    rearrange(output_mv.color,"b v c h w -> (b v) c h w"),
                ).mean().item()
                self.test_step_outputs["psnr"].append(psnr_mv)
                self.test_step_outputs["ssim"].append(ssim_mv)
                self.test_step_outputs["lpips"].append(lpips_mv)

            
        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in rgb_mv],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()


    def configure_optimizers(self):
        params = [p for n, p in self.named_parameters() if "encoder" not in n]
        optimizer = optim.Adam(params, lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": warm_up,
            #     "interval": "step",
            #     "frequency": 1,
            # },
        }

    
    def on_load_checkpoint(self, checkpoint):
        ignored_keys ={
            "refiner.means", "refiner.harmonics", "refiner.opacities", "refiner.scales", "refiner.rotations"
        }
        
        # Remove unwanted keys before loading state_dict
        for key in ignored_keys:
            if key in checkpoint["state_dict"]:
                del checkpoint["state_dict"][key]

        print("✅ Ignored unwanted keys:", ignored_keys)