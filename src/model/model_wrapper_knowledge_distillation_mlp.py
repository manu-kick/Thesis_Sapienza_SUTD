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
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

clip_normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])

# Step 1: Load CLIP model + preprocessing
def load_clip_model(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained
    )
    model.eval()
    return model, preprocess

# Step 2: Get per-cluster feature descriptor
def get_features(cluster_img_tensor, model, target_size=224):
    """
    cluster_img_tensor: torch.Tensor, [3, H, W], values in [0, 1]
    model: CLIP model
    Returns: [1, D] normalized feature
    """
    if cluster_img_tensor.max() > 1.0:
        cluster_img_tensor = cluster_img_tensor / 255.0

    # Resize to match CLIP input size
    resize = Resize((target_size, target_size))
    cluster_img_tensor = resize(cluster_img_tensor)

    # Normalize and batch
    image_tensor = clip_normalize(cluster_img_tensor).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)

    return image_features

# Step 4: Match cluster descriptor with refinement descriptors
def match_and_extract(cluster_feature, refinement_features, top_k=1):
    """
    cluster_feature: [1, D]
    refinement_features: List of [1, D]
    """
    all_ref_features = torch.cat(refinement_features, dim=0)  # [R, D]
    sims = F.cosine_similarity(cluster_feature, all_ref_features)  # [R]
    topk = torch.topk(sims, k=top_k)

    matched_feats = all_ref_features[topk.indices]  # [top_k, D]
    fused_feature = matched_feats.mean(dim=0, keepdim=True)  # [1, D]

    return fused_feature  # fused semantic feature for this cluster



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
        self.clip_model, self.clip_preprocess = load_clip_model()
        self.clip_model = self.clip_model.to(next(self.gaussian_mlp.parameters()).device)
        
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
            
        # Run the refinement process 
        
        # Repeat the gaussians for each target view
        gaussians = Gaussians(
            means=repeat(gaussians.means.clone().detach(), "b g xyz -> (b v) g xyz", v=t).detach(), # bs,g,3
            harmonics=repeat(gaussians.harmonics.clone().detach(), "b g c d_sh -> (b v) g c d_sh", v=t).detach(), # bs,g,3,25
            opacities=repeat(gaussians.opacities.clone().detach(), "b g -> (b v) g", v=t).detach(),  # bs,g
            scales=repeat(gaussians.scales.clone().detach(), "b g xyz-> (b v) g xyz", v=t).detach(), # bs,g,3
            rotations=repeat(gaussians.rotations.clone().detach(), "b g xyzw -> (b v) g xyzw", v=t).detach(), # bs,g,4
            covariances=repeat(gaussians.covariances.clone().detach(), "b g i j -> (b v) g i j", v=t).detach(),
        ) 
        
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
        refined_gaussians = []
        predicted_gaussians = []
        
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
        
        
        # Rearrange the batch refinment data
        _, _, r , _, _= batch['refinement']['extrinsics'].shape
        batch['refinement']['extrinsics'] = rearrange(batch['refinement']['extrinsics'], "b t r i j -> (b t) r i j", r=r) 
        batch['refinement']['intrinsics'] = rearrange(batch['refinement']['intrinsics'], "b t r i j -> (b t) r i j", r=r)
        batch['refinement']['image'] = rearrange(batch['refinement']['image'], "b t r c h w -> (b t) r c h w", r=r)
        batch['refinement']['near'] = rearrange(batch['refinement']['near'], "b t -> (b t) ")
        batch['refinement']['far'] = rearrange(batch['refinement']['far'], "b t -> (b t)")    
        
        # Rearrange the target data
        batch['target']['extrinsics'] = rearrange(batch['target']['extrinsics'], "b t i j -> (b t) i j")
        batch['target']['intrinsics'] = rearrange(batch['target']['intrinsics'], "b t i j -> (b t) i j")
        batch['target']['image'] = rearrange(batch['target']['image'], "b t c h w -> (b t) c h w")
        batch['target']['near'] = rearrange(batch['target']['near'], "b t -> (b t)")
        batch['target']['far'] = rearrange(batch['target']['far'], "b t -> (b t)")
        
        refinement_features = self.encoder.backbone(batch['refinement']['image'])[0]
        clusters = cluster_gaussians(self.gaussian_mlp.prepare_gaussians(gaussians), exclude_z=False)
        
        # Decode an image for each cluster for a target view
        cluster_images_targets = []
        for t_i in range(b*t):
            cluster_images_target = []
            for i, cluster in enumerate(clusters[batch_idx]):
                output = self.decoder.forward(
                    gaussians=cluster,
                    extrinsics=batch["target"]["extrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    intrinsics=batch["target"]["intrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    near=batch["target"]["near"][t_i].unsqueeze(0).unsqueeze(0),
                    far=batch["target"]["far"][t_i].unsqueeze(0).unsqueeze(0),
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )
                cluster_images_target.append(output.color[0][0])
            cluster_images_targets.append(torch.stack(cluster_images_target))
        
        # Create the Comparison with number of column equal to the number of targets and the first row is cothe original target image while the other rows are the cluster images
        for t_i in range(len(cluster_images_targets)):
            # Add the original target image to the given target cluster images
            cluster_images_targets[t_i] = torch.cat((batch['target']['image'][t_i].unsqueeze(0), cluster_images_targets[t_i]), dim=0)
        
        comparison = hcat(
            add_label(vcat(*cluster_images_targets[0]), "Target 0"),
            add_label(vcat(*cluster_images_targets[1]), "Target 1"),
            add_label(vcat(*cluster_images_targets[2]), "Target 2"),
            add_label(vcat(*cluster_images_targets[3]), "Target 3"),
        )
        
        self.logger.log_image(
            'Cluster plot for targets',
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
        
        
        enriched_cluster_features = []

        for cluster_img in cluster_images_targets[t_i][1:]:  # skip first image (full target image)
            cluster_feature = get_features(cluster_img, self.clip_model)

            refinement_feats = []
            for r_i in range(r):
                refinement_img = batch['refinement']['image'][t_i][r_i]  # [3, H, W] or PIL
                refinement_feat = get_features(refinement_img, self.clip_model, self.clip_preprocess)
                refinement_feats.append(refinement_feat)

            # Match and fuse
            enriched_feat = match_and_extract(cluster_feature, refinement_feats, top_k=1)
            enriched_cluster_features.append(enriched_feat)
        
        for t_i in range(b*t):  # Iterate over target views spread across batches
            current_target_gaussians = Gaussians(
                means=gaussians.means[t_i].unsqueeze(0),
                harmonics=gaussians.harmonics[t_i].unsqueeze(0),
                opacities=gaussians.opacities[t_i].unsqueeze(0),
                scales=gaussians.scales[t_i].unsqueeze(0),
                rotations=gaussians.rotations[t_i].unsqueeze(0),
                covariances=None
            )
        
            _, _, _ , _ = self.refiner.forward(
                {   
                    "extrinsics": batch['refinement']['extrinsics'][t_i].unsqueeze(0), 
                    "intrinsics": batch['refinement']['intrinsics'][t_i].unsqueeze(0),
                    "near": batch['refinement']['near'][t_i].unsqueeze(0),
                    "far": batch['refinement']['far'][t_i].unsqueeze(0),
                    "image": batch['refinement']['image'][t_i].unsqueeze(0),
                }, 
                current_target_gaussians, # gaussian of
                self.global_step, 
                return_history=True
            )
            
            # Get the refined Gaussians
            refined_gaussians.append(Gaussians(
                means=self.refiner.means, 
                harmonics=self.refiner.harmonics, 
                opacities=self.refiner.opacities, 
                scales=self.refiner.scales, 
                rotations=self.refiner.rotations,
                covariances=None  # We use scales and rotations
            ))
            
            output_t_i = self.decoder.forward(
                gaussians=refined_gaussians[-1],
                extrinsics=batch['target']['extrinsics'][t_i].unsqueeze(0).unsqueeze(0),
                intrinsics=batch['target']['intrinsics'][t_i].unsqueeze(0).unsqueeze(0),
                near=batch['target']['near'][t_i].unsqueeze(0).unsqueeze(0),
                far=batch['target']['far'][t_i].unsqueeze(0).unsqueeze(0),
                image_shape=(h, w),
                depth_mode=None,
                use_scale_and_rotation=True,
            ) # [bs, 1, c, h, w]
            
            # Compute evaluation metrics
            metrics['psnr_refined'].append(compute_psnr(
                batch['target']['image'][t_i].unsqueeze(0),
                output_t_i.color.squeeze(0),
            ).item())
            metrics['ssim_refined'].append(compute_ssim(
                batch['target']['image'][t_i].unsqueeze(0),
                output_t_i.color.squeeze(0),
            ).item())
            metrics['lpips_refined'].append(compute_lpips(
                batch['target']['image'][t_i].unsqueeze(0),
                output_t_i.color.squeeze(0),
            ).item())
            
            # Compute the improvement
            metrics['improvement']['psnr'].append(metrics['psnr_refined'][-1] - metrics['psnr_mv'][t_i])
            metrics['improvement']['ssim'].append(metrics['ssim_refined'][-1] - metrics['ssim_mv'][t_i])
            metrics['improvement']['lpips'].append(metrics['lpips_refined'][-1] - metrics['lpips_mv'][t_i])
            
            # Predict the gaussians using the MLP
            gaussian_tensor = self.gaussian_mlp.prepare_gaussians(refined_gaussians)
            # rearrange from 1, 131072 , 86 -> 1 ,2, 256, 256, 86
            p = batch['context']['image'].shape[1]
            gaussiasn_tensor = rearrange(gaussian_tensor, "b g a -> b p h w a", b=b, p=p, h=h, w=w, a=gaussian_tensor.shape[-1])
            predicted_gaussians.append(self.gaussian_mlp(current_target_gaussians))
            
            # torch.cat((rearrange(refinement_features[0][0],'r d x y -> r d (x y)'), self.gaussian_mlp.prepare_gaussians(current_target_gaussians)),dim=0)
            
            
            # Compute the loss between the predicted gaussians and the refined gaussians (l1 )
            l1_means = F.l1_loss(predicted_gaussians[-1][:, :3], refined_gaussians[-1].means).item()
            l1_harmonics = F.l1_loss(predicted_gaussians[-1][:, 3:78], refined_gaussians[-1].harmonics).item()
            l1_opacities = F.l1_loss(predicted_gaussians[-1][:, 78:79], refined_gaussians[-1].opacities).item()
            l1_scales = F.l1_loss(predicted_gaussians[-1][:, 79:82], refined_gaussians[-1].scales).item()
            l1_rotations = F.l1_loss(predicted_gaussians[-1][:, 82:86], refined_gaussians[-1].rotations).item()
            metrics['loss_predicted_refined'].append(l1_means + l1_harmonics + l1_opacities + l1_scales + l1_rotations)
    
        total_loss = torch.mean(torch.tensor(metrics['loss_predicted_refined']))
        
        return  total_loss
                    

            

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