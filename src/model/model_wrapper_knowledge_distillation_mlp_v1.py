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
from ..misc.utilities import save_gaussians
from .types import Gaussians
from .model_wrapper import OptimizerCfg, TestCfg, TrainCfg, TrajectoryFn


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
            
        
        # Repeat
        gaussians = Gaussians(
            means=repeat(gaussians.means.clone().detach(), "b g xyz -> (b v) g xyz", v=t).detach(), # bs,g,3
            harmonics=repeat(gaussians.harmonics.clone().detach(), "b g c d_sh -> (b v) g c d_sh", v=t).detach(), # bs,g,3,25
            opacities=repeat(gaussians.opacities.clone().detach(), "b g -> (b v) g", v=t).detach(),  # bs,g
            scales=repeat(gaussians.scales.clone().detach(), "b g xyz-> (b v) g xyz", v=t).detach(), # bs,g,3
            rotations=repeat(gaussians.rotations.clone().detach(), "b g xyzw -> (b v) g xyzw", v=t).detach(), # bs,g,4
            covariances=repeat(gaussians.covariances.clone().detach(), "b g i j -> (b v) g i j", v=t).detach(),
        ) 
    
        psnr_impr , ssim_impr, lpips_impr = [], [], []
        incremental_predicted_images = []
        metrics = {
            'psnr_prediction': [], # There will be an list for each target, each element of the inner list will be the psnr for the incremental prediction
            'ssim_prediction': [],
            'lpips_prediction': [],    
            'psnr_mv': [], # psnr that obtained mv applied directly to the raw gaussian over the target views
            'ssim_mv': [],
            'lpips_mv': [],
        } #image space ---> TODO: check if all the prediction metric are > than the mv metric
        losses = [] #Gaussian space for the MLP
        refined_history = []
        
        # Rearrange the batch refinment data from batch target ref_views ... to 
        _, _, r , _, _= batch['refinement']['extrinsics'].shape
        batch['refinement']['extrinsics'] = rearrange(batch['refinement']['extrinsics'], "b t r i j -> (b t) r i j", r=r) 
        batch['refinement']['intrinsics'] = rearrange(batch['refinement']['intrinsics'], "b t r i j -> (b t) r i j", r=r)
        batch['refinement']['image'] = rearrange(batch['refinement']['image'], "b t r c h w -> (b t) r c h w", r=r)
        batch['refinement']['near'] = rearrange(batch['refinement']['near'], "b t -> (b t) ")
        batch['refinement']['far'] = rearrange(batch['refinement']['far'], "b t -> (b t)")    
        
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
        
        
        # Rearrange the target data from batch target views ... to
        batch['target']['extrinsics'] = rearrange(batch['target']['extrinsics'], "b t i j -> (b t) i j")
        batch['target']['intrinsics'] = rearrange(batch['target']['intrinsics'], "b t i j -> (b t) i j")
        batch['target']['image'] = rearrange(batch['target']['image'], "b t c h w -> (b t) c h w")
        batch['target']['near'] = rearrange(batch['target']['near'], "b t -> (b t)")
        batch['target']['far'] = rearrange(batch['target']['far'], "b t -> (b t)")
        
        
        for t_i in range(b*t):  # Iterate over target views spread across batches
            current_target_gaussians = Gaussians(
                means=gaussians.means[t_i].unsqueeze(0),
                harmonics=gaussians.harmonics[t_i].unsqueeze(0),
                opacities=gaussians.opacities[t_i].unsqueeze(0),
                scales=gaussians.scales[t_i].unsqueeze(0),
                rotations=gaussians.rotations[t_i].unsqueeze(0),
                covariances=None
            )
            psnr_improvement,ssim_improvement, lpips_improvement, history = self.refiner.forward(
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
            psnr_impr.append(psnr_improvement)
            ssim_impr.append(ssim_improvement)
            lpips_impr.append(lpips_improvement)
            refined_history.append(history)
            psnr_prediction = []
            ssim_prediction = []
            lpips_prediction = []
            incr_for_target = [] #incremental images for the target view
            
            # First step of MLP process (from raw gaussians to the first refinement step)
            raw_gaussian_features = self.gaussian_mlp.prepare_gaussians(current_target_gaussians)
            history_item = self.gaussian_mlp.prepare_gaussians(history[0])
            # Predict the first incremental update
            incremental_prediction = self.gaussian_mlp(raw_gaussian_features)
            # Compute L2 loss for first step
            losses.append(F.mse_loss((incremental_prediction + raw_gaussian_features), history_item))
            
            # add the first incremental image
            predicted_gaussians = self.gaussian_mlp.add_to_prediction(current_target_gaussians, incremental_prediction)
            output = self.decoder.forward(
                gaussians=predicted_gaussians,
                extrinsics=batch['target']['extrinsics'][t_i].unsqueeze(0).unsqueeze(0),
                intrinsics=batch['target']['intrinsics'][t_i].unsqueeze(0).unsqueeze(0),
                near=batch['target']['near'][t_i].unsqueeze(0).unsqueeze(0),
                far=batch['target']['far'][t_i].unsqueeze(0).unsqueeze(0),
                image_shape=(h, w),
                depth_mode=None,
                use_scale_and_rotation=True,
            )
            incr_for_target.append(output.color[0])
            
            
            # Compute the metrics on the image
            # and also save it
            psnr_prediction.append(compute_psnr(batch['target']['image'][t_i].unsqueeze(0), output.color[0]).item())
            ssim_prediction.append(compute_ssim(batch['target']['image'][t_i].unsqueeze(0), output.color[0]).item())
            lpips_prediction.append(compute_lpips(batch['target']['image'][t_i].unsqueeze(0), output.color[0]).item())
            

            # Loop through the rest of the refinement steps
            for i in range(1, len(history)):
                previous_gaussians = self.gaussian_mlp.prepare_gaussians(history[i-1])
                # Predict next incremental change
                incremental_prediction = self.gaussian_mlp(previous_gaussians)
                # Apply predicted update
                predicted_gaussians = previous_gaussians + incremental_prediction
                # Compute L2 loss against the next refinement step
                target_gaussians = history[i]
                step_loss = F.mse_loss(predicted_gaussians, self.gaussian_mlp.prepare_gaussians(target_gaussians))
                # Store loss
                losses.append(step_loss)
                
                # Output the image
                output = self.decoder.forward(
                    gaussians=target_gaussians,
                    extrinsics=batch['target']['extrinsics'][t_i].unsqueeze(0).unsqueeze(0),
                    intrinsics=batch['target']['intrinsics'][t_i].unsqueeze(0).unsqueeze(0),
                    near=batch['target']['near'][t_i].unsqueeze(0).unsqueeze(0),
                    far=batch['target']['far'][t_i].unsqueeze(0).unsqueeze(0),
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )
                incr_for_target.append(output.color[0])
                psnr_prediction.append(compute_psnr(batch['target']['image'][t_i].unsqueeze(0), output.color[0]).item())
                ssim_prediction.append(compute_ssim(batch['target']['image'][t_i].unsqueeze(0), output.color[0]).item())
                lpips_prediction.append(compute_lpips(batch['target']['image'][t_i].unsqueeze(0), output.color[0]).item())
                
                print(f"Step {i} Loss: {step_loss.item()}")
                

            #Add the predicted images to the list
            incremental_predicted_images.append(incr_for_target)
            metrics['psnr_prediction'].append(psnr_prediction)
            metrics['ssim_prediction'].append(ssim_prediction)
            metrics['lpips_prediction'].append(lpips_prediction)
            
        # Compute final loss
        total_loss = sum(losses) / len(losses)  # Normalize to avoid scale issues
        
        # Save the overview Image
        index_refinement_labels = [f"{i}" for i in rearrange(batch["refinement"]["index"],'b t r->(b t r)').tolist()]
        index_refinement_labels = " ".join(index_refinement_labels)
        index_context_labels = [f"{i}" for i in rearrange(batch["context"]["index"], 'b v -> (b v)').tolist()]
        index_context_labels = " ".join(index_context_labels)
        index_target_labels = [f"{i}" for i in rearrange(batch["target"]["index"], 'b v -> (b v)').tolist()]
        index_target_labels = " ".join(index_target_labels)
        
        context_rearranged = rearrange(batch["context"]["image"], "b v c h w -> (b v) c h w")
        # target_rearranged = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")
        refinement_rearranged = rearrange(batch["refinement"]["image"], "b v c h w -> (b v) c h w")
        output_mv_rearranged = rearrange(output_mv.color, "b v c h w -> (b v) c h w")    
        # Using the predicted gaussians from the mlp
        lenght_history_per_target_predicted = [len(history) for history in incremental_predicted_images]
        lenght_history_per_target_predicted = " ".join([str(i) for i in lenght_history_per_target_predicted])
        predicted_history_imgs = torch.cat([torch.stack(history) for history in incremental_predicted_images], dim=0).squeeze(1)
        # Using the history of the gaussians coming from the refiner
        lenght_history_per_target = [len(history) for history in refined_history]
        lenght_history_per_target = " ".join([str(i) for i in lenght_history_per_target])
        # render the images
        history_imgs_refiner = []
        for t_idx ,history in enumerate(refined_history):
            for g in history:                
                history_imgs_refiner.append(self.decoder.forward(
                    gaussians=g,
                    extrinsics=batch["target"]["extrinsics"][t_idx].unsqueeze(0).unsqueeze(0),
                    intrinsics=batch["target"]["intrinsics"][t_idx].unsqueeze(0).unsqueeze(0),
                    near=batch["target"]["near"][t_idx].unsqueeze(0).unsqueeze(0),
                    far=batch["target"]["far"][t_idx].unsqueeze(0).unsqueeze(0),
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                ).color[0])
        history_imgs_refiner = torch.stack(history_imgs_refiner).squeeze(1)
               
        comparison = hcat(
            add_label(vcat(*context_rearranged), "Context ("+index_context_labels+')'),
            add_label(vcat(*refinement_rearranged), "Refinement ("+index_refinement_labels+')'),   
            add_label(vcat(*batch["target"]["image"]), "Target ("+index_target_labels+')'),     
            add_label(vcat(*output_mv_rearranged), "MV Prediction ("+index_target_labels+')'),
            add_label(vcat(*predicted_history_imgs), "MLP History Pred. ("+lenght_history_per_target_predicted+')'),
            add_label(vcat(*history_imgs_refiner), "Refiner History Pred. ("+lenght_history_per_target+')'),
        )
        
        self.logger.log_image(
            "MLP Method Logs",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
        
        return total_loss
                    

            

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