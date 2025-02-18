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
import wandb

class ModelWrapper_KD(LightningModule):
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
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
    
    @rank_zero_only
    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, t, _ , h, w = batch["target"]["image"].shape

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
        
        # --- Refinement Process ---
        mse_losses = []
        if self.refiner is not None:
            psnr_impr , ssim_impr, lpips_impr = [], [], []
            for t_i in range(t):  # Iterate over target views
                psnr_improvement,ssim_improvement, lpips_improvement = self.refiner.forward(
                    {   
                        "extrinsics": batch['refinement']['extrinsics'][:, t_i],
                        "intrinsics": batch['refinement']['intrinsics'][:, t_i],
                        "near": batch['refinement']['near'][:, t_i],
                        "far": batch['refinement']['far'][:, t_i],
                        "image": batch['refinement']['image'][:, t_i],
                    }, 
                    gaussians, 
                    self.global_step
                )
                psnr_impr.append(psnr_improvement)
                ssim_impr.append(ssim_improvement)
                lpips_impr.append(lpips_improvement)

                # Get the refined Gaussians
                refined_gaussians = Gaussians(
                    means=self.refiner.means, 
                    harmonics=self.refiner.harmonics, 
                    opacities=self.refiner.opacities, 
                    scales=self.refiner.scales, 
                    rotations=self.refiner.rotations,
                    covariances=None  # We use scales and rotations
                )

                # Compute MSE loss between original and refined Gaussians
                mse_loss = (
                    F.mse_loss(refined_gaussians.means, original_gaussians.means) +
                    F.mse_loss(refined_gaussians.harmonics, original_gaussians.harmonics) +
                    F.mse_loss(refined_gaussians.opacities, original_gaussians.opacities) +
                    F.mse_loss(refined_gaussians.scales, original_gaussians.scales) +
                    F.mse_loss(refined_gaussians.rotations, original_gaussians.rotations)
                ) 
                mse_losses.append(mse_loss)
                
        # Compute final loss (mean across target views)
        loss = torch.mean(torch.stack(mse_losses)) if mse_losses else torch.tensor(0.0, device=self.device)
        
        final_psnr_impr = np.mean(psnr_impr)
        final_ssim_impr = np.mean(ssim_impr)
        final_lpips_impr = np.mean(lpips_impr)
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
        
    
        self.log("train_loss", loss, on_step=True, sync_dist=True)
        self.logger.log_metrics({
            "train/loss": loss.item(),
            "train/psnr_improvement": final_psnr_impr,
            "train/ssim_improvement": final_ssim_impr,
            "train/lpips_improvement": final_lpips_impr,
            "train/psnr_mean": psnr_mv,
            "train/ssim_mean": ssim_mv,
            "train/lpips_mean": lpips_mv
        },
        step=self.global_step)

        return loss

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

        # --- Compute Splatting and Metrics for Each Target View ---
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        rendered_images = []
        
        for t_i in range(t):  # Iterate over target views
            output = self.decoder.forward(
                gaussians=original_gaussians,
                extrinsics=batch["target"]["extrinsics"][:, t_i].unsqueeze(1),
                intrinsics=batch["target"]["intrinsics"][:, t_i].unsqueeze(1),
                near=batch["target"]["near"][:, t_i].unsqueeze(1),
                far=batch["target"]["far"][:, t_i].unsqueeze(1),
                image_shape=(h, w),
                depth_mode=None,
                use_scale_and_rotation=True,
            )
            
            rgb_splatted = output.color[0][0]  # Extract the rendered view
            rendered_images.append(rgb_splatted)

            # Ground truth target view
            rgb_gt = batch["target"]["image"][0][t_i]

            # Compute Metrics
            psnr = compute_psnr(rgb_gt.unsqueeze(0), rgb_splatted.unsqueeze(0)).item()
            ssim = compute_ssim(rgb_gt.unsqueeze(0), rgb_splatted.unsqueeze(0)).item()
            lpips = compute_lpips(rgb_gt.unsqueeze(0), rgb_splatted.unsqueeze(0)).item()
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips)

        # --- Compute Mean Metrics Across Target Views ---
        psnr_mean = np.mean(psnr_values)
        ssim_mean = np.mean(ssim_values)
        lpips_mean = np.mean(lpips_values)

        self.logger.log_metrics({
            "val/psnr_mean": psnr_mean,
            "val/ssim_mean": ssim_mean,
            "val/lpips_mean": lpips
            },
        step=self.global_step)
    
        # --- Construct Comparison Image ---
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*batch["target"]["image"][0]), "Target (GT)"),
            add_label(vcat(*rendered_images), "Rendered (Original Gaussians)"),
        )

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # --- Render Projections ---
        projections = hcat(*render_projections(
                                original_gaussians,
                                256,
                                extra_label="(Original Gaussians)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
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
