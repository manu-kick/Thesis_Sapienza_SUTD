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
from ..misc.nn_module_tools import convert_to_buffer

from lpips import LPIPS

def mse_loss(pred, target):
    delta = pred - target
    return 1.0 * (delta**2).mean()


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Float[Tensor, ""]:
      
        loss = self.lpips.forward(
            rearrange(prediction, "b v c h w -> (b v) c h w"),
            rearrange(target, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return self.cfg.weight * loss.mean()



class ModelWrapper_KD_IMGS(LightningModule):
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

        self.lpips_loss = LossLpips(LossLpipsCfgWrapper(LossLpipsCfg(weight=0.05, apply_after_step=0)))
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
        _, _, r, _, _, _ = batch["refinement"]["image"].shape
        
        # Export this batch as a torch file in the /outputs/saved_batches
        # torch.save(batch, f"outputs/batches_refinement_UniformSampling/batch_{batch_idx}.pt")

        raw_gaussians = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        
        target_mv_output = self.decoder.forward(
            gaussians=raw_gaussians,
            extrinsics=batch["target"]["extrinsics"],
            intrinsics=batch["target"]["intrinsics"],
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
            depth_mode=None,
            use_scale_and_rotation=True,
        )
        
        raw_gaussians = Gaussians(
            means=repeat(raw_gaussians.means, "b g xyz -> (b v) g xyz", v=t), # bs,g,3
            harmonics=repeat(raw_gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=t), # bs,g,3,25
            opacities=repeat(raw_gaussians.opacities, "b g -> (b v) g", v=t),  # bs,g
            scales=repeat(raw_gaussians.scales, "b g xyz-> (b v) g xyz", v=t), # bs,g,3
            rotations=repeat(raw_gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=t), # bs,g,4
            covariances=repeat(raw_gaussians.covariances, "b g i j -> (b v) g i j", v=t),
        ) 
        
        
        if self.refiner is not None: # Refinement is enabled
            batch['refinement']['extrinsics'] = rearrange(batch['refinement']['extrinsics'], "b t r i j -> (b t) r i j", r=r) 
            batch['refinement']['intrinsics'] = rearrange(batch['refinement']['intrinsics'], "b t r i j -> (b t) r i j", r=r)
            batch['refinement']['image'] = rearrange(batch['refinement']['image'], "b t r c h w -> (b t) r c h w", r=r)
            batch['refinement']['index'] = rearrange(batch['refinement']['index'], "b t r -> (b t) r")
            batch["refinement"]['near'] = repeat(batch['refinement']['near'],'b t -> b t r', r=r)
            batch["refinement"]['far'] = repeat(batch['refinement']['far'],'b t -> b t r', r=r)
            
            batch['refinement']['near'] = repeat(batch['refinement']['near'], "b t r -> (b t) r")
            batch['refinement']['far'] = repeat(batch['refinement']['far'], "b t r -> (b t) r")    
            
    
            context_rearranged = rearrange(batch["context"]["image"], "b v c h w -> (b v) c h w")
            target_rearranged = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")
            
            psnr_impr , ssim_impr, lpips_impr = [], [], []
            refinment_losses = []
            refinement_target_predictions = []
            for t_i in range(b*t):  # Iterate over target views spread across batches
                current_target_gaussians = Gaussians(
                    means=raw_gaussians.means[t_i].unsqueeze(0).clone().detach(),
                    harmonics=raw_gaussians.harmonics[t_i].unsqueeze(0).clone().detach(),
                    opacities=raw_gaussians.opacities[t_i].unsqueeze(0).clone().detach(),
                    scales=raw_gaussians.scales[t_i].unsqueeze(0).clone().detach(),
                    rotations=raw_gaussians.rotations[t_i].unsqueeze(0).clone().detach(),
                    covariances=None
                )
                
                # Call the refiner
                psnr_improvement,ssim_improvement, lpips_improvement, _ = self.refiner.forward(
                    {   
                        "extrinsics": batch['refinement']['extrinsics'][t_i].unsqueeze(0), 
                        "intrinsics": batch['refinement']['intrinsics'][t_i].unsqueeze(0),
                        "near": batch['refinement']['near'][t_i].unsqueeze(0),
                        "far": batch['refinement']['far'][t_i].unsqueeze(0),
                        "image": batch['refinement']['image'][t_i].unsqueeze(0),
                    }, 
                    current_target_gaussians, # gaussian of
                    self.global_step
                )
                psnr_impr.append(psnr_improvement)
                ssim_impr.append(ssim_improvement)
                lpips_impr.append(lpips_improvement)

                # Get the refined Gaussians
                refined_gaussians = Gaussians(
                    means=self.refiner.means.clone().detach(), 
                    harmonics=self.refiner.harmonics.clone().detach(),
                    opacities=self.refiner.opacities.clone().detach(), 
                    scales=self.refiner.scales.clone().detach(),
                    rotations=self.refiner.rotations.clone().detach(),
                    covariances=None  # We use scales and rotations
                )
                
                # Splat the refinment cameras using the refined gaussians
                current_target_gaussians = Gaussians(
                    means=raw_gaussians.means[t_i].unsqueeze(0),
                    harmonics=raw_gaussians.harmonics[t_i].unsqueeze(0),
                    opacities=raw_gaussians.opacities[t_i].unsqueeze(0),
                    scales=raw_gaussians.scales[t_i].unsqueeze(0),
                    rotations=raw_gaussians.rotations[t_i].unsqueeze(0),
                    covariances=None
                )
                raw_mv_output = self.decoder.forward( #check if are still in the graph the raw gaussians
                    gaussians=current_target_gaussians,
                    extrinsics=batch["refinement"]["extrinsics"][t_i].unsqueeze(0),
                    intrinsics=batch["refinement"]["intrinsics"][t_i].unsqueeze(0),
                    near=batch["refinement"]["near"][t_i].unsqueeze(0), #
                    far=batch["refinement"]["far"][t_i].unsqueeze(0), #
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )
                refined_mv_output = self.decoder.forward(
                    gaussians=refined_gaussians,
                    extrinsics=batch["refinement"]["extrinsics"][t_i].unsqueeze(0),
                    intrinsics=batch["refinement"]["intrinsics"][t_i].unsqueeze(0),
                    near=batch["refinement"]["near"][t_i].unsqueeze(0),
                    far=batch["refinement"]["far"][t_i].unsqueeze(0),
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )
                
                # Loss Computation for the refinement of the current target view
                loss = 0
                loss += mse_loss(
                    refined_mv_output.color,
                    raw_mv_output.color
                )
                loss += self.lpips_loss(
                    refined_mv_output.color,
                    raw_mv_output.color
                )
                refinment_losses.append(loss)
                
                # Comparision for the target view and the refined target view
                index_refinement_labels = [f"{i}" for i in batch["refinement"]["index"][t_i]]
                index_refinement_labels = " ".join(index_refinement_labels)
                comparison = hcat(
                    add_label(vcat(*context_rearranged), "Context"),
                    add_label(vcat(*target_rearranged[t_i].unsqueeze(0)), "Target GT"),
                    add_label(vcat(*batch["refinement"]["image"][t_i]), "Refinement GT ("+index_refinement_labels+")"),
                    add_label(vcat(*raw_mv_output.color[0]), "MV Refinement prediction"),
                    add_label(vcat(*refined_mv_output.color[0]), "GS Refinement prediction"),
                )
                self.logger.log_image(
                    "Comparison refinement and target",
                    [prep_image(add_border(comparison))],
                    step=self.global_step,
                    caption=batch["scene"],
                )
                # continue
                
                final_psnr_impr = np.mean(psnr_impr)
                final_ssim_impr = np.mean(ssim_impr)
                final_lpips_impr = np.mean(lpips_impr)
                # Mean over the refinement losses
                final_refinment_loss = torch.mean(torch.stack(refinment_losses))
               
                # Log the distillation-dependent metrics
                self.logger.log_metrics({
                    "train/psnr_improvement": final_psnr_impr,
                    "train/ssim_improvement": final_ssim_impr,
                    "train/lpips_improvement": final_lpips_impr,
                    "train/refinement_loss": final_refinment_loss.item(),
                }, step=self.global_step)
            
        # Refinement is disabled, run mv
        # Loss Computation for the target view
        target_loss = 0
        for loss_fn in self.losses:
            loss_i = loss_fn.forward(target_mv_output, batch, None, self.global_step)
            self.logger.log_metrics({
                f"train/loss{loss_fn.name}": loss_i.item()
            },
            step=self.global_step
            )
            target_loss += loss_i
        self.log("train_loss", target_loss, sync_dist=True)
        psnr_mv = compute_psnr(
                rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                rearrange(target_mv_output.color, "b v c h w -> (b v) c h w"),
            ).mean().item()
        ssim_mv = compute_ssim(
            rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
            rearrange(target_mv_output.color,"b v c h w -> (b v) c h w"),
        ).mean().item()
        lpips_mv = compute_lpips(
            rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
            rearrange(target_mv_output.color,"b v c h w -> (b v) c h w"),
        ).mean().item() 
        
        if self.refiner is not None:
            total_loss = target_loss + self.refiner_cfg['refinement_lambda_loss'] * final_refinment_loss
            self.log("train/ref_loss", total_loss, sync_dist=True)
       
        self.logger.log_metrics({
            "train/loss": target_loss.item(),
            "train/psnr_mean": psnr_mv,
            "train/ssim_mean": ssim_mv,
            "train/lpips_mean": lpips_mv,
        }, step=self.global_step)
        
        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )
        
        # Construct comparison image for target.
        target_labels = [f"{i}" for i in batch["target"]["index"][0]]
        target_labels = " ".join(target_labels)
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*batch["target"]["image"][0]), "Target (Ground Truth "+target_labels+")"),
            add_label(vcat(*target_mv_output.color[0]), "MV Target prediction"),
        )
        self.logger.log_image(
            "Comparison context and target",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
            
        return total_loss if self.refiner is not None else target_loss
            


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
        output = self.decoder.forward(
            gaussians=original_gaussians,
            extrinsics=batch["target"]["extrinsics"],
            intrinsics=batch["target"]["intrinsics"],
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
            depth_mode=None,
            use_scale_and_rotation=True,
        ) 
        psnr_mean = compute_psnr(
            rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        ).mean().item()
        ssim_mean = compute_ssim(
            rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
            rearrange(output.color,"b v c h w -> (b v) c h w"),
        ).mean().item()
        lpips_mean = compute_lpips(
            rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
            rearrange(output.color,"b v c h w -> (b v) c h w"),
        ).mean().item()
            

        self.logger.log_metrics({
            "val/psnr_mean": psnr_mean,
            "val/ssim_mean": ssim_mean,
            "val/lpips_mean": lpips_mean
            },
        step=self.global_step)
        
        #Rearrange the target
        batch['target']['extrinsics'] = rearrange(batch['target']['extrinsics'], "b t i j -> (b t) i j")
        batch['target']['intrinsics'] = rearrange(batch['target']['intrinsics'], "b t i j -> (b t) i j")
        batch['target']['image'] = rearrange(batch['target']['image'], "b t c h w -> (b t) c h w")
        batch['target']['index'] = rearrange(batch['target']['index'], "b t -> (b t)")
        batch['target']['near'] = rearrange(batch['target']['near'], "b t-> (b t)")
        batch['target']['far'] = rearrange(batch['target']['far'], "b t -> (b t)")
        
        if self.refiner is not None: # Refinement is enabled
            # Obtain the refined gaussians for the target view
            r = batch["refinement"]["image"].shape[2]
            batch['refinement']['extrinsics'] = rearrange(batch['refinement']['extrinsics'], "b t r i j -> (b t) r i j", r=r) 
            batch['refinement']['intrinsics'] = rearrange(batch['refinement']['intrinsics'], "b t r i j -> (b t) r i j", r=r)
            batch['refinement']['image'] = rearrange(batch['refinement']['image'], "b t r c h w -> (b t) r c h w", r=r)
            batch['refinement']['index'] = rearrange(batch['refinement']['index'], "b t r -> (b t) r")
            batch['refinement']['near'] = rearrange(batch['refinement']['near'], "b r -> (b r)")
            batch['refinement']['far'] = rearrange(batch['refinement']['far'], "b r -> (b r)")    

    
            
            # Repeat the raw gaussian for the target views
            raw_gaussians = Gaussians(
                means=repeat(original_gaussians.means, "b g xyz -> (b v) g xyz", v=t), # bs,g,3
                harmonics=repeat(original_gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=t), # bs,g,3,25
                opacities=repeat(original_gaussians.opacities, "b g -> (b v) g", v=t),  # bs,g
                scales=repeat(original_gaussians.scales, "b g xyz-> (b v) g xyz", v=t), # bs,g,3
                rotations=repeat(original_gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=t), # bs,g,4
                covariances=repeat(original_gaussians.covariances, "b g i j -> (b v) g i j", v=t),
            ) 
            
            refined_images = []
            psnr_refined, ssim_refined, lpips_refined = [], [], []
            for t_i in range(b*t):  # Iterate over target views spread across batches
                current_target_gaussians = Gaussians(
                    means=raw_gaussians.means[t_i].unsqueeze(0).clone().detach(),
                    harmonics=raw_gaussians.harmonics[t_i].unsqueeze(0).clone().detach(),
                    opacities=raw_gaussians.opacities[t_i].unsqueeze(0).clone().detach(),
                    scales=raw_gaussians.scales[t_i].unsqueeze(0).clone().detach(),
                    rotations=raw_gaussians.rotations[t_i].unsqueeze(0).clone().detach(),
                    covariances=None
                )
                
                _, _, _, _ = self.refiner.forward(
                    {   
                        "extrinsics": batch['refinement']['extrinsics'][t_i].unsqueeze(0), 
                        "intrinsics": batch['refinement']['intrinsics'][t_i].unsqueeze(0),
                        "near": batch['refinement']['near'][t_i].unsqueeze(0),
                        "far": batch['refinement']['far'][t_i].unsqueeze(0),
                        "image": batch['refinement']['image'][t_i].unsqueeze(0),
                    }, 
                    current_target_gaussians, # gaussian of
                    self.global_step
                )
                
                # Get the refined Gaussians
                refined_gaussians = Gaussians(
                    means=self.refiner.means.clone().detach(), 
                    harmonics=self.refiner.harmonics.clone().detach(),
                    opacities=self.refiner.opacities.clone().detach(), 
                    scales=self.refiner.scales.clone().detach(),
                    rotations=self.refiner.rotations.clone().detach(),
                    covariances=None  # We use scales and rotations
                )
                
                refined_target_output = self.decoder.forward( 
                    gaussians=refined_gaussians,
                    extrinsics=batch["target"]["extrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    intrinsics=batch["target"]["intrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    near=batch["target"]["near"][t_i].unsqueeze(0).unsqueeze(0),
                    far=batch["target"]["far"][t_i].unsqueeze(0).unsqueeze(0),    
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )
                
                
                refined_images.append(refined_target_output.color[0])
                
                # Compute evaluation metrics
                psnr_refined.append(compute_psnr(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    refined_target_output.color[0]
                ).item())
            
                ssim_refined.append(compute_ssim(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    refined_target_output.color[0]
                ).item())
                
                lpips_refined.append(compute_lpips(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    refined_target_output.color[0]
                ).item())
                
            # Comparison for the target view and the refined target view
            index_refinement_labels = [f"{i}" for i in batch["refinement"]["index"][t_i]]
            index_refinement_labels = " ".join(index_refinement_labels)
            index_target_labels = [f"{i}" for i in batch["target"]["index"]]
            index_target_labels = " ".join(index_target_labels)
            index_context_labels = [f"{i}" for i in batch["context"]["index"][0]]
            index_context_labels = " ".join(index_context_labels)
            refined_images = torch.stack(refined_images)
            # Save the comparison
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"].squeeze(0)), "Context ("+index_context_labels+")"),
                add_label(vcat(*batch["target"]["image"]), "Target GT ("+index_target_labels+")"),
                add_label(vcat(*refined_images.squeeze(1)), "Refined Target prediction ("+index_refinement_labels+")"),
                add_label(vcat(*output.color[0]), "MV Target prediction"),
            )
            self.logger.log_image(
                "Validation Comparison max_bounds",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
            
            # Compute the average metrics over the target views for the refinement
            psnr_mean_refined = np.mean(psnr_refined)
            ssim_mean_refined = np.mean(ssim_refined)
            lpips_mean_refined = np.mean(lpips_refined)
            
            self.logger.log_metrics({
                "val/psnr_mean_refined": psnr_mean_refined,
                "val/ssim_mean_refined": ssim_mean_refined,
                "val/lpips_mean_refined": lpips_mean_refined
                },
            step=self.global_step)
        else:
            
            # Log the image comparison for the target view produced by the MV
            index_target_labels = [f"{i}" for i in batch["target"]["index"]]
            index_target_labels = " ".join(index_target_labels)
            index_context_labels = [f"{i}" for i in rearrange(batch['context']['index'],'b t -> (b t)')]
            index_context_labels = " ".join(index_context_labels)
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"].squeeze(0)), "Context ("+index_context_labels+")"),
                add_label(vcat(*batch["target"]["image"]), "Target GT ("+index_target_labels+")"),
                add_label(vcat(*output.color[0]), "MV Target prediction"),
            )
            self.logger.log_image(
                "Validation Comparison max_bounds",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )    
        
        self.log("val/psnr_mean", psnr_mean, sync_dist=True)

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
       
                
        if self.test_cfg.compute_scores:
            if "psnr" not in self.test_step_outputs:
                self.test_step_outputs["psnr"] = []
                psnr_mv = compute_psnr(
                    rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                    rearrange(output_mv.color, "b v c h w -> (b v) c h w"),
                ).mean().item()
                self.test_step_outputs["psnr"].append(psnr_mv)
            if "ssim" not in self.test_step_outputs:
                self.test_step_outputs["ssim"] = []
                ssim_mv = compute_ssim(
                    rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
                    rearrange(output_mv.color,"b v c h w -> (b v) c h w"),
                ).mean().item()
                self.test_step_outputs["ssim"].append(ssim_mv)
            if "lpips" not in self.test_step_outputs:
                self.test_step_outputs["lpips"] = []
                lpips_mv = compute_lpips(
                    rearrange(batch["target"]["image"],"b v c h w -> (b v) c h w"), 
                    rearrange(output_mv.color,"b v c h w -> (b v) c h w"),
                ).mean().item()
                self.test_step_outputs["lpips"].append(lpips_mv)
            
            
        if self.refiner is not None:
            
            batch['target']['extrinsics'] = rearrange(batch['target']['extrinsics'], "b t i j -> (b t) i j")
            batch['target']['intrinsics'] = rearrange(batch['target']['intrinsics'], "b t i j -> (b t) i j")
            batch['target']['image'] = rearrange(batch['target']['image'], "b t c h w -> (b t) c h w")
            batch['target']['index'] = rearrange(batch['target']['index'], "b t -> (b t)")
            batch['target']['near'] = rearrange(batch['target']['near'], "b t-> (b t)")
            batch['target']['far'] = rearrange(batch['target']['far'], "b t -> (b t)")
            
            # Obtain the refined gaussians for the target view
            r = batch["refinement"]["image"].shape[2]
            batch['refinement']['extrinsics'] = rearrange(batch['refinement']['extrinsics'], "b t r i j -> (b t) r i j", r=r) 
            batch['refinement']['intrinsics'] = rearrange(batch['refinement']['intrinsics'], "b t r i j -> (b t) r i j", r=r)
            batch['refinement']['image'] = rearrange(batch['refinement']['image'], "b t r c h w -> (b t) r c h w", r=r)
            batch['refinement']['index'] = rearrange(batch['refinement']['index'], "b t r -> (b t) r")
            batch['refinement']['near'] = rearrange(batch['refinement']['near'], "b r -> (b r)")
            batch['refinement']['far'] = rearrange(batch['refinement']['far'], "b r -> (b r)")   
            
            # Repeat the raw gaussian for the target views
            raw_gaussians = Gaussians(
                means=repeat(original_gaussians.means, "b g xyz -> (b v) g xyz", v=t), # bs,g,3
                harmonics=repeat(original_gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=t), # bs,g,3,25
                opacities=repeat(original_gaussians.opacities, "b g -> (b v) g", v=t),  # bs,g
                scales=repeat(original_gaussians.scales, "b g xyz-> (b v) g xyz", v=t), # bs,g,3
                rotations=repeat(original_gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=t), # bs,g,4
                covariances=repeat(original_gaussians.covariances, "b g i j -> (b v) g i j", v=t),
            ) 
            
            refined_images = []
            psnr_refined, ssim_refined, lpips_refined = [], [], []
            for t_i in range(t):  # Iterate over target views
                current_target_gaussians = Gaussians(
                    means=raw_gaussians.means[t_i].unsqueeze(0).clone().detach(),
                    harmonics=raw_gaussians.harmonics[t_i].unsqueeze(0).clone().detach(),
                    opacities=raw_gaussians.opacities[t_i].unsqueeze(0).clone().detach(),
                    scales=raw_gaussians.scales[t_i].unsqueeze(0).clone().detach(),
                    rotations=raw_gaussians.rotations[t_i].unsqueeze(0).clone().detach(),
                    covariances=None
                )
                
                _,_,_,_, = self.refiner.forward(
                    {   
                        "extrinsics": batch['refinement']['extrinsics'][t_i].unsqueeze(0), 
                        "intrinsics": batch['refinement']['intrinsics'][t_i].unsqueeze(0),
                        "near": batch['refinement']['near'][t_i].unsqueeze(0),
                        "far": batch['refinement']['far'][t_i].unsqueeze(0),
                        "image": batch['refinement']['image'][t_i].unsqueeze(0),
                    }, 
                    current_target_gaussians, 
                    self.global_step
                )
                
                refined_gaussians = Gaussians(
                    means=self.refiner.means.clone().detach(), 
                    harmonics=self.refiner.harmonics.clone().detach(),
                    opacities=self.refiner.opacities.clone().detach(), 
                    scales=self.refiner.scales.clone().detach(),
                    rotations=self.refiner.rotations.clone().detach(),
                    covariances=None  # We use scales and rotations
                )
                
                refined_target_output = self.decoder.forward( 
                    gaussians=refined_gaussians,
                    extrinsics=batch["target"]["extrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    intrinsics=batch["target"]["intrinsics"][t_i].unsqueeze(0).unsqueeze(0),
                    near=batch["target"]["near"][t_i].unsqueeze(0).unsqueeze(0),
                    far=batch["target"]["far"][t_i].unsqueeze(0).unsqueeze(0),    
                    image_shape=(h, w),
                    depth_mode=None,
                    use_scale_and_rotation=True,
                )                
                refined_images.append(refined_target_output.color[0])
                
                # Compute evaluation metrics
                psnr_refined.append(compute_psnr(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    refined_target_output.color[0]
                ).item())
            
                ssim_refined.append(compute_ssim(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    refined_target_output.color[0]
                ).item())
                
                lpips_refined.append(compute_lpips(
                    batch["target"]["image"][t_i].unsqueeze(0),
                    refined_target_output.color[0]
                ).item())
            
             
            # Compute the average metrics over the target views for the refinement
            self.test_step_outputs["psnr_refined"] = psnr_refined
            self.test_step_outputs["ssim_refined"] = ssim_refined
            self.test_step_outputs["lpips_refined"] = lpips_refined
                
                
                
                # ------------------------------

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

    
    def on_load_checkpoint(self, checkpoint):
        ignored_keys ={
            "refiner.means", "refiner.harmonics", "refiner.opacities", "refiner.scales", "refiner.rotations"
        }
        
        # Remove unwanted keys before loading state_dict
        for key in ignored_keys:
            if key in checkpoint["state_dict"]:
                del checkpoint["state_dict"][key]

        print("✅ Ignored unwanted keys:", ignored_keys)