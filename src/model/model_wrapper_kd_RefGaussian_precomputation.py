from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import os

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

def prepare_gaussians(gaussians):
        """
        transform the gaussians in a single tensor.
        """
        return torch.cat([
            gaussians.means, # bs,g,3
            rearrange(gaussians.harmonics,'b g c h -> b g (c h)',b=gaussians.harmonics.shape[0], g=gaussians.harmonics.shape[1], c=gaussians.harmonics.shape[2], h=gaussians.harmonics.shape[3] ), # bs,g,3,75
            gaussians.opacities.unsqueeze(-1), # bs,g,1
            gaussians.scales, # bs,g,3
            gaussians.rotations #bs,g,4
        ], dim=-1)

class ModelWrapper_KD_RefinementPrecomputation(LightningModule):
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
    
    def on_train_start(self):
        self.refined_gaussians_save_path = Path('/home/tesista10/Thesis_Sapienza_SUTD/datasets/re10k/train/precomputed_refinement/precomputed_gaussians.torch')
        
        # Ensure we don't accidentally append to an old file
        if os.path.exists(self.refined_gaussians_save_path):
            raise Exception('Ensure to delete old file first!')
            os.remove(self.refined_gaussians_save_path)

        print(f"[INFO] Will save refined Gaussians to: {self.refined_gaussians_save_path}")

    
    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, t, _ , h, w = batch["target"]["image"].shape
        _, _, r, _, _, _ = batch["refinement"]["image"].shape
        
      
        raw_gaussians = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        ) #[b,g, attribute]
        
        
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
        
        
        # For each element of the batch, apply the refinement using raw gaussians just for the first target image
        # from the second to the last target image for a given batch use the previous refined gaussians to obtain the new set of 
        # refined gaussian for the given target
        
        batched_refined = []
        for b_i in range(b):
            per_batch_refined_gaussians = []
            for t_i in range(t):
                if t_i == 0:
                    current_raw_gaussians = Gaussians(
                        means=raw_gaussians.means[b_i].unsqueeze(0).clone().detach(),
                        harmonics=raw_gaussians.harmonics[b_i].unsqueeze(0).clone().detach(),
                        opacities=raw_gaussians.opacities[b_i].unsqueeze(0).clone().detach(),
                        scales=raw_gaussians.scales[b_i].unsqueeze(0).clone().detach(),
                        rotations=raw_gaussians.rotations[b_i].unsqueeze(0).clone().detach(),
                        covariances=None
                    )
                else:
                    del current_raw_gaussians
                    current_raw_gaussians = Gaussians(
                        means=self.refiner.means.clone().detach(), 
                        harmonics=self.refiner.harmonics.clone().detach(),
                        opacities=self.refiner.opacities.clone().detach(), 
                        scales=self.refiner.scales.clone().detach(),
                        rotations=self.refiner.rotations.clone().detach(),
                        covariances=None  # We use scales and rotations
                    )
                    
                _,_,_,_ = self.refiner.forward(
                    {   
                        "extrinsics": batch['refinement']['extrinsics'][t_i].unsqueeze(0), 
                        "intrinsics": batch['refinement']['intrinsics'][t_i].unsqueeze(0),
                        "near": batch['refinement']['near'][t_i].unsqueeze(0),
                        "far": batch['refinement']['far'][t_i].unsqueeze(0),
                        "image": batch['refinement']['image'][t_i].unsqueeze(0),
                    }, 
                    current_raw_gaussians, # gaussian of
                    self.global_step
                )
                
                per_batch_refined_gaussians.append(
                    Gaussians(
                        means=self.refiner.means.clone().detach(), 
                        harmonics=self.refiner.harmonics.clone().detach(),
                        opacities=self.refiner.opacities.clone().detach(), 
                        scales=self.refiner.scales.clone().detach(),
                        rotations=self.refiner.rotations.clone().detach(),
                        covariances=None  # We use scales and rotations
                    )
                )
                    
            # Stack the gaussian for the current batch
            batched_refined.append(
                {
                    'key' : batch['scene'][b_i],
                    'context_idx' : batch['context']['index'][b_i].tolist(),
                    'refinement_idx' : batch['context']['index'][b_i].tolist(),
                    'target_idx' : batch['target']['index'][b_i].tolist(),
                    'precomputed_refined_gaussians' : Gaussians(
                        means=torch.stack([g.means for g in per_batch_refined_gaussians]),
                        harmonics=torch.stack([g.harmonics for g in per_batch_refined_gaussians]),
                        opacities=torch.stack([g.opacities for g in per_batch_refined_gaussians]),
                        scales=torch.stack([g.scales for g in per_batch_refined_gaussians]),
                        rotations=torch.stack([g.rotations for g in per_batch_refined_gaussians]),
                        covariances=None
                    )
                }
            )
            print(f'Sample {b_i} done!!')
            
        # Append the current batch to the torch file and save
        existing_data = []
        if os.path.exists(self.refined_gaussians_save_path):
            try:
                existing_data = torch.load(self.refined_gaussians_save_path)
            except Exception as e:
                print(f"[WARNING] Could not load existing file: {e}")
        
        if not isinstance(existing_data, list):
            existing_data = []

        existing_data.extend(batched_refined)
        torch.save(existing_data, self.refined_gaussians_save_path)
        print(f"[INFO] Saved {len(batched_refined)} samples to {self.refined_gaussians_save_path}")

        
    
        return None
            

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