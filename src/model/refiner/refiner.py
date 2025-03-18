import torch.nn as nn
import torch
from einops import rearrange
from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
from tqdm import tqdm
from einops import rearrange, repeat
import wandb
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Literal
import wandb
from src.model.types import Gaussians

@dataclass
class RefinerCfg:
    name: Literal["refiner"]
    num_steps: int
    patience: int = 15 # for the early stopping
    min_delta: float = 1e-4 # for the early stopping
    enable_early_stopping: bool = True # 🔥 Enable early stopping
    wandb_log = True
    
    
    
lrs_HPO = { #coming from HPO
    'means': 0.000029975,
    'harmonics': 0.014955,
    'opacities': 0.019618,
    'scales': 0.00046502,
    'rotations': 0.0044503
}

class Refiner(nn.Module):
    def __init__(self, cfg , decoder, losses, lrs: dict = lrs_HPO):
        super(Refiner, self).__init__()
        self.cfg = cfg
        self.decoder = decoder
        self.losses = losses       
        self.lrs = lrs 
        self.debug = False
        
    def initialize_parameters(self, gaussians):
        self.means = nn.Parameter(gaussians.means.clone().detach(), requires_grad=True)
        self.harmonics = nn.Parameter(gaussians.harmonics.clone().detach(), requires_grad=True)
        self.opacities = nn.Parameter(gaussians.opacities.clone().detach(), requires_grad=True)
        self.scales = nn.Parameter(gaussians.scales.clone().detach(), requires_grad=True) 
        self.rotations = nn.Parameter(gaussians.rotations.clone().detach(), requires_grad=True) 
        
        
            
    def get_optimizer(self):
        param_groups = [
            {
                'params': [self.means],
                'lr': self.lrs['means'],  # Example LR; pick a suitable value
                'name': 'means'
            },
            {
                'params': [self.harmonics],
                'lr': self.lrs['harmonics'],
                'name': 'harmonics'
            },
            {
                'params': [self.opacities],
                'lr': self.lrs['opacities'],
                'name': 'opacities'
            },
            {
                'params': [self.scales],
                'lr': self.lrs['scales'],
                'name': 'scales'
            },
            {
                'params': [self.rotations],
                'lr': self.lrs['rotations'],
                'name': 'rotations'
            }
        ]
        return torch.optim.Adam(param_groups, eps=1e-15)
    
    def forward(self, batch, gaussians, step, return_history = False):
        history = []
        with torch.inference_mode(False):  # 🔥 Force autograd to work
            with torch.enable_grad():  # ✅ Ensure gradients are enabled
                self.initialize_parameters(gaussians)
                self.optimizer = self.get_optimizer()
                
                # Batch contains a set of extrinsics, intrinsics, near, far and image
                _, v, _, _ = batch['extrinsics'].shape
                batch['near'] = repeat(batch['near'], 'b -> b v', v=v)
                batch['far'] = repeat(batch['far'], 'b -> b v ', v=v)

                # 🔥 Early stopping variables
                best_loss = float("inf")
                patience_counter = 0
                patience = self.cfg.patience
                min_delta = self.cfg.min_delta

                psnr_t0 = []
                ssim_t0 = []
                lpips_t0 = []

                for i in tqdm(range(1000), desc=f"Refinement Progress | Step {step}"):
                    # Update `gaussians` parameters from learnable parameters
                    splat_gaussians = Gaussians(
                        means=self.means, 
                        harmonics=self.harmonics, 
                        opacities=self.opacities, 
                        scales=self.scales, 
                        rotations=self.rotations,
                        covariances=None  # We use scales and rotations
                    )
                    output = self.decoder.forward(
                        splat_gaussians,
                        batch['extrinsics'],
                        batch['intrinsics'],
                        batch['near'],
                        batch['far'],
                        (batch['image'].shape[-2], batch['image'].shape[-1]),
                        depth_mode=None,
                        use_scale_and_rotation=True,
                    )
                        
                    total_loss = self.compute_loss(output, batch['image'])
                    total_loss.backward()
                    
                    # Compute Metrics
                    psnr_per_view = compute_psnr(
                        rearrange(batch['image'], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    
                    ssim_per_view = compute_ssim(
                        rearrange(batch['image'], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    
                    lpips_per_view = compute_lpips(
                        rearrange(batch['image'], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    
                    if i == 0:  # Save initial values
                        psnr_t0 = psnr_per_view.tolist()
                        ssim_t0 = ssim_per_view.tolist()
                        lpips_t0 = lpips_per_view.tolist()
                        
                    with torch.no_grad():
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        
                    if i%10 == 0 and return_history and i > 0:
                        # Append the gaussian parameters to the history
                        history.append(Gaussians(
                            means=self.means.clone().detach(),
                            harmonics=self.harmonics.clone().detach(),
                            opacities=self.opacities.clone().detach(),
                            scales=self.scales.clone().detach(),
                            rotations=self.rotations.clone().detach(),
                            covariances=None
                            )
                        )
                        

                    # 🔥 EARLY STOPPING CHECK 🔥
                    if self.cfg.enable_early_stopping:
                        current_loss = total_loss.item()
                        if current_loss < best_loss - min_delta:
                            best_loss = current_loss  # Save best loss
                            patience_counter = 0  # Reset patience counter
                        else:
                            patience_counter += 1  # Increment patience counter

                        # 🔥 Stop if no improvement for `patience` steps
                        if patience_counter >= patience:
                            if self.debug:
                                print(f"🛑 Early stopping at step {i}, best loss: {best_loss:.6f}")
                            break

                # Compute Improvement (Final - Initial)
                psnr_improvement_per_view = [psnr_f - psnr_0 for psnr_f, psnr_0 in zip(psnr_per_view.tolist(), psnr_t0)]
                ssim_improvement_per_view = [ssim_f - ssim_0 for ssim_f, ssim_0 in zip(ssim_per_view.tolist(), ssim_t0)]
                lpips_improvement_per_view = [lpips_f - lpips_0 for lpips_f, lpips_0 in zip(lpips_per_view.tolist(), lpips_t0)]

                # Compute Mean Improvement Across Views
                psnr_improvement_mean = np.mean(psnr_improvement_per_view)
                ssim_improvement_mean = np.mean(ssim_improvement_per_view)
                lpips_improvement_mean = np.mean(lpips_improvement_per_view)

        return psnr_improvement_mean, ssim_improvement_mean, lpips_improvement_mean, history
    
    def compute_loss(self, output, target):
        total_loss = torch.tensor(0.0, device=output.color.device, requires_grad=True)
        delta = output.color - target
        total_loss = (delta**2).mean()
    
        return total_loss
        