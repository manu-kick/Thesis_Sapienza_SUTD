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


@dataclass
class RefinerCfg:
    name: Literal["refiner"]
    num_steps: int
    patience: int = 15 # for the early stopping
    min_delta: float = 1e-4 # for the early stopping
    
    
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
    
    def forward(self, batch, gaussians, _):
        with torch.inference_mode(False):  # 🔥 Force autograd to work
            with torch.enable_grad():  # ✅ Ensure gradients are enabled
                self.initialize_parameters(gaussians)
                self.optimizer = self.get_optimizer()
                
                # Batch contains a set of extrinsics, intrinsics, near, far and image
                b,v,_,_ = batch['extrinsics'].shape
                missing_dim = int(v / batch['near'].shape[1])
                batch['near'] = repeat(batch['near'], 'b v -> b (v d)', d=missing_dim)
                batch['far'] = repeat(batch['far'], 'b v -> b (v d)', d=missing_dim)
                
                # 🔥 Early stopping variables
                best_loss = float("inf")
                patience_counter = 0
                patience = self.cfg.patience
                min_delta = self.cfg.min_delta

                psnr_t0 = 0
                ssim_t0 = 0
                lpips_t0 = 0
                for i in tqdm(range(10000), desc="Refinement Progress"):
                    # Update `gaussians` parameters from learnable parameters
                    gaussians.means = self.means
                    gaussians.harmonics = self.harmonics
                    gaussians.opacities = self.opacities
                    gaussians.scales = self.scales
                    gaussians.rotations = self.rotations
                    
                
                    output = self.decoder.forward(
                        gaussians,
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
                    
                    psnr_probabilistic = compute_psnr(
                        rearrange(batch['image'], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    
                    ssim = compute_ssim(
                        rearrange(batch['image'], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    
                    lpips = compute_lpips(
                        rearrange(batch['image'], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                    )
                    
                    
                    if i == 0:
                        psnr_t0 = psnr_probabilistic.mean().item()
                        ssim_t0 = ssim.mean().item()
                        lpips_t0 = lpips.mean().item()
                        

                    with torch.no_grad():
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    # 🔥 EARLY STOPPING CHECK 🔥
                    current_loss = total_loss.item()
                    if current_loss < best_loss - min_delta:
                        best_loss = current_loss  # Save best loss
                        patience_counter = 0  # Reset patience counter
                    else:
                        patience_counter += 1  # Increment patience counter

                    # 🔥 Stop if no improvement for `patience` steps
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping at step {i}, best loss: {best_loss:.6f}")
                        break

                
                psnr_final = psnr_probabilistic.mean().item()
                ssim_final = ssim.mean().item()
                lpips_final = lpips.mean().item()
                print(f"PSNR: {psnr_final:.3f} | SSIM: {ssim_final:.3f} | LPIPS: {lpips_final:.3f}")
                print(f"PSNR Improvement: {psnr_final - psnr_t0:.3f} | SSIM Improvement: {ssim_final - ssim_t0:.3f} | LPIPS Improvement: {lpips_final - lpips_t0:.3f}")
                
        return psnr_probabilistic, total_loss
    
    def compute_loss(self, output, target):
        total_loss = torch.tensor(0.0, device=output.color.device, requires_grad=True)
        delta = output.color - target
        total_loss = (delta**2).mean()
    
        return total_loss
        