import torch.nn as nn
import torch
from einops import rearrange
from src.evaluation.metrics import compute_psnr
from tqdm import tqdm
import wandb
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Literal


@dataclass
class RefinerCfg:
    name: Literal["refiner"]
    num_steps: int
    
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
        self.scales = nn.Parameter(gaussians.scales_rotated.clone().detach(), requires_grad=True) 
        self.rotations = nn.Parameter(gaussians.rotations_rotated.clone().detach(), requires_grad=True) 
            
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
        # You could also set a global default lr and override it per group:
        # return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        return torch.optim.Adam(param_groups, eps=1e-15)
    
    def forward(self, batch, gaussians, global_step):
        self.initialize_parameters(gaussians)
        self.optimizer = self.get_optimizer()
        
        for i in tqdm(range(self.cfg.num_steps), desc="Refinement Progress"):
            # Update `gaussians` parameters from learnable parameters
            gaussians.means = self.means
            gaussians.harmonics = self.harmonics
            gaussians.opacities = self.opacities
            gaussians.scales = self.scales
            gaussians.rotations = self.rotations
            
            # Batch contains a set of extrinsics, intrinsics, near, far and image
            output = self.decoder.forward(
                gaussians,
                batch['extrinsics'],
                batch['intrinsics'],
                batch['near'],
                batch['far'],
                (batch['image'].shape[-2], batch['image'].shape[-1]),
                batch_size=batch['image'].shape[0],
                views=batch['image'].shape[1],
                depth_mode=None,
            )
            
            print("Output shape: ", output.shape)
            
        return 0 # for now