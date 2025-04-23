'''
Inspired by the datasetRE10K
'''
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from dacite import Config, from_dict
from ..evaluation.evaluation_index_generator import IndexEntry
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import random
import numpy as np




# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DColle

# DEBUGGER
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))


class KDPanopticViewSampler:
    def __init__(self, strategy, num_targets=4):
        self.strategy = strategy
        self.num_targets = num_targets
        if strategy not in ['cam_proximity']:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        if strategy == 'cam_proximity':
            self.sample = self.cam_proximity

    def cam_proximity(self, extrinsics, iteration=0, max_iterations=10000):
        """
        Sampling context and target views based on camera proximity.

        Args:
            extrinsics (torch.Tensor): [N, 4, 4] camera extrinsics.
            iteration (int): current training iteration (for boundary scheduling).
            max_iterations (int): maximum number of training iterations.

        Returns:
            context_idx: list of indices of 2 context cameras
            target_idx: list of indices of n target cameras
        """
        extrinsics = torch.tensor(extrinsics) if not isinstance(extrinsics, torch.Tensor) else extrinsics
        num_views = extrinsics.shape[0]

        cam_positions = extrinsics[:, :3, 3]  # [N, 3]

        # Compute pairwise distances
        distances = torch.cdist(cam_positions, cam_positions, p=2)  # [N, N]


        # --- Context View Selection ---
        first_ctx = random.randint(0, num_views - 1)
        context_idx = [first_ctx]
        
        # Compute boundary relative to first context
        dists_from_first = distances[first_ctx]
        mask = torch.ones_like(dists_from_first, dtype=bool)
        mask[first_ctx] = False

        min_d = dists_from_first[mask].min()
        max_d = dists_from_first[mask].max()

        boundary = min_d + (max_d - min_d) * (iteration / max_iterations)

        sorted_idx = torch.argsort(distances[first_ctx])  # ascending
        for idx in sorted_idx:
            if idx != first_ctx and dists_from_first[idx] >= boundary:
                context_idx.append(idx.item())
                break
            
        # Fallback if no second context found
        if len(context_idx) < 2:
            second_ctx = torch.argmax(distances[first_ctx]).item()
            context_idx.append(second_ctx)

        # Prevent degenerate case where both contexts are the same
        if context_idx[0] == context_idx[1]:
            second_ctx = torch.argmax(distances[context_idx[0]]).item()
            context_idx[1] = second_ctx

        # --- Target View Sampling: Closest to Segment ---
        pos1, pos2 = cam_positions[context_idx[0]], cam_positions[context_idx[1]]
        segment = pos2 - pos1
        seg_len_sq = segment.dot(segment)

        if seg_len_sq < 1e-6:
            print("Segment too small, fallback to random targets.")
            all_indices = list(set(range(num_views)) - set(context_idx))
            target_idx = random.sample(all_indices, min(self.num_targets, len(all_indices)))
            return context_idx, target_idx, boundary

        # Vector from pos1 to all cameras
        vecs = cam_positions - pos1.unsqueeze(0)  # [N, 3]

        # Projection scalar t for closest point on the segment
        t = (vecs @ segment) / seg_len_sq  # [N]
        t_clamped = torch.clamp(t, 0.0, 1.0).unsqueeze(1)  # Clamp to segment [0, 1]

        # Closest points on the segment
        closest_pts = pos1.unsqueeze(0) + t_clamped * segment  # [N, 3]

        # Distances to the segment
        dists_to_segment = torch.norm(cam_positions - closest_pts, dim=1)  # [N]

        # Mask out context cameras
        dists_to_segment[context_idx] = float('inf')

        # Get indices of the n closest
        sorted_indices = torch.argsort(dists_to_segment)
        target_idx = sorted_indices[:self.num_targets].tolist()


        return context_idx, target_idx, boundary

            
def load_gaussians(path):
    gaussians = {}
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    # file = dict(np.load(path))
    for folder in folders:
        gaussians[folder] = {}
        file_path = os.path.join(path,f'{folder}/params.npz')
        file = dict(np.load(file_path))
        for key in list(file.keys()):
            gaussians[folder][key] = file[key]
    
    return gaussians

@dataclass
class KDDatasetPanopticCfg(DatasetCfgCommon):
    name: Literal["kdpanoptic"]
    roots: list[Path]
    test_len: int
    max_fov: float
    test_times_per_scene: int
    baseline_epsilon: float
    make_baseline_1: bool
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    refinement: bool = False
    

class KDDatasetPanoptic(IterableDataset):
    def __init__(
        self,
        cfg: KDDatasetPanopticCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *args,  # <-- Allow extra positional arguments
        **kwargs,  # <-- Allow extra keyword arguments
    ) -> None:
        super().__init__()  # Ensure correct initialization of IterableDataset

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        chunks: list[Path]
        self.near = 1.0
        self.far = 100.0
        
        # Create the refined gaussians dict 
        self.refined_gaussians = load_gaussians(path=Path('datasets/panoptic_gaussian_parameters/Gaussian_KD_MVSPLAT'))
        
        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks= sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
            
        print(f'[INFO] -> Loaded {len(self.chunks)} chunks for panoptic')
        # Create the length attribute
        self.data_length = 0
        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path)
            self.data_length += len(chunk)
        print(f"Data length {self.data_length}")
        self.step_tracker = view_sampler.step_tracker
        self.global_step = self.step_tracker.get_step()
        self.view_sampler = KDPanopticViewSampler(strategy='cam_proximity', num_targets=2)
        if self.stage == 'test':
            print('Test Panoptic...')
            
    
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    
    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    def __iter__(self):
        
        if self.stage == 'train' or self.stage == 'val':
            for chunk_path in self.chunks:
                # Load the chunk.
                chunk = torch.load(chunk_path) 
                
                if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                    chunk = self.shuffle(chunk) 
                
                for example in chunk:
                    extrinsics, intrinsics = self.convert_poses(example["cameras"])
                    scene = example['key']
                    seq = scene.split('_')[0]
                    timestep = int(scene.split('_')[-1])
                    context_indices, target_indices, boundary = self.view_sampler.sample(extrinsics=extrinsics, iteration=self.global_step+1, max_iterations=300000)
                    
                    # context_indices = [9,4]
                    # target_indices = [18]

                    context_images = [example["images"][index] for index in context_indices]
                    context_images = self.convert_images(context_images)
                    target_images = [example["images"][index] for index in target_indices]
                    target_images = self.convert_images(target_images)
                    
                    context_extrinsics = extrinsics[context_indices]
                    if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                        a, b = context_extrinsics[:, :3, 3]
                        scale = (a - b).norm()
                        if scale < self.cfg.baseline_epsilon:
                            print(
                                f"Skipped {scene} because of insufficient baseline "
                                f"{scale:.6f}"
                            )
                            continue
                        extrinsics[:, :3, 3] /= scale
                    else:
                        scale = 1
                    
                    nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                    
                    ref = {
                        'means3D': self.refined_gaussians[seq]['means3D'][timestep],
                        'rotations': torch.nn.functional.normalize(torch.tensor(self.refined_gaussians[seq]['unnorm_rotations'][timestep])),
                        # fixed values for different frames
                        'opacities': torch.sigmoid(torch.tensor(self.refined_gaussians[seq]['logit_opacities'])), 
                        'scales': torch.exp(torch.tensor(self.refined_gaussians[seq]['log_scales'])),
                    }
                    
                   
                    # Construct the example
                    example = {
                            "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices)) / nf_scale,
                            "far": self.get_bound("far", len(context_indices)) / nf_scale,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices)) / nf_scale,
                            "far": self.get_bound("far", len(target_indices)) / nf_scale,
                            "index": target_indices,
                        },
                        "scene": scene,
                        "ctx_boundary" : boundary,
                        'refined_gaussians': ref,
                    }
                    
                    yield apply_crop_shim(example, tuple(self.cfg.image_shape))
        else:
            for chunk_path in self.chunks:
                chunk = torch.load(chunk_path) 
                
                for example in chunk:
                    extrinsics, intrinsics = self.convert_poses(example["cameras"])
                    scene = example['key']
                    # context_indices, target_indices, boundary = self.view_sampler.sample(extrinsics=extrinsics, iteration=self.global_step+1, max_iterations=300000)
                    context_indices = [24,25]
                    target_indices = [0]
                    context_images = [example["images"][index] for index in context_indices]
                    context_images = self.convert_images(context_images)
                    target_images = [example["images"][index] for index in target_indices]
                    target_images = self.convert_images(target_images)
                    
                    context_extrinsics = extrinsics[context_indices]
                    if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                        a, b = context_extrinsics[:, :3, 3]
                        scale = (a - b).norm()
                        if scale < self.cfg.baseline_epsilon:
                            print(
                                f"Skipped {scene} because of insufficient baseline "
                                f"{scale:.6f}"
                            )
                            continue
                        extrinsics[:, :3, 3] /= scale
                    else:
                        scale = 1
                    
                    nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                    
                   
                    # Construct the example
                    example = {
                            "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices)) / nf_scale,
                            "far": self.get_bound("far", len(context_indices)) / nf_scale,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices)) / nf_scale,
                            "far": self.get_bound("far", len(target_indices)) / nf_scale,
                            "index": target_indices,
                        },
                        "scene": scene,
                        "ctx_boundary" : -1
                    }
                    yield apply_crop_shim(example, tuple(self.cfg.image_shape))

                    

            raise Exception(f'Not implemented the stage {self.stage} for panoptic')
        

            
            
    def convert_poses( self, poses, ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        poses = torch.stack(poses)
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics     
                
    def convert_images( self, images: list[UInt8[Tensor, "..."]], ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
    
    def __len__(self) -> int:
        return self.data_length
        