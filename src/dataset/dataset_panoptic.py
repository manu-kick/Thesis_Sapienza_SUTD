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

sequences = {
    'train' : ['basketball','boxes','football','juggle'],
    'test'  : ['softball','tennis'],
    'val'  : ['softball','tennis']
} 



class PanopticViewSampler:
    def __init__(self, strategy, num_targets=4):
        self.strategy = strategy
        self.num_targets = num_targets

        if strategy not in ['cam_proximity','cam_proximity_with_syntetic_extrinsics']:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        if strategy == 'cam_proximity':
            self.sample = self.cam_proximity
        if strategy == 'cam_proximity_with_syntetic_extrinsics':
            self.sample = self.cam_proximity_with_syntetic_extrinsics
            

    def init_camera(self, y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
        """
        Initialize a synthetic camera.

        Args:
            y_angle (float): Unused in this simple version (could be rotation later).
            center_dist (float): Distance from origin along X axis.
            cam_height (float): Height of camera in Y.
            f_ratio (float): Normalized focal ratio (relative to image size).

        Returns:
            k (torch.Tensor): [1,3,3] Normalized intrinsic matrix.
        """
        

        # Define normalized intrinsic
        fx = f_ratio
        fy = f_ratio
        cx = 0.5
        cy = 0.5

        k = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]

        return k


    def cam_proximity_with_syntetic_extrinsics(self, extrinsics, iteration=0, max_iterations=10000):
        k = self.init_camera()
        extrinsics = torch.tensor(extrinsics) if not isinstance(extrinsics, torch.Tensor) else extrinsics
        num_views = extrinsics.shape[0]
        cam_positions = extrinsics[:, :3, 3]

        # Step 1: Randomly choose the first context camera
        first_ctx = random.randint(0, num_views - 1)
        dists_from_first = torch.norm(cam_positions - cam_positions[first_ctx], dim=1)
        
        # Step 2: Select the second context based on boundary
        min_d, max_d = dists_from_first[dists_from_first > 0].min(), dists_from_first.max()
        boundary = min_d + (max_d - min_d) * (iteration / max_iterations)

        second_ctx = None
        for idx in torch.argsort(dists_from_first):
            if idx != first_ctx and dists_from_first[idx] >= boundary:
                second_ctx = idx.item()
                break
        if second_ctx is None:
            second_ctx = torch.argmax(dists_from_first).item()

        context_idx = [first_ctx, second_ctx]

        # Step 3: Sample synthetic extrinsics along the segment
        pos1, pos2 = cam_positions[first_ctx], cam_positions[second_ctx]
        direction = pos2 - pos1
        rot1 = extrinsics[first_ctx][:3, :3]
        rot2 = extrinsics[second_ctx][:3, :3]
        q1 = rotation_matrix_to_quaternion(rot1)
        q2 = rotation_matrix_to_quaternion(rot2)
        
        synthetic_extrinsics = []
        synthetic_count = 20
        for i in range(1,synthetic_count):
            t = i / (synthetic_count)
            interp_position = (1 - t) * pos1 + t * pos2
            q_interp = slerp(q1, q2, t)
            rot_interp = quaternion_to_rotation_matrix(q_interp)
            mid_extrinsic = torch.eye(4)
            mid_extrinsic[:3, :3] = rot_interp
            mid_extrinsic[:3, 3] = interp_position
            synthetic_extrinsics.append(mid_extrinsic)

        # Step 4: Select one synthetic extrinsic as a target
        synthetic_extrinsics = torch.stack(synthetic_extrinsics)
        # synthetic_target_idx = random.randint(0, synthetic_extrinsics.shape[0] - 1)
        # selected_target = synthetic_extrinsics[synthetic_target_idx].unsqueeze(0)

        # Step 4.1: Select only one syntetic extrinsinc choosing one index from the syntetic extrinsix stacked, choose it basedon the current iteration
        # when the iteration is at the start, choos the first, when is at the at choose the last.
        # Considering as the total number of available extrinsics, the syntehtic + the other context
        # Also return a flag into the dict that is a signal that, can be used the other context if the iteration in the last bin of iters (the one dedicated to the other context) 
        # TODO
        # Step 4.1: Select one synthetic extrinsic based on iteration progress
        total_extrinsics = synthetic_extrinsics.shape[0] + 1  # synthetic + second context
        # Map iteration to index range [0, total_extrinsics - 1]
        extrinsic_idx = int((iteration / max_iterations) * (total_extrinsics - 1))
        extrinsic_idx = min(extrinsic_idx, synthetic_extrinsics.shape[0])  # cap to max synthetic index

        if extrinsic_idx == synthetic_extrinsics.shape[0]:
            # Final step: use the second context camera itself
            synthetic_extrinsics = extrinsics[second_ctx].unsqueeze(0)
            use_second_ctx = True
        else:
            synthetic_extrinsics = synthetic_extrinsics[extrinsic_idx].unsqueeze(0)
            use_second_ctx = False
        
        # Step 5: Use the same logic as cam_proximity to select real targets
        if use_second_ctx:
            seg_end = pos2
        else:
            seg_end = synthetic_extrinsics[0, :3, 3]

        segment = seg_end - pos1
        seg_len_sq = segment.dot(segment)

        vecs = cam_positions - pos1.unsqueeze(0)
        t = (vecs @ segment) / seg_len_sq
        t_clamped = torch.clamp(t, 0.0, 1.0).unsqueeze(1)
        closest_pts = pos1.unsqueeze(0) + t_clamped * segment
        dists_to_segment = torch.norm(cam_positions - closest_pts, dim=1)
        # dists_to_segment[context_idx] = float('inf')  # exclude context views
        dists_to_segment[torch.tensor(context_idx, device=dists_to_segment.device)] = float('inf')

        sorted_indices = torch.argsort(dists_to_segment)
        target_idx = sorted_indices[:self.num_targets].tolist()


        return {
                'context_indices': context_idx, 
                'target_indices': target_idx, 
                'ctx_boundary': boundary, 
                'synthetic_extrinsics': synthetic_extrinsics,
                'synthetic_intrinsics': k, #not used
                'use_second_ctx': use_second_ctx,
                'synthetic_extrinsic_idx' : extrinsic_idx
        }

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


        return {
            'context_idx': context_idx,
            'target_idx': target_idx,
            'ctx_boundary': boundary
        }

            
def load_gaussians(path, stage):
    gaussians = {}
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    folders = [f for f in folders if f in sequences[stage]]
    # file = dict(np.load(path))
    print("Loading the precomputed gaussians")
    for folder in folders:
        gaussians[folder] = {}
        file_path = os.path.join(path,f'{folder}/params.npz')
        file = dict(np.load(file_path))
        for key in list(file.keys()):
            gaussians[folder][key] = file[key]
        print(f"\tParameters for --> {folder} loaded")
    
    # Check for each scene that the number of gaussian is the same (example checking the first dimension of means3D)
    # get the minumum and random sample the gaussian to have the same number of gaussians over all the scenes
    # Find the minimum number of Gaussians across all scenes
    min_g = min([gaussians[folder]['means3D'].shape[1] for folder in folders])

    # Now, for each scene, randomly sample min_g Gaussians
    for folder in folders:
        num_gaussians = gaussians[folder]['means3D'].shape[1]
        if num_gaussians != min_g:
            # Generate random indices to sample
            indices = np.random.permutation(num_gaussians)[:min_g]
            
            # sample random min_g gaussian over the dimension 1 of mean3d
            gaussians[folder]['means3D'] = gaussians[folder]['means3D'][:,indices,:]
            gaussians[folder]['rgb_colors'] = gaussians[folder]['rgb_colors'][:,indices,:]
            gaussians[folder]['seg_colors'] = gaussians[folder]['seg_colors'][indices,:]
            gaussians[folder]['unnorm_rotations'] = gaussians[folder]['unnorm_rotations'][:,indices,:]
            gaussians[folder]['logit_opacities'] = gaussians[folder]['logit_opacities'][indices,:]
            gaussians[folder]['log_scales'] = gaussians[folder]['log_scales'][indices,:]

            # means3D has shape 150 gaussians 3
            # rgb_colors 150 gaussians 3    
            # seg_colors gaussian 3
            # unnorm_rotations 150 gaussian 4
            # logit_opacities gaussian 1
            # log_scales gaussian 3
            
    return gaussians

@dataclass
class DatasetPanopticCfg(DatasetCfgCommon):
    name: Literal["panoptic"]
    roots: list[Path]
    test_len: int
    max_fov: float
    test_times_per_scene: int
    baseline_epsilon: float
    make_baseline_1: bool
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    refinement: bool = False
    
    
    

class DatasetPanoptic(IterableDataset):
    def __init__(
        self,
        cfg: DatasetPanopticCfg,
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
        self.w = 256
        self.h = 144
        
        # Create the refined gaussians dict 
        self.refined_gaussians = load_gaussians(path=Path('datasets/panoptic_gaussian_parameters/Gaussian_KD_MVSPLAT'), stage=self.data_stage)
        
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
        # self.view_sampler = PanopticViewSampler(strategy='cam_proximity', num_targets=2)
        self.view_sampler_strategy = 'cam_proximity_with_syntetic_extrinsics'
        self.view_sampler = PanopticViewSampler(strategy=self.view_sampler_strategy, num_targets=2)
        
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
                    
                    result = self.view_sampler.sample(extrinsics=extrinsics, iteration=self.global_step+1, max_iterations=300000)
                    context_indices = result['context_indices']
                    target_indices = result['target_indices']
                    
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
                        'colors_precomp': self.refined_gaussians[seq]['rgb_colors'][timestep]
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
                        'refined_gaussians': ref,
                    }
                    
                    # Trick to add all different info coming out from the sampling
                    keys = list(result.keys())
                    for key in keys:
                        if key not in ['context_indices','target_indices']:
                            example[key] = result[key]
                    
                    
                    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    Input: [3,3] tensor
    Output: [4] tensor
    """
    m = R
    t = m.trace()
    if t > 0.0:
        s = torch.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return torch.tensor([w, x, y, z], dtype=torch.float32)

def slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions.
    Args:
        q0 (Tensor): (4,)
        q1 (Tensor): (4,)
        t (float): interpolation factor [0,1]
    Returns:
        Tensor: interpolated quaternion (4,)
    """
    q0 = q0 / q0.norm()
    q1 = q1 / q1.norm()

    dot = (q0 * q1).sum()

    # If dot product is negative, invert one quaternion to take the shorter path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Quaternions are very close, use linear interpolation
        result = q0 + t*(q1 - q0)
        return result / result.norm()

    theta_0 = torch.acos(dot)         # angle between input quaternions
    theta = theta_0 * t               # angle at interpolation

    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)
     
def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (w, x, y, z) to rotation matrix (3x3).
    """
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=torch.float32)
    return R
