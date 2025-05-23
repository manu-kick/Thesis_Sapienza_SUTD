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

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from .view_sampler.view_sampler_evaluation import ViewSamplerEvaluation
from .view_sampler import get_view_sampler

import torch.distributed as dist


# DEBUGGER
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    refinement_cfg: dict
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    refinement: bool = False

class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler
    refinement_view_sampler: ViewSampler | None

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        refinement_view_sampler: ViewSampler | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            # make sure that the root exists (debugger adaptation)
            if not root.exists():
                # add to the absoulte prefix the folders 'mvsplat' and 'src' then the path
                root = Path(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))) / root
                 
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
            
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
            
        if self.cfg.refinement:
            self.refinement_view_sampler = refinement_view_sampler

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # To enable data sharding across multiple gpus
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # example_idx = 0

        
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.test_times_per_scene
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]

                # Context and target indices.
                try:
                    context_indices, target_indices,  = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue
                
                # Refinement data selection
                if self.cfg.refinement:
                    refinement_indices = self.refinement_view_sampler.sample(
                        scene,
                        extrinsics,
                        target_indices if self.cfg.refinement_cfg['name'] != 'refinement_uniform_sampling' else context_indices, # The unifmorm sampling is basedon the context
                    )
                    
                    refinement_intr = []
                    refinement_extr = []
                    refinement_img = []
                    
                    # Cases: [RefinementViewSamplerContext]
                    if len(refinement_indices.shape) == 1: 
                        refinement_indices = repeat(refinement_indices, "r -> t r", t=target_indices.shape[0]) 
                        
                    for i in range(refinement_indices.shape[0]): # for each target view
                        for j in range(refinement_indices.shape[1]): # for each refinement view
                            refinement_intr.append(intrinsics[refinement_indices[i,j]])
                            refinement_extr.append(extrinsics[refinement_indices[i,j]])
                            refinement_img.append(example["images"][refinement_indices[i,j]])
                    
                    refinement_intrinsics = torch.stack(refinement_intr) 
                    refinement_extrinsics = torch.stack(refinement_extr)
                    refinement_images = self.convert_images(refinement_img)
                    
                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    if self.stage == 'val':
                        print(f"Skip {self.stage} ==> scene, FOV too wide!")
                    continue

                # Load the images.
                context_images = [example["images"][index.item()] for index in context_indices]
                context_images = self.convert_images(context_images)
                target_images = [example["images"][index.item()] for index in target_indices]
                target_images = self.convert_images(target_images)
               
                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                refinement_images_invalid = refinement_images.shape[1:] != (3, 360, 640) if self.cfg.refinement else False
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid or refinement_images_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
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
                target_count = target_indices.shape[0]
                refinement_count = self.cfg.refinement_cfg['num_refinement_views'] if self.cfg.refinement_cfg['name'] != "refinement_context" else refinement_indices.shape[1]
                
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
                    "refinement": {
                        "extrinsics": rearrange(refinement_extrinsics, "(t r) h w -> t r h w", t=target_count, r=refinement_count),
                        "intrinsics": rearrange(refinement_intrinsics, "(t r) h w -> t r h w", t=target_count, r=refinement_count),
                        "image": rearrange(refinement_images, "(t r) c h w -> t r c h w", t=target_count, r=refinement_count),
                        "near": self.get_bound("near", target_count) / nf_scale ,
                        "far": self.get_bound("far", target_count) / nf_scale,
                        "index": refinement_indices,
                    } if self.cfg.refinement else None,
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment and self.cfg.refinement==False: #Note when we refine we don't augment otherwise we screw up  
                    example = apply_augmentation_shim(example)
                # if example_idx % world_size == rank:
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))
                # example_idx += 1

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape #143 18

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

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
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

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                if not root.exists():
                    # add to the absoulte prefix the folders 'mvsplat' and 'src' then the path
                    root = Path(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))) / root
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )
