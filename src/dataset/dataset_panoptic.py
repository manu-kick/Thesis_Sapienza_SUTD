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


# DEBUGGER
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))


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
        
        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks= sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
            
        if self.stage == 'test':
            # Import the evaluation index for panoptic
            dacite_config = Config(cast=[tuple])
            path_eval_index = Path(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))) / "assets" / "evaluation_index_basketball_nctx2.json"
            with path_eval_index.open("r") as f:
                self.eval_idxs = {
                    k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                    for k, v in json.load(f).items()
                }
            print(f"Loaded evaluation index from {path_eval_index}.")
    
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
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)
            
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]
        
        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path)
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)
            
            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)
                
            if self.stage == "test":
                todo_key = list(self.eval_idxs.keys())
                # we have only one example in the chunk for now
                # cam_ids[], cameras[], images[], key
                
                extrinsics, intrinsics = self.convert_poses(chunk["cameras"])
                for key in todo_key:
                    # retrieve the example in the chunk using the scene 
                    try:
                        context_indices, target_indices = self.view_sampler.sample(
                            key,
                            extrinsics,
                            intrinsics,
                        )
                    except ValueError:
                        # Skip because the example doesn't have enough frames.
                        raise ValueError(f"Skipped {key} because of insufficient frames.")
                        continue
                    
                    if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                        continue
                    
                    # Load the images.
                    context_images = [ chunk["images"][index.item()] for index in context_indices ]
                    context_images = self.convert_images(context_images)
                    target_images = [ chunk["images"][index.item()] for index in target_indices ]
                    target_images = self.convert_images(target_images)
                    
                    # Resize the world to make the baseline 1.
                    context_extrinsics = extrinsics[context_indices]
                    if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                        a, b = context_extrinsics[:, :3, 3]
                        scale = (a - b).norm()
                        if scale < self.cfg.baseline_epsilon:
                            print(
                                f"Skipped {key} because of insufficient baseline "
                                f"{scale:.6f}"
                            )
                            continue
                        extrinsics[:, :3, 3] /= scale
                    else:
                        scale = 1
                        
                    
                    nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
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
                        "scene": key,
                    }
                    yield apply_crop_shim(example, tuple(self.cfg.image_shape))
                    
                        
            else:
                raise NotImplementedError(f"Stage {self.stage} not implemented for Panoptic dataset.")
                times_per_scene = self.cfg.test_times_per_scene
                for run_idx in range(int(times_per_scene * len(chunk))):
                    example = chunk
                    extrinsics, intrinsics = self.convert_poses(example["cameras"])
                    scene = example["key"]
                    try:
                        context_indices, target_indices = self.view_sampler.sample(
                            scene,
                            extrinsics,
                            intrinsics,
                        )
                    except ValueError:
                        # Skip because the example doesn't have enough frames.
                        raise ValueError(f"Skipped {scene} because of insufficient frames.")
                        continue
                    
                    if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                        continue
                    
            
            
            
            # Check On the shapes, maybe not useful here
            
            
                
                
           
                if self.stage == "train" and self.cfg.augment:
                    raise NotImplementedError("Stage training still to be fixed.")
                    example = apply_augmentation_shim(example)
                    
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

            
            
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
            min(len(self.index.keys()) * self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )