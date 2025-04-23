# This is the file that create a single 'cameras' dict for a given scene to then be used with
# the scene visualizer to see those camera in the 3d space

from pathlib import Path
import torch
from torch import Tensor
from PIL import Image
from jaxtyping import Float, UInt8, Int64
import torchvision.transforms as tf
from io import BytesIO
from typing import Literal
from einops import rearrange, repeat
import os
import sys
from typing import Callable, Literal, TypedDict, Optional
import numpy as np
from compute_fov_overlap import compute_fov_overlap_matrix

# --------------------------------------------------------
# ------------------------ SHIM CODE ---------------------
# --------------------------------------------------------
class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    refinement: Optional[BatchedViews]  # Now optional
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    refinement: Optional[UnbatchedViews]  # Now optional
    scene: str

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews

def rescale(
    image,
    shape,
):
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images,
    intrinsics,
    shape,
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics

def rescale_and_crop(
    images,
    intrinsics,
    shape,
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)

def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }

def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Apply crop shim to context, target, and optionally refinement data."""
    cropped_example = {
        **example,
        "cameras": apply_crop_shim_to_views(example["cameras"], shape),
    }

    return cropped_example
# --------------------------------------------------------
# ------------------------ SHIM CODE ---------------------
# --------------------------------------------------------



chunks = []
roots = ["datasets/panoptic_torch/test"]
image_shape = (360, 640)

def convert_images(images: list[UInt8[Tensor, "..."]], ) -> Float[Tensor, "batch 3 height width"]:
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(tf.ToTensor()(image))
    return torch.stack(torch_images)

def get_bound(
    bound: Literal["near", "far"],
    num_views: int,
) -> Float[Tensor, " view"]:
    val = 1.0 if bound == "near" else 100.0
    value = torch.tensor(val, dtype=torch.float32)
    return repeat(value, "-> v", v=num_views)

def convert_poses( poses, ) -> tuple[
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


for root in roots:
    root = Path(root)
    root_chunks = sorted(
        [path for path in root.iterdir() if path.suffix == ".torch"]
    )
    chunks.extend(root_chunks)
print(f'[INFO] -> Loaded {len(chunks)} chunks for panoptic')

for chunk_path in chunks:
    chunk = torch.load(chunk_path)
    print(f"[INFO] -> Chunk {chunk_path}: {len(chunk)} samples")
    for example in chunk:
        # Example has keys dict_keys(['cam_ids', 'cameras', 'images', 'key'])
        scene = example["key"]
        num_views = len(example["images"])

        # Convert all images
        images = convert_images(example["images"])

        # Convert all poses
        extrinsics, intrinsics = convert_poses(example["cameras"])

        # Resize the world if required
        scale = 1.0
        nf_scale = 1.0

        # Yield unified example
        example = apply_crop_shim(
            {
                "cameras": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": images,
                    "near": get_bound("near", num_views) / nf_scale,
                    "far": get_bound("far", num_views) / nf_scale,
                    "index": list(range(num_views)),
                },
                "scene": scene,
            },
            tuple(image_shape),
        )
        
        # Save the example as a .torch file
        torch.save(example, f"outputs/panoptic_scenes/{scene}.torch")
        print(f"[INFO] -> Saved example {scene} ")
        
        
        
        # Compute the FOV overlapping between the cameras
        fov_overlap = compute_fov_overlap_matrix(intrinsics, extrinsics)
        
        print(f"[INFO] -> FOV overlap matrix for {scene}:")\
        
        exit(0)