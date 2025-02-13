import subprocess
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
import argparse
from tqdm import tqdm
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input dtu raw directory")
parser.add_argument("--output_dir", type=str, help="output directory")
args = parser.parse_args()

SEQUENCE = 'basketball'
BASE_DATASET_DIR = 'datasets/panoptic'
OUTPUT_DIR = Path('datasets/panoptic_torch')
timestep = 0

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

def build_camera_info(timestep,md,cam):
    w = 640
    h = 360
    # Search the index of the camera in the md['cam_id']['timestep']
    idx = md['cam_id'][timestep].index(cam)
    k = md['k'][timestep][idx]
    extrinsics = md['w2c'][timestep][idx]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    intrinsics = [
                [fx / w, 0.0, cx / w],
                [0.0, fy / h, cy / h],
                [0.0, 0.0, 1.0],
            ]

    near = 1
    far = 100
    
    return extrinsics, intrinsics, near, far

def get_size(path: Path) -> int:
    """Get file or folder size in bytes, cross-platform."""
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0  # Return 0 if path does not exist

def load_metadata(intrinsics, extrinsics):
    cameras = []
    
    for cam_id, intr in intrinsics.items():
        saved_fx = intr[0][0]
        saved_fy = intr[1][1]
        saved_cx = intr[0][2]
        saved_cy = intr[1][2]
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        w2c = extrinsics[cam_id]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))
        
    cam_ids = torch.tensor(list(intrinsics.keys()), dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    
    return {
        'cam_ids': cam_ids,
        'cameras': cameras
    }
    
if __name__ == "__main__":
    # we only use Panoptic for testing, not for training
    test_views = [0, 10, 15, 30]
    train_views = [i for i in range(31) if i not in test_views]
    data = {}
    chunk_size = 0
    chunk_index = 0
    chunk = []
    
    for stage in ["train", "test"]:
        selected_views = train_views if stage == "train" else test_views
        metadata = json.load(open(os.path.join(BASE_DATASET_DIR, f"{SEQUENCE}/{stage}_meta.json"), 'r'))
        IMAGE_DIR = os.path.join(BASE_DATASET_DIR, f"{SEQUENCE}/ims")   
        
        intrinsics = {}
        extrinsics = {}
        images = {}

        for cam in selected_views:
            extr, intr, near, far = build_camera_info(timestep, metadata, cam)

            # Store camera parameters
            intrinsics[cam] = intr
            extrinsics[cam] = torch.tensor(extr, dtype=torch.float32)

            # Load image as raw bytes
            image_path = os.path.join(BASE_DATASET_DIR, f"{SEQUENCE}/ims/{cam}/{timestep:06d}.jpg")
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found. Skipping camera {cam}...")
                continue

            images[cam] = torch.tensor(np.memmap(image_path, dtype="uint8", mode="r"), dtype=torch.uint8)

        # Sort all data by camera ID
        sorted_cam_ids = sorted(intrinsics.keys())

        # Create the final dictionary in the required format
        example = load_metadata(intrinsics, extrinsics)
        example["images"] = [images[cam_id] for cam_id in sorted_cam_ids]  # Ordered list of images
        example["key"] = f"{SEQUENCE}_{stage}"  # e.g., "basketball_train"

        # Save the single torch file
        torch_file_path = Path(OUTPUT_DIR) / f"{SEQUENCE}_{stage}.torch"
        torch_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([example], torch_file_path)

        print(f"Saved torch file: {torch_file_path} ({torch_file_path.stat().st_size / 1e6:.2f} MB).")
    


