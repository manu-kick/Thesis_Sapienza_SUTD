''' 
    This differ from the convert panoptic 1 because this gives as output a unique chunck file placing in the same chunk
    all the test and train cameras
'''
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


SEQUENCES_TRAIN = ['basketball','boxes','football','juggle']
SEQUENCES_TEST = ['softball','tennis']
SEQUENCES_ALL = {
   'train' : SEQUENCES_TRAIN, 
   'test' : SEQUENCES_TEST
}
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

def construct_camera_info(extrinsics, intrinsics):
    camera = torch.empty(18)
    camera[0] = intrinsics[0][0]
    camera[1] = intrinsics[1][1]
    camera[2] = intrinsics[0][2]
    camera[3] = intrinsics[1][2]
    camera[4] = 0.0
    camera[5] = 0.0
    camera[6:] = extrinsics[:3].flatten()
    
    return camera
    
    
if __name__ == "__main__":
    # Define camera sets
    test_views = [0, 10, 15, 30]
    train_views = [i for i in range(31) if i not in test_views]

    for generation_stage in ["train", "test"]:
        SEQUENCES = SEQUENCES_ALL[generation_stage]
        
        all_intrinsics = {}
        all_extrinsics = {}
        all_images = {}
        dataset = []
        
        # Process both sets and combine them
        for SEQUENCE in SEQUENCES:
            all_intrinsics[SEQUENCE] = {}
            all_extrinsics[SEQUENCE] = {}
            all_images[SEQUENCE] = {}
            
            for t in range(150):
                all_intrinsics[SEQUENCE][t] = {}
                all_extrinsics[SEQUENCE][t] = {}
                all_images[SEQUENCE][t] = {}
            
            for stage in ["train", "test"]:
                selected_views = train_views if stage == "train" else test_views
                metadata = json.load(open(os.path.join(BASE_DATASET_DIR, f"{SEQUENCE}/{stage}_meta.json"), 'r'))
                frames = len(metadata['k'])
                
                for t in range(frames):    
                    for cam in selected_views:
                        extr, intr, near, far = build_camera_info(t, metadata, cam)

                        # Store camera parameters
                        all_intrinsics[SEQUENCE][t][cam] = intr
                        all_extrinsics[SEQUENCE][t][cam] = torch.tensor(extr, dtype=torch.float32)

                        # Load image as raw bytes
                        image_path = os.path.join(BASE_DATASET_DIR, f"{SEQUENCE}/ims/{cam}/{t:06d}.jpg")
                        if not os.path.exists(image_path):
                            print(f"Warning: Image {image_path} not found. Skipping camera {cam}...")
                            continue

                        all_images[SEQUENCE][t][cam] = torch.tensor(np.array(np.memmap(image_path, dtype="uint8", mode="r")), dtype=torch.uint8)

            # After processing all cams for all frames:
            for t in range(frames):
                sorted_cam_ids = sorted(all_intrinsics[SEQUENCE][t].keys())

                # Some timesteps might be missing images (e.g., from skipped cams)
                if len(sorted_cam_ids) == 0:
                    continue

                try:
                    sample_t = {
                        "cam_ids": torch.tensor(sorted_cam_ids, dtype=torch.int64),
                        "cameras": [construct_camera_info(all_extrinsics[SEQUENCE][t][cam], all_intrinsics[SEQUENCE][t][cam]) for cam in sorted_cam_ids],
                        "images": [all_images[SEQUENCE][t][cam] for cam in sorted_cam_ids],
                        "key": f"{SEQUENCE}_{t}"
                    }
                    dataset.append(sample_t)
                except KeyError as e:
                    print(f"Missing data for SEQUENCE {SEQUENCE}, frame {t}, skipping. Error: {e}")

            print(f'{SEQUENCE} done')
        

        # Save as a single `.torch` file
        torch_file_path = Path(OUTPUT_DIR) / f"{'panoptic'}_{generation_stage}.torch"
        torch_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, torch_file_path)

        print(f"Saved torch file: {torch_file_path} ({torch_file_path.stat().st_size / 1e6:.2f} MB).")

    
    