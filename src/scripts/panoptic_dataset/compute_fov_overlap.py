import torch
import numpy as np
from einops import repeat
from typing import Tuple

def generate_camera_rays(intrinsic: torch.Tensor, extrinsic: torch.Tensor, image_size: Tuple[int, int], grid_res=16):
    """
    Generate world-space rays from a camera's intrinsics and extrinsics.
    """
    H, W = image_size
    u = torch.linspace(0, W - 1, grid_res)
    v = torch.linspace(0, H - 1, grid_res)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    pixels = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).reshape(-1, 3).T  # (3, N)

    K_inv = torch.inverse(intrinsic)
    cam_rays = K_inv @ pixels  # Camera-space rays
    cam_rays = cam_rays / cam_rays.norm(dim=0, keepdim=True)

    R = extrinsic[:3, :3]
    rays_world = R @ cam_rays  # (3, N)
    origin = extrinsic[:3, 3:4]  # (3, 1)

    return origin, rays_world

def rays_in_fov(
    origin_i, rays_i, extrinsic_j, intrinsic_j, image_size: Tuple[int, int]
) -> float:
    """
    Estimate how many of the rays_i from origin_i fall inside camera j's FoV.
    """
    H, W = image_size
    N = rays_i.shape[1]
    
    # Sample points along rays
    d = 3.0  # Fixed depth to intersect frustum plane
    pts_world = origin_i + rays_i * d  # (3, N)

    # Transform points into camera j’s coordinate system
    R_j = extrinsic_j[:3, :3]
    t_j = extrinsic_j[:3, 3:4]
    pts_cam = R_j @ (pts_world - t_j)  # (3, N)

    # Filter out points behind camera
    valid = pts_cam[2] > 0
    pts_cam = pts_cam[:, valid]

    if pts_cam.shape[1] == 0:
        return 0.0

    # Project to image plane
    K = intrinsic_j
    pts_2d = K @ pts_cam  # (3, N)
    pts_2d = pts_2d[:2] / pts_2d[2:]  # normalize

    x, y = pts_2d
    in_x = (x >= 0) & (x < W)
    in_y = (y >= 0) & (y < H)
    in_view = in_x & in_y

    return in_view.sum().item() / N


def compute_fov_overlap_matrix(intrinsics, extrinsics, image_size=(360, 640), grid_res=16):
    N = len(intrinsics)
    fov_matrix = np.zeros((N, N))

    for i in range(N):
        origin_i, rays_i = generate_camera_rays(
            intrinsics[i], extrinsics[i], image_size, grid_res=grid_res
        )
        for j in range(N):
            if i == j:
                continue

            overlap = rays_in_fov(
                origin_i,
                rays_i,
                extrinsics[j],
                intrinsics[j],
                image_size,
            )
            fov_matrix[i, j] = overlap * 100  # percentage

    return fov_matrix
