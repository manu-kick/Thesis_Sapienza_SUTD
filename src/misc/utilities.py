import os
import numpy as np
from ..model.types import Gaussians
from typing import Union
from pathlib import Path
import os
import numpy as np
import torch  # only needed if you're using PyTorch tensors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from einops import rearrange


# Assuming Gaussians is defined elsewhere:
# class Gaussians:
#     means: Float[Tensor, "batch gaussian dim"]
#     covariances: Float[Tensor, "batch gaussian dim dim"]
#     harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
#     opacities: Float[Tensor, "batch gaussian"]

def save_gaussians(gaussians: "Gaussians", path: Union[str, Path]) -> None:
    """
    Save the parameters of a Gaussians object to an npz file.

    Parameters:
        gaussians: A Gaussians object with attributes 'means', 'covariances',
                   'harmonics', and 'opacities'.
        path: The file path (or Path object) where the parameters will be saved.
    """
    
    # Convert the path to a string if it's a Path object
    path_str = str(path)

    # Helper function to convert tensors (or any array-like) to numpy arrays.
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            # Detach and move to CPU if necessary before converting
            return x.detach().cpu().numpy()
        return np.array(x)

    # Prepare a dictionary with all Gaussian parameters.
    params = {
        "means": to_numpy(gaussians.means),
        "covariances": to_numpy(gaussians.covariances),
        "harmonics": to_numpy(gaussians.harmonics),
        "opacities": to_numpy(gaussians.opacities),
    }

    # Ensure that the directory for the path exists.
    os.makedirs(os.path.dirname(path_str), exist_ok=True)

    print(f"Saving Gaussian parameters to {path_str}")
    np.savez(path_str, **params)

def plot_camera(ax, K, R, t, color='b', scale=1.0):
    """Plot a camera in 3D space given its intrinsic and extrinsic parameters with a frustum."""
    # Define frustum points in normalized image space (image plane at z=1)
    img_corners = np.array([
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]  # Image plane corners
    ]).T  # Shape (3, 4)
    
    # Convert to world coordinates
    K_inv = np.linalg.inv(K)
    cam_corners = K_inv @ img_corners  # Transform image plane to camera space
    cam_corners = cam_corners / cam_corners[2]  # Normalize so z=1
    cam_corners *= scale  # Scale frustum

    # Convert to homogeneous coordinates
    cam_corners_h = np.vstack([cam_corners, np.ones((1, cam_corners.shape[1]))])

    # Transform to world space
    world_corners = (R @ cam_corners_h[:3, :]) + t.reshape(3, 1)  # Apply rotation and translation

    # Extract world coordinates
    img_plane = world_corners.T  # Image plane corners (4,3)
    cam_center = t.flatten()  # Camera center

    # Draw image plane as a filled polygon
    plane = Poly3DCollection([img_plane], alpha=0.2, color=color)
    ax.add_collection3d(plane)
    
    # Draw arrows for the frustum
    for corner in img_plane:
        ax.quiver(*cam_center, *(corner - cam_center), color=color, arrow_length_ratio=0.2, linewidth=1.5)

    # Plot the camera center with a marker
    ax.scatter(*cam_center, color=color, marker='o', s=100, edgecolors='black')

def visualize_batch(example):
    """Visualizes context, target, and refinement cameras in 3D space with frustums."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract data
    context_extrinsics = example["context"]["extrinsics"]  # Shape: [n_context, 4, 4]
    target_extrinsics = example["target"]["extrinsics"]  # Shape: [n_target, 4, 4]
    refinement_extrinsics = example["refinement"]["extrinsics"] if example["refinement"] else None  # Shape: [n_target, n_refinement, 4, 4]
    
    context_intrinsics = example["context"]["intrinsics"]
    target_intrinsics = example["target"]["intrinsics"]
    refinement_intrinsics = example["refinement"]["intrinsics"] if example["refinement"] else None

    # Define colors
    context_color = 'blue'
    target_colors = ['red', 'green', 'purple', 'orange', 'cyan']  # Assign different colors for targets
    refinement_colors = ['pink', 'lightgreen', 'lightblue', 'yellow', 'gray']  # Refinement colors per target

    # Plot context cameras
    for i in range(len(context_extrinsics)):
        RT = context_extrinsics[i]
        K = context_intrinsics[i]
        R = RT[:3, :3]
        t = RT[:3, 3]
        plot_camera(ax, K, R, t, color=context_color)

    # Plot target cameras
    for i in range(len(target_extrinsics)):
        RT = target_extrinsics[i]
        K = target_intrinsics[i]
        R = RT[:3, :3]
        t = RT[:3, 3]
        plot_camera(ax, K, R, t, color=target_colors[i % len(target_colors)])  # Cycle through target colors

        # Plot refinement cameras for this target
        if refinement_extrinsics is not None:
            for j in range(len(refinement_extrinsics[i])):  # Iterate over refinements for this target
                RT_ref = refinement_extrinsics[i, j]
                K_ref = refinement_intrinsics[i, j]
                R_ref = RT_ref[:3, :3]
                t_ref = RT_ref[:3, 3]
                plot_camera(ax, K_ref, R_ref, t_ref, color=refinement_colors[i % len(refinement_colors)])  # Same color per target

    # View settings
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses and Frustums (Batch Visualization)")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.show()







def cluster_gaussians(gaussians, num_clusters=5, exclude_z=False):
    means = gaussians[:, :,  :3]
    if exclude_z:
        means_xy = means[:, :, :2]
    harmonics = gaussians[:, :, 3:78]
    harmonics = rearrange(harmonics, 'b v (c d) -> b v c d', c=3)
    opacitites = gaussians[:, :, 78:79]
    scales = gaussians[:, :, 79:82]
    rotations = gaussians[:, :, 82:86]

    bs = means.shape[0]
    cluster_batches = []
    for i in range(bs):
        positions = means[i] if not exclude_z else means_xy[i]
        positions = positions.detach().cpu().numpy()   
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(positions)
        

        # Create the Gaussians object for each cluster
        cluster_gaussians = []
        for cluster_idx in range(num_clusters):
            cluster_mask = kmeans == cluster_idx
            cluster_positions = positions[cluster_mask]
            cluster_gaussians.append(Gaussians(
                means=torch.tensor(cluster_positions, device=means.device).unsqueeze(0) if not exclude_z else means[i,cluster_mask,:].unsqueeze(0),
                covariances=None,
                harmonics=harmonics[i, cluster_mask].unsqueeze(0),
                opacities=opacitites[i, cluster_mask].unsqueeze(0).squeeze(-1),
                scales=scales[i, cluster_mask].unsqueeze(0),
                rotations=rotations[i, cluster_mask].unsqueeze(0),
            ))
            
        cluster_batches.append(cluster_gaussians)
         
    return cluster_batches


def project_points_to_image(points_3d, intrinsics, extrinsics):
    """
    points_3d: (N, 3)
    intrinsics: (3, 3)
    extrinsics: (3, 4)
    Returns:
        projected_coords: (N, 2) in image space
        depths: (N,)
    """
    # Convert 3D points to homogeneous (N, 4)
    ones = torch.ones_like(points_3d[:, :1])
    points_3d_h = torch.cat([points_3d, ones], dim=-1)  # (N, 4)

    # Transform to camera coordinates
    cam_coords = (extrinsics @ points_3d_h.T).T  # (N, 3)

    # Project using intrinsics
    pixel_coords = (intrinsics @ cam_coords.T).T  # (N, 3)

    # Normalize by depth
    pixel_coords_2d = pixel_coords[:, :2] / (pixel_coords[:, 2:3] + 1e-5)  # avoid div by 0
    depths = cam_coords[:, 2]

    return pixel_coords_2d, depths

   