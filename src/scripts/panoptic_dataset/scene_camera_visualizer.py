# This file plots the cameras in 3d Space for the panoptic dataset's example

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from pathlib import Path

DRAW_POINTING_DIRECTION = True

def get_label(color):
    if color == "blue":
        return "Camera"
    else:
        return "Unknown"

def plot_camera(ax, K, R, t, color='b', scale=1.0, size=100):
    img_corners = np.array([
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]).T

    K_inv = np.linalg.inv(K)
    cam_corners = K_inv @ img_corners
    cam_corners /= cam_corners[2]
    cam_corners *= scale
    cam_corners_h = np.vstack([cam_corners, np.ones((1, cam_corners.shape[1]))])
    world_corners = (R @ cam_corners_h[:3, :]) + t.reshape(3, 1)

    img_plane = world_corners.T
    cam_center = t.flatten()

    if DRAW_POINTING_DIRECTION:
        direction = R[:, 2] * scale * 1.5
        ax.quiver(*cam_center, *direction, color=color, arrow_length_ratio=0.3, linewidth=2)

    ax.scatter(*cam_center, color=color, marker='o', s=size, edgecolors='black')

def visualize_cameras(K_list, RT_list, colors, selected_batch):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    camera_positions = []
    legend_handles = {}

    for i, (K, RT) in enumerate(zip(K_list, RT_list)):
        R = RT[:3, :3]
        t = RT[:3, 3].reshape(3, 1)
        plot_camera(ax, K, R, t, color=colors[i], size=60)
        camera_positions.append(t.flatten())

        if colors[i] not in legend_handles:
            legend_handles[colors[i]] = ax.scatter([], [], [], color=colors[i], label=get_label(colors[i]))

    camera_positions = np.array(camera_positions)
    if len(camera_positions) > 0:
        centroid = np.mean(camera_positions, axis=0)
        min_bounds = np.min(camera_positions, axis=0)
        max_bounds = np.max(camera_positions, axis=0)
        padding = np.max(max_bounds - min_bounds) * 0.2
        ax.set_xlim(min_bounds[0] - padding, max_bounds[0] + padding)
        ax.set_ylim(min_bounds[1] - padding, max_bounds[1] + padding)
        ax.set_zlim(min_bounds[2] - padding, max_bounds[2] + padding)
        ax.view_init(elev=20, azim=30)

    ax.legend(handles=legend_handles.values(), loc='upper right', title="Camera Types")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Batch {selected_batch} Visualization")
    plt.show(block=True)

# --- MAIN ENTRY ---
selected_batch = 0
TORCH_FILE = Path(f"/Users/emanuelerucci/Desktop/Sapienza/Altri progetti/tesi/panoptic_scenes/softball_0.torch")

file = torch.load(TORCH_FILE, map_location='cpu')

# Unified camera view structure
extrinsics = file["cameras"]["extrinsics"].cpu().numpy()
intrinsics = file["cameras"]["intrinsics"].cpu().numpy()

# Prepare camera lists
K_list = []
RT_list = []
colors = []

for i in range(len(extrinsics)):
    K_list.append(intrinsics[i])
    RT_list.append(extrinsics[i])
    colors.append("blue")  # Unified "cameras" are all blue by default

# Debug index info
indices = file["cameras"]["index"]
print(f"Camera indices: {indices}")

visualize_cameras(K_list, RT_list, colors, selected_batch)
