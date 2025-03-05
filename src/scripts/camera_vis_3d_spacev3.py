import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from pathlib import Path

DRAW_POINTING_DIRECTION = False

def get_label(color):
    """Returns a label for the color."""
    if color == "blue":
        return "Context"
    elif color == "green":
        return "Target"
    elif color == "purple":
        return "Refinement"
    else:
        return "Unknown"
    
def plot_camera(ax, K, R, t, color='b', scale=1.0, size=100):
    """Plot a camera in 3D space given its intrinsic and extrinsic parameters with a frustum."""
    # Define frustum points in normalized image space (image plane at z=1)
    img_corners = np.array([
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]  # Image plane corners
    ]).T  # Shape (3, 4)
    
    # Convert to world coordinates
    K_inv = np.linalg.inv(K)
    cam_corners = K_inv @ img_corners  # Transform image plane to camera space
    cam_corners /= cam_corners[2]  # Normalize so z=1
    cam_corners *= scale  # Scale frustum

    cam_corners_h = np.vstack([cam_corners, np.ones((1, cam_corners.shape[1]))])

    # Transform to world space
    world_corners = (R @ cam_corners_h[:3, :]) + t.reshape(3, 1)  # Apply rotation and translation

    # Extract world coordinates
    img_plane = world_corners.T  # Image plane corners (4,3)
    cam_center = t.flatten()  # Camera center

    if DRAW_POINTING_DIRECTION:
        # Draw a direction arrow from the camera center forward
        direction = R[:, 2] * scale * 1.5  # Camera direction (Z-axis in world space)
        ax.quiver(*cam_center, *direction, color=color, arrow_length_ratio=0.3, linewidth=2)

    # Plot the camera center with a marker (smaller for refinement cameras)
    ax.scatter(*cam_center, color=color, marker='o', s=size, edgecolors='black')

def visualize_cameras(K_list, RT_list, colors, selected_batch):
    """Visualizes multiple cameras in 3D space with frustums."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    camera_positions = []  # Store camera positions
    legend_handles = {}  # Store legend handles for color labels

    # if target selected =  all, RT.shape = 4x4
    # if target selected = integer idx, RT.shape = 1x4x4
    for i, (K, RT) in enumerate(zip(K_list, RT_list)):
        R = RT[:3, :3]  # 3x3 rotation matrix
        t = RT[:3, 3].reshape(3, 1)  # 3x1 translation vector
        
        # Reduce size for refinement cameras
        # size = 40 if colors[i] == "purple" else 100  # Refinement cameras are smaller
        size = 40
        
        plot_camera(ax, K, R, t, color=colors[i], size=size)  # Assign color and size
        camera_positions.append(t.flatten())  # Collect camera centers

        # Add legend entry only once per color
        if colors[i] not in legend_handles:
            legend_handles[colors[i]] = ax.scatter([], [], [], color=colors[i], label=get_label(colors[i]))

    camera_positions = np.array(camera_positions)
    if len(camera_positions) > 0:
        # Compute centroid (mean of all camera positions)
        centroid = np.mean(camera_positions, axis=0)

        # Compute bounding box (min/max coordinates)
        min_bounds = np.min(camera_positions, axis=0)
        max_bounds = np.max(camera_positions, axis=0)

        # Determine axis limits dynamically with padding
        padding = np.max(max_bounds - min_bounds) * 0.2  # 20% padding
        ax.set_xlim(min_bounds[0] - padding, max_bounds[0] + padding)
        ax.set_ylim(min_bounds[1] - padding, max_bounds[1] + padding)
        ax.set_zlim(min_bounds[2] - padding, max_bounds[2] + padding)

        # Adjust view to look towards the centroid
        ax.view_init(elev=20, azim=30)  # Adjust viewing angles as needed

    # Color Legend: Blue (Context), Green (Target), Purple (Refinement)
    ax.legend(handles=legend_handles.values(), loc='upper right', title="Camera Types")

    # View settings
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Batch {selected_batch} Visualization")
    plt.show(block=True)

    

# Load the .pt file
selected_batch = 1
selected_target = 'all' # idx | 'all' Select the first target camera
TORCH_FILE = Path(f'C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/outputs/saved_batches/batch_{selected_batch}.pt')
file = torch.load(TORCH_FILE, map_location='cpu')

# Extract extrinsics and intrinsics from context, target, and refinement cameras
context_extrinsics = file["context"]["extrinsics"].squeeze(0).cpu().numpy()  # Shape: (n_context, 4, 4)
context_intrinsics = file["context"]["intrinsics"].squeeze(0).cpu().numpy()  # Shape: (n_context, 3, 3)

if selected_target != 'all':
    target_extrinsics = file["target"]["extrinsics"].squeeze(0)[selected_target].unsqueeze(0).cpu().numpy()  # Shape: (n_target, 4, 4)
    target_intrinsics = file["target"]["intrinsics"].squeeze(0)[selected_target].unsqueeze(0).cpu().numpy()  # Shape: (n_target, 3, 3)
    refinement_extrinsics = file["refinement"]["extrinsics"].squeeze(0)[selected_target].unsqueeze(0).cpu().numpy()  # Shape: (n_target, n_refinement, 4, 4)
    refinement_intrinsics = file["refinement"]["intrinsics"].squeeze(0)[selected_target].unsqueeze(0).cpu().numpy()  # Shape: (n_target, n_refinement, 3, 3)
else:
    target_extrinsics = file["target"]["extrinsics"].squeeze(0).cpu().numpy()  # Shape: (n_target, 4, 4)
    target_intrinsics = file["target"]["intrinsics"].squeeze(0).cpu().numpy()
    refinement_extrinsics = file["refinement"]["extrinsics"].squeeze(0).cpu().numpy()  # Shape: (n_target, n_refinement, 4, 4) 
    refinement_intrinsics = file["refinement"]["intrinsics"].squeeze(0).cpu().numpy()
    

# Flatten refinement cameras: (n_target, n_refinement, 4, 4) → (n_target * n_refinement, 4, 4)
refinement_extrinsics = refinement_extrinsics.reshape(-1, 4, 4) if len(refinement_extrinsics) > 0 else []
refinement_intrinsics = refinement_intrinsics.reshape(-1, 3, 3) if len(refinement_intrinsics) > 0 else []

# Prepare lists for visualization
K_list = []
RT_list = []
colors = []

# Assign colors
context_color = "blue"
target_color = "green"
refinement_color = "purple"

# Add context cameras
for i in range(len(context_extrinsics)):
    K_list.append(context_intrinsics[i])
    RT_list.append(context_extrinsics[i])
    colors.append(context_color)

# Add target cameras
for i in range(len(target_extrinsics)):
    K_list.append(target_intrinsics[i])
    RT_list.append(target_extrinsics[i])
    colors.append(target_color)

# Add refinement cameras
for i in range(len(refinement_extrinsics)):
    K_list.append(refinement_intrinsics[i])
    RT_list.append(refinement_extrinsics[i])
    colors.append(refinement_color)

# -------------- Visualize -----------------------
target_idx = file['target']['index'].reshape(-1).tolist()
ref_idx = file['refinement']['index'].reshape(-1).tolist()
context_idx = file['context']['index'].reshape(-1).tolist()
# Check if overlap between target and refinement indices
overlap = set(target_idx).intersection(set(ref_idx))
print('Overlap target-refinment:', overlap)

# Check if overlap between context and target indices
overlap = set(context_idx).intersection(set(target_idx))
print('Overlap target-context:', overlap)
# ----------------------------------------------
visualize_cameras(K_list, RT_list, colors, selected_batch)
