import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        ax.quiver(*cam_center, *(corner - cam_center), color='gray', arrow_length_ratio=0.2, linewidth=1.5)

    # Plot the camera center with a marker
    ax.scatter(*cam_center, color=color, marker='o', s=100, edgecolors='black')

def visualize_cameras(K_list, RT_list):
    """Visualizes multiple cameras in 3D space with frustums."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (K, RT) in enumerate(zip(K_list, RT_list)):
        R = RT[:, :3]  # Rotation matrix
        t = RT[:, 3].reshape(3, 1)  # Ensure t is a column vector
        plot_camera(ax, K, R, t, color=f'C{i}')
    
    # View settings
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses and Frustums")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.show()

# Example Usage
K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Sample intrinsic matrix
RT1 = np.hstack([np.eye(3), np.array([[0], [0], [-5]])])   # Camera at (0,0,-5)
RT2 = np.hstack([np.eye(3), np.array([[2], [1], [-5]])])   # Camera at (2,1,-5)

visualize_cameras([K, K], [RT1, RT2])
