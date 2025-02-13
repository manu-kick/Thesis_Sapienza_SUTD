import torch
import numpy as np
import open3d as o3d
import time
from pathlib import Path
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult, build_rotation
# from colormap import colormap
from copy import deepcopy
from typing import Tuple, Dict


RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
REMOVE_BACKGROUND = False  # False or True
FORCE_LOOP = False  # False or True
w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15

def_pix = torch.tensor(np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        fov_x=2 * np.arctan(w / (2 * fx)),
        fov_y=2 * np.arctan(h / (2 * fy)),
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        focal_x=fx,
        focal_y=fy,
        camera_center=cam_center,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def load_scene_data(seq: str, exp: str) -> Dict[str, torch.Tensor]:
    """
    Load the Gaussian parameters for a single frame from an .npz file.
    """
    # Load params from npz
    params_path = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/outputs/test/placeholder/basketball_test_15/gaussians/ctx_000014.json.npz')
    params = dict(np.load(str(params_path)))

    # Convert to CUDA tensors with explicit dtype
    gaussians = {
        "means": torch.tensor(params["means"], dtype=torch.float32, device="cuda"),
        "covariances": torch.tensor(params["covariances"], dtype=torch.float32, device="cuda"),
        "harmonics": torch.tensor(params["harmonics"], dtype=torch.float32, device="cuda"),
        "opacities": torch.tensor(params["opacities"], dtype=torch.float32, device="cuda")
    }

    return gaussians


def render(w2c: np.ndarray, k: np.ndarray, gaussians: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a single frame using the Gaussian rasterizer.
    """
    with torch.no_grad():
        # cam = setup_camera(w, h, k, w2c, near, far)
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3], # 
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        im, _, depth = Renderer(raster_settings=cam)(**gaussians)
        return im, depth


def visualize(seq: str, exp: str):
    """
    Visualize a single frame using Open3D.
    """
    gaussians = load_scene_data(seq, exp)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=True)

    # Initialize camera
    w2c, k = np.eye(4), np.eye(3)  # Identity matrices (adjust if needed)
    im, depth = render(w2c, k, gaussians)

    # Convert rendered data to point cloud
    pcd = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(gaussians["means"].cpu().numpy())
    colors = o3d.utility.Vector3dVector(im.permute(1, 2, 0).reshape(-1, 3).cpu().numpy())

    pcd.points = pts
    pcd.colors = colors
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()
    
if __name__ == "__main__":
    exp_name = "exp1"
    sequence = "basketball"
    visualize(sequence, exp_name)