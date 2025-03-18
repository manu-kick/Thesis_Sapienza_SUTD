import torch
from dataclasses import dataclass
from typing import Literal
from torch import Tensor
from . import ViewSampler

@dataclass
class RefinementViewSamplerCameraRtCfg:
    name: Literal["refinement_camera_R_t"]
    index_path: str  # Not used in this implementation
    num_refinement_views: int
    refinement_loss: str 
    lambda_fov: float = 0.5  # Weight for field-of-view similarity (0 to 1)
    random_target_selection: bool = False


class RefinementViewSamplerCameraRt(ViewSampler[RefinementViewSamplerCameraRtCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Tensor,  # Shape: [num_views, 4, 4]
        target_indices: Tensor,  # Indices of target views
        device: torch.device = torch.device("cpu"),
    ):
        num_views, _, _ = extrinsics.shape
        extrinsics_inv = extrinsics.inverse()  # Convert world-to-camera to camera-to-world
        cam_positions = extrinsics_inv[:, :3, 3]  # Extract camera positions [num_views, 3]
        optical_axes = extrinsics[:, :3, 2]  # Extract camera optical axes (third column of R)

        
        refinement_indices = []
        for i in target_indices.tolist():
            # Compute cosine similarity between optical axes
            similarity = torch.nn.functional.cosine_similarity(
                optical_axes[i].unsqueeze(0), optical_axes, dim=1
            )

            # Compute Euclidean distance between camera positions
            distances = torch.norm(cam_positions[i] - cam_positions, dim=1)  # Shape: [num_views]

            # Normalize distances (to range 0-1)
            distances = distances / distances.max()

            # Compute final score (higher is better)
            lambda_fov = self.cfg.lambda_fov
            score = lambda_fov * similarity - (1 - lambda_fov) * distances

            # Sort based on score (descending)
            sorted_indices = torch.argsort(score, descending=True)
            if str(sorted_indices[0].item()) == str(i):
                closest_views = sorted_indices[1:self.cfg.num_refinement_views + 1]
            else:
                closest_views = sorted_indices[0:self.cfg.num_refinement_views]  
            refinement_indices.append(closest_views)

        refinement_indices = torch.stack(refinement_indices)  # Convert to tensor
        return refinement_indices.to(device)


    @property
    def num_refinement_views(self) -> int:
        return self.cfg.num_refinement_views

    @property
    def num_target_views(self) -> int:
        raise NotImplementedError("RefinementViewSamplerCameraKE does not have target views")

    @property
    def num_context_views(self) -> int:
        raise NotImplementedError("RefinementViewSamplerCameraKE does not have context views")
