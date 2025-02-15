from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ..view_sampler import ViewSampler


@dataclass
class RefinementViewSamplerBoundedCfg:
    name: Literal["refinement_bounded"]
    num_refinement_views: int


class RefinementViewSamplerBounded(ViewSampler[RefinementViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        target_indices: Int64[Tensor, "targets"],
        device: torch.device = torch.device("cpu"),
    ): # indices for refinement views
        num_views, _, _ = extrinsics.shape
        extrinsics = extrinsics.inverse()
        extrinsics_inv = extrinsics.inverse()  # Convert world-to-camera to camera-to-world
        cam_positions = extrinsics_inv[:, :3, 3]  # Extract camera positions [num_views, 3]

        if self.stage == "test":
            # Compute pairwise distances between camera positions
            distances = torch.cdist(cam_positions, cam_positions, p=2)  # Euclidean distance [num_views, num_views]

            refinement_indices = []
            for i in target_indices.tolist():
                sorted_indices = torch.argsort(distances[i])  # Sort views by closest distance
                closest_views = sorted_indices[1:self.cfg.num_refinement_views + 1]  # Skip self and take N closest
                refinement_indices.append(closest_views)

            refinement_indices = torch.stack(refinement_indices)  # Convert to tensor

            return refinement_indices.to(device)

        else:
            raise NotImplementedError("Training and validation are not implemented yet")

    

    @property
    def num_refinment_views(self) -> int:
        return self.cfg.num_refinement_views

    @property
    def num_target_views(self) -> int:
        raise NotImplementedError ("RefinementViewSamplerBounded does not have target views")
    
    @property
    def num_context_views(self) -> int:
        raise NotImplementedError ("RefinementViewSamplerBounded does not have context views")
