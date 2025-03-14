import torch
from dataclasses import dataclass
from typing import Literal
from torch import Tensor
from . import ViewSampler

@dataclass
class RefinementViewSamplerUniformSamplerCfg:
    name: Literal["refinement_uniform_sampling"]
    index_path: str  # Not used in this implementation
    num_refinement_views: int
    refinement_loss: str
    random_target_selection: bool = False


class RefinementViewSamplerUniformSampler(ViewSampler[RefinementViewSamplerUniformSamplerCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Tensor,  # Shape: [num_views, 4, 4]
        context_indices: Tensor,  # Indices of context views
        device: torch.device = torch.device("cpu"),
    ):
        num_views, _, _ = extrinsics.shape
        min_distance_to_context_views = 0
        # uniform sampling of refinement cameras in the contexts bound
        # refinement_indices = torch.randint(
        #     context_indices.min().item() + min_distance_to_context_views,
        #     context_indices.max().item() +1 - min_distance_to_context_views,
        #     size=(self.cfg.num_refinement_views,),
        #     device=device,
        # )
        # import torch



        # Compute the step size based on available distance and required number of points
        available_range =  context_indices.max().item() -  context_indices.min().item() - 1  # Excluding the min and max
        step_size = available_range / (self.cfg.num_refinement_views + 1)  # Divide by N+1 to keep min/max excluded

        # Generate samples, ensuring they stay within bounds
        refinement_indices = torch.linspace(context_indices.min().item() + step_size, context_indices.max().item() - step_size, self.cfg.num_refinement_views).round().int()
    

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
