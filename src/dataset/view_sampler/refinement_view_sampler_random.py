import torch
import random
from dataclasses import dataclass
from typing import Literal
from torch import Tensor
from . import ViewSampler

@dataclass
class RefinementViewSamplerRandomCfg:
    name: Literal["refinement_random"]
    index_path: str  # Not used in this implementation
    num_refinement_views: int


class RefinementViewSamplerRandom(ViewSampler[RefinementViewSamplerRandomCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Tensor,  # Shape: [num_views, 4, 4]
        target_indices: Tensor,  # Indices of target views
        device: torch.device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        num_views, _, _ = extrinsics.shape

        if self.stage == "test":
            refinement_indices = []
            for i in target_indices.tolist():
                available_views = list(range(num_views))
            
                # Randomly select k cameras
                selected_views = random.sample(available_views, self.cfg.num_refinement_views)
                refinement_indices.append(torch.tensor(selected_views, dtype=torch.int64))

            refinement_indices = torch.stack(refinement_indices)  # Convert to tensor
            return refinement_indices.to(device)

        else:
            raise NotImplementedError("Training and validation are not implemented yet")

    @property
    def num_refinement_views(self) -> int:
        return self.cfg.num_refinement_views

    @property
    def num_target_views(self) -> int:
        raise NotImplementedError("RefinementViewSamplerRandom does not have target views")

    @property
    def num_context_views(self) -> int:
        raise NotImplementedError("RefinementViewSamplerRandom does not have context views")
