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
    min_distance_between_refinement_views: int
    max_distance_between_refinement_views: int
    min_distance_to_refinement_views: int
    warm_up_steps: int
    initial_min_distance_between_refinement_views: int
    initial_max_distance_between_refinement_views: int
    num_steps: int


class RefinementViewSamplerBounded(ViewSampler[RefinementViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> Int64[Tensor, " refinement_view"]: # indices for refinement views
        num_views, _, _ = extrinsics.shape

        # Compute the refinement view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_refinement_views
            min_gap = self.cfg.max_distance_between_refinement_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_refinement_views,
                self.cfg.max_distance_between_refinement_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_refinement_views,
                self.cfg.min_distance_between_refinement_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_refinement_views
            min_gap = self.cfg.min_distance_between_refinement_views

        # Pick the gap between the refinement views.
        # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, min_gap)
        min_gap = max(2 * self.cfg.min_distance_to_refinement_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        refinement_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right refinement indices.
        index_refinement_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - refinement_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_refinement_left = index_refinement_left * 0
        index_refinement_right = index_refinement_left + refinement_gap

        if self.is_overfitting:
            index_refinement_left *= 0
            index_refinement_right *= 0
            index_refinement_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_refinement_left,
                index_refinement_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            index_target = torch.randint(
                index_refinement_left + self.cfg.min_distance_to_refinement_views,
                index_refinement_right + 1 - self.cfg.min_distance_to_refinement_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_refinement_right %= num_views

        return (
            torch.tensor((index_refinement_left, index_refinement_right)),
            index_target,
        )

    @property
    def num_refinment_views(self) -> int:
        return self.cfg.num_refinement_views

    @property
    def num_target_views(self) -> int:
        raise NotImplementedError ("RefinementViewSamplerBounded does not have target views")
    
    @property
    def num_context_views(self) -> int:
        raise NotImplementedError ("RefinementViewSamplerBounded does not have context views")
