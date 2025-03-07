from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerMaximumBoundCfg:
    name: Literal["maximum_bound"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int


class ViewSampleMaximumBound(ViewSampler[ViewSamplerMaximumBoundCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape

        # Just use the full gap.
        max_gap = self.cfg.max_distance_between_context_views
        min_gap = self.cfg.max_distance_between_context_views


        # Pick the gap between the context views.
        # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs
        # NOTE 2: we soleve the bug
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap) #bug solved
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        index_context_right = index_context_left + context_gap

        # We can teak einspiration  here to uniformely sampling the refinment cameras
        # if self.stage == "test":
        #     # When testing, pick all.
        #     index_target = torch.arange(
        #         index_context_left,
        #         index_context_right + 1,
        #         device=device,
        #     )
        # else:
        # When training or validating (visualizing), pick at random.
        index_target = torch.randint(
            index_context_left + self.cfg.min_distance_to_context_views,
            index_context_right + 1 - self.cfg.min_distance_to_context_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        return (
            torch.tensor((index_context_left, index_context_right)),
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
