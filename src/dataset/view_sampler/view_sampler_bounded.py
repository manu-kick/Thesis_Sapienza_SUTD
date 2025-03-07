from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler

# Configuration class for the ViewSamplerBounded.
# This defines various hyperparameters controlling the selection of views.
@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]  # Name identifier for this sampler type.
    num_context_views: int  # Number of context views to be sampled.
    num_target_views: int  # Number of target views to be sampled.
    min_distance_between_context_views: int  # Minimum spacing between context views.
    max_distance_between_context_views: int  # Maximum spacing between context views.
    min_distance_to_context_views: int  # Minimum distance from context to target views.
    warm_up_steps: int  # Number of warm-up steps where distances increase progressively.
    initial_min_distance_between_context_views: int  # Initial min distance (before warm-up ends).
    initial_max_distance_between_context_views: int  # Initial max distance (before warm-up ends).

# A bounded view sampler class inheriting from ViewSampler.
# It selects a set of context and target views based on the defined configuration.
class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    
    # Gradually adjusts a value over time using a scheduling function.
    # This is used to progressively increase min/max distances during training.
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps  # Compute progress in warm-up steps.
        return min(initial + int((final - initial) * fraction), final)  # Linear interpolation.

    # Samples context and target views given scene camera parameters.
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],  # Extrinsics matrix for each view (4x4 transformation).
        intrinsics: Float[Tensor, "view 3 3"],  # Intrinsics matrix for each view (3x3 camera parameters).
        device: torch.device = torch.device("cpu"),  # Device on which to run computations.
    ) -> tuple[
        Int64[Tensor, " context_view"],  # Indices of sampled context views.
        Int64[Tensor, " target_view"],  # Indices of sampled target views.
    ]:
        num_views, _, _ = extrinsics.shape  # Get total number of available views.

        # Determine the spacing between context views based on training/testing stage.
        if self.stage == "test":
            # During testing, the full maximum gap is always used.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            # During training, if warm-up is enabled, adjust the min/max gaps dynamically.
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            # If no warm-up, use the final specified min/max gaps.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # BUG NOTE: The original implementation contains an incorrect line which is left unchanged 
        # to match previous configurations.
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, min_gap)  # BUG: Should be `max_gap = min(num_views - 1, max_gap)`
        
        # Ensure the minimum gap is large enough based on context constraints.
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)

        # If the computed gap range is invalid, raise an error.
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")

        # Randomly sample a context view spacing within the allowed range.
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,  # `+1` because randint is exclusive on the upper bound.
            size=tuple(),
            device=device,
        ).item()

        # Randomly select a left context view index.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()

        # During testing, always pick the first available view.
        if self.stage == "test":
            index_context_left = index_context_left * 0
        
        # Right context index is offset by the chosen gap.
        index_context_right = index_context_left + context_gap

        # Special handling for overfitting mode: fix indices to a fixed range.
        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap  # Fixed to max_gap.

        # Select target view indices.
        if self.stage == "test":
            # In testing, target views span the full range between the context views.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )
        else:
            # During training, target views are randomly selected within the context range,
            # while maintaining a minimum distance from context views.
            index_target = torch.randint(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right + 1 - self.cfg.min_distance_to_context_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modular arithmetic for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        return (
            torch.tensor((index_context_left, index_context_right)),  # Context view indices.
            index_target,  # Target view indices.
        )

    @property
    def num_context_views(self) -> int:
        return 2  # Always select two context views (left & right).

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views  # Defined by configuration.
