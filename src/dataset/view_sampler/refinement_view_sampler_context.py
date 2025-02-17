from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from pathlib import Path


from . import ViewSampler
from ...evaluation.evaluation_index_generator import IndexEntry
from ..types import Stage
from ...misc.step_tracker import StepTracker
from dacite import Config, from_dict
import json


@dataclass
class RefinementViewSamplerContextCfg:
    name: Literal["refinement_context"]
    index_path: Path | str
    num_refinement_views: int # actually not useful here because we get directly form the evaluation file
    


class RefinementViewSamplerContext(ViewSampler[RefinementViewSamplerContextCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: RefinementViewSamplerContextCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        dacite_config = Config(cast=[tuple])
        with Path(cfg.index_path).open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }
            
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
        
       
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        refinement_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        
       
        return refinement_indices.to(device)

    

    @property
    def num_refinment_views(self) -> int:
        return self.cfg.num_refinement_views

    @property
    def num_target_views(self) -> int:
        raise NotImplementedError ("RefinementViewSamplerBounded does not have target views")
    
    @property
    def num_context_views(self) -> int:
        raise NotImplementedError ("RefinementViewSamplerBounded does not have context views")
