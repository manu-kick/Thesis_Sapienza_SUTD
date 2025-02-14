from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_panoptic import DatasetPanoptic, DatasetPanopticCfg
from .types import Stage
from .view_sampler import get_view_sampler
from .view_sampler.refinement_view_sampler_bounded import RefinementViewSamplerBoundedCfg

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "panoptic": DatasetPanoptic
}

DATASET_CONFIGS = {
    "re10k": DatasetRE10kCfg,
    "panoptic": DatasetPanopticCfg
}

DatasetCfg = DatasetRE10kCfg | DatasetPanopticCfg

def get_dataset(
    cfg: DatasetRE10kCfg | DatasetPanopticCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    DatasetCfg = DATASET_CONFIGS[cfg.name]
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    
    refinement_view_sampler = None
    if cfg.refinement:
        refinement_cfg_dict = cfg.refinement_cfg  # This is a dictionary

        if isinstance(refinement_cfg_dict, dict):
            # Convert dict to an instance of RefinementViewSamplerBoundedCfg
            refinement_cfg = RefinementViewSamplerBoundedCfg(**refinement_cfg_dict)
        else:
            refinement_cfg = refinement_cfg_dict  # Already an object
            
        refinement_view_sampler = get_view_sampler(
            refinement_cfg,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        
    
    return DATASETS[cfg.name](cfg, stage, view_sampler, refinement_view_sampler)