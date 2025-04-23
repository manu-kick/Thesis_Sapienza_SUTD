from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_panoptic import DatasetPanoptic, DatasetPanopticCfg
from .dataset_kdpanoptic import KDDatasetPanoptic, KDDatasetPanopticCfg
from .types import Stage
from .view_sampler import get_view_sampler
from .view_sampler.refinement_view_sampler_camera_proximity import RefinementViewSamplerCameraProximityCfg
from .view_sampler.refinement_view_sampler_context import RefinementViewSamplerContextCfg
from .view_sampler.refinement_view_sampler_camera_R_t import RefinementViewSamplerCameraRtCfg
from .view_sampler.refinement_view_sampler_uniform_sampling import RefinementViewSamplerUniformSamplerCfg
from .view_sampler.refinement_view_sampler_random import RefinementViewSamplerRandomCfg

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "panoptic": DatasetPanoptic,
    "kdpanoptic": KDDatasetPanoptic,
}

DATASET_CONFIGS = {
    "re10k": DatasetRE10kCfg,
    "panoptic": DatasetPanopticCfg,
    "kdpanoptic": KDDatasetPanopticCfg,
}

DatasetCfg = DatasetRE10kCfg | DatasetPanopticCfg | KDDatasetPanopticCfg

def get_dataset(
    cfg: DatasetRE10kCfg | DatasetPanopticCfg | KDDatasetPanopticCfg,
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
        
        # Convert dict to an instance of 
        if isinstance(refinement_cfg_dict, dict):     
            if refinement_cfg_dict["name"] == "refinement_camera_proximity":
                refinement_cfg = RefinementViewSamplerCameraProximityCfg(**refinement_cfg_dict)
            elif refinement_cfg_dict["name"] == "refinement_context":
                refinement_cfg = RefinementViewSamplerContextCfg(**refinement_cfg_dict)
            elif refinement_cfg_dict["name"] == "refinement_camera_R_t":
                refinement_cfg = RefinementViewSamplerCameraRtCfg(**refinement_cfg_dict)
            elif refinement_cfg_dict["name"] == "refinement_random":
                refinement_cfg = RefinementViewSamplerRandomCfg(**refinement_cfg_dict)
            elif refinement_cfg_dict["name"] == "refinement_uniform_sampling":
                refinement_cfg = RefinementViewSamplerUniformSamplerCfg(**refinement_cfg_dict)
            else:
                raise ValueError(f"Unknown refinement view sampler {refinement_cfg_dict['name']}")
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