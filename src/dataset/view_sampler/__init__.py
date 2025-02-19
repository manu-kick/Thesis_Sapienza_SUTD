from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .refinement_view_sampler_camera_proximity import RefinementViewSamplerCameraProximity, RefinementViewSamplerCameraProximityCfg
from .refinement_view_sampler_context import RefinementViewSamplerContext, RefinementViewSamplerContextCfg
from .refinement_view_sampler_camera_K_E import RefinementViewSamplerCameraKE, RefinementViewSamplerCameraKECfg
from .refinement_view_sampler_random import RefinementViewSamplerRandom, RefinementViewSamplerRandomCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "refinement_camera_proximity": RefinementViewSamplerCameraProximity,
    "refinement_context": RefinementViewSamplerContext, # Gives to the refiner the same set of views as the one used for obtain the raw gaussians from mv splat
    "refinement_camera_K_E": RefinementViewSamplerCameraKE,
    "refinement_random": RefinementViewSamplerRandom,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | RefinementViewSamplerContextCfg
    | RefinementViewSamplerCameraProximityCfg
    | RefinementViewSamplerCameraKECfg
    | RefinementViewSamplerRandomCfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
