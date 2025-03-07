from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_maximum_bound import ViewSampleMaximumBound, ViewSamplerMaximumBoundCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .refinement_view_sampler_camera_proximity import RefinementViewSamplerCameraProximity, RefinementViewSamplerCameraProximityCfg
from .refinement_view_sampler_context import RefinementViewSamplerContext, RefinementViewSamplerContextCfg
from .refinement_view_sampler_camera_R_t import RefinementViewSamplerCameraRt, RefinementViewSamplerCameraRtCfg
from .refinement_view_sampler_uniform_sampling import RefinementViewSamplerUniformSampler, RefinementViewSamplerUniformSamplerCfg
from .refinement_view_sampler_random import RefinementViewSamplerRandom, RefinementViewSamplerRandomCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    'maximum_bound': ViewSampleMaximumBound,
    "evaluation": ViewSamplerEvaluation,
    "refinement_camera_proximity": RefinementViewSamplerCameraProximity,
    "refinement_context": RefinementViewSamplerContext, # Gives to the refiner the same set of views as the one used for obtain the raw gaussians from mv splat
    "refinement_camera_R_t": RefinementViewSamplerCameraRt,
    'refinement_uniform_sampling': RefinementViewSamplerUniformSampler,
    "refinement_random": RefinementViewSamplerRandom,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerMaximumBoundCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | RefinementViewSamplerContextCfg
    | RefinementViewSamplerCameraProximityCfg
    | RefinementViewSamplerCameraRtCfg
    | RefinementViewSamplerUniformSamplerCfg
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
