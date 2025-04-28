from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"] | Float[Tensor, "batch views gaussian dim"] 
    covariances: Float[Tensor, "batch gaussian dim dim"] | None | Float[Tensor, "batch views gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] | None | Float[Tensor, "batch views gaussian 3 d_sh"] #none if using color precomputed
    opacities: Float[Tensor, "batch gaussian"] | Float[Tensor, "batch views gaussian"]
    scales: Float[Tensor, "batch gaussian 3"] | Float[Tensor, "batch views gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"] | Float[Tensor, "batch views gaussian 4"]
    colors_precomp: Float[Tensor, 'batch gaussian 3'] | None | Float[Tensor, "batch views gaussian 3"] # none if using harmonics
