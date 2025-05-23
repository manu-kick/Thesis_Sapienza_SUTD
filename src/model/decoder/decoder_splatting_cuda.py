from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        use_scale_and_rotation: bool = False,
        use_sh: bool = True
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color = render_cuda(
            extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
            intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
            near=rearrange(near, "b v -> (b v)"),
            far=rearrange(far, "b v -> (b v)"),
            image_shape=image_shape,
            background_color=repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            gaussian_means=repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            gaussian_covariances=repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v) if gaussians.covariances is not None else None,
            gaussian_sh_coefficients=repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) if use_sh else None,
            colors_precomp=repeat(gaussians.colors_precomp, 'b g c -> (b v) g c',v=v) if not use_sh else None,
            gaussian_opacities=repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            gaussian_scales=repeat(gaussians.scales, "b g xyz-> (b v) g xyz", v=v), #added
            gaussian_rotations=repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v), #added
            use_scale_and_rotation=use_scale_and_rotation,
            use_sh=use_sh
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
