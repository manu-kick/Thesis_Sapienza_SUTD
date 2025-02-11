import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


# DEBUGGER
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))


@dataclass
class DatasetPanoptic(DatasetCfgCommon):
    name: Literal["panoptic"]


class DatasetPanoptic(IterableDataset):
    def __init__() -> None:
        cfg: DatasetPanoptic
        super().__init__()
        print("DatasetPanoptic")
