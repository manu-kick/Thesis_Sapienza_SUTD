from ...dataset import DatasetCfg
from .refiner import Refiner
from .refiner import RefinerCfg

REFINERS = {
    "refiner": Refiner,
}
Refiner_Cfg = RefinerCfg

def get_decoder(refiner_cfg: Refiner_Cfg, dataset_cfg: DatasetCfg):
    return REFINERS[refiner_cfg.name](refiner_cfg, dataset_cfg)
