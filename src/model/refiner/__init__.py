from ...dataset import DatasetCfg
from .refiner import Refiner
from .refiner import RefinerCfg

REFINERS = {
    "refiner": Refiner,
}
Refiner_Cfg = RefinerCfg

def get_refiner(name, decoder, losses) -> Refiner:
    if name == "refiner":
        return Refiner(Refiner_Cfg, decoder, losses)
    if name == "refiner_precomputation":
        NotImplementedError('The refiner precomputation still be implemented')
    else:
        NotImplementedError(f"Refiner {name} not implemented.")
