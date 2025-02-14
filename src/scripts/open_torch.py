import torch
from pathlib import Path

#TORCH FILE
# TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/dtu/test/000000.torch')
# TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/panoptic_torch/test/basketball_all.torch')
TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/re10k/train/000000.torch')
TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/re10kpc/point_cloud_figure/000217.torch')

#LOAD TORCH FILE to check the content
file = torch.load(TORCH_FILE)
print(file)

