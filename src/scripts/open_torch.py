import torch
from pathlib import Path

#TORCH FILE
# TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/dtu/test/000000.torch')
# TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/panoptic_torch/test/basketball_all.torch')
# TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/re10k/train/000000.torch')
# TORCH_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/datasets/re10kpc/point_cloud_figure/000217.torch')
TORCH_FILE = Path('/home/tesista10/Thesis_Sapienza_SUTD/datasets/re10k/train/precomputed_refinement/precomputed_gaussians.torch')

# #LOAD TORCH FILE to check the content
file = torch.load(TORCH_FILE)
print(file)



#open a ckpt file
# CKPT_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/outputs/2025-02-20/18-50-57/checkpoints/best_val_psnr-v1.ckpt')
# #open a ckpt file
# file = torch.load(CKPT_FILE)
# print(file)

















import torch
from pathlib import Path
TORCH_FILE = Path('/home/tesista10/Thesis_Sapienza_SUTD/datasets/re10k/train/precomputed_refinement/precomputed_gaussians.torch')

# #LOAD TORCH FILE to check the content
file = torch.load(TORCH_FILE)
print(file)

