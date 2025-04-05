



#open a ckpt file
# CKPT_FILE = Path('C:/Users/rucci/OneDrive/Desktop/manu/uni/Mv_Thesis/outputs/2025-02-20/18-50-57/checkpoints/best_val_psnr-v1.ckpt')
# #open a ckpt file
# file = torch.load(CKPT_FILE)
# print(file)

















import torch
from pathlib import Path
# TORCH_FILE = Path('/home/tesista10/Thesis_Sapienza_SUTD/datasets/re10k/test/000000.torch')
TORCH_FILE = Path('/home/tesista10/Thesis_Sapienza_SUTD/datasets/panoptic_torch/panoptic_train.torch')
# TORCH_FILE = Path('/home/tesista10/Thesis_Sapienza_SUTD/datasets/panoptic_torch/basketball_all.torch')

# #LOAD TORCH FILE to check the content
file = torch.load(TORCH_FILE)
print(file)

