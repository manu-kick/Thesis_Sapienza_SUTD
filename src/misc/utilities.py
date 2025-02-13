import os
import numpy as np
from ..model.types import Gaussians
from typing import Union
from pathlib import Path
import os
import numpy as np
import torch  # only needed if you're using PyTorch tensors

# Assuming Gaussians is defined elsewhere:
# class Gaussians:
#     means: Float[Tensor, "batch gaussian dim"]
#     covariances: Float[Tensor, "batch gaussian dim dim"]
#     harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
#     opacities: Float[Tensor, "batch gaussian"]

def save_gaussians(gaussians: "Gaussians", path: Union[str, Path]) -> None:
    """
    Save the parameters of a Gaussians object to an npz file.

    Parameters:
        gaussians: A Gaussians object with attributes 'means', 'covariances',
                   'harmonics', and 'opacities'.
        path: The file path (or Path object) where the parameters will be saved.
    """
    
    # Convert the path to a string if it's a Path object
    path_str = str(path)

    # Helper function to convert tensors (or any array-like) to numpy arrays.
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            # Detach and move to CPU if necessary before converting
            return x.detach().cpu().numpy()
        return np.array(x)

    # Prepare a dictionary with all Gaussian parameters.
    params = {
        "means": to_numpy(gaussians.means),
        "covariances": to_numpy(gaussians.covariances),
        "harmonics": to_numpy(gaussians.harmonics),
        "opacities": to_numpy(gaussians.opacities),
    }

    # Ensure that the directory for the path exists.
    os.makedirs(os.path.dirname(path_str), exist_ok=True)

    print(f"Saving Gaussian parameters to {path_str}")
    np.savez(path_str, **params)