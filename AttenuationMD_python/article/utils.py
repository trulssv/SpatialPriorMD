from article.attenuations import PEPVC_TO_WAIO
import torch
from pathlib import Path
import os
from scipy.io import loadmat
import numpy as np

def convert_pepvc_to_waio(a: torch.Tensor) -> torch.Tensor:
    """
    Convert a PE/PVC image basis pair to a Water/Iodine image basis pair using the PEPVC_TO_WAIO transformation matrix.
    Args:
        a (torch.Tensor): A tensor of shape [2, H, W] representing the PE/PVC image basis pair.
    Returns:
        torch.Tensor: A tensor of shape [2, H, W] representing the Water/Iodine image basis pair.
    """
    return torch.einsum("ij,jhw->ihw", PEPVC_TO_WAIO, a)  # [2 x H x W]

def load_dir(path: Path) -> dict[str, torch.Tensor]:
    """
    Load all .mat files in a directory into a dictionary of torch tensors.
    Args:
        path (Path): The path to the directory containing .mat files.
    Returns:
        dict[str, torch.Tensor]: A dictionary where keys are file names (without extension) and values are the loaded torch tensors.
    """

    def _is_arr(x) -> bool:
        
        if isinstance(x, np.ndarray):
            if x.dtype in [np.float32, np.float64, np.int32, np.int64, np.uint8]:
                return True
            else:
                print(f"Warning: Unsupported numpy dtype {x.dtype}.")
                return False
        elif isinstance(x, torch.Tensor):
            if x.dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
                return True
            else:
                print(f"Warning: Unsupported torch dtype {x.dtype}.")
                return False
        else:
            print(f"Warning: Unsupported type {type(x)}.")
            return False

    data = {}
    for file in os.listdir(path):
        if file.endswith(".mat"):
            key = file[:-4]  # Remove the .mat extension
            mat_data = loadmat(path / file)
            # Assuming the .mat file contains a single variable with the same name as the file

            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if len(keys) == 1:
                if _is_arr(mat_data[keys[0]]):
                    data[key] = torch.tensor(mat_data[keys[0]])
                else:
                    print(f"Warning: The variable in {file} is not a valid array. Skipping.")
            elif len(keys) > 1:
                # If there are multiple keys, you might want to handle this differently
                # For now, we'll just take the first key that doesn't start with "__"
                if _is_arr(mat_data[keys[0]]):
                    data[key] = torch.tensor(mat_data[keys[0]])
                else:
                    print(f"Warning: The variable in {file} is not a valid array. Skipping.")
            else:   
                print(f"Warning: No valid keys found in {file}. Skipping.")
    return data 

