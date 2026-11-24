
from pathlib import Path
from nibabel.nifti1 import Nifti1Image, Nifti1Header, load, save
import numpy as np
import torch

from config import *

import scipy.io as sio
from typing import Any

def _load_mat(path: str | Path) -> dict[str, Any]:  
    """Load a .mat file; returns a plain dict with no MATLAB meta-keys."""
    path = str(path)
    try:
        mat = sio.loadmat(path)
    except NotImplementedError:
        # HDF5 / v7.3 format
        import h5py
        mat = {}
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                mat[k] = np.array(v)
    # Strip MATLAB meta keys
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _extract_mid_slice(vol: torch.Tensor, num_slices: int = 1) -> torch.Tensor:
    """Extract the central slice, averaging over num_slices if requested."""
    Z = vol.shape[0]
    mid = Z // 2
    half = num_slices // 2
    lo = max(0, mid - half)
    hi = min(Z, lo + num_slices)
    lo = max(0, hi - num_slices)
    return vol[lo:hi].mean(dim=0)

def segment_id(scan_id: int, data_root: Path | None = None, refine_mask: bool=True, num_slices: int = 1) -> dict[str, torch.Tensor]:
    """Return a dictionary of scan metadata for the given scan ID."""

    output = {}

    assert isinstance(scan_id, int), f"scan_id must be an integer, got {type(scan_id)}"

    ### PATHS and Params

    if data_root is None:
        data_root = UZB_DATA
    elif not isinstance(data_root, Path):
        data_root = Path(data_root)

    from config import SCANS # Maps scan IDs to scan directories in data root
    scan_path = data_root / SCANS[scan_id]
    f_names = {"Basis": ("water_basis_vol", "iodine_basis_vol"),        # Initial water/iodine 2MD, always loaded
               "GT": ("iodine_gt", "gadolinium_gt", "water_gt"),        # Ground truth maps, loaded if available. NOTE: not all scans have GT water maps. 
                "mask": "mask_vol"}                                     # Segmentation mask, always loaded
    background_values: dict[str, float | torch.Tensor] = {"water": 1.0, "iodine": 0.0}  # Background values for water and iodine

    # Normalize the basis volumes to mg/mL using the known material densities

    basis_normalization = {"water": DENSITY_WA, "iodine": DENSITY_I}  # Material densities in mg/mL
    ground_truth_normalization = {"iodine": DENSITY_I, "gadolinium": DENSITY_GD, "water": DENSITY_WA}  # Material densities in mg/mL

    # Load Basis volumes
    a = {}
    for fname in f_names["Basis"]:
        path = scan_path / f"{fname}.nii"
        if not path.exists():
            raise FileNotFoundError(f"Expected file {fname}.nii not found in {scan_path}")

        vol = torch.from_numpy(load(path).get_fdata().astype(np.float32))
        a[fname] = _extract_mid_slice(vol, num_slices=num_slices)
   
    # Load Ground Truth volumes if available
    a_GT = {}
    has_all_gt = True
    has_wa_gt = False  # Flag to indicate if water_gt is available
    for rho, fname in zip(ground_truth_normalization.values(), f_names["GT"]):
        path = scan_path / f"{fname}.nii"
        if not path.exists():
            has_all_gt = False if fname == "water_gt" else has_all_gt  # Only water_gt is optional
            print(f"Warning: {fname}.nii not found in {scan_path}. Initializing ground truth with zeros...\n")

            a_GT[fname] = torch.zeros_like(a[f_names["Basis"][0]])  # Placeholder for missing ground truth
        else:
            print(f"Loading {fname} from {path}...")
            vol = torch.from_numpy(load(path).get_fdata().astype(np.float32))
            mid_slice = _extract_mid_slice(vol, num_slices=num_slices)
            a_GT[fname] = mid_slice / rho
            has_wa_gt = True if fname == "water_gt" else has_wa_gt  # Set the flag if water_gt is available

    # Load Segmentation Mask

    fname = f_names["mask"]
    path = scan_path / f"{fname}.nii"
    if not path.exists():
        raise FileNotFoundError(f"Expected file {fname}.nii not found in {scan_path}")
    print(f"Loading {fname} from {path}...")
    vol = torch.from_numpy(load(path).get_fdata().astype(np.float32))
    mask = torch.Tensor(_extract_mid_slice(vol, num_slices=num_slices)).to(torch.bool)  # Convert to boolean mask


    # add air thresholding to mask
    air_threshold = 0.0  # Adjust this threshold based on your data
    # mask = mask & (a[f_names["Basis"][0]] > air_threshold)



    # If refine_mask is True, use ground truth maps (if available): 

    if True:
        print("Refining mask using ground truth maps...")
        # Combine ground truth maps to refine the mask
        

        # update ground truth maps to be masked

        for fname in f_names["GT"]:
            a_GT[fname] = a_GT[fname] * mask.float()  # Apply the refined mask to ground truth maps

    assert torch.sum(mask) > 0, "The loaded mask is empty. Please check the mask file."

    
    # Define background 

    bkg: dict[str, torch.Tensor] = {}

    signal: dict[str, torch.Tensor] = {}

    # update water background value if water_gt is available
    if has_wa_gt:
        water_gt = a_GT["water_gt"]
        background_values["water"] = mask.float()
        background_values["water"][water_gt > 0] = a_GT["water_gt"][water_gt > 0]  # Use ground truth water values where available

    # remove water_gt from a_GT:

    del a_GT["water_gt"]



    assert mask.shape == a["water_basis_vol"].shape, f"Mask shape {mask.shape} does not match basis volume shape {a['water_basis_vol'].shape}"
    assert len(background_values) == len(a), f"Background values {background_values.keys()} do not match basis volumes {a.keys()}"

    for val, (name, arr) in zip(background_values.values(), a.items()):
        signal[name] = (arr - val) * mask.float()
        bkg[name] = arr - signal[name]

    # Convert to torch tensors and store in output dictionary
    signal_tensor = torch.stack(list(signal.values()), dim=0)  # [2 x H x W]
    bkg_tensor = torch.stack(list(bkg.values()), dim=0)  # [2 x H x W]
    a_GT_tensor = torch.stack(list(a_GT.values()), dim=0)  # [2 x H x W] (iodine, gadolinium)
    mask = mask.float()  # Convert boolean mask to float tensor


    output["signal"] = signal_tensor
    output["background"] = bkg_tensor
    output["ground_truth"] = a_GT_tensor
    output["mask"] = mask

    return output
    
    
def load_materials(usePePvc=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    materials_dir = FILES_DIR / "materials"
    materials = {}

    mats = ["MuWaimm", "MuIimm", "MuGdimm"] if not usePePvc else ["MuPEimm", "MuPVCimm", "MuIimm" ,"MuGdimm"]

    for fname in mats:
        if not (materials_dir / f"{fname}.mat").exists():
            print(f"Expected material file {fname}.mat not found in {materials_dir}")
            raise FileNotFoundError
        
        print(f"Loading material data from {materials_dir / f'{fname}.mat...'}")

        material = list(_load_mat(materials_dir / f"{fname}.mat").values())[0]  # Assuming the .mat file has a single variable
        
        materials[fname] = torch.from_numpy(material)

    materials = {k: v.squeeze() for k, v in materials.items()}  # Remove singleton dimensions


    fB = torch.stack([materials["MuWaimm"], materials["MuIimm"]], dim=0)  if not usePePvc else torch.stack([materials["MuPEimm"], materials["MuPVCimm"]], dim=0)  # [2 x nE]
    fC = torch.stack([materials["MuIimm"], materials["MuGdimm"]], dim=0)  # [2 x nE]


    return fB, fB, fC # fB = fR

def load_energies() -> torch.Tensor:
    """Load the energy values corresponding to the material attenuation curves."""
    
    Emin = 15
    Emax = 120

    Evec = torch.linspace(Emin, Emax, 106)  # [nE]
    return Evec

def load_weights() -> torch.Tensor:
    
    weights  = list(_load_mat(FILES_DIR / "weights.mat").values())[0]  # Assuming the .mat file has a single variable

    weights = weights[::3, :]

    return torch.from_numpy(weights)
