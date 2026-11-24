from operators.tomo import SliceWiseRayTrafoModule
from optim.solvers import NewtonSolver
import numpy as np
import torch
import os

# Nifti support

import nibabel as nib
from nibabel.nifti1 import Nifti1Image, Nifti1Header, load, save


from scipy.io import loadmat
from pathlib import Path
from data_processing.utils import load_materials, load_weights
from tqdm import tqdm

# Typing imports

from typing import Any, TypedDict, cast
from numpy.typing import NDArray
from dataclasses import dataclass

# Plotting imports

from matplotlib import pyplot as plt


@dataclass
class MaterialData:
    vol: NDArray[np.float32] | None = None
    affine: NDArray[np.float32] | None = None
    header: Nifti1Header | None = None
    mu: NDArray[np.float32] | None = None
    scale: float | None = None
    rho: float | None = None


PLOT_DIR = Path.home() / "Documents/AttenuationMD/AttenuationMD_python/article/figs/plots"
SAVE_DIR = Path.home() / "Documents/AttenuationMD/AttenuationMD_python/data"
MATERIAL_SCALES = {
    "i": {"scale": 20, "rho": 4930},
    "gd": {"scale": 200, "rho": 7900},
    "water": {"scale": 1000, "rho": 1000},
    "ca": {"scale": 100, "rho": 1500},
}


def plot_log_counts(log_n: np.ndarray) -> None:
    assert log_n.ndim == 4, f"Expected 4-D array for predicted photon counts, got shape {log_n.shape}"
    
    b, a, z, y = log_n.shape

    assert b==8 or b ==24, f"Expected 8 energy bins, got {b} bins"

    print(f"Plotting log counts with shape {log_n.shape}...")

    print(f"Diagnostics for log(n): logn min: {log_n.min()}, logn max: {log_n.max()}, logn mean: {log_n.mean()}, logn std: {log_n.std()}")

    # pick mid slice to plot all bins

    mid_slice = z // 2
    fig, axes = plt.subplots(2, b // 2, figsize=(5 * ((b + 1) // 2), 10))
    if b == 1:
        axes = np.array([[axes]])

    assert isinstance(axes, (list, np.ndarray)), f"Expected axes to be a list or ndarray, got {type(axes)}"

    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    for i in range(b):
        im = axes[i].imshow(log_n[i, :, mid_slice, :], cmap="gray")
        axes[i].set_title(f"Log counts for bin {i} at mid slice")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], orientation='vertical', label='Log counts')
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving predicted photon counts plot to {PLOT_DIR / 'log_counts.png'}... \n")
    fig.savefig(PLOT_DIR / "log_counts.png")
    plt.close(fig)

def plot_material_basis_sinograms(A: np.ndarray, mats: list[str] | tuple[str, ...]) -> None:

    assert A.ndim == 4, f"Expected 4-D array for sinograms, got shape {A.shape}"

    m, a, z, y = A.shape

    print(f"Plotting material basis sinograms with shape {A.shape} for materials {mats}...")

    # pick mid slice to plot all materials

    mid_slice = z // 2

    fig, axes = plt.subplots(m, 1, figsize=(10, 5 * m))

    if m == 1:
        axes = [axes]

    assert isinstance(axes, (list, np.ndarray)), f"Expected axes to be a list or ndarray, got {type(axes)}"
    
    for i in range(m):
        im = axes[i].imshow(A[i, :, mid_slice, :], cmap="gray")
        axes[i].set_title(f"Material basis sinogram for {mats[i]} at mid slice")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], orientation='vertical', label='Sinogram values')
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving material basis sinograms plot to {PLOT_DIR / 'material_basis_sinograms.png'}... \n")
    fig.savefig(PLOT_DIR / "material_basis_sinograms.png")
    plt.close(fig)


class PolyChromaticForwardModel():

    def __init__(self, batch_size: int = 2048, with_contrast: bool = True, with_noise: bool = True) -> None:
        self.batch_size = batch_size
        self.with_contrast = with_contrast
        self.with_noise = with_noise


        # Set material scales and densities for conversion to volume fractions

        self.mats: dict[str, MaterialData] = {}

        for mat, params in MATERIAL_SCALES.items():
            self.mats[mat] = MaterialData(
                vol=None,  # Placeholder for the volume data
                affine=None,  # Placeholder for the affine transformation matrix
                header=None,  # Placeholder for the NIfTI header
                mu=None,  # Placeholder for the attenuation coefficients
                scale=params["scale"],
                rho=params["rho"]
            )

        self.ray_trafo_params = {
            "physical_size": [305, 471, 471], 
            "shape": [611, 512, 512], 
            "slices_considered": 10,
            "n_angles": 512, 
            "detector_physical_size": [500, 500],
            "detector_shape": [512, 512],
            "geometry": 'cone', 
            "src_radius": 800,
            "det_radius": 800,
            "axis": [1, 0, 0],
            "geometry_type": 'parallel'
        }

        

        self.md_mats = ("water", "i")
        self.forward_mats =("water", "i", "gd", "ca") if self.with_contrast else ("water", "ca")

        self.data_dir = Path("/mnt/data/MDdata/Antropomorphic/case_00000/")
        self.files_dir = Path.home() / "Documents/AttenuationMD/files/"

        assert self.data_dir.exists(), f"Data directory {self.data_dir} does not exist."
        assert self.files_dir.exists(), f"Files directory {self.files_dir} does not exist."
        assert set(self.md_mats).issubset(set(MATERIAL_SCALES.keys())), f"MD materials {self.md_mats} must be a subset of available materials {list(MATERIAL_SCALES.keys())}."

    def _update_mode(self, with_contrast: bool, with_noise: bool) -> None:
        self.with_contrast = with_contrast
        self.with_noise = with_noise
        self.forward_mats = ("water", "i", "gd", "ca") if self.with_contrast else ("water", "ca")

    def _setup_ray_trafo(self):
        water = self.mats["water"]
        assert water.vol is not None and water.header is not None

        slice_shape = tuple(water.vol.shape[1:])
        slice_spacing = water.header.get_zooms()[1:3]
        slice_physical_size = tuple(np.asarray(slice_shape) * np.asarray(slice_spacing))
        print(
            "Building slice-wise parallel-beam transform with "
            f"slice_shape={slice_shape}, physical_size={slice_physical_size}...\n"
        )
        self.ray_trafo = SliceWiseRayTrafoModule(
            physical_size=slice_physical_size,
            shape=slice_shape,
            n_angles=self.ray_trafo_params["n_angles"],
            n_detectors=self.ray_trafo_params["detector_shape"][1],
        )

    def _get_newton_solver(self):
        """
        Create a Newton solver for multi-material decomposition (MD) using the loaded materials and their attenuation coefficients.
        """
        assert hasattr(self, "mats") and isinstance(self.mats, dict), "Materials must be loaded and stored in self.mats before creating the Newton solver."
        
        fMD_list = []

        for mat in self.md_mats:
            mu = self.mats[mat].mu

            assert mu is not None, f"Attenuation coefficient for {mat} is not loaded."

            fMD_list.append(mu.squeeze())

        fMD = np.stack(fMD_list, axis=0)

        # Convert fMD and weights to torch tensor for the Newton solver
        fMD = torch.from_numpy(fMD).float()
        weights = torch.from_numpy(self.weights).float()

        assert isinstance(fMD, torch.Tensor), f"fMD must be a torch.Tensor but got {type(fMD)}"
        assert isinstance(weights, torch.Tensor), f"weights must be a torch.Tensor but got {type(weights)}"

        return NewtonSolver(
            fB=fMD, 
            fR=fMD, 
            fC=fMD, 
            weights=weights, 
            verbose=False
        )  

    def _normalize_mats(self, mats: dict[str, MaterialData]) -> dict[str, MaterialData]:
        """
        Convert material concentrations to volume fractions for the attenuation model.

        Concentrations are physically nonnegative. Some input volumes contain
        reconstruction noise below zero; keeping it would turn the negative
        attenuation exponent positive and make the photon rate unbounded.
        """
        assert isinstance(mats, dict), "Materials must be a dictionary of MaterialData with string keys."
        assert all(isinstance(m_data, MaterialData) for m_data in mats.values()), "All materials must be instances of MaterialData."

        for m, m_data in mats.items():
            vol = m_data.vol
            rho = m_data.rho
            scale = m_data.scale

            assert vol is not None and rho is not None and scale is not None, f"Material '{m}' must have 'vol', 'rho', and 'scale' defined."

            # Convert concentration to volume fraction
            vol_fraction = vol * scale / (rho) 

            # Remove negative values to avoid unphysical results in the attenuation model
            vol_fraction[vol_fraction < 0] = 0

            mats[m].vol = vol_fraction.astype(np.float32)

        return mats

    def _denormalize_mats(self, mats: dict[str, MaterialData]) -> dict[str, MaterialData]:
        """
        Convert volume fractions back to material concentrations for visualization or further analysis.
        """
        assert isinstance(mats, dict), "Materials must be a dictionary of MaterialData with string keys."
        assert all(isinstance(m_data, MaterialData) for m_data in mats.values()), "All materials must be instances of MaterialData."

        for m, m_data in mats.items():
            vol_fraction = m_data.vol
            rho = m_data.rho
            scale = m_data.scale

            assert vol_fraction is not None and rho is not None and scale is not None, f"Material '{m}' must have 'vol', 'rho', and 'scale' defined."

            # Convert volume fraction back to concentration
            concentration = vol_fraction * (rho) / scale

            mats[m].vol = concentration.astype(np.float32)

        return mats

    def _denormalize_reconstruction(self, aC: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert reconstructed volume fractions to each material's input units."""
        values = aC.detach().cpu().numpy() if isinstance(aC, torch.Tensor) else aC
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != len(self.md_mats):
            raise ValueError(f"Expected {len(self.md_mats)} reconstructed materials, got {values.shape[0]}.")

        concentrations = []
        for material_name, volume_fraction in zip(self.md_mats, values):
            material = self.mats[material_name]
            assert material.rho is not None and material.scale is not None
            concentrations.append(volume_fraction * material.rho / material.scale)
        return np.stack(concentrations, axis=0).astype(np.float32)

    def _extract_slices(self, a: np.ndarray, slices_considered: int) -> np.ndarray:
        """
        Extract the specified number of slices from the 3D volume.
        """
        mid_slice = a.shape[0] // 2
        start_slice = max(0, mid_slice - slices_considered // 2)
        end_slice = start_slice + slices_considered
        return a[start_slice:end_slice, :, :]

    def _load_weights(self) -> None:
        """
        Loads the PCCT weight functions from a .mat file and returns them as a numpy array.
        """
        weights_path = self.files_dir / "weights.mat"
        weights = loadmat(weights_path)["weights"].astype(np.float32)

        # Extract every third energy bin to reduce the number of bins from 24 to 8
        weights = weights[::3, :]


        print(f"Loaded weights from {weights_path} with shape {weights.shape} and dtype {weights.dtype} and saved to self.weights.")
        print(f"Diagnostics for weights: min: {np.min(weights)}, max: {np.max(weights)}, mean: {np.mean(weights)}, std: {np.std(weights)} , total photons per bin: {np.sum(weights, axis=1)}, total photons overall: {np.sum(weights)}")
        self.weights = weights
        
    def _load_materials(self) -> None:
        """
        Load material maps and their corresponding attenuation coefficients from .nii.gz and .mat files.
        """

        mats_to_mu = {"ca": "MuCaimm", "water": "MuWaimm", "i": "MuIimm", "gd": "MuGdimm"}
        skip = ["segmentation", "water_recon", "i_recon"]
        # Get .nii.gz files in the data directory
        #print(os.listdir(self.data_dir))
        #print([f for f in os.listdir(self.data_dir) if f.endswith('.nii.gz')])
        mat_files = [f for f in os.listdir(self.data_dir) if f.endswith('.nii.gz')]
        for mat_file in mat_files:
            mat_name = os.path.splitext(os.path.splitext(mat_file)[0])[0]
            # print(f"mat_name: {mat_name} for mat_file: {mat_file}")
            if mat_name in skip:
                continue
            print(f"Loading {mat_name} from {mat_file}...")

            # Save the raw material maps and metadata

            
            data = cast(
                Nifti1Image,
                load(self.data_dir / mat_file),
            )
            vol: NDArray[np.float32] = data.get_fdata()

            # Extract the specified number of slices and convert to float32

            vol = self._extract_slices(vol, self.ray_trafo_params["slices_considered"]).astype(np.float32)

            # Extract metadata: affine transformation matrix and header information

            affine = data.affine
            assert affine is not None
            affine = affine.astype(np.float32)

            header = data.header

            # Update header information to reflect the new shape after slicing
            header.set_data_shape(vol.shape)

            # Store the volume and metadata in the mats dictionary

     
           
            # Load attenuation coefficients from .mat file
            print(f"Loading attenuation coefficients for {mat_name}...")
            mu_path = self.files_dir / "materials" / (mats_to_mu[mat_name] + ".mat")
            mu = np.asarray(
                        loadmat(mu_path)[mats_to_mu[mat_name]], 
                        dtype=np.float32
                        )

            # Update material data with volume, affine, header, and attenuation coefficients

            self.mats[mat_name].__setattr__("vol", vol)
            self.mats[mat_name].__setattr__("affine", affine)
            self.mats[mat_name].__setattr__("header", header)
            self.mats[mat_name].__setattr__("mu", mu)

        print(f"Loaded materials: {list(self.mats.keys())}... \n")

    def prep(self):
        print("Loading materials and weights... \n")
        self._load_materials()
        self._load_weights()
        self._setup_ray_trafo()

        weights = self.weights
        mats = self.mats

        # Scale materials to volume fractions for the attenuation model

        print("Normalizing materials... \n")
        mats = self._normalize_mats(mats)
        a_np = []
        f_np = []
        for mat in self.forward_mats:
            vol = mats[mat].vol
            mu = mats[mat].mu
            assert vol is not None and mu is not None, f"Material '{mat}' volume or attenuation data is not loaded."
            a_np.append(vol)
            f_np.append(mu.squeeze())

        # Stack the materials into a single array for forward projection
        a_np = np.stack(a_np, axis=0)  # Shape: (n_materials, D, H, W)
        f_np = np.stack(f_np, axis=0)  # Shape: (n_materials, n_energies)

        if weights.shape[1] != f_np.shape[1]:
            raise ValueError(
                f"Weights have {weights.shape[1]} energy samples, but attenuation curves have {f_np.shape[1]}."
            )
        if np.any(weights.sum(axis=1) <= 0):
            raise ValueError("Every energy bin must have a positive open-beam count.")
        return a_np, f_np, weights

    def pcct_forward_project(self, a_np: np.ndarray, f_np: np.ndarray, weights: np.ndarray) -> np.ndarray:


        """
        Forward projection of the materials to get the predicted AC.
        """
        # Convert the dictionary of materials to a single 3D array

        print("Forward projecting... \n")

        A_np = self.ray_trafo.forward(torch.from_numpy(a_np)).numpy()  # Shape: (n_angles, detector_shape[0], detector_shape[1], n_materials)

        if A_np.shape[0] != a_np.shape[0]:
            raise ValueError(f"Expected material-first projection output, got shape {A_np.shape}.")

        print(f"Diagnostics for A_np: min: {np.min(A_np, axis=(1, 2, 3))}, max: {np.max(A_np, axis=(1, 2, 3))}")
        print("Computing attenuation... \n")

        att = np.exp(-np.einsum('mazy,me->azye', A_np, f_np))
        if not np.isfinite(att).all():
            raise ValueError("Non-finite attenuation exponent from projected material path lengths.")

        print(att.shape, att.min(), att.max())

        print("Applying weights... \n")
        
        lam = np.einsum("be,azye->bazy", weights, att)  # Apply weights


        
        if not np.isfinite(lam).all() or np.any(lam < 0):
            raise ValueError("Predicted photon rates must be finite and nonnegative before Poisson sampling.")
        open_beam = weights.sum(axis=1)
        if np.any(lam > open_beam[:, None, None, None] * (1 + 1e-5)):
            raise ValueError("Attenuation increased the photon rate; check material values and attenuation units.")

        print("Simulating Poisson noise... \n")

        ep=1e-2            
        n = np.random.poisson(lam) + ep if self.with_noise else lam + ep  # Add a small constant to avoid log(0)
        print("Forward projection complete. \n")
        return n

    def _interpolate_zero_counts(self, n: np.ndarray) -> np.ndarray:
        """
        Interpolate zero counts in the predicted photon counts n to avoid issues in the Newton solver.
        """ 
        zeros = n == 0

        print(f"Interpolating zero counts in predicted photon counts... Number of zeros: {np.sum(zeros)} out of {n.size} voxels\n")

        if not np.any(zeros):
            return n  # No zeros to interpolate

        n_interp = n.copy()
        for b in range(n.shape[0]):  # Iterate over bins
            for z in range(n.shape[2]):  # Iterate over slices
                for y in range(n.shape[3]):  # Iterate over rows
                    zero_indices = np.where(zeros[b, :, z, y])[0]
                    if zero_indices.size > 0:
                        # Interpolate using linear interpolation
                        valid_indices = np.where(~zeros[b, :, z, y])[0]
                        if valid_indices.size > 1:
                            n_interp[b, :, z, y] = np.interp(
                                np.arange(n.shape[1]),
                                valid_indices,
                                n[b, valid_indices, z, y]
                            )
                        elif valid_indices.size == 1:
                            n_interp[b, :, z, y] = n[b, valid_indices[0], z, y]
                        else:
                            n_interp[b, :, z, y] = 1e-6  # Fallback to a small positive value



        return n_interp
    
    def pcct_forward_md(self, n: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Forward model for multi-material decomposition (MD) using the predicted photon counts n.
        
        Args:
            n (torch.Tensor or np.ndarray): Predicted photon counts with shape [nBins, nAngles, Z, Y]."""

        # Load solver

        solver = self._get_newton_solver()

        if isinstance(n, np.ndarray):
            n = torch.from_numpy(n).float()

        assert isinstance(n, torch.Tensor), f"Input n must be a torch.Tensor but got {type(n)}"

        with torch.no_grad():

            
            # Flatten n from [nBins, nAngles, Z, Y] to [P, nBins] where P = nAngles * Z * Y
            n_flat = n.permute(1, 2, 3, 0).reshape(-1, n.shape[0])  # Shape: [P, nBins]

            # Iterate over batches of n_flat to compute AC for each batch
            P = n_flat.shape[0]
            n_batches = (P + self.batch_size - 1) // self.batch_size
            AC_list = []
            for i in tqdm(range(n_batches), desc="Computing AC in batches", unit="batch"):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, P)
                n_batch = n_flat[start:end]  # Shape: [batch_size, nBins]

                # Solve for AC using the Newton solver

                sol = solver.solve_from_counts(n_batch, AC_init=torch.zeros(n_batch.shape[0], len(self.md_mats), device=n.device, dtype=n.dtype))
                AC_list.append(sol["AC"])  # Shape: [batch_size, nMaterials]



            # Concatenate all batches to get the final AC tensor
            AC = torch.cat(AC_list, dim=0)  # Shape: [P, nMaterials]
            # Reshape AC back to [nAngles, Z, Y, nMaterials] and permute to [nMaterials, nAngles, Z, Y]
            AC = AC.view(n.shape[1], n.shape[2], n.shape[3], -1).permute(3, 0, 1, 2)

            # plot the material sinograms for the first slice:

            plot_material_basis_sinograms(AC.cpu().numpy(), self.md_mats)

            # Backproject the AC to get the reconstructed material maps

            print(f"Backprojecting AC to get reconstructed material maps... \n")

            aC = self.ray_trafo.backward(AC)  # Shape: [nMaterials, D, H, W]

            print(f"Reconstructed material maps shape: {aC.shape}, min: {aC.min()}, max: {aC.max()}")

        return aC

    def forward(self, output_path: Path | None = None) -> np.ndarray:


        print(f"Running forward model {'with noise' if self.with_noise else 'without noise'} and {self.ray_trafo_params['slices_considered']} slices using materials: {self.forward_mats} and solving for materials {self.md_mats}... \n")

        a_np, f_np, weights = self.prep()

        n = self.pcct_forward_project(a_np, f_np, weights) # 8 x 512 x 2 x 512

        # Interpolate zero counts in n to avoid issues in the Newton solver

        n = self._interpolate_zero_counts(n)
        n = np.clip(n, 1e-2, None)  # Ensure all counts are positive to avoid log(0)

        # Visualize the predicted photon counts n

        log_n = np.log(n + 1e-6)  # Add a small constant to avoid log(0)

        plot_log_counts(log_n)  # Plot the log of predicted photon counts for all bins and slices



        aC = self.pcct_forward_md(n)

        if isinstance(aC, torch.Tensor):
            aC = aC.cpu().numpy().astype(np.float32)

        if output_path is None:
            output_path = SAVE_DIR

        output_path.mkdir(parents=True, exist_ok=True)
        self._save_nib(aC, output_path) 



        return aC

    def _save_nib(self, aC: torch.Tensor | np.ndarray, output_path: Path) -> None:
        """
        Save a 3D numpy array as a NIfTI file using the provided affine and header.
        """

        assert hasattr(self, "mats") and isinstance(self.mats, dict), "Materials must be loaded and stored in self.mats before saving."

        suffix = "_recon_contrast.nii.gz" if self.with_contrast else "_recon_no_contrast.nii.gz"

        aC = self._denormalize_reconstruction(aC)

        for m_name, md_vol in zip(self.md_mats, aC):
            if m_name not in self.mats:
                raise ValueError(f"Material '{m_name}' not found in self.mats. Available materials: {list(self.mats.keys())}")

            affine = self.mats[m_name].affine
            header = self.mats[m_name].header

            # Create a NIfTI image and save it
            img = Nifti1Image(md_vol.astype(np.float32), affine, header)
            save(img, output_path / f"{m_name}{suffix}")
            print(f"Saved reconstructed material map for '{m_name}' to {output_path / (m_name + suffix)}\n")
     




def main():

    model = PolyChromaticForwardModel(batch_size=2048, with_contrast=True)
    aC = model.forward()
    model._update_mode(with_contrast=False, with_noise=False)
    aC_no_contrast = model.forward()
    


if __name__ == "__main__":
    main()
