from __future__ import annotations
from typing import List, Optional, Tuple, List

from operators.tomo import RayTrafoModule
from data_processing.utils import segment_id

from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import distance_transform_edt, label as nd_label
from tqdm import tqdm


from article.attenuations import LinearMaterialDecomposition
from optim.solvers import NewtonSolver, ResidualSolver

from config import *


ARTIFACT_DIR = REPO_ROOT / "AttenuationMD_python" / "artifacts"
DEBUG_DIR = REPO_ROOT / "AttenuationMD_python" / "debug"


class ShallowCalibrationNet(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Forward pass through the calibration network. """
        
        assert x.dim() in (2, 3) and x.shape[-1] == self.net[0].in_features, f"Expected input to have shape [N, {self.net[0].in_features}], got {x.shape}"

        # Flatten the input if it has more than 2 dimensions
        if x.dim() > 2:
            original_shape = x.shape
            x = x.view(-1, self.net[0].in_features) 
            y = self.net(x)
            y = y.view(*original_shape[:-1], self.net[-1].out_features)
        else:
            y = self.net(x)
        return y


class CalibratedForwardModel:
    SUPPORTED_CALIBRATION_MODES = {"shallow-net", "residual-net"}


    def __init__(
            self, 
            fB: torch.Tensor, 
            fR: torch.Tensor, 
            fC: torch.Tensor, 
            weights: torch.Tensor, 
            Eactvector: torch.Tensor, 
            output_dir: Path, 
            batch_size=2048, 
            num_mats=2, 
            num_contrast=2, 
            normalize_materials: bool = False, 
            objective: str = "KL", # Options: "L1", "L2", "Lasso", "KL", "KLsym",
            debug: bool = False,
            ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device} for computations... \n")

        
        self.fB = fB.to(device=self.device)
        self.fR = fR.to(device=self.device)
        self.fC = fC.to(device=self.device)

        self.weights = weights.to(device=self.device)
        self.Eactvector = Eactvector.to(device=self.device)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.normalize_materials = normalize_materials
        self.objective = objective
        self.debug = debug
        # Store the number of materials and contrast agents for later use

        self.num_mats = num_mats
        self.num_contrast = num_contrast

        self.cache = {}  # Cache for storing intermediate results to avoid redundant computations
        self.calibration_scan_id = None  # Scan ID used for the initial calibrator training, if any

        # Set global constants

        self.MATERIALS = ["Iodine", "Gadolinium"]
        self.DENSITIES = [
            4933,  # Iodine
            7870,  # Gadolinium
        ]
        self.WINDOW_SETTINGS = [
            (0, 15),  # Iodine
            (0, 15),  # Gadolinium
        ]


    def _resolve_calibration_mode(self, calibration_mode: str | None, default: str = "shallow-net") -> str:
        mode = default if calibration_mode is None else str(calibration_mode)
        if mode not in self.SUPPORTED_CALIBRATION_MODES:
            raise ValueError(f"Unsupported calibration mode: {mode}. Expected one of {sorted(self.SUPPORTED_CALIBRATION_MODES)}")
        return mode

    def _extract_gt(self, X: torch.Tensor) -> torch.Tensor:
        """Extract the ground truth AC from the input sinogram X. The input X is expected to have shape [2 * num_mats + num_contrast x P]."""
        assert X.shape[0] in (2 * self.num_mats + self.num_contrast, 2 * self.num_mats), f"Expected input to have {2 * self.num_mats + self.num_contrast} or {2 * self.num_mats} channels, got {X.shape[0]}"

        if X.shape[0] == 2 * self.num_mats:
            # Return a zero tensor if ground truth AC is not present in the input
            return torch.zeros((self.num_contrast, X.shape[1]), dtype=X.dtype, device=self.device)
        return X[2*self.num_mats:2*self.num_mats+self.num_contrast]  # Ground truth AC

    def _extract_roi_components(self, mask: np.ndarray, erosion: float = 1.0) -> list[np.ndarray]:
          
        def _validate_roi_erosion(erosion: float) -> float:
            erosion = float(erosion)
            if not (0.0 < erosion <= 1.0):
                raise ValueError(f"ROI erosion must be in (0, 1], got {erosion}")
            return erosion

        def _plot_mask_and_components(mask: np.ndarray, components: list[np.ndarray], output_path: Path) -> None:
            """Plot the original mask and its connected components for debugging purposes."""
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(mask, cmap='gray')
            plt.title("Original Mask")
            plt.axis('off')


            # Ensure the output directory exists before saving the plot

            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving ROI components plot to {output_path}... \n")
            plt.savefig(output_path)
            plt.close()

        erosion = _validate_roi_erosion(erosion)
        labeled, n_roi = nd_label(mask > 0.95)
        components: list[np.ndarray] = []

        for idx in range(1, n_roi + 1):
            component = labeled == idx
            if not np.any(component):
                continue

            if erosion < 1.0:
                area = float(component.sum())
                eq_radius = max(np.sqrt(area / np.pi), 1.0)
                shrink_margin = (1.0 - erosion) * eq_radius
                distances = distance_transform_edt(component)
                eroded = distances >= shrink_margin
                if not np.any(eroded):
                    assert isinstance(distances, np.ndarray), f"Expected distances to be a numpy array, got {type(distances)}"
                    max_dist = distances.max()
                    eroded = distances == max_dist if max_dist > 0 else component
                component = eroded

            components.append(component)

        if not components:
            raise ValueError("No connected ROI components found in the mask")
        

        _plot_mask_and_components(mask, components, output_path=DEBUG_DIR / "roi_components.png")

        return components

    def _get_ray_trafo(self, scan_id: int | None=None, **kwargs) -> RayTrafoModule:

        if scan_id is not None:
            if scan_id not in TOMO:
                raise ValueError(f"Scan ID {scan_id} not found in TOMO configuration. Available scan IDs: {list(TOMO.keys())}")
            kwargs.update(TOMO[scan_id])

        return RayTrafoModule(**kwargs)

    def _get_uncalibrated_forward_model(
            self, fB: torch.Tensor | None = None, 
            fR: torch.Tensor | None = None, 
            fC: torch.Tensor | None = None, 
            weights: torch.Tensor | None = None,
            **kwargs
            ) -> ResidualSolver | NewtonSolver:

        fB = fB if fB is not None else self.fB
        fR = fR if fR is not None else self.fR
        fC = fC if fC is not None else self.fC
        weights = weights if weights is not None else self.weights

        uncal_forward =  ResidualSolver(fB, fR, fC, weights, objective=self.objective, **kwargs)

        print(f"Initialized uncalibrated forward model of type {type(uncal_forward).__name__} with objective: {self.objective}...  \n")

        return uncal_forward

    def _get_calibrator(self, in_dim: int = 2, hidden_dim: int = 16, out_dim: int = 2) -> ShallowCalibrationNet:
        return ShallowCalibrationNet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(self.device)

    def _normalize_material_stack(
        self,
        A: torch.Tensor,
        f: torch.Tensor,
        is_flattened: bool,
        contains_ground_truth: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep = 1e-6

        if not self.normalize_materials:
            print("Material normalization is disabled. Using scales of 1 for all materials... \n")
            scales = torch.ones((f.shape[0], 1), dtype=f.dtype, device=self.device)
            return A, f, scales


        print("Normalizing material stack. Computing scales based on the mean of the material basis functions... \n")

        scales = torch.mean(f, dim=1, keepdim=True).to(device=self.device) + ep

        if contains_ground_truth:
            A_normalized = A * scales if is_flattened else A * scales.view(-1, 1, 1)
        else:
            active_scales = scales[0: 2 * self.num_mats, :]
            A_normalized = A * active_scales if is_flattened else A * active_scales.view(-1, 1, 1)

        f_normalized = f / scales
        return A_normalized, f_normalized, scales
  
    def _get_active_ray_mask(self, AR: torch.Tensor, threshold: list[float | torch.Tensor] | None = None) -> torch.Tensor:
        """Returns a boolean mask indicating which flattened rays are active."""
        if AR.dim() != 2 or AR.shape[0] != self.num_mats:
            raise ValueError(f"Expected AR to have shape [{self.num_mats}, P], got {AR.shape}")
        assert torch.sum(AR) > 0, "The input AR tensor is empty. Cannot compute active ray mask."



        if threshold is not None:
            assert len(threshold) == AR.shape[0], "There should be one threshold value for each material in AR"
        else:
            threshold = []
            for m in range(AR.shape[0]):
                non_zero_AR = AR[m, ...][AR[m, ...] > 0]
                threshold_m = 1e-2 * torch.mean(non_zero_AR) if non_zero_AR.numel() > 0 else 0.0
                threshold.append(threshold_m)

        threshold_tensor = torch.tensor(threshold, dtype=AR.dtype, device=self.device)
        active_mask = (AR > threshold_tensor.view(-1, 1)).any(dim=0)

        print(f"Found {active_mask.sum().item()} active rays out of {AR.shape[1]} total rays... \n")
    
        return active_mask

    def _get_linear_forward_model(self, fB: torch.Tensor | None = None, fR: torch.Tensor | None = None, fC: torch.Tensor | None = None, weights: torch.Tensor | None = None, fWa: torch.Tensor | None = None, tWa: float | None = 0) -> LinearMaterialDecomposition:
        """Get the linear forward model for material decomposition. If fWa and tWa are not provided, they will be initialized to default values."""

    
        
        fR = fR if fR is not None else self.fR
        fC = fC if fC is not None else self.fC
        weights = weights if weights is not None else self.weights

        fWa = fR[0, :] if fWa is None else fWa

        # move to numpy for LinearMaterialDecomposition

        fR_np = fR.cpu().detach().numpy()
        fC_np = fC.cpu().detach().numpy()
        fWa_np = fWa.cpu().detach().numpy() if fWa is not None else fB[0, :].cpu().detach().numpy()
        weights_np = weights.cpu().detach().numpy()

        if tWa is None:
            tWa = 200
        print(f"Initialized linear forward model with water thickness {tWa}... \n")
        return LinearMaterialDecomposition(fC_np, fR_np, weights_np, f_wa=fWa_np, pathlength_wa=tWa) # We compute the inverse mapping since it is more stable for some reason.

    def _calibration_loss(self, AC_pred: torch.Tensor, AC_gt: torch.Tensor, alpha: float = 0.0, energy_vals: list[float] | None = None, data_loss_scale: torch.Tensor = torch.tensor(1.0), spectral_loss_scale: torch.Tensor = torch.tensor(1.0)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the calibration loss as a combination of MSE and spectral regularization."""
        
        
        def _energy_to_index(EkeV: float) -> int:
            """Convert energy in keV to the corresponding index in the energy grid."""
            Evector_np = self.Eactvector.cpu().detach().numpy()
            if EkeV < Evector_np[0] or EkeV > Evector_np[-1]:
                raise ValueError(f"Energy {EkeV} keV is out of bounds for the energy grid [{Evector_np[0]}, {Evector_np[-1]}]")
            return int(np.round((EkeV - Evector_np[0]) / (Evector_np[1] - Evector_np[0])))
        
        def _spectral_loss(AC_pred: torch.Tensor, AC_gt: torch.Tensor, energy_indices: list[int]) -> torch.Tensor:
            """Compute the spectral regularization loss based on selected energy indices."""
            if not energy_indices:
                return torch.tensor(0.0, device=self.device, dtype=AC_pred.dtype)
            
            fC_selected = self.fC.to(device=self.device, dtype=AC_pred.dtype).index_select(1, torch.tensor(energy_indices, device=self.device, dtype=torch.long))
            pred_spectrum = torch.einsum('mp, me -> pe', AC_pred, fC_selected)  # [P x nSelectedE]
            gt_spectrum = torch.einsum('mp, me -> pe', AC_gt, fC_selected)  # [P x nSelectedE]
            return torch.mean((pred_spectrum - gt_spectrum) ** 2)


        energy_indices = [ _energy_to_index(E) for E in energy_vals ] if energy_vals is not None else []


        ep = 1e-6
        data_loss_scale = torch.clip(torch.as_tensor(data_loss_scale, device=self.device, dtype=AC_pred.dtype), min=ep)
        spectral_loss_scale = torch.clip(torch.as_tensor(spectral_loss_scale, device=self.device, dtype=AC_pred.dtype), min=ep)

        data_loss = nn.MSELoss(reduction='mean')(AC_pred, AC_gt) / data_loss_scale
        spectral_loss = _spectral_loss(AC_pred, AC_gt, energy_indices) / spectral_loss_scale

        total_loss = data_loss + alpha * spectral_loss
        return total_loss, data_loss, spectral_loss

    def _uncalibrated_forward(self, A: torch.Tensor, ACinit: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """The uncalibrated forward model. It takes a sinogram A of shape [4 x P] or [6 x P] and returns the predicted AC of shape [2 x P]. I"""


        if A.device != self.device:
            print(f"Moving input sinogram A from device {A.device} to {self.device} for computations... \n")
            A = A.to(device=self.device)
        if ACinit is not None and ACinit.device != self.device:
            print(f"Moving initial guess ACinit from device {ACinit.device} to {self.device} for computations... \n")
            ACinit = ACinit.to(device=self.device)


        assert A.dim() in (2, 3) and A.shape[0] in (2 * self.num_mats, 2 * self.num_mats + self.num_contrast), \
            f"A should either be a sinogram or a flattened sinogram with batch dimension equal to f{self.num_mats + self.num_contrast} or f{2 * self.num_mats + self.num_contrast} but got shape {A.shape}"



        def _prepare_input(A: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
            """Prepares the input sinogram for the forward model. It 

            1.) Selects the active rays based on the mask
            2.) Divides the data into batches specified by self.batch_size


            """
            
            A_active = A[:, mask]  # Select active rays

            num_batches = (A_active.shape[1] + self.batch_size - 1) // self.batch_size

            A_batches = [A_active[:, i * self.batch_size:(i + 1) * self.batch_size] for i in range(num_batches)]

            return A_batches



            

        is_flattened = A.dim() == 2
        contains_ground_truth = A.shape[0] == (2 * self.num_mats + self.num_contrast)

        # Normalize the input sinogram and the material basis functions

        f = torch.cat([self.fB, self.fR, self.fC], dim=0) if contains_ground_truth else torch.cat([self.fB, self.fR], dim=0)
        A, f, scales = self._normalize_material_stack(A, f, is_flattened, contains_ground_truth)


        # Get forward model

        fB = f[0:self.num_mats]
        fR = f[self.num_mats:2*self.num_mats]
        fC = f[2*self.num_mats:2*self.num_mats+self.num_contrast] if contains_ground_truth else self.fC


        forward_model = self._get_uncalibrated_forward_model(fB=fB, fR=fR, fC=fC, weights=self.weights, E=self.Eactvector, **kwargs)
        
        # Preprocess the input sinogram for the forward model

        if not is_flattened:

            sino_sz = A.shape[1:3]  # [nAngls,e nDet]

            # Flatten the batch dimension
            A = A.contiguous().view(A.shape[0], -1)  # [2 *nMat + nContrast, nAngles*nDet]


        # Get the active mask based on the AR values

        AR = A[self.num_mats:2*self.num_mats]

        assert torch.sum(AR > 0) > 0, "The input AR tensor is empty. Cannot compute active ray mask."

        mask = self._get_active_ray_mask(AR, threshold=None if "threshold" not in kwargs else kwargs["threshold"])

        # prepare data for the forward model

        A_batches = _prepare_input(A, mask)

        # Preprocess the initial guess for AC if provided

        if ACinit is not None:
            print(f"Using provided initial guess for AC with shape {ACinit.shape}... \n")
            assert ACinit.shape[0] == self.num_contrast, f"Expected ACinit to have {self.num_contrast} channels, got {ACinit.shape[0]}"

            if not is_flattened:
                ACinit = ACinit.view(ACinit.shape[0], -1)  # [num_contrast x P]

            assert ACinit.shape[0] == self.num_contrast, f"Expected ACinit to have {self.num_contrast} channels, got {ACinit.shape[0]}"
            
            # Normalize the initial guess for AC if provided

            ACinit = ACinit * scales[4:4+self.num_contrast].view(-1, 1) if (contains_ground_truth) else ACinit

            ACinit_batches = _prepare_input(ACinit, mask)
        else:
            print("No initial guess for AC provided. It will be initialized to zeros in the solver... \n")
            ACinit_batches = [torch.zeros(self.num_contrast, A_batch.shape[1], dtype=A.dtype, device=self.device) for A_batch in A_batches]

        # Solve

        AC_pred = []
        for i, (A_batch, ACinit_batch) in enumerate(tqdm(zip(A_batches, ACinit_batches), desc=f"Solving batches", total=len(A_batches))):
            AB_batch = A_batch[0:self.num_mats]
            AR_batch = A_batch[self.num_mats:2*self.num_mats]

            sol = forward_model.newton_solve(
                AB_batch.T,
                AR_batch.T,
                AC_init=ACinit_batch.T,
            )            # Transpose to match input shape [P x nMat]

            


            # Plot the optimization process for debugging purposes if requested
            
            if self.debug:
                # forward_model.plot_AC_optim(AB_batch.T, AR_batch.T, output_dir=self.output_dir / f"batch_{i}")  # Transpose to match input shape [P x nMat]
                forward_model.plot_AC_optim_with_curves(AB_batch.T, AR_batch.T, fR, fC, output_dir=self.output_dir / f"batch_{i}")  # Transpose to match input shape [P x nMat]
                # input(f"Press Enter to continue after inspecting the optimization plots for batch {i}... \n")
                # forward_model.plot_metrics(output_dir=self.output_dir / f"batch_{i}")  # Save metrics plot for the current batch

            
            AC_batch = sol["AC"]
            assert isinstance(AC_batch, torch.Tensor), f"Expected AC_batch to be a torch.Tensor, got {type(AC_batch)}"
            AC_pred.append(AC_batch.T)                                                                             # Transpose back to match output shape [nMat x P_batch]

        # Postprocess the output to match the input shape

        AC_pred_active = torch.cat(AC_pred, dim=1)  # [nMat x P_active]

        # Remove extreme outliers (4 sigma) from the uncalibrated predictions for better training stability

        AC_pred_mean, AC_pred_std = torch.mean(AC_pred_active, dim=1, keepdim=True), torch.std(AC_pred_active, dim=1, keepdim=True)
        AC_pred_active = torch.clamp(AC_pred_active, min=AC_pred_mean - 4 * AC_pred_std, max=AC_pred_mean + 4 * AC_pred_std)



        # zero fill unprocessed rays

        AC = torch.zeros((self.num_contrast, A.shape[1]), dtype=A.dtype, device=self.device)  # [2 x P]

        AC[:, mask] = AC_pred_active

        # Reshape the output to match the input shape

        if not is_flattened:
            AC = AC.view(self.num_contrast, *sino_sz)  # [num_contrast x nAngles x nDet]

        # Scale the output back to the original units


        AC = AC / scales[4:4+self.num_contrast].view(-1, 1, 1) if (contains_ground_truth & (not is_flattened)) else AC / scales[4:4+self.num_contrast].view(-1, 1) if (contains_ground_truth & is_flattened) else AC

        return AC, mask

    def _calibrated_forward(self, A, **kwargs):
        """The calibrated forward model. It takes a sinogram A of shape [4 x P] or [6 x P] and returns the predicted AC of shape [2 x P]. I"""

        AC_pred, _ = self._uncalibrated_forward(A, **kwargs)
        if kwargs.get("use_learned_residual", False) and not hasattr(self, "calibrator"):
            return AC_pred
        assert hasattr(self, "calibrator"), "Calibrator network is not initialized. Please call _get_calibrator() first."
        AC_cal = self.calibrator(AC_pred.T).T  # Transpose to match input shape
        return AC_cal

    def _linear_forward(self, aR, **kwargs):
        """The linear forward model. It takes a sinogram A of shape [4 x P] or [6 x P] and returns the predicted AC of shape [2 x P]. I"""

        # move tensor to numPy for LinearMaterialDecomposition

        aR_np = aR.cpu().detach().numpy()
        
        if not hasattr(self, "linear_forward_model"):
            self.linear_forward_model = self._get_linear_forward_model(**kwargs)

        
        
        aC_pred_linear = self.linear_forward_model.apply_inverse(aR_np)  # Transpose to match input shape
        return aC_pred_linear

    def _load_scan_from_id(self, scan_id: int, refine_mask: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the scan data for a given scan ID."""



        #aB, aR, aC_gt_img, mask = load_data(scan_id, create_mask=create_mask)

        out = segment_id(scan_id, refine_mask=refine_mask)

        aB, aR, aC_gt_img, mask = out["background"], out["signal"], out["ground_truth"], out["mask"]


        assert torch.sum(aR) > 0, f"The input aR tensor should be strictly positive. Got sum(aR) = {torch.sum(aR)}"

        a = torch.cat([aB, aR], dim=0) if aC_gt_img is None else torch.cat([aB, aR, aC_gt_img], dim=0)

        # Rotate 90 degrees clockwise (k=-1) so the table is at the bottom; applied before the forward
        # projection, so sinograms, ground truth and masks all stay consistent with the rotated maps

        if scan_id in ROTATE_CW_SCANS:
            a = torch.rot90(a, k=-1, dims=(-2, -1)).contiguous()
            mask = torch.rot90(mask, k=-1, dims=(-2, -1)).contiguous()


        # a = torch.cat([aB, aR], dim=0) if aC_gt_img is None else torch.cat([aB, aR, aC_gt_img], dim=0)

        # Move the data to the appropriate device for computations

        a = a.to(device=self.device)
        mask = mask.to(device=self.device)

        return a, mask

    def calibrate_from_id(self, cal_id: int, ray_trafo_params: dict = {}, calibrator_params: dict | None =None, include_linear_model: bool = True) -> torch.Tensor | None:
        """Calibrate the forward model using a calibration scan with ID cal_id. Returns the calibrated AC predictions."""
   
        
        def train_calibrator(AC_pred: torch.Tensor, AC_gt: torch.Tensor, calibrator: ShallowCalibrationNet, num_epochs: int = 500, learning_rate: float = 1e-3, energy_vals = [60, 75]) -> tuple[ShallowCalibrationNet, dict[str, list[float]]]:
            """Train the calibrator network to map uncalibrated AC predictions to ground truth AC values."""
            optimizer = optim.Adam(calibrator.parameters(), lr=learning_rate)
            loss_hist = {"loss": [], "data_loss": [], "spectral_loss": []}


            num_batches = (AC_pred.shape[1] + self.batch_size - 1) // self.batch_size

            bacthed_data = [(AC_pred[:, i*self.batch_size:(i+1)*self.batch_size], AC_gt[:, i*self.batch_size:(i+1)*self.batch_size]) for i in range(num_batches)]

            # get loss scales

            _, data_scale, spectral_scale = self._calibration_loss(AC_pred, AC_gt, energy_vals=energy_vals)


            for epoch in range(num_epochs):


                for AC_pred_batch, AC_gt_batch in bacthed_data:

                    optimizer.zero_grad()
                    AC_cal = calibrator(AC_pred_batch.T)                                                                                     # Transpose to match input shape
                    loss, data_loss, spectral_loss = self._calibration_loss(AC_cal.T, AC_gt_batch, energy_vals=energy_vals, data_loss_scale=data_scale, spectral_loss_scale=spectral_scale)                   # Transpose to match target shape
                    loss.backward() 
                    optimizer.step()

                    # Store loss history for analysis and debugging

                    loss_hist["loss"].append(loss.item())
                    loss_hist["data_loss"].append(data_loss.item())
                    loss_hist["spectral_loss"].append(spectral_loss.item())

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}, Data Loss: {data_loss.item()}, Spectral Loss: {spectral_loss.item()}")

            return calibrator, loss_hist
        
        def plot_loss(loss_hist: dict[str, list[float]], output_path: Path) -> None:
            """Plot the training loss history and save it to a file."""
            plt.figure(figsize=(10, 6))
            plt.plot(loss_hist["loss"], label="Total Loss")
            plt.plot(loss_hist["data_loss"], label="Data Loss")
            plt.plot(loss_hist["spectral_loss"], label="Spectral Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Calibrator Training Loss History")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

        # Initialize operators if they haven't been initialized yet

        self.ray_trafo = self._get_ray_trafo(scan_id=cal_id, **ray_trafo_params)

        if not hasattr(self, "calibrator") and calibrator_params is not None:
            self.calibrator = self._get_calibrator(**calibrator_params)
            self.calibration_scan_id = cal_id

        if include_linear_model and not hasattr(self, "calibrator_linear") and calibrator_params is not None:
            self.calibrator_linear = self._get_calibrator(**calibrator_params)

        # load the calibration scan data

        a, mask_img = self._load_scan_from_id(cal_id, refine_mask=True)
        a0 = a[0:2] + a[2:4]  # Initial material maps
        aC_gt = self._extract_gt(a)  # Ground truth AC

        aR = a[self.num_mats:2*self.num_mats]  # Extract the AR sinogram
        assert torch.sum(aR > 0) > 0, "The input AR tensor is empty. Cannot compute active ray mask."

        with torch.no_grad():

            # forward projection to get the predicted AC

            ray_trafo = self.ray_trafo if hasattr(self, "ray_trafo") else self._get_ray_trafo(scan_id=cal_id, **ray_trafo_params)
            A = ray_trafo.forward(a)
            AC_gt = self._extract_gt(A)  # Ground truth AC

            # Run uncalibrated forward model to get the predicted AC

            AC_pred_uncal, mask = self._uncalibrated_forward(A, ACinit=None)


            aC_pred_uncal =  ray_trafo.backward(AC_pred_uncal)

            # flatten the AC_pred and AC_gt for training the calibrator

            AC_pred_flat = AC_pred_uncal.view(self.num_contrast, -1)            # [2 x P]
            AC_gt_flat = AC_gt.view(self.num_contrast, -1)                      # [2 x P]

            # mask out the inactive rays for training

            AC_pred_masked = AC_pred_flat[:, mask]
            AC_gt_masked = AC_gt_flat[:, mask]

        if calibrator_params is not None:

            # Train the calibrator network (needs gradients enabled, so this runs outside torch.no_grad())

            self.calibrator, loss_hist = train_calibrator(AC_pred_masked, AC_gt_masked, self.calibrator)

            with torch.no_grad():
                AC_pred_cal = self.calibrator(AC_pred_uncal.permute(1, 2, 0)).permute(2, 0, 1)  # Transpose to match input shape

                plot_loss(loss_hist, PLOT_DIR / f"calibrator_loss_{cal_id}.png")

                aC_pred_cal = ray_trafo.backward(AC_pred_cal)
        else:
            print(f"Warning: no calibrator parameters provided: only uncalibrated predictions will be cached for scan ID {cal_id}... \n")
            AC_pred_cal, aC_pred_cal = None, None

        if include_linear_model:
            with torch.no_grad():
                print(f"Computing linear forward model predictions for scan ID {cal_id}... \n")
                # Compute the linear forward model predictions for comparison

                AR = A[self.num_mats:2*self.num_mats]  # Extract the AR sinogram
                AC_pred_linear = torch.from_numpy(self._linear_forward(AR)).to(device=self.device, dtype=A.dtype)  # Convert to torch.Tensor and move to the appropriate device
                aC_pred_linear = ray_trafo.backward(AC_pred_linear)

            if calibrator_params is not None:

                # Train a separate calibrator network for the linear model predictions (needs gradients enabled)

                AC_pred_linear_flat = AC_pred_linear.view(self.num_contrast, -1)  # [2 x P]
                AC_pred_linear_masked = AC_pred_linear_flat[:, mask]

                self.calibrator_linear, loss_hist_linear = train_calibrator(AC_pred_linear_masked, AC_gt_masked, self.calibrator_linear)

                with torch.no_grad():
                    AC_pred_linear_cal = self.calibrator_linear(AC_pred_linear.permute(1, 2, 0)).permute(2, 0, 1)  # Transpose to match input shape

                    plot_loss(loss_hist_linear, PLOT_DIR / f"calibrator_linear_loss_{cal_id}.png")

                    aC_pred_linear_cal = ray_trafo.backward(AC_pred_linear_cal)
            else:
                AC_pred_linear_cal, aC_pred_linear_cal = None, None
        else:
            aC_pred_linear = None
            AC_pred_linear_cal, aC_pred_linear_cal = None, None

        with torch.no_grad():

            # Save the calibration to cache to avoid redundant computations in future evaluations

            self.cache[cal_id] = {
                "mode": "cal",
                "calibration_mode": "shallow-net",
                "AC_pred_cal": AC_pred_cal.cpu().detach().numpy() if isinstance(AC_pred_cal, torch.Tensor) else AC_pred_cal,
                "AC_gt": AC_gt.cpu().detach().numpy() if isinstance(AC_gt, torch.Tensor) else AC_gt,
                "AC_pred_uncal": AC_pred_uncal.cpu().detach().numpy() if isinstance(AC_pred_uncal, torch.Tensor) else AC_pred_uncal,
                "aC_pred_cal": aC_pred_cal.cpu().detach().numpy() if isinstance(aC_pred_cal, torch.Tensor) else aC_pred_cal,
                "aC_gt": aC_gt.cpu().detach().numpy() if isinstance(aC_gt, torch.Tensor) else aC_gt,
                "aC_pred_uncal": aC_pred_uncal.cpu().detach().numpy() if isinstance(aC_pred_uncal, torch.Tensor) else aC_pred_uncal,
                "aC_pred_linear": aC_pred_linear.cpu().detach().numpy() if isinstance(aC_pred_linear, torch.Tensor) else aC_pred_linear,
                "AC_pred_linear_cal": AC_pred_linear_cal.cpu().detach().numpy() if isinstance(AC_pred_linear_cal, torch.Tensor) else AC_pred_linear_cal,
                "aC_pred_linear_cal": aC_pred_linear_cal.cpu().detach().numpy() if isinstance(aC_pred_linear_cal, torch.Tensor) else aC_pred_linear_cal,
                "a0": a0.cpu().detach().numpy() if isinstance(a0, torch.Tensor) else a0,
                "mask": mask_img.cpu().detach().numpy() if isinstance(mask_img, torch.Tensor) else mask_img,
                }

        return AC_pred_cal

    def eval_from_id(self, eval_id) -> torch.Tensor | None:
        """Evaluate the calibrated forward model on a scan with ID eval_id. Returns the calibrated AC predictions."""
        
        assert hasattr(self, "calibrator"), "Calibrator network is not initialized. Please call calibrate_from_id() first."

        # Create the transform based on the eval_id

        ray_trafo = self._get_ray_trafo(scan_id=eval_id) 
                

        # load the evaluation scan data

        if eval_id in self.cache:
           print(f"Using cached results for scan ID {eval_id}... \n")
           cached = self.cache[eval_id]
           return cached["AC_pred_cal"] if isinstance(cached["AC_pred_cal"], torch.Tensor) else torch.from_numpy(cached["AC_pred_cal"])

        self.calibrator.eval()
        if hasattr(self, "calibrator_linear"):
            self.calibrator_linear.eval()

        with torch.no_grad():
            a, mask_img = self._load_scan_from_id(eval_id, refine_mask=True)
            a0 = a[0:2] + a[2:4]  # Initial material maps
            assert a.shape[0] == (2 * self.num_mats + self.num_contrast), f"Expected input to have {2 * self.num_mats + self.num_contrast} channels, got {a.shape[0]}"

            aC_gt = a[2*self.num_mats:2*self.num_mats+self.num_contrast]  # Ground truth AC

            # forward projection to get the predicted AC

            A = ray_trafo.forward(a)
            AC_gt = A[2*self.num_mats:2*self.num_mats+self.num_contrast]  # Ground truth AC

            # Run calibrated forward model to get the predicted AC


            AC_pred_uncal = self._uncalibrated_forward(A, ACinit=None)[0]

            AC_pred_cal = self.calibrator(AC_pred_uncal.permute(1, 2, 0)).permute(2, 0, 1)  # Transpose to match input shape

            # Backproject the AC predictions to get the reconstructed images

            ac_pred_cal = ray_trafo.backward(AC_pred_cal)
            ac_pred_uncal = ray_trafo.backward(AC_pred_uncal)

            # Calculate the linear forward model predictions for comparison

            aR = a[self.num_mats:2*self.num_mats]  # Extract the AR sinogram
            aC_pred_linear = self._linear_forward(aR)

            # Calculate the calibrated linear model predictions, in the same (sinogram) domain the
            # calibrator_linear network was trained on in calibrate_from_id, then backproject

            aC_pred_linear_cal = None
            if hasattr(self, "calibrator_linear"):
                AR = A[self.num_mats:2*self.num_mats]  # Extract the AR sinogram
                AC_pred_linear = torch.from_numpy(self._linear_forward(AR)).to(device=self.device, dtype=A.dtype)
                AC_pred_linear_cal = self.calibrator_linear(AC_pred_linear.permute(1, 2, 0)).permute(2, 0, 1)  # Transpose to match input shape
                aC_pred_linear_cal = ray_trafo.backward(AC_pred_linear_cal)

            # Save the evaluation results to cache to avoid redundant computations in future evaluations

            self.cache[eval_id] = {
                "mode": "eval",
                "AC_pred_cal": AC_pred_cal.cpu().detach().numpy() if isinstance(AC_pred_cal, torch.Tensor) else AC_pred_cal,
                "AC_pred_uncal": AC_pred_uncal.cpu().detach().numpy() if isinstance(AC_pred_uncal, torch.Tensor) else AC_pred_uncal,
                "AC_gt": AC_gt.cpu().detach().numpy() if isinstance(AC_gt, torch.Tensor) else AC_gt,
                "aC_pred_cal": ac_pred_cal.cpu().detach().numpy() if isinstance(ac_pred_cal, torch.Tensor) else ac_pred_cal,
                "aC_pred_uncal": ac_pred_uncal.cpu().detach().numpy() if isinstance(ac_pred_uncal, torch.Tensor) else ac_pred_uncal,
                "aC_gt": aC_gt.cpu().detach().numpy() if isinstance(aC_gt, torch.Tensor) else aC_gt,
                "mask": mask_img.cpu().detach().numpy() if isinstance(mask_img, torch.Tensor) else mask_img,
                "aC_pred_linear": aC_pred_linear.cpu().detach().numpy() if isinstance(aC_pred_linear, torch.Tensor) else aC_pred_linear,
                "aC_pred_linear_cal": aC_pred_linear_cal.cpu().detach().numpy() if isinstance(aC_pred_linear_cal, torch.Tensor) else aC_pred_linear_cal,
                "a0": a0.cpu().detach().numpy() if isinstance(a0, torch.Tensor) else a0,
            }

            return AC_pred_cal
    
    def eval_direct(self, a: torch.Tensor, eval_id: int | None = None) -> torch.Tensor | None:
        """Evaluate the calibrated forward model on a given sinogram a. Returns the calibrated AC predictions."""
        

        assert hasattr(self, "calibrator"), "Calibrator network is not initialized. Please call calibrate_from_id() first."


        aC_gt = self._extract_gt(a)  # Ground truth AC

        # forward projection to get the predicted AC

        ray_trafo = self.ray_trafo if hasattr(self, "ray_trafo") else self._get_ray_trafo(eval_id=eval_id)
        A = ray_trafo.forward(a)
        AC_gt = self._extract_gt(A)  # Ground truth AC

        # Calibrated and uncalibrated forward model predictions

        AC_uncal = self._uncalibrated_forward(A)[0]
        AC_cal = self.calibrator(AC_uncal.T).T  # Transpose to match input shape

        # Backproject the AC predictions to get the reconstructed images

        aC_cal, aC_uncal = ray_trafo.backward(AC_cal), ray_trafo.backward(AC_uncal)

        # Save the evaluation results to cache to avoid redundant computations in future evaluations

        out = {
            "mode": "eval",
            "AC_pred_cal": AC_cal,
            "AC_pred_uncal": AC_uncal,
            "AC_gt": AC_gt,
            "aC_pred_cal": aC_cal,
            "aC_pred_uncal": aC_uncal,
            "aC_gt": aC_gt,
            "mask": None,  # No mask available for direct evaluation
        }

        
        if eval_id is not None:
            self.cache[eval_id] = {k: v.detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
            print(f"Evaluation results cached for eval ID {eval_id}")

        return aC_uncal

    def plot_predictions(
            self,
            A: np.ndarray,
            A_gt: np.ndarray | None = None,
            densities:List[float]|None = None, 
            title_prefixes: List[str] | None = None, 
            window_settings: List[Tuple[float, float]] | None = None,
            output_path: Path | None = None,
            ) -> None:
        """Plot the sinograms A and optionally A_gt. A is expected to have shape [4 x nAngles x nDet] or [6 x nAngles x nDet]."""
        
        nMats = A.shape[0]
        assert title_prefixes is None or len(title_prefixes) == nMats, f"Expected title_prefixes to have length {nMats}, got {len(title_prefixes)}"
        assert window_settings is None or (len(window_settings) == 2 and all(len(ws) == nMats for ws in window_settings)), f"Expected window_settings to have length 2 with each element having length {nMats}, got {window_settings}"
        assert isinstance(A, np.ndarray),f"Expected A to be a numpy array, got {type(A)}"

        # Rescale by densities if provided

        if densities is not None:
            
            assert len(densities) == nMats, f"Expected densities to have length {nMats}, got {len(densities)}"


            for i in range(nMats):
                A[i] = A[i] * densities[i]
    
                if A_gt is not None:
                    A_gt[i] = A_gt[i] * densities[i]


        fig, ax = plt.subplots(3 if A_gt is not None else 1, nMats, figsize=(16 if A_gt is not None else 8, 8*nMats))

        if A_gt is not None:
            for i in range(nMats):
                im = ax[0, i].imshow(A[i], cmap='gray', vmin=window_settings[i][0] if window_settings is not None else None, vmax=window_settings[i][1] if window_settings is not None else None)
                ax[0, i].set_title(f"{title_prefixes[i]} A[{i}]") if title_prefixes is not None else ax[0, i].set_title(f"A[{i}]")
                ax[0, i].axis('off')
                plt.colorbar(im, ax=ax[0, i], orientation='vertical', fraction=0.046, pad=0.04)

                im = ax[1, i].imshow(A_gt[i], cmap='gray', vmin=window_settings[i][0] if window_settings is not None else None, vmax=window_settings[i][1] if window_settings is not None else None)
                ax[1, i].set_title(f"{title_prefixes[i]} A_gt[{i}]") if title_prefixes is not None else ax[1, i].set_title(f"A_gt[{i}]")
                ax[1, i].axis('off')
                plt.colorbar(im, ax=ax[1, i], orientation='vertical', fraction=0.046, pad=0.04)

                diff = np.abs(A[i] - A_gt[i])
                im = ax[2, i].imshow(diff, cmap='gray', vmin=window_settings[i][0] if window_settings is not None else None, vmax=window_settings[i][1] if window_settings is not None else None)
                ax[2, i].set_title(f"{title_prefixes[i]} Diff[{i}]") if title_prefixes is not None else ax[2, i].set_title(f"Diff[{i}]")
                ax[2, i].axis('off')
                plt.colorbar(im, ax=ax[2, i], orientation='vertical', fraction=0.046, pad=0.04)
        
        else:
            for i in range(nMats):
                im = ax[i].imshow(A[i], cmap='gray', vmin=window_settings[i][0] if window_settings is not None else None, vmax=window_settings[i][1] if window_settings is not None else None)
                ax[i].set_title(f"{title_prefixes[i]} A[{i}]") if title_prefixes is not None else ax[i].set_title(f"A[{i}]")
                ax[i].axis('off')
                plt.colorbar(im, ax=ax[i], orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.tight_layout()

        if output_path is not None:
            print(f"Saving plot to {output_path}... \n")
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_roi_scatter(
            self,
            aC_pred_list: List[np.ndarray], # A list of predicted aC for different methods
            aC_pred_methods: List[str], # A list of method names corresponding to the predictions
            aC_gt: np.ndarray,
            mask: np.ndarray,
            title_prefixes: List[str] | None = None,
            erode_fraction: float = 0.5,
            concentration_percentual_error: float = 0.02,
            fs: int = 18,
            output_path: Path | None = None
    )->None:
        """Plot scatter plots of the predicted vs ground truth AC values for each ROI."""
        
        assert len(aC_pred_list) == len(aC_pred_methods) and all(isinstance(item, np.ndarray) for item in aC_pred_list) and all(isinstance(item, str) for item in aC_pred_methods), f"Expected aC_pred_list and aC_pred_methods to have the same length, got {len(aC_pred_list)} and {len(aC_pred_methods)}"


        is_flattened = aC_pred_list[0].ndim == 2
        nMethods = len(aC_pred_list)
        nMats = aC_pred_list[0].shape[0]

        assert title_prefixes is None or len(title_prefixes) == nMats, f"Expected title_prefixes to have length {nMats}, got {len(title_prefixes)}"


        rois = self._extract_roi_components(mask, erode_fraction)

        print(f"Found {len(rois)} ROIs in the mask for scatter plot analysis... \n")


        #aC_pred_cal_panel =         {"title": "Calibrated",     "color": "blue",    "marker":"^", "mean": [], "std": [], "mse": []}
        #aC_pred_uncal_panel =       {"title": "Uncalibrated",   "color": "red",     "marker":"v", "mean": [], "std": [], "mse": []}

        colors = ["k", "r", "b", "g", "c", "m", "y"]

        aC_pred_panels = [{"title": aC_pred_methods[i], "color": colors[i % len(colors)], "marker": "o", "mean": [], "std": [], "mse": []} for i in range(nMethods)]

        aC_gt_metrics = { "mean": [],  "uncertainty": []}

        for roi in rois:
            roi_mask = roi > 0.5
            aC_pred_rois = [aC_pred_list[i][:, roi_mask] for i in range(nMethods)]
            aC_gt_roi = aC_gt[:, roi_mask]

            # Ground truth metrics

            aC_gt_mean = aC_gt_roi.mean(axis=1)
            aC_gt_metrics["mean"].append(aC_gt_mean)            
            aC_gt_metrics["uncertainty"].append(np.abs(aC_gt_mean) * concentration_percentual_error)

            for i in range(nMethods):
                aC_pred_panels[i]["mean"].append(aC_pred_rois[i].mean(axis=1))
                aC_pred_panels[i]["std"].append(aC_pred_rois[i].std(axis=1))

                aC_pred_panels[i]["mse"].append(((aC_pred_rois[i] - aC_gt_roi) ** 2).mean(axis=1))

        # Convert lists to numpy arrays for easier plotting and calculations

        aC_gt_metrics["mean"], aC_gt_metrics["uncertainty"] = np.array(aC_gt_metrics["mean"]), np.array(aC_gt_metrics["uncertainty"])
        aC_pred_panels = [{**panel, "mean": np.array(panel["mean"]), "std": np.array(panel["std"]), "mse": np.array(panel["mse"])} for panel in aC_pred_panels]

        fig, ax = plt.subplots(1, nMats, figsize=(16, 10))
        if nMats == 1:
            ax = np.array([ax])
        assert isinstance(ax, np.ndarray), f"Expected ax to be a numpy array, got {type(ax)}"

        for i in range(nMats):  
            
            # Plot the line y=x for reference
            min_val = min(min(aC_gt_metrics["mean"][:, i].min(), min(panel["mean"][:, i].min() for panel in aC_pred_panels)), 0)
            max_val = max(aC_gt_metrics["mean"][:, i].max(), max(panel["mean"][:, i].max() for panel in aC_pred_panels))
            ax[i].plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-.', linewidth=1)


            for panel in aC_pred_panels:
                
                ax[i].errorbar(
                    aC_gt_metrics["mean"][:, i], 
                    panel["mean"][:, i],
                    xerr=aC_gt_metrics["uncertainty"][:, i], 
                    yerr=panel["std"][:, i], 
                    fmt=panel["marker"], 
                    color=panel["color"], 
                    elinewidth=0.2, 
                    markersize=8, 
                    alpha=0.5,
                    capsize=3, 
                    label=f"{panel['title']}: RMSE={np.sqrt(np.sum(panel['mse'][:, i])):.2f} mg/mL"
                )
                ax[i].set_title(f"{title_prefixes[i]}", fontsize=fs+2)
                ax[i].set_xlabel(f"Ground Truth", fontsize=fs)
                ax[i].set_ylabel(f"Predicted", fontsize=fs)
                ax[i].legend(loc="upper left", fontsize=fs-2)
                ax[i].grid()


            ax[i].tick_params(axis='both', which='major', labelsize=fs)
            

        fig.tight_layout()

        if output_path is not None:
            print(f"Saving ROI scatter plot to {output_path}... \n")
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
            plt.close()

    def _roi_method_metrics(
            self,
            data: dict,
            method_specs: list[tuple[str, str, str, str]],
    ) -> list[tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray, str, str, float]]]] | None:
        """Compute per-ROI ground-truth means and per-method prediction means/SEM/MSE for both materials.

        `method_specs` is a list of (method_name, cache_key, marker, color) tuples. Returns, per
        material, a (gt_means, predictions) tuple where predictions is a list of
        (method_name, pred_means, pred_sem, marker, color, mse) for the methods present in `data`
        (methods whose cache_key is missing/None are skipped). Returns None if the scan's mask has
        no ROI components.

        MSE (mean squared error of the per-ROI means vs. ground truth) is used instead of R^2
        because R^2 is undefined/degenerate when a scan's ROI ground-truth values are all equal
        (e.g. a phantom with a single true concentration per material).
        """
        rois = self._extract_roi_components(data["mask"], erosion=0.5)
        if not rois:
            return None

        roi_masks = [roi > 0.5 for roi in rois]
        n_pixels = np.array([roi_mask.sum() for roi_mask in roi_masks])

        materials = []
        for material_idx in range(2):
            gt_means = np.array([data["aC_gt"][material_idx, roi_mask].mean() for roi_mask in roi_masks])
            predictions = []
            for method_name, key, marker, color in method_specs:
                prediction = data.get(key)
                if prediction is None:
                    continue
                pred_means = np.array([prediction[material_idx, roi_mask].mean() for roi_mask in roi_masks])
                pred_stds = np.array([prediction[material_idx, roi_mask].std() for roi_mask in roi_masks])
                pred_sem = pred_stds / np.sqrt(n_pixels)

                mse = float(np.mean((pred_means - gt_means) ** 2))

                predictions.append((method_name, pred_means, pred_sem, marker, color, mse))
            materials.append((gt_means, predictions))
        return materials

    def _plot_cached_roi_scatters(self, cached_scans: list[dict], output_path: Path) -> None:
        """Plot calibrated, uncalibrated, and linear ROI statistics for every cached scan."""
        if not cached_scans:
            return
        limits = [-2, 20]
        font_size = 32
        method_specs = [
            ("Optimization + Calibration", "aC_pred_cal", "o", "tab:green"),
            ("Linear", "aC_pred_linear", "s", "tab:blue"),
            ("Non-linear Optimization", "aC_pred_uncal", "^", "tab:red")
        ]

        sorted_scans = sorted(cached_scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        scan_metrics = []
        for data in sorted_scans:
            materials = self._roi_method_metrics(data, method_specs)
            if materials is None:
                continue
            scan_metrics.append({"scan_id": data["scan_id"], "materials": materials})

        if not scan_metrics:
            return


        fig, axes = plt.subplots(
            len(scan_metrics),
            2,
            figsize=(18, 7 * len(scan_metrics)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for row, metrics in enumerate(scan_metrics):
            is_calibration_scan = metrics["scan_id"] == self.calibration_scan_id
            for material_idx, (gt_means, predictions) in enumerate(metrics["materials"]):
                axis = axes[row, material_idx]
                lower, upper = limits
                axis.plot([lower, upper], [lower, upper], color="black", linestyle="-.", linewidth=2)
                for method_name, pred_means, pred_sem, marker, color, mse in predictions:
                    axis.errorbar(
                        gt_means,
                        pred_means,
                        xerr=0.02 * np.abs(gt_means),
                        yerr=pred_sem,
                        fmt=marker,
                        color=color,
                        markersize=14,
                        capsize=4,
                        elinewidth=1.2,
                        alpha=0.75,
                        label=f"{method_name} (MSE={mse:.2f})",
                    )

                axis.set_xlim(lower, upper)
                axis.set_ylim(lower, upper)

                axis.grid()
                axis.tick_params(axis="both", labelsize=font_size - 6)
                axis.legend(loc="upper left", fontsize=font_size - 10)
                if row == 0:
                    axis.set_title(("Iodine" if material_idx == 0 else "Gadolinium") + " [mg/mL]", fontsize=font_size + 2)
                if row == len(scan_metrics) - 1:
                    axis.set_xlabel("Ground truth [mg/mL]", fontsize=font_size)
                if material_idx == 1 and is_calibration_scan:
                    axis.text(
                        0.97, 0.97, "Calibration phantom",
                        transform=axis.transAxes, ha="right", va="top",
                        fontsize=font_size - 10, fontweight="bold", color="darkred",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkred", alpha=0.85),
                    )
            axes[row, 0].set_ylabel(f"{SCAN_TO_LABEL[metrics['scan_id']]}", fontsize=font_size)


        fig.subplots_adjust(left=0.14, right=0.95, bottom=0.08, top=0.95, wspace=0.1, hspace=0.05)
        print(f"Saving combined ROI scatter plot to {output_path}...\n")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def _cross_material_bias_values(
            self,
            cached_scans: list[dict],
            bias_material_idx: int,
            gt_material_idx: int,
            prediction_key: str,
    ) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str | None]]:
        """Compute per-ROI cross-material bias data for every cached scan.

        Bias is predicted (from `prediction_key`) minus ground truth for `bias_material_idx`, per
        ROI. Returns a list of (scan_id, gt_x_means, bias, bias_sem, mode) tuples, one per scan
        with usable data, sorted alphabetically by SCAN_TO_LABEL, where bias_sem is the standard
        error of the mean (std / sqrt(n_pixels_in_ROI)) and mode is the scan's cache mode
        ("cal"/"eval").
        """
        sorted_scans = sorted(cached_scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        results = []
        for data in sorted_scans:
            if data.get(prediction_key) is None or data.get("aC_gt") is None:
                continue

            rois = self._extract_roi_components(data["mask"], erosion=0.5)
            if not rois:
                continue

            roi_masks = [roi > 0.5 for roi in rois]
            n_pixels = np.array([roi_mask.sum() for roi_mask in roi_masks])

            gt_x_means = np.array([data["aC_gt"][gt_material_idx, roi_mask].mean() for roi_mask in roi_masks])
            gt_bias_means = np.array([data["aC_gt"][bias_material_idx, roi_mask].mean() for roi_mask in roi_masks])
            pred_bias_means = np.array([data[prediction_key][bias_material_idx, roi_mask].mean() for roi_mask in roi_masks])
            pred_bias_stds = np.array([data[prediction_key][bias_material_idx, roi_mask].std() for roi_mask in roi_masks])
            pred_bias_sem = pred_bias_stds / np.sqrt(n_pixels)

            bias = pred_bias_means - gt_bias_means

            results.append((data["scan_id"], gt_x_means, bias, pred_bias_sem, data.get("mode")))
        return results

    def _plot_cached_cross_material_bias(
            self,
            cached_scans: list[dict],
            bias_material_idx: int,
            bias_material_name: str,
            gt_material_idx: int,
            gt_material_name: str,
            output_path: Path,
            prediction_key: str = "aC_pred_uncal",
            method_label: str = "Uncalibrated",
            ylim: tuple[float, float] | None = None,
            show_flags: bool = False,
    ) -> None:
        """Plot the prediction bias of one material vs. ground truth concentration of another material.

        Bias is defined as predicted (from `prediction_key`) minus ground truth for
        `bias_material_idx`, plotted against the ground truth of `gt_material_idx`, per ROI across
        all cached scans. Only ROIs with non-zero ground truth for `gt_material_idx` are included.
        Each scan is color/marker-coded as in article/compare_scans.py, and labeled via SCAN_TO_LABEL.
        The y-error bars show the standard error of the mean. Pass `ylim` to keep the y-axis
        consistent across multiple calls (e.g. uncalibrated vs. calibrated). When `show_flags` is
        True, the legend marks the calibration-phantom scan and any evaluation-mode scans.
        """
        font_size = 22

        scan_data = self._cross_material_bias_values(cached_scans, bias_material_idx, gt_material_idx, prediction_key)
        if not scan_data:
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        self._draw_cross_material_bias(ax, scan_data, show_flags=show_flags)

        ax.set_xlabel(f"Ground truth {gt_material_name.lower()} [mg/mL]", fontsize=font_size)
        ax.set_ylabel(f"{bias_material_name} bias (predicted - ground truth) [mg/mL]", fontsize=font_size)
        ax.set_title(f"{method_label}: {bias_material_name} bias", fontsize=font_size + 4)
        ax.tick_params(axis="both", labelsize=font_size - 4)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper left", fontsize=font_size - 8, frameon=False)
        ax.grid()

        fig.tight_layout()
        print(f"Saving combined {bias_material_name.lower()} bias scatter plot ({method_label}) to {output_path}...\n")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def _draw_cross_material_bias(
            self,
            ax: plt.Axes,
            scan_data: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str | None]],
            show_flags: bool = False,
    ) -> None:
        """Draw per-scan bias error bars (from `_cross_material_bias_values`) and a zero-bias line on `ax`."""
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        markers = ["o", "s", "^", "D", "v"]

        for i, (scan_id, gt_x_means, bias, pred_bias_sem, mode) in enumerate(scan_data):
            label = SCAN_TO_LABEL[scan_id]
            if show_flags:
                if scan_id == self.calibration_scan_id:
                    label = f"{label} (Calibration phantom)"
                elif mode == "eval":
                    label = f"{label} (Evaluation)"
            ax.errorbar(
                gt_x_means,
                bias,
                xerr=0.02 * np.abs(gt_x_means),
                yerr=pred_bias_sem,
                fmt=markers[i % len(markers)],
                color=colors[i % len(colors)],
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=4,
                elinewidth=1.2,
                alpha=0.8,
                label=label,
            )

        # ax.axhline(0, color="black", linestyle="-.", linewidth=1.5, label="Zero bias")

    def _plot_cached_calibrated_bias_side_by_side(
            self,
            cached_scans: list[dict],
            output_path: Path,
            ylim: tuple[float, float] | None = None,
            tolerance: float = 1.0,
    ) -> None:
        """Plot calibrated iodine and gadolinium cross-material bias side by side.

        Left: iodine bias vs. ground truth gadolinium. Right: gadolinium bias vs. ground truth
        iodine. The band -`tolerance` < bias < `tolerance` (mg/mL) is shaded, and both panels
        share the y-axis and a single compact legend.
        """
        font_size = 22
        panels = [
            (0, "Iodine", 1, "Gadolinium"),
            (1, "Gadolinium", 0, "Iodine"),
        ]

        panel_data = [
            self._cross_material_bias_values(cached_scans, bias_idx, gt_idx, "aC_pred_cal")
            for bias_idx, _, gt_idx, _ in panels
        ]
        if not any(panel_data):
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

        for ax, scan_data, (_, bias_name, _, gt_name) in zip(axes, panel_data, panels):
            ax.axhspan(-tolerance, tolerance, color="tab:gray", alpha=0.2, linewidth=0, zorder=0, label=f"$\\pm${tolerance:g} mg/mL")
            self._draw_cross_material_bias(ax, scan_data, show_flags=True)
            ax.set_xlabel(f"Ground truth {gt_name.lower()} [mg/mL]", fontsize=font_size)
            ax.set_title(f"{bias_name} bias", fontsize=font_size + 2)
            ax.tick_params(axis="both", labelsize=font_size - 4)
            ax.grid()

        axes[0].set_ylabel("Bias (predicted - ground truth) [mg/mL]", fontsize=font_size)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].legend(loc="upper left", fontsize=font_size - 10, frameon=False, markerscale=0.8, handletextpad=0.3, borderaxespad=0.3)

        fig.tight_layout()
        print(f"Saving calibrated side-by-side bias plot to {output_path}...\n")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def _hu_per_mg_ml(self, energy_keV: float) -> tuple[np.ndarray, float]:
        """Return the HU contrast per mg/mL of each contrast material at the energy closest to
        `energy_keV`, together with that grid energy.

        A concentration c [mg/mL] of material m is a volume fraction c / DENSITIES[m], so it adds
        1000 * fC[m, E] * c / (DENSITIES[m] * mu_water(E)) HU on top of water.
        """
        energies = self.Eactvector.detach().cpu().numpy() if isinstance(self.Eactvector, torch.Tensor) else self.Eactvector
        f_water = self.fB[0].detach().cpu().numpy() if isinstance(self.fB, torch.Tensor) else self.fB[0]
        f_contrast = self.fC.detach().cpu().numpy() if isinstance(self.fC, torch.Tensor) else self.fC
        energy_idx = int(np.argmin(np.abs(energies - energy_keV)))
        hu_per_mg_ml = 1000 * f_contrast[:, energy_idx] / (np.asarray(self.DENSITIES) * f_water[energy_idx])
        return hu_per_mg_ml, float(energies[energy_idx])

    def _relative_hu_error_values(
            self,
            cached_scans: list[dict],
            hu_per_mg_ml: np.ndarray,
            prediction_key: str,
            min_hu: float = 20.0,
            selection_hu_per_mg_ml: list[np.ndarray] | None = None,
    ) -> list[list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str | None]]]:
        """Compute per-ROI relative HU errors for each contrast material and every cached scan.

        For each ROI, the ground truth contrast enhancement is HU_gt = sum_m k_m * c_m,gt and the
        relative error of material m is k_m * bias_m / HU_gt (in %), where k = `hu_per_mg_ml`.
        ROIs are kept only if HU_gt >= `min_hu` for every conversion in `selection_hu_per_mg_ml`
        (default: just `hu_per_mg_ml`), so passing the conversions of several energies keeps the
        same ROIs across plots at those energies. Returns one list per material of
        (scan_id, hu_gt, rel_err, rel_err_sem, mode) tuples, in the format used by
        `_draw_cross_material_bias`.
        """
        sorted_scans = sorted(cached_scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        results = [[] for _ in hu_per_mg_ml]
        for data in sorted_scans:
            if data.get(prediction_key) is None or data.get("aC_gt") is None:
                continue

            rois = self._extract_roi_components(data["mask"], erosion=0.5)
            if not rois:
                continue

            roi_masks = [roi > 0.5 for roi in rois]
            n_pixels = np.array([roi_mask.sum() for roi_mask in roi_masks])

            gt_means = np.array([[data["aC_gt"][m, roi_mask].mean() for roi_mask in roi_masks] for m in range(len(hu_per_mg_ml))])
            pred_means = np.array([[data[prediction_key][m, roi_mask].mean() for roi_mask in roi_masks] for m in range(len(hu_per_mg_ml))])
            pred_sem = np.array([[data[prediction_key][m, roi_mask].std() for roi_mask in roi_masks] for m in range(len(hu_per_mg_ml))]) / np.sqrt(n_pixels)

            hu_gt = hu_per_mg_ml @ gt_means
            selection = selection_hu_per_mg_ml if selection_hu_per_mg_ml is not None else [hu_per_mg_ml]
            keep = np.all([k_sel @ gt_means >= min_hu for k_sel in selection], axis=0)
            if (~keep).any():
                print(f"Scan {data['scan_id']}: dropping {(~keep).sum()}/{keep.size} ROIs with ground truth contrast below {min_hu:g} HU")
            if not keep.any():
                continue

            for m, k in enumerate(hu_per_mg_ml):
                rel_err = 100 * k * (pred_means[m] - gt_means[m]) / hu_gt
                rel_err_sem = 100 * k * pred_sem[m] / hu_gt
                results[m].append((data["scan_id"], hu_gt[keep], rel_err[keep], rel_err_sem[keep], data.get("mode")))
        return results

    def _plot_cached_calibrated_relative_hu_error(
            self,
            cached_scans: list[dict],
            output_dir: Path,
            filename: str = "cached_calibrated_relative_hu_error",
            energies_keV: tuple[float, ...] = (50, 70),
            tolerance: float = 10.0,
            min_hu: float = 20.0,
    ) -> None:
        """Plot calibrated iodine and gadolinium errors relative to the ground truth mixture contrast.

        Each material's bias is converted to HU and divided by the ground truth contrast
        enhancement of the mixture (HU above water), see `_relative_hu_error_values`. One figure
        is saved per energy in `energies_keV` as `{filename}_{E}keV.png`, with iodine and
        gadolinium side by side. The same ROIs (those above `min_hu` at every energy) and the same
        y-axis are used in all figures. The band -`tolerance` < error < `tolerance` (%) is shaded.
        """
        font_size = 22

        conversions = [self._hu_per_mg_ml(energy_keV) for energy_keV in energies_keV]
        selection = [hu_per_mg_ml for hu_per_mg_ml, _ in conversions]
        figures = [
            (grid_energy, self._relative_hu_error_values(cached_scans, hu_per_mg_ml, "aC_pred_cal", min_hu=min_hu, selection_hu_per_mg_ml=selection))
            for hu_per_mg_ml, grid_energy in conversions
        ]
        if not any(scan_data for _, material_data in figures for scan_data in material_data):
            return

        max_abs_err = max(
            (np.max(np.abs(rel_err) + rel_err_sem) for _, material_data in figures for scan_data in material_data for _, _, rel_err, rel_err_sem, _ in scan_data),
            default=0.0,
        )
        ylim = max(1.05 * max_abs_err, 1.2 * tolerance)

        for grid_energy, material_data in figures:
            fig, axes = plt.subplots(1, len(self.MATERIALS), figsize=(18, 8.5), sharey=True)

            for ax, scan_data, material_name in zip(axes, material_data, self.MATERIALS):
                ax.axhspan(-tolerance, tolerance, color="tab:gray", alpha=0.2, linewidth=1, zorder=0)
                # ax.axhline(-tolerance, color="black", linestyle="-.", linewidth=1.5)
                
                self._draw_cross_material_bias(ax, scan_data, show_flags=True)
                ax.set_xlabel("Contrast [HU]", fontsize=font_size)
                ax.set_title(f"{material_name}", fontsize=font_size + 2)
                ax.tick_params(axis="both", labelsize=font_size - 4)
                ax.grid()

            axes[0].set_ylabel("Relative error: \n Bias / Contrast [%]", fontsize=font_size)
            axes[0].set_ylim(-ylim, ylim)
            fig.suptitle(f"Calibrated relative error at {grid_energy:g} keV", fontsize=font_size + 4)

            # Legend above the panels so it never hides data points.
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=font_size - 8, frameon=False, markerscale=0.8, handletextpad=0.3)
            fig.tight_layout(rect=(0, 0, 1, 0.88))

            output_path = output_dir / f"{filename}_{grid_energy:g}keV.png"
            print(f"Saving calibrated relative HU error plot ({grid_energy:g} keV) to {output_path}...\n")
            fig.savefig(output_path, dpi=300)
            plt.close(fig)

    def _plot_cached_reconstructions(
            self,
            cached_scans: list[dict],
            output_path: Path,
            prediction_key: str = "aC_pred_uncal",
            method_label: str = "Uncalibrated",
            show_flags: bool = False,
    ) -> None:
        """Plot predictions (from `prediction_key`), ground truth, and absolute errors for every cached scan.

        When `show_flags` is True, a label is added below the calibration-phantom scan's columns
        and below any evaluation-mode scan's columns. The bottom margin always reserves room for
        that label (whether or not `show_flags` is set) and the figure is saved at a fixed size
        without bbox-tight cropping, so the calibrated and uncalibrated reconstruction plots have
        pixel-identical image grids -- e.g. for flipping between them in a slide deck.
        """

        scans = [data for data in cached_scans if data.get(prediction_key) is not None and data["aC_gt"] is not None]
        if not scans:
            return

        scans = sorted(scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        assert all("a0" in data for data in scans), "All cached scans must contain 'a0' for reconstruction plotting."

        # Size the figure so every grid cell is square, given the subplots_adjust rect below
        n_rows, n_cols = 3, 2 * len(scans)
        left, right, bottom, top = 0.045, 0.91, 0.11, 0.85
        fig_height = 9
        fig_width = fig_height * (top - bottom) / n_rows * n_cols / (right - left)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        material_names = ["Iodine", "Gadolinium"]
        limits = (0, 20)
        row_names = ["Prediction", "Ground truth", "Difference"]
        image_handle = None

        for scan_idx, data in enumerate(scans):
            for material_idx, material_name in enumerate(material_names):
                col = 2 * scan_idx + material_idx
                prediction = data[prediction_key][material_idx]
                ground_truth = data["aC_gt"][material_idx]


                images = [prediction, ground_truth, np.abs(prediction - ground_truth)]
                for row, image in enumerate(images):
                    axis = axes[row, col]
                    image_handle = axis.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1], aspect="equal")
                    axis.set_axis_off()
                    if row == 0:
                        axis.set_title(f" {SCAN_TO_LABEL[data['scan_id']]}\n{material_name}", fontsize=24)

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.0, hspace=0.0)
        for row, row_name in enumerate(row_names):
            row_position = 0.5 * (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1)
            fig.text(0.035, row_position, row_name, rotation=90, ha="center", va="center", fontsize=22)

        if show_flags:
            for scan_idx, data in enumerate(scans):
                if data["scan_id"] == self.calibration_scan_id:
                    flag_label, color = "Calibration phantom", "darkred"
                elif data.get("mode") == "eval":
                    flag_label, color = "Evaluation", "tab:blue"
                else:
                    continue
                col_left, col_right = 2 * scan_idx, 2 * scan_idx + 1
                pos_left = axes[-1, col_left].get_position()
                pos_right = axes[-1, col_right].get_position()
                x_center = 0.5 * (pos_left.x0 + pos_right.x1)
                fig.text(
                    x_center, pos_left.y0 - 0.025, flag_label,
                    ha="center", va="top", fontsize=16, fontweight="bold", color=color,
                )

        colorbar_axis = fig.add_axes([0.92, 0.12, 0.018, 0.68])
        colorbar = fig.colorbar(image_handle, cax=colorbar_axis, ticks=[0, 5, 10, 15, 20])
        colorbar.set_label("mg/mL", fontsize=20)
        colorbar.ax.tick_params(labelsize=16)
        print(f"Saving combined reconstruction plot ({method_label}) to {output_path}...\n")
        # No bbox_inches="tight" here: a fixed figure size + fixed subplots_adjust rect is what
        # keeps the image grid pixel-identical between the calibrated and uncalibrated plots.
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def _plot_cached_initial_maps(self, cached_scans: list[dict], output_path: Path) -> None:
        """Plot initial water, iodine, and 70-keV VMI maps for every cached scan."""
        scans = [data for data in cached_scans if data.get("a0") is not None]
        if not scans:
            return

        scans = sorted(scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        energies = self.Eactvector.detach().cpu().numpy() if isinstance(self.Eactvector, torch.Tensor) else self.Eactvector
        material_functions = self.fB.detach().cpu().numpy() if isinstance(self.fB, torch.Tensor) else self.fB
        vmi_energy_idx = int(np.argmin(np.abs(energies - 70)))
        f_water, f_iodine = material_functions[:2, vmi_energy_idx]
        row_specs = [
            ("Water", "mg/mL", (0, 1500), [0, 500, 1000, 1500]),
            ("Iodine", "mg/mL", (0, 20), [0, 5, 10, 15, 20]),
            ("VMI (70 keV)", "HU", (-1000, 1500), [-1000, -500, 0, 500, 1000, 1500]),
        ]

        # Size the figure so every grid cell is square, given the subplots_adjust rect below
        n_rows, n_cols = 3, len(scans)
        left, right, bottom, top = 0.045, 0.88, 0.04, 0.88
        fig_height = 12
        fig_width = fig_height * (top - bottom) / n_rows * n_cols / (right - left)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        row_images = [None, None, None]
        for scan_idx, data in enumerate(scans):
            water, iodine = data["a0"][:2]
            vmi = f_water * water + f_iodine * iodine
            images = [
                water * 1000,
                iodine * self.DENSITIES[0],
                1000 * (vmi - f_water) / f_water,
            ]
            for row, image in enumerate(images):
                _, _, (vmin, vmax), _ = row_specs[row]
                axis = axes[row, scan_idx]
                row_images[row] = axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
                axis.set_axis_off()
                if row == 0:
                    axis.set_title(f" {SCAN_TO_LABEL[data['scan_id']]}", fontsize=26, fontweight='bold')

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.0, hspace=0.0)
        for row, (row_name, unit, _, ticks) in enumerate(row_specs):
            position = axes[row, 0].get_position()
            row_position = 0.5 * (position.y0 + position.y1)
            fig.text(0.028, row_position, row_name, rotation=90, ha="center", va="center", fontsize=26, fontweight='bold')
            colorbar_height = 0.82 * position.height
            colorbar_axis = fig.add_axes([0.885, row_position - 0.5 * colorbar_height, 0.018, colorbar_height])
            colorbar = fig.colorbar(row_images[row], cax=colorbar_axis, ticks=ticks)
            colorbar.set_label(unit, fontsize=18)
            colorbar.ax.tick_params(labelsize=14)

        print(f"Saving combined initial-map plot to {output_path}...\n")
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _annotate_roi_concentration_labels(
            self,
            ax,
            rois: list[np.ndarray],
            aC_gt: np.ndarray,
            font_size: int = 12,
    ) -> None:
        """Label each ROI, centered on its centroid, with its ground-truth iodine/gadolinium
        concentration as `iodine/gadolinium` (numbers only, no units)."""
        for roi in rois:
            roi_mask = roi > 0.5
            center = np.argwhere(roi_mask).mean(axis=0)  # (row, col)
            iodine_gt = aC_gt[0, roi_mask].mean()
            gadolinium_gt = aC_gt[1, roi_mask].mean()

            ax.text(
                center[1], center[0], f"{iodine_gt:.1f}/{gadolinium_gt:.1f}",
                color="yellow", fontsize=font_size, ha="center", va="center", fontweight="bold",
            )

    def _plot_cached_vmi_roi_labels(self, cached_scans: list[dict], output_dir: Path) -> None:
        """Save a 70 keV VMI reconstruction for every cached scan, one file per scan, with every
        ROI labeled by its ground-truth iodine/gadolinium concentration (see
        _annotate_roi_concentration_labels)."""
        scans = [
            data for data in cached_scans
            if data.get("a0") is not None and data.get("mask") is not None and data.get("aC_gt") is not None
        ]
        if not scans:
            return

        scans = sorted(scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        energies = self.Eactvector.detach().cpu().numpy() if isinstance(self.Eactvector, torch.Tensor) else self.Eactvector
        material_functions = self.fB.detach().cpu().numpy() if isinstance(self.fB, torch.Tensor) else self.fB
        vmi_energy_idx = int(np.argmin(np.abs(energies - 70)))
        f_water, f_iodine = material_functions[:2, vmi_energy_idx]

        for data in scans:
            rois = self._extract_roi_components(data["mask"])
            if not rois:
                continue

            water, iodine = data["a0"][:2]
            vmi = 1000 * (f_water * water + f_iodine * iodine - f_water) / f_water  # HU

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(vmi, cmap="gray", vmin=-1000, vmax=1500, aspect="equal")
            ax.set_axis_off()
            ax.set_title(
                f"{SCAN_TO_LABEL[data['scan_id']]}: VMI (70 keV) with ground truth (I/Gd mg/mL)",
                fontsize=18, fontweight="bold",
            )

            self._annotate_roi_concentration_labels(ax, rois, data["aC_gt"])

            output_path = output_dir / f"scan_{data['scan_id']}_{data.get('mode', 'cal')}_vmi_roi_labels.png"
            print(f"Saving VMI ROI label plot for scan {data['scan_id']} to {output_path}...\n")
            fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

    def _plot_cached_linear_comparison(self, cached_scans: list[dict], output_path: Path) -> None:
        """Compare the non-linear and linear forward models (each with its own calibration) per ROI.

        Produces an Nscans x 4 grid: columns 0-1 are Iodine/Gadolinium ROI scatter plots for the
        non-linear optimization method (uncalibrated + calibrated overlaid), columns 2-3 are the
        same pair of plots for the linear model (uncalibrated + calibrated overlaid). Only scans
        with linear-model predictions are included.
        """
        scans = [data for data in cached_scans if data.get("aC_pred_linear") is not None]
        if not scans:
            return

        scans = sorted(scans, key=lambda data: SCAN_TO_LABEL.get(data["scan_id"], str(data["scan_id"])))

        limits = [-2, 20]
        font_size = 30
        non_linear_specs = [("Non-linear", "aC_pred_uncal", "^", "tab:red"), ("Calibrated", "aC_pred_cal", "o", "tab:green")]
        linear_specs = [("Linear", "aC_pred_linear", "s", "tab:blue"), ("Calibrated", "aC_pred_linear_cal", "o", "tab:purple")]
        column_specs = [
            ("Iodine", "Non-linear", 0),
            ("Gadolinium", "Non-linear", 1),
            ("Iodine", "Linear", 0),
            ("Gadolinium", "Linear", 1),
        ]

        fig, axes = plt.subplots(
            len(scans), 4,
            figsize=(28, 7 * len(scans)),
            sharex=True, sharey=True, squeeze=False,
        )

        for row, data in enumerate(scans):
            metrics_by_group = {
                "Non-linear": self._roi_method_metrics(data, non_linear_specs),
                "Linear": self._roi_method_metrics(data, linear_specs),
            }
            for col, (material_name, method_group, material_idx) in enumerate(column_specs):
                axis = axes[row, col]
                lower, upper = limits
                axis.plot([lower, upper], [lower, upper], color="black", linestyle="-.", linewidth=2)

                materials = metrics_by_group[method_group]
                if materials is not None:
                    gt_means, predictions = materials[material_idx]
                    for method_name, pred_means, pred_sem, marker, color, mse in predictions:
                        axis.errorbar(
                            gt_means, pred_means,
                            xerr=0.02 * np.abs(gt_means), yerr=pred_sem,
                            fmt=marker, color=color, markersize=12, capsize=4,
                            elinewidth=1.2, alpha=0.75,
                            label=f"{method_name} (RMSE={np.sqrt(mse):.2f} mg/mL)",
                        )
                    axis.legend(loc="upper left", fontsize=font_size - 12)

                axis.set_xlim(lower, upper)
                axis.set_ylim(lower, upper)
                axis.grid()
                axis.tick_params(axis="both", labelsize=font_size - 8)

                if row == 0:
                    axis.set_title(f"{method_group}\n{material_name} [mg/mL]", fontsize=font_size)
                if row == len(scans) - 1:
                    axis.set_xlabel("Ground truth [mg/mL]", fontsize=font_size)
            axes[row, 0].set_ylabel(f"{SCAN_TO_LABEL[data['scan_id']]}\nMeasured [mg/mL]", fontsize=font_size)

        # Leave a margin on the right so a per-row flag (calibration phantom / evaluation scan)
        # can be placed just outside column 3, then compute its position from the final layout.
        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.06, top=0.92, wspace=0.12, hspace=0.08)

        for row, data in enumerate(scans):
            if data["scan_id"] == self.calibration_scan_id:
                row_label, color = "Calibration phantom", "darkred"
            elif data.get("mode") == "eval":
                row_label, color = "Evaluation", "tab:blue"
            else:
                continue
            position = axes[row, 3].get_position()
            row_position = 0.5 * (position.y0 + position.y1)
            fig.text(
                0.935, row_position, row_label,
                rotation=90, ha="center", va="center",
                fontsize=font_size - 10, fontweight="bold", color=color,
            )

        print(f"Saving combined linear-vs-non-linear comparison plot to {output_path}...\n")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def plot_cache(self, output_dir: Path = ARTIFACT_DIR) -> None:
        """Plot the cached predictions and ground truth for all scans in the cache."""
        output_dir.mkdir(parents=True, exist_ok=True)

        def _apply_physical(A: np.ndarray | list[np.ndarray], densities: List[float]) -> torch.Tensor:
            """Apply physical scaling to the sinogram or reconstructed images based on the provided densities."""

            if isinstance(A, list):
                for i in range(len(A)):
                    assert A[i] is None or A[i].shape[0] == len(densities), f"Expected A[{i}] to have {len(densities)} channels, got {A[i].shape[0]}"

                    if A[i] is not None:
                        for j in range(A[i].shape[0]):
                            A[i][j] = A[i][j] * densities[j]
                    else: 
                        A[i] = None
                return A
            else:
                assert isinstance(A, np.ndarray), f"Expected A to be a np.ndarray or list of np.ndarray, got {type(A)}"
                assert A.shape[0] == len(densities), f"Expected A to have {len(densities)} channels, got {A.shape[0]}"
                for j in range(A.shape[0]):
                    A[j] = A[j] * densities[j]
                return A


        cached_scans = []
        for scan_id, data in self.cache.items():
            mode = data["mode"]
            print(f"Plotting cached results for scan {scan_id} in mode {mode}... \n")
            AC_pred_cal = data["AC_pred_cal"].copy() if data["AC_pred_cal"] is not None else None
            AC_pred_uncal = data["AC_pred_uncal"].copy()
            AC_gt = data["AC_gt"].copy()
            aC_pred_cal = data["aC_pred_cal"].copy() if data["aC_pred_cal"] is not None else None
            aC_pred_uncal = data["aC_pred_uncal"].copy()
            aC_pred_linear = data["aC_pred_linear"].copy() if data.get("aC_pred_linear") is not None else None
            aC_pred_linear_cal = data["aC_pred_linear_cal"].copy() if data.get("aC_pred_linear_cal") is not None else None
            aC_gt = data["aC_gt"].copy()
            a0 = data["a0"].copy() if data.get("a0") is not None else None
            mask = data["mask"]

            # Apply physical scaling to the sinograms and reconstructed images

            AC_pred_cal, AC_pred_uncal, AC_gt, aC_pred_cal, aC_pred_uncal, aC_pred_linear, aC_pred_linear_cal, aC_gt = _apply_physical(
                [AC_pred_cal, AC_pred_uncal, AC_gt, aC_pred_cal, aC_pred_uncal, aC_pred_linear, aC_pred_linear_cal, aC_gt], self.DENSITIES
            )


            # Plot sinograms

            if AC_pred_cal is not None and AC_gt is not None:
                print(f"Plotting sinograms for scan {scan_id}... \n")

                self.plot_predictions(
                    A=AC_pred_cal,
                    A_gt=AC_gt,
                    densities = None, # Densities already applied
                    window_settings = [(0, 500), (0, 300)],
                    title_prefixes=[f"Scan {scan_id} - Calibrated", f"Scan {scan_id} - Ground Truth"],
                    output_path=output_dir / f"scan_{scan_id}_{mode}_sinograms.png"
                )

                # Plot uncalibrated sinograms
                self.plot_predictions(
                    A=AC_pred_uncal,
                    A_gt=AC_gt,
                    densities = None, # Densities already applied
                    window_settings = [(0, 500), (0, 300)],
                    title_prefixes=[f"Scan {scan_id} - Uncalibrated", f"Scan {scan_id} - Ground Truth"],
                    output_path=output_dir / f"scan_{scan_id}_{mode}_uncal_sinograms.png"
                )

            if aC_pred_cal is not None and aC_gt is not None:
                print(f"Plotting reconstructed images for scan {scan_id}... \n")

                # Plot reconstructed images
                self.plot_predictions(
                    A=aC_pred_cal,
                    A_gt=aC_gt,
                    densities = None, # Densities already applied
                    window_settings = [(0, 15), (0, 15)],
                    title_prefixes=[f"Scan {scan_id} - Calibrated", f"Scan {scan_id} - Ground Truth"],
                    output_path=output_dir / f"scan_{scan_id}_{mode}_reconstructions.png"
                )

                # Plot uncalibrated reconstructed images
                self.plot_predictions(
                    A=aC_pred_uncal,
                    A_gt=aC_gt,
                    densities = None, # Densities already applied
                    window_settings = [(0, 15), (0, 15)],
                    title_prefixes=[f"Scan {scan_id} - Uncalibrated", f"Scan {scan_id} - Ground Truth"],
                    output_path=output_dir / f"scan_{scan_id}_{mode}_uncal_reconstructions.png"
                )


            methods, aC_list = zip(*[(m, aC) for m, aC in zip(["Optimization + Calibration", "Non-linear Optimization", "Linear"], [aC_pred_cal, aC_pred_uncal, aC_pred_linear]) if aC is not None])
            print(f"Plotting ROI scatter plots for scan {scan_id} with methods: {methods}... \n")
            # Plot ROI scatter plots
            self.plot_roi_scatter(
                aC_pred_list=aC_list,
                aC_pred_methods=methods,
                aC_gt=aC_gt,
                mask=mask,
                title_prefixes=["Iodine [mg/mL]", "Gadolinium [mg/mL]"],
                output_path=output_dir / f"scan_{scan_id}_{mode}_roi_scatter.png"
            )

            cached_scans.append(
                {
                    "scan_id": scan_id,
                    "mode": mode,
                    "aC_pred_cal": aC_pred_cal,
                    "aC_pred_uncal": aC_pred_uncal,
                    "aC_pred_linear": aC_pred_linear,
                    "aC_pred_linear_cal": aC_pred_linear_cal,
                    "aC_gt": aC_gt,
                    "mask": mask,
                    "a0": a0,
                }
            )

        self._plot_cached_roi_scatters(cached_scans, output_dir / "cached_roi_scatters.png")

        bias_combos = [
            (1, "Gadolinium", 0, "Iodine", "cached_gadolinium_bias_vs_iodine"),
            (0, "Iodine", 1, "Gadolinium", "cached_iodine_bias_vs_gadolinium"),
        ]
        bias_prediction_specs = [("aC_pred_uncal", "Uncalibrated", "", False)]
        if cached_scans and all(data.get("aC_pred_cal") is not None for data in cached_scans):
            bias_prediction_specs.append(("aC_pred_cal", "Calibrated", "_calibrated", True))

        # Share one y-axis range across all cross-material bias plots so they're directly comparable.
        max_abs_bias = 0.0
        for bias_material_idx, _, gt_material_idx, _, _ in bias_combos:
            for prediction_key, _, _, _ in bias_prediction_specs:
                for _, _, bias, bias_sem, _ in self._cross_material_bias_values(cached_scans, bias_material_idx, gt_material_idx, prediction_key):
                    if bias.size:
                        max_abs_bias = max(max_abs_bias, np.max(np.abs(bias) + bias_sem))
        bias_ylim = (-1.05 * max_abs_bias, 1.05 * max_abs_bias) if max_abs_bias > 0 else None

        for bias_material_idx, bias_material_name, gt_material_idx, gt_material_name, filename in bias_combos:
            for prediction_key, method_label, suffix, show_flags in bias_prediction_specs:
                self._plot_cached_cross_material_bias(
                    cached_scans, bias_material_idx=bias_material_idx, bias_material_name=bias_material_name,
                    gt_material_idx=gt_material_idx, gt_material_name=gt_material_name,
                    output_path=output_dir / f"{filename}{suffix}.png",
                    prediction_key=prediction_key, method_label=method_label, ylim=bias_ylim,
                    show_flags=show_flags,
                )
        if len(bias_prediction_specs) > 1:
            self._plot_cached_calibrated_bias_side_by_side(
                cached_scans, output_dir / "cached_calibrated_bias_side_by_side.png", ylim=bias_ylim,
            )
            self._plot_cached_calibrated_relative_hu_error(
                cached_scans, output_dir, energies_keV=(50, 70),
            )
        self._plot_cached_reconstructions(
            cached_scans, output_dir / "cached_reconstructions.png",
            prediction_key="aC_pred_uncal", method_label="Uncalibrated",
        )
        if cached_scans and all(data.get("aC_pred_cal") is not None for data in cached_scans):
            self._plot_cached_reconstructions(
                cached_scans, output_dir / "cached_reconstructions_calibrated.png",
                prediction_key="aC_pred_cal", method_label="Calibrated", show_flags=True,
            )
        self._plot_cached_initial_maps(cached_scans, output_dir / "cached_initial_maps.png")
        self._plot_cached_vmi_roi_labels(cached_scans, output_dir)
        if any(data.get("aC_pred_linear") is not None for data in cached_scans):
            self._plot_cached_linear_comparison(cached_scans, output_dir / "cached_linear_comparison.png")
      