from data_processing.utils import load_materials, load_weights, load_energies, segment_id
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from typing import List
from scipy.ndimage import label as nd_label, distance_transform_edt
from pathlib import Path
from config import *

PLOT_DIR = REPO_ROOT / "AttenuationMD_python" / "article" / "figs" / "plots"
DEBUG_DIR = REPO_ROOT / "AttenuationMD_python" / "debug"

def _extract_roi_components(mask: np.ndarray | torch.Tensor, erosion: float = 1.0, scan_id:str|None = None) -> list[np.ndarray]:
        
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


        for idx, component in enumerate(components):
            plt.contour(component, colors=[plt.cm.jet(idx / len(components))], linewidths=1.5, alpha=0.7)

            # Add a legend entry for each component
            plt.text(np.mean(np.where(component)[1]), np.mean(np.where(component)[0]), f'ROI {idx + 1}', color='white', fontsize=8, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


        plt.title("Original Mask")
        plt.axis('off')


        # Ensure the output directory exists before saving the plot

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving ROI components plot to {output_path}... \n")
        plt.savefig(output_path)
        plt.close()

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
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
                max_dist = distances.max()
                eroded = distances == max_dist if max_dist > 0 else component
            component = eroded

        components.append(component)

    if not components:
        raise ValueError("No connected ROI components found in the mask")

    # Plot the original mask and its connected components for debugging
    plot_path = PLOT_DIR / f"roi_components_{scan_id}.png" if scan_id is not None else PLOT_DIR / "roi_components.png"
    _plot_mask_and_components(mask, components, plot_path)

    return components

def _load_scan_from_id(scan_id: int, create_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the scan data for a given scan ID."""

        out = segment_id(scan_id, refine_mask=create_mask)
        aB, aR, aC_gt_img, mask = out["background"], out["signal"], out["ground_truth"], out["mask"]
        a = torch.cat([aB, aR], dim=0) if aC_gt_img is None else torch.cat([aB, aR, aC_gt_img], dim=0)
        return a, mask


def _plot_signal_maps_with_rois(
    water_map: torch.Tensor,
    iodine_map: torch.Tensor,
    rois: list[np.ndarray],
    scan_id: int,
) -> None:
    """Show the signal maps with equal-area circles for the mean-calculation ROIs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    maps = [
        (water_map.detach().cpu().numpy(), "Water signal [mg/mL]"),
        (iodine_map.detach().cpu().numpy(), "Iodine signal [mg/mL]"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for axis, (image, title) in zip(axes, maps):
        image_handle = axis.imshow(image, cmap="gray")
        axis.set_title(title, fontsize=12)
        axis.axis("off")
        fig.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04, label="mg/mL")

        for index, (roi, color) in enumerate(zip(rois, colors), start=1):
            rows, columns = np.nonzero(roi)
            center_x = float(columns.mean())
            center_y = float(rows.mean())
            radius = float(np.sqrt(roi.sum() / np.pi))
            axis.add_patch(Circle((center_x, center_y), radius, fill=False, color=color, linewidth=1.8))
            axis.text(
                center_x,
                center_y,
                str(index),
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
            )

    fig.suptitle(f"Scan {scan_id}: ROI circles used for signal means", fontsize=13)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"scan_{scan_id}_signal_maps_rois.png"
    print(f"Saving signal maps with ROI circles to {plot_path}")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roi_attenuations(
    scan_id: int,
    roi_index: int,
    water_signal: float,
    iodine_signal: float,
    iodine_gt: float,
    gadolinium_gt: float,
    energies: np.ndarray,
    water_attenuation: np.ndarray,
    iodine_attenuation: np.ndarray,
    gadolinium_attenuation: np.ndarray,
) -> None:
    """Plot the signal and ground-truth attenuation spectra for one ROI."""
    signal_spectrum = (
        water_signal / DENSITY_WA * water_attenuation
        + iodine_signal / DENSITY_I * iodine_attenuation
    )
    ground_truth_spectrum = (
        iodine_gt / DENSITY_I * iodine_attenuation
        + gadolinium_gt / DENSITY_GD * gadolinium_attenuation
    )
    water_iodine_basis = np.column_stack([water_attenuation, iodine_attenuation])
    least_squares_coefficients, _, _, _ = np.linalg.lstsq(
        water_iodine_basis,
        ground_truth_spectrum,
        rcond=None,
    )
    least_squares_spectrum = water_iodine_basis @ least_squares_coefficients

    fig, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    axis.plot(energies, signal_spectrum, color="tab:blue", linewidth=1, label="Water/iodine signal")
    axis.plot(energies, ground_truth_spectrum, color="tab:red", linewidth=1, label="Iodine/gadolinium GT")
    axis.plot(
        energies,
        least_squares_spectrum,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Least squares approximation",
    )
    axis.set_xlabel("Energy [keV]")
    axis.set_ylabel("Attenuation [mm$^{-1}$]")
    axis.set_ylim(0, 0.1)
    axis.set_title(f"Scan {scan_id}, ROI {roi_index}: attenuation spectra")
    axis.grid(axis="y", color="#d9dee3", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    output_dir = PLOT_DIR / f"scan_{scan_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"roi_{roi_index}_attenuations.png"
    print(f"Saving ROI attenuation plot to {plot_path}")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_scans(scan_ids: List, correctGd=False)->None:

    fig, ax = plt.subplots(2, 1, figsize=(10, 16), layout="constrained", sharex=True, sharey=False)
    rho_wa = DENSITY_WA
    rho_i = DENSITY_I
    rho_gd = DENSITY_GD
    fB, _, fC = load_materials()
    energies = load_energies().detach().cpu().numpy()
    water_attenuation = fB[0].detach().cpu().numpy()
    iodine_attenuation = fC[0].detach().cpu().numpy()
    gadolinium_attenuation = fC[1].detach().cpu().numpy()


    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers = ["o", "s", "^", "D", "v"]
    for i, scan_id in enumerate(scan_ids):
        plot_maps = False
        # Load the scan data (replace this with your actual data loading code)
        a, mask = _load_scan_from_id(scan_id, create_mask=False)

        aWa = a[2, :, :] * rho_wa  # Water attenuation image
        aI = a[3, :, :] * rho_i  # Iodine attenuation image

        aI_gt = a[4, :, :] * rho_i  # Iodine ground truth image
        aGd_gt = a[5, :, :] * rho_gd  # Gadolinium ground truth image

        if correctGd:

            if scan_id in [448, 456]:
                aGd_gt = aGd_gt * 2


        if plot_maps:
            # DEBUG: plot the maps:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(aWa, cmap='gray')
            # add colorbar
            plt.colorbar(label='Water Signal [mg/mL]')
            plt.title(f'{SCAN_TO_LABEL[scan_id]} - Water Signal')
            plt.axis('off') 
            plt.subplot(2, 2, 2)
            plt.imshow(aI, cmap='gray')
            plt.colorbar(label='Iodine Signal [mg/mL]')
            plt.title(f'{SCAN_TO_LABEL[scan_id]} - Iodine Signal')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(aI_gt, cmap='gray')
            plt.colorbar(label='Iodine Ground Truth [mg/mL]')
            plt.title(f'{SCAN_TO_LABEL[scan_id]} - Iodine Ground Truth')
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(aGd_gt, cmap='gray')
            plt.colorbar(label='Gadolinium Ground Truth [mg/mL]')
            plt.title(f'{SCAN_TO_LABEL[scan_id]} - Gadolinium Ground Truth')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(DEBUG_DIR / f"scan_{SCAN_TO_LABEL[scan_id]}_maps.png", dpi=300, bbox_inches="tight")

            plt.clf()

        rois = _extract_roi_components(mask, erosion=0.5, scan_id=str(scan_id))
        _plot_signal_maps_with_rois(aWa, aI, rois, scan_id)

        water_signal_avgs = [0]
        iodine_signal_avgs = []
        gadolinium_gt_avgs = [0]
        iodine_gt_avgs = []

        for roi_index, roi in enumerate(rois, start=1):
            water_signal_avg = aWa[roi].mean().item()
            iodine_signal_avg = aI[roi].mean().item()
            gadolinium_gt_avg = aGd_gt[roi].mean().item()
            iodine_gt_avg = aI_gt[roi].mean().item()

            water_signal_avgs.append(water_signal_avg)
            iodine_signal_avgs.append(iodine_signal_avg)
            gadolinium_gt_avgs.append(gadolinium_gt_avg)
            iodine_gt_avgs.append(iodine_gt_avg)
            _plot_roi_attenuations(
                scan_id=scan_id,
                roi_index=roi_index,
                water_signal=water_signal_avg,
                iodine_signal=iodine_signal_avg,
                iodine_gt=iodine_gt_avg,
                gadolinium_gt=gadolinium_gt_avg,
                energies=energies,
                water_attenuation=water_attenuation,
                iodine_attenuation=iodine_attenuation,
                gadolinium_attenuation=gadolinium_attenuation,
            )

        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        s = 120  # Marker size for scatter plot
        # Error bars represent within-ROI standard deviations of the signal maps.
        ax[0].scatter(
            gadolinium_gt_avgs,
            water_signal_avgs,
            s=s,
            marker=m,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.8,
            color=c,
        )
        ax[1].scatter(
            iodine_gt_avgs,
            iodine_signal_avgs,
            s=s,
            marker=m,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.8,
            color=c,
            label=SCAN_TO_LABEL[scan_id],
        )

        gd_values = np.asarray(gadolinium_gt_avgs, dtype=float)
        water_values = np.asarray(water_signal_avgs, dtype=float)
        valid = np.isfinite(gd_values) & np.isfinite(water_values)
        if valid.sum() >= 2:

            # Fit a slope with zero intercept to the water signal vs. gadolinium data.
            slope = np.sum(gd_values[valid] * water_values[valid]) / np.sum(gd_values[valid] ** 2)
            
            gd_fit = np.linspace(gd_values[valid].min(), gd_values[valid].max(), 100)
            ax[0].plot(
                gd_fit,
                slope * gd_fit,
                color=c,
                linewidth=1.5,
                linestyle="-.",
                label=f"{SCAN_TO_LABEL[scan_id]}: y = {slope:.3g}x",
                alpha=1
            )
    # Plot the line y=x for reference in the iodine plot

    ax[1].plot(
        [0, 15],
        [0, 15],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="y = x",
        alpha=1
    )    
    
    ax[1].plot()

    ax[0].set_xlabel('Gadolinium GT [mg/mL]', fontsize=24)
    ax[0].set_ylabel('Water signal [mg/mL]', fontsize=24)
    ax[0].set_title("Water signal vs. gadolinium", fontsize=28, pad=10)
    ax[0].legend(
        frameon=False,
        fontsize=20,
        title="Linear fits",
        title_fontsize=24,
        loc="lower right",
    )
    ax[1].set_xlabel('Iodine GT [mg/mL]', fontsize=24)
    ax[1].set_ylabel('Iodine signal [mg/mL]', fontsize=24)
    ax[1].set_title("Iodine signal vs. iodine", fontsize=28, pad=10)
    ax[1].legend(
        frameon=False,
        fontsize=20,
        title="Scans",
        title_fontsize=24,
        loc="lower right",
    )
    

    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d9dee3", linewidth=0.7, alpha=0.8)
        axis.tick_params(axis="both", labelsize=18)

    plot_path = PLOT_DIR / "compare_scans.png"
    print(f"Saving comparison plot to {plot_path}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    scan_ids = [161, 448, 449, 456]
    compare_scans(scan_ids)

if __name__ == "__main__":
    main()
    