from cal.cal import CalibratedForwardModel
from config import *
import torch
from matplotlib import pyplot as plt
from operators.tomo import RayTrafoModule
from operators.denoising import PairedContrastCovarianceDRUNetDenoiser
from data_processing.utils import load_materials, load_weights, load_energies
import cv2
import argparse
from article.utils import load_dir
from article.attenuations import LinearMaterialDecomposition
import numpy as np
import nibabel as nib
from nibabel.nifti1 import load
from matplotlib.patches import Circle
from scipy.ndimage import binary_fill_holes


DATA_DIR = REPO_ROOT / "data"

GAMMEX_PATH = DATA_DIR / "gammex" / "phantomtype_1"
ANTROPOMORPHIC_PATH  = DATA_DIR / "antropomorphic_phantoms" / "case_50"
PLOT_DIR = REPO_ROOT / "AttenuationMD_python" / "article" / "figs" / "plots"




gammex_data = load_dir(GAMMEX_PATH)
antropomorphic_data = load_dir(ANTROPOMORPHIC_PATH)


rhoW = 1000
rhoI = 4930
rhoGd = 7900


def load_all_materials():


    fB, _, fC = load_materials()

    fWa = fB[0, :].numpy()  # [nE]
    fI = fC[0, :].numpy()
    fGd = fC[1, :].numpy()

    fB, _, _ = load_materials(usePePvc=True)

    fPE = fB[0, :].numpy()  # [nE]
    fPVC  = fB[1, :].numpy()  # [nE]

    return fWa, fI, fGd, fPE, fPVC

def circle_mask(shape, center_xy, radius):
    h, w = shape
    cx, cy = center_xy
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

def pick_circular_rois(image, title="Click center then edge for each ROI, middle-click to finish"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    ax.set_title(title)

    rois = []

    while True:
        pts = plt.ginput(2, timeout=-1)
        if len(pts) < 2:
            break

        center = np.array(pts[0])
        edge = np.array(pts[1])
        radius = np.linalg.norm(edge - center)

        rois.append((center, radius))

        circle = Circle(
            (center[0], center[1]),
            radius,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
            linestyle="-",
        )
        ax.add_patch(circle)
        ax.plot(center[0], center[1], marker="+", color="yellow", markersize=8)

        fig.canvas.draw()

    plt.close(fig)
    return rois

def annotate_rois(ax, image, rois, unit="mg/mL", text_offsets=None):
    if text_offsets is None:
        text_offsets = [(-100, 100), (-100, -100)] 

    stats = []
    ROI_names = ["Aorta", "Endovascular sack"]

    for idx, ((center, radius), offset) in enumerate(zip(rois, text_offsets), start=1):
        mask = circle_mask(image.shape, center, radius)
        values = image[mask]
        mean_val = float(np.mean(values)) if values.size else np.nan

        circle = Circle(
            (center[0], center[1]),
            radius,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
        ax.add_patch(circle)

        text_xy = (center[0] + offset[0], center[1] + offset[1])
        ax.annotate(
            f"{ROI_names[idx-1]}\nmean = {mean_val:.2f} {unit}",
            xy=(center[0], center[1]),
            xytext=text_xy,
            color="yellow",
            fontsize=16,
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color="yellow",
                lw=1.5,
            ),
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="black",
                ec="yellow",
                alpha=0.6,
            ),
        )

        stats.append(
            {
                "roi": idx,
                "center_xy": (float(center[0]), float(center[1])),
                "radius": float(radius),
                "mean": mean_val,
            }
        )

    return stats

def add_gt_overlay(ax, title, values, unit="mg/mL"):
    lines = [title]
    roi_names = ["Aorta", "Endovascular sack"]
    for idx, value in enumerate(values):
        label = roi_names[idx] if idx < len(roi_names) else f"ROI {idx+1}"
        lines.append(f"{label}: {value:.2f} {unit}")

    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        color="yellow",
        fontsize=15,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="black",
            ec="yellow",
            alpha=0.6,
        ),
    )

def plot_nested_image_panels(
    panels,
    outer_shape=(2, 2),
    figsize=(20, 20),
    save_path=None,
    suptitle=None,
    constrained_layout=False,
    outer_wspace=0.1,
    outer_hspace=0.1,
    inner_wspace=0.02,
    inner_hspace=0.02,
    row_labels=None,
    row_label_fontsize=20,
    row_label_xpad=0.02,
    panel_labels=("(A)", "(B)", "(C)", "(D)"),
    panel_label_fontsize=28,
    square_cells=True,
):
    if square_cells:
        # Keep the figure width and pick the height so every inner grid cell is square, so images
        # drawn with aspect="auto" are not stretched. GridSpec spacing is a fraction of the cell size.
        inner_rows = max(panel["shape"][0] for panel in panels)
        inner_cols = max(panel["shape"][1] for panel in panels)
        width_frac = plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"]
        height_frac = plt.rcParams["figure.subplot.top"] - plt.rcParams["figure.subplot.bottom"]
        cols_span = (outer_shape[1] + outer_wspace * (outer_shape[1] - 1)) * (inner_cols + inner_wspace * (inner_cols - 1))
        rows_span = (outer_shape[0] + outer_hspace * (outer_shape[0] - 1)) * (inner_rows + inner_hspace * (inner_rows - 1))
        cell_size = figsize[0] * width_frac / cols_span
        figsize = (figsize[0], cell_size * rows_span / height_frac)

    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    outer_grid = fig.add_gridspec(
        outer_shape[0],
        outer_shape[1],
        wspace=outer_wspace,
        hspace=outer_hspace,
    )

    if suptitle is not None:
        fig.suptitle(suptitle)

    default_cbar_kwargs = {"fraction": 0.046, "pad": 0.04}
    panel_axes = {}

    for panel_idx, panel in enumerate(panels):
        row = panel.get("outer_row", panel_idx // outer_shape[1])
        col = panel.get("outer_col", panel_idx % outer_shape[1])
        panel_grid = outer_grid[row, col].subgridspec(
            panel["shape"][0],
            panel["shape"][1],
            wspace=panel.get("inner_wspace", inner_wspace),
            hspace=panel.get("inner_hspace", inner_hspace),
        )

        axes = np.empty(panel["shape"], dtype=object)
        image_handles = np.empty(panel["shape"], dtype=object)
        for inner_row in range(panel["shape"][0]):
            for inner_col in range(panel["shape"][1]):
                axes[inner_row, inner_col] = fig.add_subplot(panel_grid[inner_row, inner_col])

        panel_axes[(row, col)] = axes

        if panel.get("title"):
            left = min(ax.get_position().x0 for ax in axes.ravel())
            right = max(ax.get_position().x1 for ax in axes.ravel())
            top = max(ax.get_position().y1 for ax in axes.ravel())
            fig.text(
                0.5 * (left + right),
                top + panel.get("title_pad", 0.01),
                panel["title"],
                fontsize=panel.get("title_fontsize", 16),
                fontweight=panel.get("title_fontweight", "bold"),
                ha="center",
                va="bottom",
            )

        column_titles = panel.get("column_titles")
        if column_titles is not None:
            top = max(ax.get_position().y1 for ax in axes[0, :].ravel())
            for inner_col, column_title in enumerate(column_titles):
                col_axes = axes[:, inner_col].ravel()
                left = min(ax.get_position().x0 for ax in col_axes)
                right = max(ax.get_position().x1 for ax in col_axes)
                fig.text(
                    0.5 * (left + right),
                    top + panel.get("column_title_pad", 0.008),
                    column_title,
                    fontsize=panel.get("column_title_fontsize", 22),
                    fontweight=panel.get("column_title_fontweight", "bold"),
                    ha="center",
                    va="bottom",
                )

        for item_idx, item in enumerate(panel["items"]):
            inner_row = item.get("row", item_idx // panel["shape"][1])
            inner_col = item.get("col", item_idx % panel["shape"][1])
            ax = axes[inner_row, inner_col]

            im = ax.imshow(
                item["image"],
                cmap=item.get("cmap", "gray"),
                vmin=item.get("vmin"),
                vmax=item.get("vmax"),
                aspect=item.get("aspect", "auto"),
            )
            image_handles[inner_row, inner_col] = im
            ax.set_title(item.get("title", ""), fontsize=item.get("title_fontsize", 24))

            if item.get("axis_off", True):
                ax.axis("off")

            if item.get("colorbar", panel.get("colorbar", False)):
                cbar_kwargs = default_cbar_kwargs.copy()
                cbar_kwargs.update(item.get("colorbar_kwargs", {}))
                fig.colorbar(im, ax=ax, **cbar_kwargs)

        for empty_slot in panel.get("empty_slots", []):
            axes[empty_slot[0], empty_slot[1]].axis("off")

        # Drawn inside the top-left image (axes coordinates) so it doesn't affect layout or spacing
        if panel_labels is not None and panel_idx < len(panel_labels):
            axes[0, 0].text(
                0.03,
                0.97,
                panel_labels[panel_idx],
                transform=axes[0, 0].transAxes,
                color="yellow",
                fontsize=panel_label_fontsize,
                fontweight="bold",
                ha="left",
                va="top",
            )

        for colorbar_spec in panel.get("column_colorbars", []):
            col_idx = colorbar_spec["col"]
            mappable = image_handles[colorbar_spec.get("row", 0), col_idx]
            target_axes = [axes[row_idx, col_idx] for row_idx in range(panel["shape"][0])]
            cbar_kwargs = {
                "location": "bottom",
                "fraction": 0.08,
                "pad": 0.08,
                "shrink": colorbar_spec.get("shrink", 0.95),
            }
            cbar_kwargs.update(colorbar_spec.get("colorbar_kwargs", {}))
            cbar = fig.colorbar(mappable, ax=target_axes, **cbar_kwargs)
            if "label" in colorbar_spec:
                cbar.set_label(colorbar_spec["label"], fontsize=colorbar_spec.get("label_fontsize", 16))
            if "ticks" in colorbar_spec:
                cbar.set_ticks(colorbar_spec["ticks"])
            cbar.ax.tick_params(labelsize=colorbar_spec.get("tick_labelsize", 18))

        panel_colorbar = panel.get("panel_colorbar")
        if panel_colorbar is not None:
            mappable = image_handles[panel_colorbar.get("row", 0), panel_colorbar.get("col", 0)]
            left = min(ax.get_position().x0 for ax in axes.ravel())
            right = max(ax.get_position().x1 for ax in axes.ravel())
            bottom = min(ax.get_position().y0 for ax in axes.ravel())
            cbar_height = panel_colorbar.get("height", 0.014)
            cbar_pad = panel_colorbar.get("pad", 0.018)
            cax = fig.add_axes([left, bottom - cbar_pad - cbar_height, right - left, cbar_height])
            cbar_kwargs = {"orientation": "horizontal"}
            cbar_kwargs.update(panel_colorbar.get("colorbar_kwargs", {}))
            cbar = fig.colorbar(mappable, cax=cax, **cbar_kwargs)
            if "label" in panel_colorbar:
                cbar.set_label(panel_colorbar["label"], fontsize=panel_colorbar.get("label_fontsize", 22))
            if "ticks" in panel_colorbar:
                cbar.set_ticks(panel_colorbar["ticks"])
            cbar.ax.tick_params(labelsize=panel_colorbar.get("tick_labelsize", 18))

    if row_labels is not None:
        row_positions: list[float] = []
        n_global_rows = outer_shape[0] * max(panel["shape"][0] for panel in panels)
        for global_row in range(n_global_rows):
            top_group = global_row // 2
            inner_row = global_row % 2
            row_axes = []
            for outer_col in range(outer_shape[1]):
                axes = panel_axes.get((top_group, outer_col))
                if axes is None or inner_row >= axes.shape[0]:
                    continue
                row_axes.extend(list(axes[inner_row, :].ravel()))
            if row_axes:
                y0 = min(ax.get_position().y0 for ax in row_axes)
                y1 = max(ax.get_position().y1 for ax in row_axes)
                row_positions.append(0.5 * (y0 + y1))

        leftmost = min(ax.get_position().x0 for axes in panel_axes.values() for ax in axes.ravel())
        for label, ypos in zip(row_labels, row_positions):
            fig.text(
                leftmost - row_label_xpad,
                ypos,
                label,
                rotation=90,
                fontsize=row_label_fontsize,
                ha="center",
                va="center",
            )


    if save_path is not None:
        print(f"Saving figure to: {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

def analyse_gammex(gammex_data=gammex_data):

    fWa, fI, fGd, fPE, fPVC = load_all_materials()

    # Debug: plot attenuation curves

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))

    ax[0].plot(fWa, label="Water")
    ax[0].plot(fPE, label="PE")
    ax[0].plot(fPVC, label="PVC")
    ax[0].legend()

    ax[1].plot(fI, label="Iodine")
    ax[1].plot(fGd, label="Gadolinium")
    ax[1].legend()
    plt.savefig(PLOT_DIR / "attenuations.png", dpi=300)


    # weights = torch.softmax(load_weights(), dim=1).numpy()  # [nBins x nE]
    weights = load_weights().numpy()  # [nBins x nE]


    pepvc_to_waio = LinearMaterialDecomposition(np.stack([fPE, fPVC], axis=0), np.stack([fWa, fI], axis=0), weights, f_wa=fWa, pathlength_wa=100, plot=True, mats_dom=["PE", "PVC"], mats_range=["Water", "Iodine"])



    # Gammex

    # convert torch tensors to numpy arrays

    gammex_data = {k: v.numpy() for k, v in gammex_data.items()}

    PE = gammex_data["PE"]
    PVC = gammex_data["PVC"]
    I_gt = gammex_data["I"] 
    Gd_gt = gammex_data["Gd"]
    mask = (I_gt > 1e-4) | (Gd_gt > 1e-5)

    # Plot initial PE/PVC images

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(PE, cmap="gray")
    ax[0].set_title("PE")
    ax[0].axis("off")
    ax[1].imshow(PVC, cmap="gray")
    ax[1].set_title("PVC")
    ax[1].axis("off")
    plt.savefig(PLOT_DIR / "gammex_pepvc.png", dpi=300, bbox_inches="tight")

    mask_comps = cv2.connectedComponents(mask.astype(np.uint8))

    # Convert PE/PVC to Water/Iodine

    aPEPVC = np.stack([PE, PVC], axis=0)

    aWaI = pepvc_to_waio.apply(aPEPVC)

    Wa = aWaI[0, :]
    I = aWaI[1, :]


    # Perform segmentation

    # Get circular centerd mask

    C = np.array(mask.shape) / 2
    R = min(C) * 0.3

    Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
    dist_from_center = np.sqrt((X - C[1])**2 + (Y - C[0])**2)
    circular_mask = dist_from_center <= R

    # Obtained background and signal

    I_mean = np.mean(I[circular_mask])
    Wa_mean = np.mean(Wa[circular_mask])

    WaR = (Wa - Wa_mean) * mask 
    IR = (I - I_mean) * mask 

    aR = np.stack([WaR, IR], axis=0)

    WaB = Wa - WaR
    IB = I - IR

    aB = np.stack([WaB, IB], axis=0)
    aR = np.stack([WaR, IR], axis=0)
    aC_gt = np.stack([I_gt, Gd_gt], axis=0)

    # Perform nonlinear optimization using Newton's method



    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]
    Eactvector = load_energies()  # [nE]

    ray_trafo = RayTrafoModule(physical_size=(220, 220), shape=(512, 512), n_angles=512, n_detectors=512, geometry='parallel', src_radius=40, det_radius=40)

    calibrated_model = CalibratedForwardModel(fB, fR, fC, weights, Eactvector, output_dir=PLOT_DIR)

    AB = ray_trafo.forward(torch.from_numpy(aB).to(torch.float32))
    AR = ray_trafo.forward(torch.from_numpy(aR).to(torch.float32))
    AC_gt = ray_trafo.forward(torch.from_numpy(aC_gt).to(torch.float32))

    A = torch.cat([AB, AR, AC_gt], dim=0)  # [6 x H x W]

    print(A.shape, AB.shape, AR.shape, AC_gt.shape)

    AB_plot = AB.detach().cpu().numpy()
    AR_plot = AR.detach().cpu().numpy()

    sino_sz = AB.shape[1:]

    AB = AB.permute(1, 2, 0).contiguous().view(-1, 2)  # [P x 2]
    AR = AR.permute(1, 2, 0).contiguous().view(-1, 2)  # [P x 2]



    AC_pred_nonlinear = calibrated_model._uncalibrated_forward(A, ACinit=None)[0]

    print(AC_pred_nonlinear.shape, AC_gt.shape)
    
    #AC_pred_nonlinear = AC_pred_nonlinear.view(*sino_sz, 2).permute(2, 0, 1).contiguous() # [2 x H x W]

    Ilim = [0, 0.1]
    Gdlim = [0, 0.1]

    aC_pred_nonlinear = ray_trafo.backward(AC_pred_nonlinear).detach().cpu().numpy()  # [2 x H x W]

    # The model may compute on CUDA; move the sinograms to CPU numpy for plotting
    AC_pred_nonlinear = AC_pred_nonlinear.detach().cpu().numpy()
    AC_gt = AC_gt.detach().cpu().numpy()



    aC_pred_nonlinear[0, :] = aC_pred_nonlinear[0, :] * rhoI
    aC_pred_nonlinear[1, :] = aC_pred_nonlinear[1, :] * rhoGd
    aC_gt[0, :] = aC_gt[0, :] * rhoI
    aC_gt[1, :] = aC_gt[1, :] * rhoGd

    # post process predicted images
    aI_pred_nonlinear = aC_pred_nonlinear[0, :]
    aGd_pred_nonlinear = aC_pred_nonlinear[1, :]
    I_gt = aC_gt[0, :]
    Gd_gt = aC_gt[1, :]
    # ROI scatter plot

    pred_vals = {"I": [], "Gd": []}
    gt_vals = {"I": [], "Gd": []}
    for val in range(1, mask_comps[0]):
        
        cc = (mask_comps[1] == val).astype(np.uint8)  # Component mask
    
        # Compute predicted values for the component
        

        pred_vals["I"].append(aI_pred_nonlinear[cc == 1].mean())
        pred_vals["Gd"].append(aGd_pred_nonlinear[cc == 1].mean())
        gt_vals["I"].append(I_gt[cc == 1].mean())
        gt_vals["Gd"].append(Gd_gt[cc == 1].mean())


    # plot scatter plot of predicted vs ground truth values

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(gt_vals["I"], pred_vals["I"], color="red", label="Iodine", marker="o", s=20, alpha=0.7)
    ax[0].plot([0, 24], [0, 24], color="k", linestyle="-.", linewidth=0.6)

    ax[0].set_xlabel("Ground Truth Iodine (mg/mL)")

    ax[0].set_ylabel("Predicted Iodine (mg/mL)")

    ax[0].legend()
    ax[1].scatter(gt_vals["Gd"], pred_vals["Gd"], color="blue", label="Gadolinium", marker="o", s=20, alpha=0.7)
    ax[1].plot([0, 4], [0, 4], color="k", linestyle="-.", linewidth=0.6)
    ax[1].set_xlabel("Ground Truth Gadolinium (mg/mL)")
    ax[1].set_ylabel("Predicted Gadolinium (mg/mL)")
    ax[1].legend()

    print("Saving Gammex scatter plot to:", PLOT_DIR / "gammex_scatter_nonlinear.png")
    fig.savefig(PLOT_DIR / "gammex_scatter_nonlinear.png", dpi=300, bbox_inches="tight")

    # Plot nonlinear results







    Ilim = [-20, 20]
    Gdlim = [-5, 5]
    diff_nonlinear = np.abs(aC_pred_nonlinear - aC_gt)

    gammex_panels = [
        {
            "title": None,
            "shape": (2, 2),
            "items": [
                {"image": WaB, "title": "Background (Water)"},
                {"image": IB, "title": "Background (Iodine)"},
                {"image": WaR, "title": "Signal (Water)"},
                {"image": IR, "title": "Signal (Iodine)"},
            ],
        },
        {
            "title": None,
            "shape": (2, 2),
            "items": [
                {"image": AB_plot[0, :], "title": "Background (Water)"},
                {"image": AB_plot[1, :], "title": "Background (Iodine)"},
                {"image": AR_plot[0, :], "title": "Signal (Water)"},
                {"image": AR_plot[1, :], "title": "Signal (Iodine)"},
            ],
        },
        {
            "title": None,
            "shape": (2, 2),
            "items": [
                {"image": AC_gt[0, :], "title": "Iodine gt", "vmin": 0, "vmax": 0.3},
                {"image": AC_gt[1, :], "title": "Gadolinium gt", "vmin": 0, "vmax": 0.05},
                {"image": AC_pred_nonlinear[0, :], "title": "Iodine pred", "vmin": 0, "vmax": 0.3},
                {"image": AC_pred_nonlinear[1, :], "title": "Gadolinium pred", "vmin": 0, "vmax": 0.05},
            ],
        },
        {
            "title": None,
            "shape": (3, 2),
            "items": [
                {"image": aC_gt[0, :], "title": "Iodine gt", "vmin": 0, "vmax": 20},
                {"image": aC_gt[1, :], "title": "Gadolinium gt", "vmin": 0, "vmax": 5},
                {"image": aC_pred_nonlinear[0, :], "title": "Iodine pred", "vmin": 0, "vmax": 20},
                {"image": aC_pred_nonlinear[1, :], "title": "Gadolinium pred", "vmin": 0, "vmax": 5},
                {"image": diff_nonlinear[0, :], "title": "Iodine diff", "vmin": 0, "vmax": 20},
                {"image": diff_nonlinear[1, :], "title": "Gadolinium diff", "vmin": 0, "vmax": 5},
            ],
        },
    ]

    overview_path = PLOT_DIR / "gammex_overview_nested.png"
    print("Saving combined Gammex overview to:", overview_path)
    plot_nested_image_panels(
        gammex_panels,
        outer_shape=(2, 2),
        figsize=(16, 18),
        save_path=overview_path,
        suptitle=None,
        constrained_layout=False,
        outer_wspace=0.0,
        outer_hspace=0.0,
        inner_wspace=0.0,
        inner_hspace=0.0,
    )

def analyse_antropomorphic(average_slice=True):
    def organs_to_mask(seg: np.ndarray, organs: list[str]) -> np.ndarray:


        organ_labels = [labels[organ] for organ in organs if organ in labels]
        print(f"Organ labels for {organs}: {organ_labels}")
        mask = np.isin(seg, organ_labels)
        return mask

    RECON_DIR =  REPO_ROOT / "AttenuationMD_python" / "data"
    DATA_DIR = ANTROPOMORPHIC_PATH



    filenames = ["water_recon_contrast.nii.gz", 
                 "i_recon_contrast.nii.gz",
                 "water_recon_no_contrast.nii.gz",
                 "i_recon_no_contrast.nii.gz",
    ]
    gt_filenames = ["i.nii.gz", "gd.nii.gz"]



    data = {fname: load(RECON_DIR / fname).get_fdata() for fname in filenames}
    data_gt = {fname: load(DATA_DIR / fname).get_fdata() for fname in gt_filenames}

    mid_slice = data_gt["i.nii.gz"].shape[0] // 2
    data_gt = {fname: vol[mid_slice, :, :].squeeze() for fname, vol in data_gt.items()}

    # Segmentation and labels

    seg = load(RECON_DIR / "segmentation.nii.gz").get_fdata()   
    labels = torch.load(RECON_DIR / "labels.pt")["segmentation"]

    # transpose keys and values in the labels dictionary: initilly it is {value, name} we want {name, value}


    labels = {v: k for k, v in labels.items()}
    seg = seg[mid_slice, :, :].squeeze()

    organs = ["kidney_right", "kidney_left", "liver", "spleen", "gallbladder", "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", 
              # "small_bowel", 
              # "duodenum", 
              # "colon", 
              "kidney_cyst_right", "aorta", "inferior_vena_cava", "blood"]
    mask = organs_to_mask(seg, organs)
    # {1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'pancreas', 8: 'adrenal_gland_right', 9: 'adrenal_gland_left', 10: 'lung_upper_lobe_left', 11: 'lung_lower_lobe_left', 12: 'lung_middle_lobe_right', 13: 'lung_lower_lobe_right', 14: 'esophagus', 15: 'small_bowel', 16: 'duodenum', 17: 'colon', 18: 'kidney_cyst_right', 29: 'aorta', 30: 'inferior_vena_cava', 31: 'portal_vein_and_splenic_vein', 32: 'iliac_artery_left', 33: 'iliac_artery_right', 34: 'iliac_vena_left', 35: 'iliac_vena_right', 38: 'spinal_cord', 39: 'gluteus_maximus_left', 40: 'gluteus_maximus_right', 41: 'gluteus_medius_left', 42: 'gluteus_medius_right', 43: 'gluteus_minimus_left', 44: 'gluteus_minimus_right', 45: 'autochthon_left', 46: 'autochthon_right', 47: 'iliopsoas_left', 48: 'iliopsoas_right', 62: 'costal_cartilages', 64: 'intervertebral_discs', 65: 'subcutaneous_fat', 66: 'torso_fat', 67: 'skeletal_muscle', 68: 'intermuscular_fat', 69: 'lung_vessels', 70: 'blood', 71: 'bone'}, 'body': {1: 'body_trunc'}

    print("Data shapes:")
    for fname, vol in data.items():
        print(f"{fname}: {vol.shape}")

    if average_slice:

        data = {fname: vol.mean(axis=0) for fname, vol in data.items()}
    else:
        # Extract middle slice of the 3D GT volime
        mid_slice = data[filenames[0]].shape[0] // 2
        data = {fname: vol[mid_slice, :, :].squeeze() for fname, vol in data.items()}
        
    # Antropomorphic

    rhoI = 4930
    scaleI = 20
    scaleGd = 200
    rhoGd = 7900

    # convert torch tensors to numpy arrays

    Wa_contrast = data[filenames[0]]
    I_contrast = data[filenames[1]] * scaleI / rhoI
    Wa_no_contrast = data[filenames[2]]
    I_no_contrast = data[filenames[3]] * scaleI / rhoI


    
    # Estimate covariance from the paired acquisition, then denoise the contrast pair.
    pca_sigma = 0.0
    contrast_images = np.stack([Wa_contrast, I_contrast], axis=0)
    no_contrast_images = np.stack([Wa_no_contrast, I_no_contrast], axis=0)
    denoiser = PairedContrastCovarianceDRUNetDenoiser(
        sigmas=pca_sigma,
        blur_sigma=3.0,
        trim_fraction=0.2,
    )
    if pca_sigma > 0:
        denoised_contrast_images = denoiser(contrast_images, no_contrast_images)
        print(f"Estimated paired-acquisition noise covariance:\n{denoiser.covariance}")
        print(f"Estimated paired-acquisition noise correlation: {denoiser.correlation[0, 1]:.3f}")
    else:
        denoised_contrast_images = contrast_images
        print("No denoising applied (pca_sigma=0).")

    Wa_contrast_denoised, I_contrast_denoised = denoised_contrast_images

    # Air mask

    soft_mask = (Wa_contrast_denoised < 0.95)
    # remove soft mask from mask

    mask = mask & (~soft_mask)

    # plot segmentation and mask

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(seg, cmap="tab20")
    ax[0].set_title("Segmentation")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")
    print(f"Saving segmentation and mask to {PLOT_DIR / 'antropomorphic_segmentation_and_mask.png'}")
    plt.savefig(PLOT_DIR / "antropomorphic_segmentation_and_mask.png", dpi=300, bbox_inches="tight")

    
    WaS = (Wa_contrast_denoised - Wa_no_contrast) * mask
    IS = (I_contrast_denoised - I_no_contrast) * mask
    WaB = Wa_no_contrast
    IB = I_no_contrast

    # remove negative values from the signal images and add to the background images
    neg = (IS < 0) | (WaS < 0)
    
    #WaB[neg] = WaB[neg] - WaS[neg]
    #IB[neg] = IB[neg] - IS[neg]
    WaS[neg] = 0
    IS[neg] = 0
    
    # Plot initial contrast images and their denoised versions

    fig, ax  = plt.subplots(2, 2, figsize=(18, 12))
    im=ax[0, 0].imshow(Wa_contrast, cmap="gray", aspect="auto")
    ax[0, 0].set_title("Water contrast")
    ax[0, 0].axis("off")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    im=ax[1, 0].imshow(Wa_contrast_denoised, cmap="gray", aspect="auto")
    ax[1, 0].set_title("Water contrast (denoised)")
    ax[1, 0].axis("off")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
    im=ax[0, 1].imshow(I_contrast, cmap="gray", aspect="auto")
    ax[0, 1].set_title("Iodine contrast")
    ax[0, 1].axis("off")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)
    im=ax[1, 1].imshow(I_contrast_denoised, cmap="gray", aspect="auto")
    ax[1, 1].set_title("Iodine contrast (denoised)")
    ax[1, 1].axis("off")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
    print(f"Saving denoised antropomorphic images to {PLOT_DIR / 'antropomorphic_denoised_images.png'}")
    plt.savefig(PLOT_DIR / f"antropomorphic_denoised_contrast_images_covariance_sigma={pca_sigma}.png", dpi=300, bbox_inches="tight")
    
    # Plot initial antropomorphic images and their difference

    fig = plt.figure(figsize=(12, 13))
    outer_grid = fig.add_gridspec(2, 1, height_ratios=(1, 0.035), hspace=0.12)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.1, top=0.92)
    image_grid = outer_grid[0].subgridspec(2, 2, wspace=0, hspace=0)
    colorbar_grid = outer_grid[1].subgridspec(1, 2, wspace=0.2)
    ax = np.array([
        [fig.add_subplot(image_grid[0, 0]), fig.add_subplot(image_grid[0, 1])],
        [fig.add_subplot(image_grid[1, 0]), fig.add_subplot(image_grid[1, 1])],
    ])

    water_images = (Wa_contrast * rhoW, Wa_no_contrast * rhoW)
    iodine_images = (I_contrast * rhoI, I_no_contrast * rhoI)
    water_im = ax[0, 0].imshow(water_images[0], cmap="gray", vmin=0, vmax=1500, aspect="auto")
    ax[1, 0].imshow(water_images[1], cmap="gray", vmin=0, vmax=1500, aspect="auto")
    iodine_im = ax[0, 1].imshow(iodine_images[0], cmap="gray", vmin=0, vmax=20, aspect="auto")
    ax[1, 1].imshow(iodine_images[1], cmap="gray", vmin=0, vmax=20, aspect="auto")

    ax[0, 0].set_title("Water", fontsize=28, pad=18)
    ax[0, 1].set_title("Iodine", fontsize=28, pad=18)
    ax[0, 0].set_ylabel("Full simulation", fontsize=26, labelpad=24)
    ax[1, 0].set_ylabel("Background simulation", fontsize=26, labelpad=24)
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    water_colorbar = fig.colorbar(water_im, cax=fig.add_subplot(colorbar_grid[0, 0]), orientation="horizontal")
    water_colorbar.set_label("Water [mg/ml]", fontsize=24, labelpad=10)
    water_colorbar.ax.tick_params(labelsize=20)
    iodine_colorbar = fig.colorbar(iodine_im, cax=fig.add_subplot(colorbar_grid[0, 1]), orientation="horizontal")
    iodine_colorbar.set_label("Iodine [mg/ml]", fontsize=24, labelpad=10)
    iodine_colorbar.ax.tick_params(labelsize=20)
    print(f"Saving initial antropomorphic images to {PLOT_DIR / f'antropomorphic_initial_images_covariance_sigma={pca_sigma}.png'}")
    plt.savefig(PLOT_DIR / f"antropomorphic_initial_images_covariance_sigma={pca_sigma}.png", dpi=300, bbox_inches="tight")


    # Obtained background and signal


    aR = np.stack([WaS, IS], axis=0)

    aB = np.stack([WaB, IB], axis=0)
    aC_gt = np.stack([data_gt["i.nii.gz"] * scaleI, data_gt["gd.nii.gz"] * scaleGd], axis=0)


    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]
    Eactvector = load_energies()  # [nE]

    # Forward projection of background and signal

    ray_trafo = RayTrafoModule( 
                physical_size = (471, 471), 
                shape = (512, 512), 
                n_angles = 512, 
                n_detectors= 512)

    AR = ray_trafo.forward(torch.from_numpy(aR).to(torch.float32))
    AB = ray_trafo.forward(torch.from_numpy(aB).to(torch.float32))
    AC_gt = ray_trafo.forward(torch.from_numpy(aC_gt).to(torch.float32))


    calibrated_model = CalibratedForwardModel(fB, fR, fC, weights, Eactvector, output_dir=PLOT_DIR, normalize_materials=True)

    
    igd_to_waio = LinearMaterialDecomposition(fC.numpy(), fB.numpy(), weights.numpy(), f_wa=fB[0, :].numpy(), pathlength_wa=100, plot=False, mats_dom=["Water", "Iodine"], mats_range=["Water", "Gadolinium"])

    sino_sz = AB.shape[1:]

    #AB = AB.permute(1, 2, 0).contiguous().view(-1, 2)  # [P x 2]
    #AR = AR.permute(1, 2, 0).contiguous().view(-1, 2)  # [P x 2]

    AC_pred_nonlinear = calibrated_model._uncalibrated_forward(A=torch.cat([AB, AR], dim=0))[0]
    #AC_pred_nonlinear = AC_pred_nonlinear.view(*sino_sz, 2).permute(2, 0, 1).contiguous() # [2 x H x W]

    # Backproject predicted sinogram to obtain predicted images

    aC_pred_nonlinear = ray_trafo.backward(AC_pred_nonlinear).cpu().numpy()  # [2 x H x W]
    aC_pred_linear = igd_to_waio.apply_inverse(aR)  # [2 x H x W]

    # Apply mask

    aC_pred_nonlinear = aC_pred_nonlinear * np.expand_dims(mask, axis=0)
    aC_pred_linear = aC_pred_linear * np.expand_dims(mask, axis=0)


    aC_pred_nonlinear[0, :] = aC_pred_nonlinear[0, :] * rhoI
    aC_pred_nonlinear[1, :] = aC_pred_nonlinear[1, :] * rhoGd

    aC_pred_linear[0, :] = aC_pred_linear[0, :] * rhoI 
    aC_pred_linear[1, :] = aC_pred_linear[1, :] * rhoGd 

    AC_pred_mgml = AC_pred_nonlinear.detach().cpu().numpy().copy()
    AC_pred_mgml[0] *= rhoI
    AC_pred_mgml[1] *= rhoGd
    AC_gt_mgml = AC_gt.detach().cpu().numpy()
    image_limits = [(0, 10), (0, 5)]
    sinogram_limits = [(0, 400), (0, 200)]
    panels = [
        {
            "shape": (2, 2),
            "column_titles": ["Water", "Iodine"],
            "items": [
                {"image": aB[0], "title": ""},
                {"image": aB[1], "title": ""},
                {"image": aR[0], "title": ""},
                {"image": aR[1], "title": ""},
            ],
        },
        {
            "shape": (2, 2),
            "column_titles": ["Water", "Iodine"],
            "items": [
                {"image": AB[0].detach().cpu().numpy(), "title": "", "aspect": "auto"},
                {"image": AB[1].detach().cpu().numpy(), "title": "", "aspect": "auto"},
                {"image": AR[0].detach().cpu().numpy(), "title": "", "aspect": "auto"},
                {"image": AR[1].detach().cpu().numpy(), "title": "", "aspect": "auto"},
            ],
        },
        {
            "shape": (2, 2),
            "column_titles": ["Iodine", "Gadolinium"],
            "panel_colorbar": {"label": "mg/mL", "ticks": [0, 100, 200, 300, 400], "label_fontsize": 22, "tick_labelsize": 18},
            "items": [
                {"image": AC_gt_mgml[0], "title": "", "vmin": sinogram_limits[0][0], "vmax": sinogram_limits[0][1], "aspect": "auto"},
                {"image": AC_gt_mgml[1], "title": "", "vmin": sinogram_limits[1][0], "vmax": sinogram_limits[1][1], "aspect": "auto"},
                {"image": AC_pred_mgml[0], "title": "", "vmin": sinogram_limits[0][0], "vmax": sinogram_limits[0][1], "aspect": "auto"},
                {"image": AC_pred_mgml[1], "title": "", "vmin": sinogram_limits[1][0], "vmax": sinogram_limits[1][1], "aspect": "auto"},
            ],
        },
        {
            "shape": (2, 2),
            "column_titles": ["Iodine", "Gadolinium"],
            "panel_colorbar": {"label": "mg/mL", "ticks": [0, 2.5, 5, 7.5, 10], "label_fontsize": 22, "tick_labelsize": 18},
            "items": [
                {"image": aC_gt[0], "title": "", "vmin": image_limits[0][0], "vmax": image_limits[0][1]},
                {"image": aC_gt[1], "title": "", "vmin": image_limits[1][0], "vmax": image_limits[1][1]},
                {"image": aC_pred_nonlinear[0], "title": "", "vmin": image_limits[0][0], "vmax": image_limits[0][1]},
                {"image": aC_pred_nonlinear[1], "title": "", "vmin": image_limits[1][0], "vmax": image_limits[1][1]},
            ],
        },
    ]
    overview_path = PLOT_DIR / f"antropomorphic_overview_covariance_sigma={pca_sigma}.png"
    print(f"Saving combined anthropomorphic overview to {overview_path}")
    plot_nested_image_panels(
        panels,
        outer_shape=(2, 2),
        figsize=(16, 18),
        save_path=overview_path,
        constrained_layout=False,
        outer_wspace=0.1,
        outer_hspace=0.14,
        inner_wspace=0.0,
        inner_hspace=0.0,
        row_labels=["Background", "Signal", "Ground Truth", "Prediction"],
        row_label_fontsize=22,
        row_label_xpad=0.035,
    )

def analyse_uzb(scan_id=449):


    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]
    Eactvector = load_energies()  # [nE]

    model = CalibratedForwardModel(fB, fR, fC, weights, Eactvector, output_dir=PLOT_DIR, normalize_materials=True, debug=False)
    model.ray_trafo = model._get_ray_trafo(scan_id=scan_id)
    
    a, mask_img = model._load_scan_from_id(scan_id, refine_mask=True)
    aB = a[0:2, :, :]  # Background (Water, Iodine)
    aR = a[2:4, :, :]  # Signal (Water, Iodine)
    aC_gt = model._extract_gt(a)  # Ground truth AC

    # forward projection to get the predicted AC

    A = model.ray_trafo.forward(a)
    AB = A[0:2, :, :]  # Background (Water, Iodine)
    AR = A[2:4, :, :]  # Signal (Water, Iodine)
    AC_gt = model._extract_gt(A)  # Ground truth AC

    # Run uncalibrated forward model to get the predicted AC

    AC_pred_uncal = model._uncalibrated_forward(A, ACinit=None)[0]
    aC_pred_uncal =  model.ray_trafo.backward(AC_pred_uncal)

    # Rescale predicted and ground truth AC to mg/mL

    AC_pred_uncal_sino = AC_pred_uncal.clone()
    AC_gt_sino = AC_gt.clone()
    AC_pred_uncal_sino[0, :] = AC_pred_uncal_sino[0, :] * rhoI
    AC_pred_uncal_sino[1, :] = AC_pred_uncal_sino[1, :] * rhoGd
    AC_gt_sino[0, :] = AC_gt_sino[0, :] * rhoI
    AC_gt_sino[1, :] = AC_gt_sino[1, :] * rhoGd

    aC_pred_uncal[0, :] = aC_pred_uncal[0, :] * rhoI * mask_img
    aC_pred_uncal[1, :] = aC_pred_uncal[1, :] * rhoGd * mask_img
    aC_gt[0, :] = aC_gt[0, :] * rhoI
    aC_gt[1, :] = aC_gt[1, :] * rhoGd

    # Move everything to CPU numpy for plotting (the model may compute on CUDA)

    def _to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    aB, aR, AB, AR, aC_gt, aC_pred_uncal, AC_gt_sino, AC_pred_uncal_sino, mask_img = map(
        _to_numpy, (aB, aR, AB, AR, aC_gt, aC_pred_uncal, AC_gt_sino, AC_pred_uncal_sino, mask_img)
    )

    # Debug: plot and save the image mask

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(mask_img, cmap='gray')
    ax.set_title("Mask Image")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(PLOT_DIR / f"scan_{scan_id}_mask_image.png", dpi=300, bbox_inches="tight")
    
    diff = np.abs(aC_pred_uncal - aC_gt)
    
    panels = [
        {
            "title": None,
            "shape": (2, 2),
            "column_titles": ["Water", "Iodine"],
            "items": [
                {"image": aB[0, :], "title": ""},
                {"image": aB[1, :], "title": ""},
                {"image": aR[0, :], "title": ""},
                {"image": aR[1, :], "title": ""},
            ],
        },
        {
            "title": None,
            "shape": (2, 2),
            "column_titles": ["Water", "Iodine"],
            "items": [
                {"image": AB[0, :], "title": ""},
                {"image": AB[1, :], "title": ""},
                {"image": AR[0, :], "title": ""},
                {"image": AR[1, :], "title": ""},
            ],
        },
        {
            "title": None,
            "shape": (2, 2),
            "column_titles": ["Iodine", "Gadolinium"],
            "panel_colorbar": {"label": "mg/mL", "ticks": [0, 200, 400, 600], "label_fontsize": 22, "tick_labelsize": 18},
            "items": [
                {"image": AC_gt_sino[0, :], "title": "", "vmin": 0, "vmax": 600, "aspect": "auto"},
                {"image": AC_gt_sino[1, :], "title": "", "vmin": 0, "vmax": 600, "aspect": "auto"},
                {"image": AC_pred_uncal_sino[0, :], "title": "", "vmin": 0, "vmax": 600, "aspect": "auto"},
                {"image": AC_pred_uncal_sino[1, :], "title": "", "vmin": 0, "vmax": 600, "aspect": "auto"},
            ],
        },
        {
            "title": None,
            "shape": (2, 2),
            "column_titles": ["Iodine", "Gadolinium"],
            "panel_colorbar": {"label": "mg/mL", "ticks": [0, 5, 10, 15], "label_fontsize": 22, "tick_labelsize": 18},
            "items": [
                {"image": aC_gt[0, :], "title": "", "vmin": 0, "vmax": 15, "aspect": "auto"},
                {"image": aC_gt[1, :], "title": "", "vmin": 0, "vmax": 15, "aspect": "auto"},
                {"image": aC_pred_uncal[0, :], "title": "", "vmin": 0, "vmax": 15, "aspect": "auto"},
                {"image": aC_pred_uncal[1, :], "title": "", "vmin": 0, "vmax": 15, "aspect": "auto"},
            ],
        },
    ]

    overview_path = PLOT_DIR / f"scan_{SCAN_TO_LABEL[scan_id]}_overview_nested.png"
    print("Saving combined UZB overview to:", overview_path)
    plot_nested_image_panels(
        panels,
        outer_shape=(2, 2),
        figsize=(16, 18),
        save_path=overview_path,
        suptitle=None,
        constrained_layout=False,
        outer_wspace=0.1,
        outer_hspace=0.14,
        inner_wspace=0.0,
        inner_hspace=0.0,
        row_labels=["Background", "Signal", "Ground Truth", "Prediction"],
        row_label_fontsize=22,
        row_label_xpad=0.035,
    )
    model.plot_cache(output_dir=PLOT_DIR)

def _analyse_endoleak_scan(scan_id, calibrated_model, slice_idx=251, num_slices=5):


    def _get_slice(slice_idx, num_slices, vol: np.ndarray):

        return np.mean(vol[slice_idx-num_slices // 2:slice_idx+num_slices // 2, :, :], axis=0)


    scan_path = UZB_DATA / SCANS[scan_id]
    
    raw_scans = {}

    # Load raw volumes and ground truth for the specified scan ID

    for fname in ["iodine_basis_vol", "water_basis_vol", "mask_vol", "mask_roi"]:
        if not (scan_path / f"{fname}.nii").exists():
            raise FileNotFoundError(f"Expected file {fname}.nii not found in {scan_path}")
        
        print(f"Loading {fname} from {scan_path / f'{fname}.nii'}")
        raw_scans[fname.replace("_vol", "")] = _get_slice(slice_idx, num_slices, nib.load(scan_path / f"{fname}.nii").get_fdata().astype(np.float32))
    

    # plot the loaded slices

    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    im = ax[0].imshow(raw_scans["water_basis"], cmap='gray')
    ax[0].set_title("Water Basis Volume Slice")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)    
    im = ax[1].imshow(raw_scans["iodine_basis"], cmap='gray')
    ax[1].set_title("Iodine Basis Volume Slice")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    im = ax[2].imshow(raw_scans["mask"], cmap='gray')
    ax[2].set_title("Mask Volume Slice")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    plt.savefig(PLOT_DIR / f"endoleak_slices_{scan_id}.png", dpi=300)


    # Obtain mask

    mask_whole = raw_scans["water_basis"] > 0.5

    # Only keep largest connected component in the mask

    num_labels, labels_im = cv2.connectedComponents(mask_whole.astype(np.uint8))

    if num_labels > 1:
        largest_component = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
        mask_whole = labels_im == largest_component

    mask_roi = raw_scans["mask_roi"] > 0.5
    mask_bkg = mask_whole & (~mask_roi)  # Background mask: largest component and not ROI

        # Plot the mask

    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].imshow(mask_whole, cmap='gray')
    ax[0].set_title("Largest Connected Component Mask")
    ax[1].imshow(mask_bkg, cmap='gray')
    ax[1].set_title("Background Mask (Largest Component & Not ROI)")
    plt.savefig(PLOT_DIR / f"endoleak_mask_{scan_id}.png", dpi=300)

    # Obtain background and signal images (assuming the background in the sack is (1, 0) wa/io)
      
    WaR = mask_roi * (raw_scans["water_basis"] - 1)
    IR = mask_roi * raw_scans["iodine_basis"]
    WaB = raw_scans["water_basis"] - WaR
    IB = raw_scans["iodine_basis"] - IR

    # Remove negative values from the background and signal images

    aB = np.stack([WaB, IB], axis=0)
    aR = np.stack([WaR, IR], axis=0)


    # plot the background and signal 

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    im = ax[0, 0].imshow(WaB, cmap='gray')
    ax[0, 0].set_title("Water Basis Background")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(IB, cmap='gray')
    ax[0, 1].set_title("Iodine Basis Background")       
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04) 
    im = ax[1, 0].imshow(WaR, cmap='gray')
    ax[1, 0].set_title("Water Basis Signal")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(IR, cmap='gray')
    ax[1, 1].set_title("Iodine Basis Signal")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
    print(f"Saving endoleak background and signal images to {PLOT_DIR / f'endoleak_bkg_signal_{scan_id}.png'}")
    plt.savefig(PLOT_DIR / f"endoleak_bkg_signal_{scan_id}.png", dpi=300)


    # Forward projection of background and signal

    ray_trafo = RayTrafoModule()

    AR = ray_trafo.forward(torch.from_numpy(aR).to(torch.float32))
    AB = ray_trafo.forward(torch.from_numpy(aB).to(torch.float32))

    # plot the sinograms

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    im = ax[0, 0].imshow(AB[0, :], cmap='gray')
    ax[0, 0].set_title("Background Sinogram (Water)")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(AB[1, :], cmap='gray')
    ax[0, 1].set_title("Background Sinogram (Iodine)")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)
    im = ax[1, 0].imshow(AR[0, :], cmap='gray')
    ax[1, 0].set_title("Signal Sinogram (Water)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(AR[1, :], cmap ='gray')
    ax[1, 1].set_title("Signal Sinogram (Iodine)")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
    print(f"Saving endoleak sinograms to {PLOT_DIR / f'endoleak_sinograms_{scan_id}.png'}")
    plt.savefig(PLOT_DIR / f"endoleak_sinograms_{scan_id}.png", dpi=300)


    # non linear optimization using Newton's method

    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]

    igd_to_waio = LinearMaterialDecomposition(fC.numpy(), fB.numpy(), weights.numpy(), f_wa=fB[0, :].numpy(), pathlength_wa=100, plot=False, mats_dom=["Water", "Iodine"], mats_range=["Water", "Gadolinium"])
    print(igd_to_waio.A)

    sino_sz = AB.shape[1:]

    #AB = AB.permute(1, 2, 0).contiguous().view(-1, 2)  # [P x 2]
    #AR = AR.permute(1, 2, 0).contiguous().view(-1, 2)  # [P x 2]

    AC_pred_nonlinear = calibrated_model._uncalibrated_forward(A=torch.cat([AB, AR], dim=0))[0]
    #AC_pred_nonlinear = AC_pred_nonlinear.view(*sino_sz, 2).permute(2, 0, 1).contiguous() # [2 x H x W]

    # Backproject predicted sinogram to obtain predicted images

    aC_pred_nonlinear = ray_trafo.backward(AC_pred_nonlinear)  # [2 x H x W]
    aC_pred_linear = igd_to_waio.apply_inverse(aR)  # [2 x H x W]

    # Apply the calibrator trained on the calibration scan (see calibrated_model.calibrate_from_id
    # in analyse_endoleak) to the non-linear predictions, in the same sinogram domain it was
    # trained on, then backproject to get the calibrated predicted images.

    aC_pred_cal = None
    if hasattr(calibrated_model, "calibrator"):
        with torch.no_grad():
            AC_pred_cal = calibrated_model.calibrator(AC_pred_nonlinear.permute(1, 2, 0)).permute(2, 0, 1)
        aC_pred_cal = ray_trafo.backward(AC_pred_cal)


    AC_pred_nonlinear = AC_pred_nonlinear.detach().cpu().numpy() if isinstance(AC_pred_nonlinear, torch.Tensor) else AC_pred_nonlinear
    aC_pred_nonlinear = aC_pred_nonlinear.detach().cpu().numpy() if isinstance(aC_pred_nonlinear, torch.Tensor) else aC_pred_nonlinear
    aC_pred_cal = aC_pred_cal.detach().cpu().numpy() if isinstance(aC_pred_cal, torch.Tensor) else aC_pred_cal

    aC_pred_nonlinear[0, :] = aC_pred_nonlinear[0, :] * rhoI * mask_roi
    aC_pred_nonlinear[1, :] = aC_pred_nonlinear[1, :] * rhoGd * mask_roi

    aC_pred_linear[0, :] = aC_pred_linear[0, :] * rhoI * mask_roi
    aC_pred_linear[1, :] = aC_pred_linear[1, :] * rhoGd * mask_roi

    if aC_pred_cal is not None:
        aC_pred_cal[0, :] = aC_pred_cal[0, :] * rhoI * mask_roi
        aC_pred_cal[1, :] = aC_pred_cal[1, :] * rhoGd * mask_roi

    # Convert tensors back to numpy


    aC_pred_linear = aC_pred_linear.detach().cpu().numpy() if isinstance(aC_pred_linear, torch.Tensor) else aC_pred_linear
    # plot predicted sinograms

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    im = ax[0].imshow(AC_pred_nonlinear[0, :], cmap='gray')
    ax[0].set_title("Predicted Iodine Basis Sinogram (Nonlinear)")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    im = ax[1].imshow(AC_pred_nonlinear[1, :], cmap='gray')
    ax[1].set_title("Predicted Gadolinium Basis Sinogram (Nonlinear)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    print(f"Saving predicted sinograms to {PLOT_DIR / f'endoleak_predicted_sinograms_{scan_id}.png'}")
    plt.savefig(PLOT_DIR / f"endoleak_predicted_sinograms_{scan_id}.png", dpi=300)

    # plot


    fig, ax = plt.subplots(3,2, figsize=(10, 10))

    im = ax[0, 0].imshow(aC_pred_nonlinear[0, :], cmap='gray')
    ax[0, 0].set_title("Predicted Iodine Basis Image")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(aC_pred_nonlinear[1, :], cmap='gray')
    ax[0, 1].set_title("Predicted Gadolinium Basis Image")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im = ax[1, 0].imshow(aC_pred_linear[0, :], cmap='gray')
    ax[1, 0].set_title("Predicted Iodine Basis Image (Linear)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(aC_pred_linear[1, :], cmap='gray')
    ax[1, 1].set_title("Predicted Gadolinium Basis Image (Linear)")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im = ax[2, 0].imshow(AC_pred_nonlinear[0, :], cmap='gray')
    ax[2, 0].set_title("Predicted Iodine Basis Sinogram")
    fig.colorbar(im, ax=ax[2, 0], fraction=0.046, pad=0.04)
    im = ax[2, 1].imshow(AC_pred_nonlinear[1, :], cmap='gray')
    ax[2, 1].set_title("Predicted Gadolinium Basis Sinogram")
    fig.colorbar(im, ax=ax[2, 1], fraction=0.046, pad=0.04)
    print(f"Saving predicted images and sinograms to {PLOT_DIR / f'endoleak_predicted_images_{scan_id}.png'}")
    plt.savefig(PLOT_DIR / f"endoleak_predicted_images_{scan_id}.png", dpi=300)




    iodine_img = aC_pred_nonlinear[0, :]
    gd_img = aC_pred_nonlinear[1, :]
    iodine_cal_img = aC_pred_cal[0, :] if aC_pred_cal is not None else None
    gd_cal_img = aC_pred_cal[1, :] if aC_pred_cal is not None else None

    i_gt = {"560": [7.7, 3.4], "561": [7.1, 7]}
    gd_gt = {"560": [5.6, 2.7], "561": [5.6, 0]}
    scan_key = str(scan_id)


    rois = pick_circular_rois(iodine_img, title=f"Scan {scan_id}: click center then edge for each ROI, middle-click to finish")
    return {
        "scan_id": scan_id,
        "iodine": iodine_img,
        "gadolinium": gd_img,
        "iodine_cal": iodine_cal_img,
        "gadolinium_cal": gd_cal_img,
        "rois": rois,
        "iodine_gt": i_gt[scan_key],
        "gadolinium_gt": gd_gt[scan_key],
    }


def _plot_endoleak_grid(scan_results, image_keys, output_path):
    """Plot the 2x2 endoleak ROI grid (rows=scans, cols=Iodine/Gadolinium) for a given pair of
    per-scan image keys (e.g. ("iodine", "gadolinium") or ("iodine_cal", "gadolinium_cal"))."""

    fig = plt.figure(figsize=(12, 13))
    outer_grid = fig.add_gridspec(2, 1, height_ratios=(1, 0.035), hspace=0.12)
    image_grid = outer_grid[0].subgridspec(2, 2, wspace=0, hspace=0)
    colorbar_grid = outer_grid[1].subgridspec(1, 2, wspace=0.2)
    axes = np.array([
        [fig.add_subplot(image_grid[0, 0]), fig.add_subplot(image_grid[0, 1])],
        [fig.add_subplot(image_grid[1, 0]), fig.add_subplot(image_grid[1, 1])],
    ])

    material_specs = [
        (image_keys[0], "iodine_gt", "Iodine"),
        (image_keys[1], "gadolinium_gt", "Gadolinium"),
    ]

    colorbar_images = [None, None]
    for row, result in enumerate(scan_results):
        for col, (material_key, gt_key, _) in enumerate(material_specs):
            axis = axes[row, col]
            im = axis.imshow(result[material_key], cmap="gray", vmin=0, vmax=10, aspect="auto")
            annotate_rois(axis, result[material_key], result["rois"], unit="mg/mL")
            add_gt_overlay(axis, f"Target {material_key}", result[gt_key], unit="mg/mL")
            axis.set_axis_off()

            if row == 1:
                colorbar_images[col] = im

        row_position = 0.5 * (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1)
        fig.text(0.055, row_position, f"{SCAN_TO_LABEL[result['scan_id']]}", rotation=90, fontsize=28, ha="center", va="center", fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.92)

    axes[0, 0].set_title("Iodine", fontsize=28, pad=18, fontweight="bold")
    axes[0, 1].set_title("Gadolinium", fontsize=28, pad=18, fontweight="bold")

    for col, im in enumerate(colorbar_images):
        cbar = fig.colorbar(im, cax=fig.add_subplot(colorbar_grid[0, col]), orientation="horizontal")
        cbar.set_label("mg/mL", fontsize=16)
        cbar.set_ticks([0, 2.5, 5, 7.5, 10])
        cbar.ax.tick_params(labelsize=14)

    print(f"Saving combined endoleak ROI figure to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def analyse_endoleak(scan_ids=(560, 561), slice_idx=251, num_slices=5, cal_id=449):
    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]
    Eactvector = load_energies()  # [nE]

    calibrated_model = CalibratedForwardModel(fB, fR, fC, weights, Eactvector, output_dir=PLOT_DIR, normalize_materials=True)
    print(f"Calibrating endoleak forward model on scan {cal_id}... \n")
    calibrated_model.calibrate_from_id(cal_id=cal_id, calibrator_params={}, include_linear_model=False)

    scan_results = [
        _analyse_endoleak_scan(scan_id, calibrated_model, slice_idx=slice_idx, num_slices=num_slices)
        for scan_id in scan_ids
    ]

    _plot_endoleak_grid(scan_results, ("iodine", "gadolinium"), PLOT_DIR / "endoleak_predicted_rois_grid.png")

    if all(result.get("iodine_cal") is not None and result.get("gadolinium_cal") is not None for result in scan_results):
        _plot_endoleak_grid(
            scan_results, ("iodine_cal", "gadolinium_cal"),
            PLOT_DIR / "endoleak_predicted_rois_grid_calibrated.png",
        )


def _split_endoleak_roi(iodine_basis, mask_roi):
    """Split mask_roi into the bright insert and the surrounding sack by Otsu thresholding the
    iodine basis inside the ROI and keeping the largest bright component (holes filled)."""

    lo, hi = np.percentile(iodine_basis[mask_roi], [0.5, 99.5])
    iodine_u8 = np.clip((iodine_basis - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(iodine_u8[mask_roi].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels_im = cv2.connectedComponents(((iodine_u8 > threshold) & mask_roi).astype(np.uint8))
    largest_component = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])

    mask_insert = binary_fill_holes(labels_im == largest_component) & mask_roi
    mask_sack = mask_roi & (~mask_insert)

    return mask_insert, mask_sack


def _analyse_endoleak_vmi_scan(scan_id, mu_E, slice_idx=251, num_slices=5):
    """Compute two VMIs (HU) for one endoleak scan: one from the primary water/iodine 2MD, and one
    from the background (sack assumed (1, 0) wa/io) plus the ground truth I/Gd concentrations."""

    scan_path = UZB_DATA / SCANS[scan_id]
    raw_scans = {}

    for fname in ["iodine_basis_vol", "water_basis_vol", "mask_roi"]:
        if not (scan_path / f"{fname}.nii").exists():
            raise FileNotFoundError(f"Expected file {fname}.nii not found in {scan_path}")

        print(f"Loading {fname} from {scan_path / f'{fname}.nii'}")
        vol = np.asarray(nib.load(scan_path / f"{fname}.nii").dataobj[slice_idx - num_slices // 2:slice_idx + num_slices // 2], dtype=np.float32)
        raw_scans[fname.replace("_vol", "")] = np.mean(vol, axis=0)

    mask_roi = raw_scans["mask_roi"] > 0.5
    mask_insert, mask_sack = _split_endoleak_roi(raw_scans["iodine_basis"], mask_roi)

    # Background: same as in _analyse_endoleak_scan, the sack is assumed to be (1, 0) wa/io

    WaB = np.where(mask_roi, 1.0, raw_scans["water_basis"])
    IB = np.where(mask_roi, 0.0, raw_scans["iodine_basis"])

    # Ground truth concentrations (mg/mL) given as [insert, sack]

    i_gt = {"560": [7.7, 3.4], "561": [7.1, 7]}
    gd_gt = {"560": [5.6, 2.7], "561": [5.6, 0]}
    scan_key = str(scan_id)

    I_gt = mask_insert * i_gt[scan_key][0] + mask_sack * i_gt[scan_key][1]
    Gd_gt = mask_insert * gd_gt[scan_key][0] + mask_sack * gd_gt[scan_key][1]

    mu_2md = raw_scans["water_basis"] * mu_E["Water"] + raw_scans["iodine_basis"] * mu_E["Iodine"]
    mu_gt = WaB * mu_E["Water"] + IB * mu_E["Iodine"] + I_gt / rhoI * mu_E["Iodine"] + Gd_gt / rhoGd * mu_E["Gadolinium"]

    to_hu = lambda mu: 1000 * (mu - mu_E["Water"]) / mu_E["Water"]
    vmi_2md, vmi_gt = to_hu(mu_2md), to_hu(mu_gt)

    stats = {}
    for region, mask in (("insert", mask_insert), ("sack", mask_sack)):
        stats[region] = {
            "2md": (vmi_2md[mask].mean(), vmi_2md[mask].std()),
            "gt": (vmi_gt[mask].mean(), vmi_gt[mask].std()),
            "diff": ((vmi_2md - vmi_gt)[mask].mean(), (vmi_2md - vmi_gt)[mask].std()),
        }
        print(f"Scan {scan_id} {region:>6}: 2MD VMI {stats[region]['2md'][0]:7.1f} ± {stats[region]['2md'][1]:5.1f} HU | "
              f"GT VMI {stats[region]['gt'][0]:7.1f} ± {stats[region]['gt'][1]:5.1f} HU | "
              f"diff {stats[region]['diff'][0]:7.1f} ± {stats[region]['diff'][1]:5.1f} HU")

    return {
        "scan_id": scan_id,
        "vmi_2md": vmi_2md,
        "vmi_gt": vmi_gt,
        "mask_insert": mask_insert,
        "mask_sack": mask_sack,
        "stats": stats,
    }


def analyse_endoleak_vmi(scan_ids=(560, 561), slice_idx=251, num_slices=5, energy_keV=70, hu_window=(-100, 500), diff_window=(-150, 150)):
    """Compare the VMI at energy_keV obtained from the primary water/iodine 2MD with the VMI obtained
    from the background + ground truth I/Gd concentrations for the endoleak scans."""

    fB, _, fC = load_materials()
    Evec = load_energies().numpy()
    e_idx = int(np.argmin(np.abs(Evec - energy_keV)))
    print(f"Using energy {Evec[e_idx]:.1f} keV (requested {energy_keV} keV)")

    mu_E = {"Water": fB[0, e_idx].item(), "Iodine": fC[0, e_idx].item(), "Gadolinium": fC[1, e_idx].item()}  # [1/mm]

    scan_results = [
        _analyse_endoleak_vmi_scan(scan_id, mu_E, slice_idx=slice_idx, num_slices=num_slices)
        for scan_id in scan_ids
    ]

    column_specs = [
        ("vmi_2md", "2md", "Water/Iodine 2MD", hu_window),
        ("vmi_gt", "gt", "Background + GT", hu_window),
        ("diff", "diff", "2MD − GT", diff_window),
    ]

    fig, axes = plt.subplots(len(scan_results), 3, figsize=(18, 6 * len(scan_results)), squeeze=False)

    for row, result in enumerate(scan_results):

        # Crop around the ROI so the insert and sack are visible

        ys, xs = np.where(result["mask_insert"] | result["mask_sack"])
        pad = 40
        crop = np.s_[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]

        for col, (key, stat_key, title, window) in enumerate(column_specs):
            axis = axes[row, col]
            image = result["vmi_2md"] - result["vmi_gt"] if key == "diff" else result[key]
            cmap = "coolwarm" if key == "diff" else "gray"

            im = axis.imshow(image[crop], cmap=cmap, vmin=window[0], vmax=window[1])
            axis.contour(result["mask_insert"][crop], levels=[0.5], colors="yellow", linewidths=0.8)
            axis.contour(result["mask_sack"][crop] | result["mask_insert"][crop], levels=[0.5], colors="cyan", linewidths=0.8)

            text = "\n".join(
                f"{region}: {result['stats'][region][stat_key][0]:.1f} ± {result['stats'][region][stat_key][1]:.1f} HU"
                for region in ("insert", "sack")
            )
            axis.text(0.02, 0.98, text, transform=axis.transAxes, fontsize=13, va="top", ha="left", color="white",
                      bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="none"))
            axis.set_axis_off()

            cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.02)
            cbar.set_label("HU", fontsize=14)

            if row == 0:
                axis.set_title(title, fontsize=22, pad=14, fontweight="bold")

        axes[row, 0].text(-0.08, 0.5, SCAN_TO_LABEL[result["scan_id"]], transform=axes[row, 0].transAxes, rotation=90,
                          fontsize=26, ha="center", va="center", fontweight="bold")

    fig.suptitle(f"VMI at {Evec[e_idx]:.0f} keV", fontsize=24, fontweight="bold")

    output_path = PLOT_DIR / f"endoleak_vmi_{Evec[e_idx]:.0f}keV.png"
    print(f"Saving endoleak VMI comparison to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return scan_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--endoleak", action="store_true", help="Generate the combined scan 560/561 endoleak figure.")
    parser.add_argument("--vmi", action="store_true", help="Compare 70 keV VMIs from the water/iodine 2MD and from background + GT for scans 560/561.")
    args = parser.parse_args()

    if args.endoleak:
        analyse_endoleak()
        raise SystemExit(0)

    if args.vmi:
        analyse_endoleak_vmi()
        raise SystemExit(0)

    analyse_uzb(scan_id=449)
    assert False
    # analyse_endoleak(scan_id=scan_id, num_slices=10, slice_idx=251)

    #analyse_gammex(gammex_data=gammex_data)

    print(f"ANTROPOMORPHIC_PATH: {ANTROPOMORPHIC_PATH}")

    # analyse_antropomorphic(average_slice=True)


    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]
    Eactvector = load_energies()  # [nE]

    analyse_scans = True
    if analyse_scans:

        scans = sorted([161, 448, 449, 456], reverse=False)  # [161, 448, 449, 456]
        # scans = [560, 561]  # [161, 448, 449, 456]
        model = CalibratedForwardModel(fB, fR, fC, weights, Eactvector, output_dir=PLOT_DIR, normalize_materials=True, debug=False)


        model.calibrate_from_id(cal_id=449, calibrator_params={})  # Initial calibration from scan 161

        for scan_id in scans:
            # analyse_uzb(scan_id=scan_id)
            

            model.eval_from_id(eval_id=scan_id) # No calibration step
            # model.calibrate_from_id(cal_id=scan_id, calibrator_params=None)  # Overwrites the cache entry with uncalibrated-only results
        model.plot_cache(output_dir=PLOT_DIR)



