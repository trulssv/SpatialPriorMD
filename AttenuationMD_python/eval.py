from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import label as nd_label

from cal.cal import ARTIFACT_DIR, _extract_roi_components, _flatten_sinogram_triplet, _validate_roi_erosion, load_trained_weight_net
from config import DENSITY_GD, DENSITY_I, PLOT_DIR, SCANS
from data_processing.utils import load_data, load_mask, load_materials
from operators.md_operators import ParameterizedMDOperator
from operators.tomo import RayTrafoModule


def _default_checkpoint() -> Path:
    checkpoints = sorted(ARTIFACT_DIR.glob("scan_*_weights_model.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ARTIFACT_DIR}")
    return checkpoints[-1]


def _eval_prefix(scan_id: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"eval_scan_{scan_id}"


def _plot_weight_functions(weight_net, output_dir: Path, scan_id: int) -> Path:
    E_keV = np.arange(15, 121, 1)
    with torch.no_grad():
        learned = weight_net(weight_net.get_energy_grid()).detach().cpu().numpy()
    reference = weight_net.w_ref.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor="black")
    fig.suptitle(f"Learned weight functions for evaluation on scan {scan_id}", color="white", fontsize=12)
    for idx, ax in enumerate(axes.flat):
        if idx >= learned.shape[0]:
            ax.set_visible(False)
            continue
        ax.plot(E_keV, reference[idx], "--", color="gray", lw=1.5, label="reference")
        ax.plot(E_keV, learned[idx], color="cyan", lw=2.0, label="trained")
        ax.set_title(f"Bin {idx}", color="white", fontsize=10)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        if idx == 0:
            ax.legend(facecolor="black", edgecolor="gray", labelcolor="white", fontsize=8)
    plt.tight_layout()
    fig_path = _eval_prefix(scan_id, output_dir).with_name(f"eval_scan_{scan_id}_weight_functions.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return fig_path


def _plot_ac_images(
    I_pred_mgml: np.ndarray,
    Gd_pred_mgml: np.ndarray,
    I_gt_mgml: np.ndarray,
    Gd_gt_mgml: np.ndarray,
    output_dir: Path,
    scan_id: int,
) -> Path:
    I_err = np.abs(I_pred_mgml - I_gt_mgml)
    Gd_err = np.abs(Gd_pred_mgml - Gd_gt_mgml)
    panels = [
        (0, 0, I_pred_mgml, "hot", 0, max(1e-6, np.percentile(I_gt_mgml, 99)), "Predicted iodine [mg/mL]"),
        (0, 1, I_gt_mgml, "hot", 0, max(1e-6, np.percentile(I_gt_mgml, 99)), "GT iodine [mg/mL]"),
        (0, 2, I_err, "hot", 0, max(1e-6, np.percentile(I_err, 99)), "|Error| iodine [mg/mL]"),
        (1, 0, Gd_pred_mgml, "cool", 0, max(1e-6, np.percentile(Gd_gt_mgml, 99)), "Predicted gadolinium [mg/mL]"),
        (1, 1, Gd_gt_mgml, "cool", 0, max(1e-6, np.percentile(Gd_gt_mgml, 99)), "GT gadolinium [mg/mL]"),
        (1, 2, Gd_err, "cool", 0, max(1e-6, np.percentile(Gd_err, 99)), "|Error| gadolinium [mg/mL]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="black")
    for row, col, data, cmap, vmin, vmax, title in panels:
        ax = axes[row, col]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    fig.suptitle(f"Predicted vs GT aC images for scan {scan_id}", color="white", fontsize=12)
    plt.tight_layout()
    fig_path = _eval_prefix(scan_id, output_dir).with_name(f"eval_scan_{scan_id}_ac_images.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return fig_path


def _plot_roi_scatter(
    I_pred_mgml: np.ndarray,
    Gd_pred_mgml: np.ndarray,
    I_gt_mgml: np.ndarray,
    Gd_gt_mgml: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    scan_id: int,
    erosion: float = 1.0,
    I_pred_uncal_mgml: np.ndarray | None = None,
    Gd_pred_uncal_mgml: np.ndarray | None = None,
) -> Path:
    components = _extract_roi_components(mask, erosion=_validate_roi_erosion(erosion))

    def roi_means(img: np.ndarray) -> np.ndarray:
        return np.array([float(img[comp].mean()) for comp in components], dtype=np.float32)

    I_pred_means = roi_means(I_pred_mgml)
    I_gt_means = roi_means(I_gt_mgml)
    Gd_pred_means = roi_means(Gd_pred_mgml)
    Gd_gt_means = roi_means(Gd_gt_mgml)
    I_pred_uncal_means = roi_means(I_pred_uncal_mgml) if I_pred_uncal_mgml is not None else None
    Gd_pred_uncal_means = roi_means(Gd_pred_uncal_mgml) if Gd_pred_uncal_mgml is not None else None

    def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        ss_res = float(np.sum((pred - gt) ** 2))
        ss_tot = float(np.sum((gt - gt.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-16)
        return rmse, r2

    I_rmse, I_r2 = metrics(I_pred_means, I_gt_means)
    Gd_rmse, Gd_r2 = metrics(Gd_pred_means, Gd_gt_means)
    if I_pred_uncal_means is not None:
        I_uncal_rmse, I_uncal_r2 = metrics(I_pred_uncal_means, I_gt_means)
    else:
        I_uncal_rmse, I_uncal_r2 = None, None
    if Gd_pred_uncal_means is not None:
        Gd_uncal_rmse, Gd_uncal_r2 = metrics(Gd_pred_uncal_means, Gd_gt_means)
    else:
        Gd_uncal_rmse, Gd_uncal_r2 = None, None

    fig, (ax_i, ax_gd) = plt.subplots(1, 2, figsize=(12, 5), facecolor="black")
    panels = [
        (ax_i, I_gt_means, I_pred_means, I_pred_uncal_means, "Iodine [mg/mL]", "#ff6b35", I_rmse, I_r2, I_uncal_rmse, I_uncal_r2),
        (ax_gd, Gd_gt_means, Gd_pred_means, Gd_pred_uncal_means, "Gadolinium [mg/mL]", "#00bfff", Gd_rmse, Gd_r2, Gd_uncal_rmse, Gd_uncal_r2),
    ]
    for ax, gt, pred, pred_uncal, title, colour, rmse, r2, uncal_rmse, uncal_r2 in panels:
        vmax = max(float(np.max(gt)) if gt.size else 0.0, float(np.max(pred)) if pred.size else 0.0, 1e-6)
        ax.set_facecolor("#111111")
        ax.plot([0, vmax], [0, vmax], "--", color="gray", lw=1.2)
        ax.scatter(gt, pred, c=colour, s=70, edgecolors="white", linewidths=0.5, marker="o", label="calibrated")
        if pred_uncal is not None:
            ax.scatter(gt, pred_uncal, c=colour, s=75, edgecolors="white", linewidths=0.5, marker="^", label="uncalibrated")
        for idx, (x, y) in enumerate(zip(gt, pred), start=1):
            ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color="white")
        title_lines = [f"{title}", f"Cal: RMSE={rmse:.3f} mg/mL  R²={r2:.3f}"]
        if pred_uncal is not None and uncal_rmse is not None and uncal_r2 is not None:
            title_lines.append(f"Uncal: RMSE={uncal_rmse:.3f} mg/mL  R²={uncal_r2:.3f}")
        ax.set_title("\n".join(title_lines), color="white", fontsize=10)
        ax.set_xlabel("GT mean [mg/mL]", color="white", fontsize=10)
        ax.set_ylabel("Predicted mean [mg/mL]", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.legend(facecolor="#111111", edgecolor="white", labelcolor="white", fontsize=8, loc="best")
    fig.suptitle(f"Per-ROI predicted vs GT aC means for scan {scan_id} (n={len(components)}, erosion={erosion:.2f})", color="white", fontsize=11)
    plt.tight_layout()
    fig_path = _eval_prefix(scan_id, output_dir).with_name(f"eval_scan_{scan_id}_roi_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return fig_path


def evaluate(checkpoint_path: str | Path, scan_id: int, output_dir: Path = PLOT_DIR, erosion: float = 1.0) -> dict[str, Path]:
    checkpoint_path = Path(checkpoint_path)
    weight_net = load_trained_weight_net(checkpoint_path)
    aB, aR, aC_gt_img = load_data(scan_id)
    mask = load_mask(scan_id).cpu().numpy()
    fB, fR, fC = load_materials()
    ray_trafo = RayTrafoModule()
    md_operator = ParameterizedMDOperator(weight_net, fB, fR, fC)

    AB = ray_trafo(aB)
    AR = ray_trafo(aR)
    AC_gt = ray_trafo(aC_gt_img)
    A = _flatten_sinogram_triplet(AB, AR)

    sol = md_operator(A, compute_grad=False)
    AC_pred = sol["AC"].detach().permute(2, 0, 1)
    aC_gt = ray_trafo.backward(AC_gt).detach()
    aC_pred = ray_trafo.backward(AC_pred).detach()

    I_pred_mgml = (aC_pred[0] * DENSITY_I).cpu().numpy()
    Gd_pred_mgml = (aC_pred[1] * DENSITY_GD).cpu().numpy()
    I_gt_mgml = (aC_gt[0] * DENSITY_I).cpu().numpy()
    Gd_gt_mgml = (aC_gt[1] * DENSITY_GD).cpu().numpy()
    I_gt_ref_mgml = (aC_gt_img[0] * DENSITY_I).cpu().numpy()
    Gd_gt_ref_mgml = (aC_gt_img[1] * DENSITY_GD).cpu().numpy()

    weight_fig = _plot_weight_functions(weight_net, output_dir, scan_id)
    ac_fig = _plot_ac_images(I_pred_mgml, Gd_pred_mgml, I_gt_mgml, Gd_gt_mgml, output_dir, scan_id)
    scatter_fig = _plot_roi_scatter(
        I_pred_mgml,
        Gd_pred_mgml,
        I_gt_mgml,
        Gd_gt_mgml,
        mask,
        output_dir,
        scan_id,
        erosion=erosion,
    )

    npz_path = _eval_prefix(scan_id, output_dir).with_name(f"eval_scan_{scan_id}_results.npz")
    np.savez(
        npz_path,
        I_pred_mgml=I_pred_mgml,
        Gd_pred_mgml=Gd_pred_mgml,
        I_gt_mgml=I_gt_mgml,
        Gd_gt_mgml=Gd_gt_mgml,
        I_gt_ref_mgml=I_gt_ref_mgml,
        Gd_gt_ref_mgml=Gd_gt_ref_mgml,
        mask=mask.astype(np.uint8),
    )

    return {
        "weights_plot": weight_fig,
        "ac_plot": ac_fig,
        "roi_scatter": scatter_fig,
        "results": npz_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained AttenuationMD_python calibration model")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a saved *_weights_model.pt checkpoint")
    p.add_argument("--scan_id", type=int, default=449, choices=sorted(SCANS.keys()))
    p.add_argument("--erosion", type=float, default=1.0, help="ROI radius fraction in (0, 1] used for ROI scatter plots")
    p.add_argument("--output_dir", type=str, default=str(PLOT_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else _default_checkpoint()
    outputs = evaluate(checkpoint, args.scan_id, output_dir=Path(args.output_dir), erosion=args.erosion)
    for name, path in outputs.items():
        print(f"Saved {name} to {path}")
