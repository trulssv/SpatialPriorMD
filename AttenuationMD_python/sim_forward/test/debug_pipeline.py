"""End-to-end diagnostics for the toy PCCT forward-model pipeline.

Run from AttenuationMD_python:
    python -m sim_forward.test.debug_pipeline

The suite writes summary.json and PNG panels under sim_forward/test/artifacts.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from operators.tomo import SliceWiseRayTrafoModule
from sim_forward.forward_model import PolyChromaticForwardModel


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "artifacts"
TEST_NAMES = ("geometry", "spectrum", "material_decomposition", "reconstruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose geometry, spectra, 2MD, and reconstruction alignment.")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=TEST_NAMES,
        default=TEST_NAMES,
        help="Diagnostics to run (default: all).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-rays", type=int, default=128, help="Rays for the 2MD round-trip test.")
    return parser.parse_args()


def save_panel(output_path: Path, panels: list[tuple[np.ndarray, str, str]]) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for axis, (image, title, cmap) in zip(axes[0], panels):
        finite = image[np.isfinite(image)]
        vmin, vmax = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)
        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
        plot = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / max(np.linalg.norm(reference), 1e-12))


def geometry_test(output_dir: Path) -> dict[str, Any]:
    """Reconstruct a known disk with the production slice-wise transform."""
    shape = (96, 96)
    transform = SliceWiseRayTrafoModule(
        physical_size=(96.0, 96.0),
        shape=shape,
        n_angles=180,
        n_detectors=144,
    )
    row, column = np.indices(shape)
    disk = (((row - 48) / 20) ** 2 + ((column - 48) / 12) ** 2 <= 1).astype(np.float32)
    reference = np.repeat(disk[None, None], 2, axis=1)
    volume = torch.from_numpy(reference)
    sinogram = transform.forward(volume)
    reconstruction = transform.backward(sinogram).detach().cpu().numpy()

    scale = float(np.vdot(reference, reconstruction) / max(np.vdot(reconstruction, reconstruction), 1e-12))
    reconstruction *= scale
    nrmse = relative_l2(reference, reconstruction)
    peak = np.unravel_index(np.argmax(reconstruction[0, 0]), reconstruction[0, 0].shape)
    save_panel(
        output_dir / "geometry_disk_roundtrip.png",
        [(reference[0, 0], "Input disk", "gray"), (sinogram[0, :, 0].detach().cpu().numpy(), "Sinogram", "magma"), (reconstruction[0, 0], "FBP reconstruction", "gray")],
    )
    return {
        "nrmse": nrmse,
        "peak_index": [int(peak[0]), int(peak[1])],
        "pass": bool(nrmse < 0.3),
        "interpretation": "A failure here means the whirl is geometric, before spectral binning or 2MD.",
    }


def prepare_model() -> tuple[PolyChromaticForwardModel, np.ndarray, np.ndarray, np.ndarray]:
    model = PolyChromaticForwardModel(batch_size=2048)
    maps, attenuation, weights = model.prep()
    return model, maps, attenuation, weights


def spectrum_test(model: PolyChromaticForwardModel, maps: np.ndarray, attenuation: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Check count dimensions and physically required open-beam bounds."""
    counts = model.pcct_forward_project(maps, attenuation, weights)
    open_beam = weights.sum(axis=1)
    pass_bounds = bool(np.all(np.isfinite(counts)) and np.all(counts >= 0) and np.all(counts <= open_beam[:, None, None, None] * (1 + 1e-5)))
    return {
        "shape": list(counts.shape),
        "count_min": float(counts.min()),
        "count_max": float(counts.max()),
        "open_beam_min": float(open_beam.min()),
        "open_beam_max": float(open_beam.max()),
        "pass": pass_bounds,
        "interpretation": "A failure means the energy-bin sinograms are not physically valid.",
    }


def material_decomposition_test(
    model: PolyChromaticForwardModel,
    maps: np.ndarray,
    attenuation: np.ndarray,
    weights: np.ndarray,
    n_rays: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate noiseless counts from real projected rays and recover their path lengths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = model.ray_trafo.forward(torch.from_numpy(maps)).detach().cpu().numpy()
    flat_paths = paths.transpose(1, 2, 3, 0).reshape(-1, paths.shape[0])
    nonzero = np.flatnonzero(flat_paths[:, 0] > 1e-3)
    if nonzero.size == 0:
        raise RuntimeError("No attenuating rays were found for the 2MD diagnostic.")
    sample_indices = nonzero[np.linspace(0, nonzero.size - 1, min(n_rays, nonzero.size), dtype=int)]
    truth = torch.from_numpy(flat_paths[sample_indices]).float()

    solver = model._get_newton_solver()
    counts = torch.einsum("be,pe->pb", solver.weights, torch.exp(-(truth @ solver.fC)))
    estimate = solver.solve_from_counts(counts, torch.zeros_like(truth), max_iter=50, gtol=1e-5)["AC"]
    error = (estimate - truth).detach().cpu().numpy()
    truth_np = truth.detach().cpu().numpy()
    relative_error = np.abs(error) / np.maximum(np.abs(truth_np), 1e-4)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for material_index, material_name in enumerate(model.md_mats):
        axes[material_index].scatter(truth_np[:, material_index], estimate[:, material_index].detach().cpu().numpy(), s=8)
        limits = [float(truth_np[:, material_index].min()), float(truth_np[:, material_index].max())]
        axes[material_index].plot(limits, limits, "k--")
        axes[material_index].set_title(f"{material_name}: recovered vs. true")
        axes[material_index].set_xlabel("True line integral")
        axes[material_index].set_ylabel("Recovered line integral")
    figure.tight_layout()
    figure.savefig(output_dir / "material_decomposition_roundtrip.png", dpi=160)
    plt.close(figure)

    return {
        "rays_tested": int(truth.shape[0]),
        "max_absolute_error": [float(value) for value in np.abs(error).max(axis=0)],
        "median_relative_error": [float(value) for value in np.median(relative_error, axis=0)],
        "pass": bool(np.all(np.median(relative_error, axis=0) < 2e-3)),
        "interpretation": "A failure means the spectral 2MD solver disagrees with the forward spectrum even before backprojection.",
    }


def best_in_plane_alignment(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, str, float]:
    candidates: list[tuple[np.ndarray, str]] = []
    for rotations, flip_rows, flip_columns in product(range(4), (False, True), (False, True)):
        candidate = np.rot90(estimate, rotations, axes=(-2, -1))
        if flip_rows:
            candidate = candidate[..., ::-1, :]
        if flip_columns:
            candidate = candidate[..., :, ::-1]
        candidates.append((candidate, f"rot{rotations * 90}_flip_rows={flip_rows}_flip_columns={flip_columns}"))

    best_candidate, best_name = min(candidates, key=lambda item: relative_l2(reference, item[0]))
    return best_candidate, best_name, relative_l2(reference, best_candidate)


def reconstruction_test(output_dir: Path) -> dict[str, Any]:
    """Compare saved maps with their inputs and diagnose simple orientation mismatches."""
    case_dir = Path("/mnt/data/MDdata/Antropomorphic/case_00000")
    reconstruction_dir = Path(__file__).resolve().parents[2] / "plots" / case_dir.name
    result: dict[str, Any] = {"reconstruction_dir": str(reconstruction_dir), "maps": {}}

    for input_name, reconstructed_name, label in (
        ("water.nii.gz", "water_recon.nii.gz", "water"),
        ("i.nii.gz", "i_recon.nii.gz", "iodine"),
    ):
        reconstructed_path = reconstruction_dir / reconstructed_name
        if not reconstructed_path.is_file():
            result["maps"][label] = {"status": "missing", "path": str(reconstructed_path)}
            continue

        reference = nib.load(case_dir / input_name).get_fdata(dtype=np.float32)
        reconstruction = nib.load(reconstructed_path).get_fdata(dtype=np.float32)
        if reference.shape[1:] == reconstruction.shape[1:] and reference.shape[0] > reconstruction.shape[0]:
            start = reference.shape[0] // 2 - reconstruction.shape[0] // 2
            reference = reference[start : start + reconstruction.shape[0]]
        if reference.shape != reconstruction.shape:
            result["maps"][label] = {"status": "shape_mismatch", "reference_shape": list(reference.shape), "reconstruction_shape": list(reconstruction.shape)}
            continue

        aligned, transform_name, alignment_nrmse = best_in_plane_alignment(reference, reconstruction)
        raw_nrmse = relative_l2(reference, reconstruction)
        middle = reference.shape[0] // 2
        save_panel(
            output_dir / f"reconstruction_{label}_comparison.png",
            [(reference[middle], f"Input {label}", "gray"), (reconstruction[middle], f"Saved {label}", "gray"), (aligned[middle], f"Best aligned: {transform_name}", "gray")],
        )
        result["maps"][label] = {
            "status": "compared",
            "raw_nrmse": raw_nrmse,
            "best_alignment": transform_name,
            "best_alignment_nrmse": alignment_nrmse,
        }

    result["pass"] = all(item.get("status") == "compared" for item in result["maps"].values())
    result["interpretation"] = "A large improvement after a rotation/flip points to an orientation bug; poor alignment under every transform points to FBP geometry/scaling."
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    model: PolyChromaticForwardModel | None = None
    maps = attenuation = weights = None

    if "geometry" in args.tests:
        summary["geometry"] = geometry_test(args.output_dir)
    if any(name in args.tests for name in ("spectrum", "material_decomposition")):
        model, maps, attenuation, weights = prepare_model()
    if "spectrum" in args.tests:
        assert model is not None and maps is not None and attenuation is not None and weights is not None
        summary["spectrum"] = spectrum_test(model, maps, attenuation, weights)
    if "material_decomposition" in args.tests:
        assert model is not None and maps is not None and attenuation is not None and weights is not None
        summary["material_decomposition"] = material_decomposition_test(
            model, maps, attenuation, weights, args.n_rays, args.output_dir
        )
    if "reconstruction" in args.tests:
        summary["reconstruction"] = reconstruction_test(args.output_dir)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
