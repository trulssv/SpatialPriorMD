"""Plot the middle slices of the initial anthropomorphic material maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


DEFAULT_DATA_DIR = Path("/mnt/data/MDdata/Antropomorphic/case_00000")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots"
MATERIALS = ("water", "ca", "i", "gd")
MATERIAL_LABELS = ("Water", "Calcium", "Iodine", "Gadolinium")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot middle slices of the initial water, calcium, iodine, and gadolinium maps."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Case directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <project>/plots/<case>_initial_maps.png)",
    )
    return parser.parse_args()


def load_volume(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Initial material map not found: {path}")

    volume = nib.load(path).get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume in {path}, got shape {volume.shape}")
    return volume


def robust_limits(volume: np.ndarray) -> tuple[float, float]:
    finite_values = volume[np.isfinite(volume)]
    if finite_values.size == 0:
        raise ValueError("Volume contains no finite values.")

    vmin, vmax = np.min(finite_values), np.max(finite_values)
    return float(vmin), float(vmax)


def plot_initial_maps(data_dir: Path, output_path: Path) -> None:
    volumes = [load_volume(data_dir / f"{material}.nii.gz") for material in MATERIALS]
    reference_shape = volumes[0].shape
    if any(volume.shape != reference_shape for volume in volumes[1:]):
        shapes = ", ".join(str(volume.shape) for volume in volumes)
        raise ValueError(f"Initial material maps must have matching shapes, got {shapes}.")

    middle_slice = reference_shape[0] // 2
    limits = [robust_limits(volume) for volume in volumes]
    figure, axes = plt.subplots(1, len(MATERIALS), figsize=(16, 4.5), layout="constrained")

    for axis, volume, label, (vmin, vmax) in zip(axes, volumes, MATERIAL_LABELS, limits):
        image = axis.imshow(volume[middle_slice], cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(label)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.9, label=f"{label} value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved initial-material visualization to {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.data_dir.name}_initial_maps.png"
    plot_initial_maps(args.data_dir, output_path)


if __name__ == "__main__":
    main()