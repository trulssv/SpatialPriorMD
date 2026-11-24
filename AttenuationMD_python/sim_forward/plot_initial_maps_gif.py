"""Animate initial anthropomorphic material maps across all axial slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from sim_forward.forward_model import MATERIAL_SCALES
from sim_forward.plot_initial_maps import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    MATERIAL_LABELS,
    MATERIALS,
    load_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GIF that steps through initial material-map slices."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Case directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames (slices) per second (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path (default: <project>/plots/<case>_initial_maps.gif)",
    )
    return parser.parse_args()


def concentration_in_mg_per_ml(volume: np.ndarray, material: str) -> np.ndarray:
    factors = MATERIAL_SCALES[material]
    return volume  * factors["scale"]


def fixed_limits(volume: np.ndarray) -> tuple[float, float]:
    finite_values = volume[np.isfinite(volume)]
    if finite_values.size == 0:
        raise ValueError("Volume contains no finite values.")

    vmin, vmax = float(finite_values.min()), float(finite_values.max())
    if vmin == vmax:
        padding = max(abs(vmin) * 0.01, 1e-6)
        return vmin - padding, vmax + padding
    return vmin, vmax


def create_initial_maps_gif(data_dir: Path, output_path: Path, fps: float = 10.0) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")

    volumes = [
        concentration_in_mg_per_ml(load_volume(data_dir / f"{material}.nii.gz"), material)
        for material in MATERIALS
    ]
    reference_shape = volumes[0].shape
    if any(volume.shape != reference_shape for volume in volumes[1:]):
        shapes = ", ".join(str(volume.shape) for volume in volumes)
        raise ValueError(f"Initial material maps must have matching shapes, got {shapes}.")

    limits = [fixed_limits(volume) for volume in volumes]
    figure, axes = plt.subplots(1, len(MATERIALS), figsize=(16, 4.5), layout="constrained")
    images = []
    for axis, volume, label, (vmin, vmax) in zip(axes, volumes, MATERIAL_LABELS, limits):
        image = axis.imshow(volume[0], cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(label)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.9, label=f"{label} (mg/mL)")
        images.append(image)

    slice_label = figure.suptitle("Slice 0")

    def update(slice_index: int) -> tuple[object, ...]:
        for image, volume in zip(images, volumes):
            image.set_data(volume[slice_index])
        slice_label.set_text(f"Slice {slice_index}")
        return (*images, slice_label)

    animation = FuncAnimation(
        figure,
        update,
        frames=reference_shape[0],
        interval=1000 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=180)
    plt.close(figure)
    print(f"Saved initial-material GIF to {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.data_dir.name}_initial_maps.gif"
    create_initial_maps_gif(args.data_dir, output_path, args.fps)


if __name__ == "__main__":
    main()