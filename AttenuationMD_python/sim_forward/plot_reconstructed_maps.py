"""Plot reconstructed water and iodine maps from an anthropomorphic case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DEFAULT_DATA_DIR = Path.home() / "Documents/AttenuationMD/AttenuationMD_python/data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "article" / "figs" / "plots"
# DEFAULT_DATA_DIR = DEFAULT_OUTPUT_DIR / "case_00000"

RECON_MATS = ("water", "i")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot evenly spaced axial slices of reconstructed water and iodine maps."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Case directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "n_slices",
        nargs="?",
        type=int,
        default=1,
        help="Number of evenly spaced axial slices to plot (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <project>/plots/<case>_reconstructed_maps.png)",
    )
    return parser.parse_args()


def load_volume(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Reconstructed map not found: {path}")

    volume = nib.load(path).get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume in {path}, got shape {volume.shape}")
    return volume


def robust_limits(volume: np.ndarray) -> tuple[float, float]:
    finite_values = volume[np.isfinite(volume)]
    if finite_values.size == 0:
        raise ValueError("Volume contains no finite values.")

    vmin, vmax = np.percentile(finite_values, (1, 99))
    if vmin == vmax:
        padding = max(abs(vmin) * 0.01, 1e-6)
        return float(vmin - padding), float(vmax + padding)
    return float(vmin), float(vmax)


def plot_maps(data_dir: Path, n_slices: int, output_path: Path) -> None:
    if n_slices < 1:
        raise ValueError("n_slices must be at least 1.")



    recons = [load_volume(data_dir / f"{mat}_recon_contrast.nii.gz") for mat in RECON_MATS]

    print(recons[0].shape, recons[1].shape)




    if recons[0].shape != recons[1].shape:
        raise ValueError(
            f"Recon volumes must have the same shape, got {recons[0].shape} and {recons[1].shape}."
        )
    if n_slices > recons[0].shape[0]:
        raise ValueError(f"n_slices ({n_slices}) exceeds the volume depth ({recons[0].shape[0]}).")

    slice_indices = np.linspace(0, recons[0].shape[0] - 1, n_slices, dtype=int)

    recon_limits = [robust_limits(recon) for recon in recons]
    M = len(RECON_MATS)
    print(f"Plotting {n_slices} slices for {M} reconstructed maps with limits: {recon_limits}")
    figure, axes = plt.subplots(
        n_slices,
        M,
        figsize=(8, max(3.5, 3 * n_slices)),
        squeeze=False,
        layout="constrained",
    )


    for row, slice_index in enumerate(slice_indices):


        for m, mat in enumerate(recons):

            im = axes[row, m].imshow(
                mat[slice_index], cmap="gray", vmin=recon_limits[m][0], vmax=recon_limits[m][1]
            )
            axes[row, m].set_axis_off()
            if m == 0:
                axes[row, 0].set_ylabel(f"z = {slice_index}", rotation=0, labelpad=35, va="center")

    axes[0, 0].set_title("Reconstructed water")
    axes[0, 1].set_title("Reconstructed iodine")

    for m, mat in enumerate(RECON_MATS):
        axes[-1, m].set_xlabel(f"{RECON_MATS[m]} value", labelpad=10)

        figure.colorbar(axes[0, m].images[0], ax=axes[:, m], shrink=0.98, label=f"{RECON_MATS[m].capitalize()} value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved visualization to {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.data_dir.name}_reconstructed_maps.png"
    plot_maps(args.data_dir, args.n_slices, output_path)


if __name__ == "__main__":
    main()