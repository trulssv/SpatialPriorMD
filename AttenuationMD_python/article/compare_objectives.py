from cal.cal import CalibratedForwardModel
from config import *
from data_processing.utils import load_materials, load_weights, load_energies

import numpy as np

SCANS = (161, 162, 448, 449, 456)
OBJECTIVES = ("L1", "L2", "Lasso", "KL", "KLsym")
SCAN_LABELS = {scan_id: str(scan_id) for scan_id in SCANS}


# These values predate the density-scaled map metric below. They are retained
# only to reproduce the previous plot; replace them with fresh output from
# compare_objectives() for the mg/mL result grid.
results = {'L1': {161: 1.8069757586964645, 162: 2.695992306281277, 448: 2.2558405717750656, 4491: 2.657702582336294, 456: 1.5363082560932473}, 
 'L2': {161: 0.31161140912814256, 162: 0.7814403997387617, 448: 0.8281793082208131, 4491: 0.4284440470498772, 456: 0.5375230300422118}, 
 'Lasso': {161: 1.7794920692131464, 162: 2.722290045362093, 448: 1.715688620528309, 4491: 2.671614743357043, 456: 1.474998437849801}, 
 'KL': {161: 0.3865026563403849, 162: 0.5587234976568782, 448: 0.38676092843803617, 4491: 0.3461736819445112, 456: 0.4562170690764089}, 
 'KLsym': {161: 0.3852266517320768, 162: 0.5592950593164969, 448: 0.39068620907852325, 4491: 0.3445884381982901, 456: 0.45637432115530446}}

def density_scaled_map_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return joint iodine/gadolinium map RMSE in mg/mL."""
    densities = np.array([DENSITY_I, DENSITY_GD], dtype=np.float64).reshape(-1, 1, 1)
    pred_mg_ml = np.asarray(pred, dtype=np.float64) * densities
    gt_mg_ml = np.asarray(gt, dtype=np.float64) * densities
    return float(np.sqrt(np.mean((pred_mg_ml - gt_mg_ml) ** 2)))

def compare_objectives():
    fB, fR, fC = load_materials()
    weights = load_weights()        # [nBins x nE]
    Eactvector = load_energies()  # [nE]

    
    results = {}
    for obj in OBJECTIVES:

        results[obj] = {}
        for scan in SCANS:
            model = CalibratedForwardModel(fB, fR, fC, weights, Eactvector, output_dir=PLOT_DIR, normalize_materials=True, debug=False, objective=obj)

            print(f"Evaluating {obj} objective for scan {scan}...")
            model.calibrate_from_id(scan, calibrator_params=None) # Perform no calibration, just to set up the model for evaluation

            pred = model.cache[scan]["aC_pred_uncal"]
            gt = model.cache[scan]["aC_gt"]
            results[obj][scan] = density_scaled_map_rmse(pred, gt)
    return results


def visualize_results(results, metric_label="RMSE [mg/mL]"):
    import matplotlib.pyplot as plt

    scan_ids = list(SCANS)
    errors = np.array(
        [[results[objective][scan_id] for objective in OBJECTIVES] for scan_id in scan_ids],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0), layout="constrained")
    image = ax.imshow(errors, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(OBJECTIVES)), labels=OBJECTIVES)
    ax.set_yticks(np.arange(len(scan_ids)), labels=[SCAN_LABELS[scan_id] for scan_id in scan_ids])
    ax.set_xlabel("Objective")
    ax.set_ylabel("Scan ID")
    ax.set_title("Uncalibrated Reconstruction Error")
    ax.tick_params(axis="both", length=0, pad=7)

    threshold = errors.min() + 0.55 * (errors.max() - errors.min())
    for row, scan_id in enumerate(scan_ids):
        for column, objective in enumerate(OBJECTIVES):
            error = errors[row, column]
            text_color = "white" if error > threshold else "#1f2933"
            ax.text(column, row, f"{error:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label(metric_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(OBJECTIVES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scan_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    output_path = PLOT_DIR / "objective_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    results = compare_objectives()
    print(results)
    visualize_results(results, metric_label="RMSE [mg/mL]")

if __name__ == "__main__":
    main()


        