from pathlib import Path

# ── resolve paths relative to this file ──────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = Path("~/Documents/AttenuationMD").expanduser()  # Adjust if your repo is located elsewhere
PLOT_DIR    = REPO_ROOT / "AttenuationMD_python" / "plots"

FILES_DIR   = REPO_ROOT / "files"
UZB_DATA    = REPO_ROOT / "UZB" / "data" / "mega_mono"
MODEL_DIR   = REPO_ROOT / "python" / "model"
DEBUG_DIR   = REPO_ROOT / "debug"

EVAL_SCAN = "CT99.456.2.1.1765364800.382548.302.12277"
CAL_SCAN  = "CT99.449.2.2.1765289969.251640.303.12253"
# EVAL_SCAN = "CT99.448.2.1.1765289335.300826.302"


MODEL = "weights_model.pt"
# Material densities [mg/cm³ = mg/mL]
DENSITY_I  = 4933.0
DENSITY_GD = 7870.0
DENSITY_WA = 1000.0
PIXELSTOMM = 200.0 / 512        # FOV 200 mm over 512 pixels (≈ 0.3906 mm/pixel)

SCANS = {
        161: "large_qrm",
        162: "small_qrm",
        448: "CT99.448.2.1.1765289335.300826.302",
        # 449_2: "CT99.449.2.2.1765289969.251640.303.12253", 
        449: "CT99.449.2.1.1765289969.251640.302.12252",
        456: "CT99.456.2.1.1765364800.382548.302.12277", 
        457: "CT99.457.2.1.1765364800.382548.302.12278",  
        560: "CT99.560.2.1.1771925082.978325.303.13299",
        561: "CT99.561.2.1.1771925824.677157.303.13438"
        }


TOMO = {
        161:
        {"physical_size": (350, 350), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', "src_radius": 40, "det_radius": 40},
        162: 
        {"physical_size": (220, 220), "shape": (875, 875), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', "src_radius": 40, "det_radius": 40},
        448:
        {"physical_size": (230, 230), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', "src_radius": 40, "det_radius": 40},
        449:
        {"physical_size": (230, 230), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', "src_radius": 40, "det_radius": 40},
        #449_2:
        #{"physical_size": (230, 230), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', "src_radius": 40, "det_radius": 40},
        456:
        {"physical_size": (230, 230), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', "src_radius": 40, "det_radius": 40},
        560:
        {"physical_size": (200, 200), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', # Actually helical, but we use parallel for simplicity 
                                                "src_radius": float('nan') , "det_radius": float('nan')},
        561:
        {"physical_size": (200, 200), "shape": (512, 512), "n_angles": 512, "n_detectors": 512, "geometry": 'parallel', # Actually helical, but we use parallel for simplicity
                                        "src_radius": float('nan'), "det_radius": float('nan')},
        }

SCAN = {
        448: 
        {"mA": 295, "kVp": 120},
        449: 
        {"mA": 295, "kVp": 120},
        456: 
        {"mA": 295, "kVp": 120},
}

# Scans whose maps are rotated 90 degrees clockwise on load, so the patient table ends up at the bottom
ROTATE_CW_SCANS = {161, 448, 449, 456}

SCAN_TO_LABEL = {
        161: "(A)",
        448: "(B)",
        449: "(C)",
        456: "(D)",
        560: "(E)",
        561: "(F)",
}