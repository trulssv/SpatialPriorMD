"""Create a small, scaled-back pipeline on the Gammex phantom and assemble a PPTX

This script performs a lightweight version of the 5 pipeline steps described
in `article/non_linear_md.py` and writes PNGs and a PPTX to `artifacts/`.

Steps produced (approximate / illustrative):
1. Initial Water/Iodine 2MD
2. Segmented background + signal (simple central ROI subtraction)
3. Forward projections (simple Radon approximation using OpenCV rotations)
4. "Optimization" step: combine sinograms to produce contrast sinogram (simple sum)
5. Backprojection (simple unfiltered backprojection by smearing columns back)

This is intentionally lightweight and avoids heavy dependencies (ODL, pretrained models).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pptx import Presentation
from pptx.util import Inches
import scipy.io as sio
from config import REPO_ROOT
import os

# Use the same PEPVC->WAIO matrix used in the repo (avoid importing torch-heavy modules)
PEPVC_TO_WAIO = np.array([[1.2465975, -0.4603571], [-0.00212777, 0.02189077]])


def load_dir(path: Path) -> dict:
    """Lightweight loader for the Gammex .mat files in `path`.
    Returns a dict of numpy arrays keyed by filename (without extension).
    """
    data = {}
    for fname in os.listdir(path):
        if fname.endswith('.mat'):
            key = fname[:-4]
            mat = sio.loadmat(path / fname)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if len(keys) >= 1:
                arr = mat[keys[0]]
                data[key] = np.asarray(arr, dtype=np.float32)
    return data


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def convert_pepvc_to_waio_numpy(a_pepvc: np.ndarray) -> np.ndarray:
    # a_pepvc is expected [2, H, W]; apply change-of-basis along the material axis
    return np.einsum('ij,j...->i...', PEPVC_TO_WAIO, a_pepvc)


def circular_center_mask(shape, frac_radius=0.3):
    h, w = shape
    cy, cx = h // 2, w // 2
    R = min(h, w) * frac_radius
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= R * R


def simple_radon(img: np.ndarray, angles: np.ndarray) -> np.ndarray:
    # produce sinogram with shape [n_angles, width]
    h, w = img.shape
    sino = []
    for a in angles:
        # rotate image around center
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        proj = rot.sum(axis=0)
        sino.append(proj)
    return np.stack(sino, axis=0)


def simple_iradon(sino: np.ndarray, angles: np.ndarray, out_shape=(512, 512)) -> np.ndarray:
    # naive unfiltered backprojection: for each angle, smear the projection back
    n_angles, w = sino.shape
    h, ow = out_shape
    recon = np.zeros(out_shape, dtype=np.float32)
    for i, a in enumerate(angles):
        proj = np.tile(sino[i:i+1, :], (h, 1))  # repeat as image
        # rotate back by -a
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -a, 1.0)
        back = cv2.warpAffine(proj.astype(np.float32), M, (ow, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        recon += back
    recon /= len(angles)
    return recon


def save_img(img: np.ndarray, path: Path, cmap="gray"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def create_ppt(images: list[tuple[str, Path]], out_path: Path):
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Gammex pipeline overview"
    slide.placeholders[1].text = "Illustrative pipeline: 2MD → segmentation → projection → optimization → backprojection"

    for caption, imgpath in images:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        left = Inches(0.5)
        top = Inches(0.8)
        pic = slide.shapes.add_picture(str(imgpath), left, top, width=Inches(9))
        tx = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(9), Inches(1))
        tf = tx.text_frame
        tf.text = caption

    prs.save(str(out_path))


def main():
    data_dir = REPO_ROOT / "data" / "gammex" / "phantomtype_1"
    data = load_dir(data_dir)
    # Expect keys: PE, PVC, I, Gd
    # convert to numpy
    data_np = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in data.items()}

    aPE = data_np.get("PE")
    aPVC = data_np.get("PVC")
    I_gt = data_np.get("I")
    Gd_gt = data_np.get("Gd")

    aPEPVC = np.stack([aPE, aPVC], axis=0)
    aWaI = convert_pepvc_to_waio_numpy(aPEPVC)
    Wa = aWaI[0]
    I = aWaI[1]

    # Step 1: initial 2MD
    img1_path = ARTIFACTS_DIR / "step1_initial_wa_io.png"
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(Wa, cmap='gray'); ax[0].set_title('Water (Wa)'); ax[0].axis('off')
    ax[1].imshow(I, cmap='gray'); ax[1].set_title('Iodine (I)'); ax[1].axis('off')
    plt.tight_layout(); plt.savefig(img1_path, dpi=200); plt.close()

    # Step 2: segmented 2MD (simple central circular ROI subtraction)
    mask = circular_center_mask(Wa.shape, frac_radius=0.25)
    Wa_mean = Wa[mask].mean()
    I_mean = I[mask].mean()
    WaR = (Wa - Wa_mean) * mask
    IR = (I - I_mean) * mask
    WaB = Wa - WaR
    IB = I - IR
    img2_path = ARTIFACTS_DIR / "step2_segmented_b_s.png"
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0,0].imshow(WaB, cmap='gray'); ax[0,0].set_title('Background Wa'); ax[0,0].axis('off')
    ax[0,1].imshow(IB, cmap='gray'); ax[0,1].set_title('Background I'); ax[0,1].axis('off')
    ax[1,0].imshow(WaR, cmap='gray'); ax[1,0].set_title('Signal Wa'); ax[1,0].axis('off')
    ax[1,1].imshow(IR, cmap='gray'); ax[1,1].set_title('Signal I'); ax[1,1].axis('off')
    plt.tight_layout(); plt.savefig(img2_path, dpi=200); plt.close()

    # Step 3: forward projection
    angles = np.linspace(0, 180, 90, endpoint=False)
    AB_sino = simple_radon(WaB, angles)
    AR_sino = simple_radon(WaR, angles)
    img3_path = ARTIFACTS_DIR / "step3_sinograms.png"
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].imshow(AB_sino, cmap='gray', aspect='auto'); ax[0].set_title('Background sinogram (Wa)'); ax[0].axis('off')
    ax[1].imshow(AR_sino, cmap='gray', aspect='auto'); ax[1].set_title('Signal sinogram (Wa)'); ax[1].axis('off')
    plt.tight_layout(); plt.savefig(img3_path, dpi=200); plt.close()

    # Step 4: optimization (illustrative: sum the sinograms)
    AC_sino = AB_sino + AR_sino
    img4_path = ARTIFACTS_DIR / "step4_contrast_sinogram.png"
    plt.figure(figsize=(8, 4)); plt.imshow(AC_sino, cmap='gray', aspect='auto'); plt.title('Contrast sinogram (sum)'); plt.axis('off'); plt.tight_layout(); plt.savefig(img4_path, dpi=200); plt.close()

    # Step 5: backprojection
    recon = simple_iradon(AC_sino, angles, out_shape=Wa.shape)
    img5_path = ARTIFACTS_DIR / "step5_backprojection.png"
    save_img(recon, img5_path)

    # Assemble PPT
    ppt_path = ARTIFACTS_DIR / "gammex_pipeline_overview.pptx"
    images = [
        ("1 — Initial Water/Iodine 2MD", img1_path),
        ("2 — Segmented background + signal", img2_path),
        ("3 — Forward projections (sinograms)", img3_path),
        ("4 — Contrast sinogram (illustrative)", img4_path),
        ("5 — Backprojection: contrast map", img5_path),
    ]
    create_ppt(images, ppt_path)

    print(f"Saved PNGs and PPTX to {ARTIFACTS_DIR}")


if __name__ == '__main__':
    main()
