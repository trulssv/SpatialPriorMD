from cal.cal import CalibrationModel
from config import *
import torch
from matplotlib import pyplot as plt
import matplotlib
from operators.tomo import RayTrafoModule
from data_processing.utils import load_materials, load_weights
import cv2
from article.utils import load_dir
from article.attenuations import LinearMaterialDecomposition
from optim.solvers import NewtonSolver
import numpy as np

DATA_DIR = REPO_ROOT / "data"

GAMMEX_PATH = DATA_DIR / "gammex" / "phantomtype_1"
ANTROPOMORPHIC_PATH  = DATA_DIR / "antropomorphic_phantoms" / "case_50"
PLOT_DIR = REPO_ROOT / "AttenuationMD_python" / "article" / "figs" / "plots"

gammex_data = load_dir(GAMMEX_PATH)
antropomorphic_data = load_dir(ANTROPOMORPHIC_PATH)
print("Gammex data loaded:", gammex_data.keys())
print("Anthropomorphic data loaded:", antropomorphic_data.keys())

def load_all_materials():


    fB, _, fC = load_materials()

    fWa = fB[0, :].numpy()  # [nE]
    fI = fC[0, :].numpy()
    fGd = fC[1, :].numpy()

    fB, _, _ = load_materials(usePePvc=True)

    fPE = fB[0, :].numpy()  # [nE]
    fPVC  = fB[1, :].numpy()  # [nE]

    return fWa, fI, fGd, fPE, fPVC


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


weights = torch.softmax(load_weights(), dim=1).numpy()  # [nBins x nE]
# weights = load_weights().numpy()  # [nBins x nE]


pepvc_to_waio = LinearMaterialDecomposition(np.stack([fPE, fPVC], axis=0), np.stack([fWa, fI], axis=0), weights, f_wa=fWa, pathlength_wa=100, plot=True, mats_dom=["PE", "PVC"], mats_range=["Water", "Iodine"])
pepvc_to_igd = LinearMaterialDecomposition(np.stack([fPE, fPVC], axis=0), np.stack([fI, fGd], axis=0), weights, f_wa=fWa, pathlength_wa=100, plot=True, mats_dom=["PE", "PVC"], mats_range=["Iodine", "Gadolinium"])
igd_to_waio = LinearMaterialDecomposition(np.stack([fI, fGd], axis=0), np.stack([fWa, fI], axis=0), weights, f_wa=fWa, pathlength_wa=100, plot=True, mats_dom=["Iodine", "Gadolinium"], mats_range=["Water", "Iodine"])
wai_to_igd = LinearMaterialDecomposition(np.stack([fWa, fI], axis=0), np.stack([ fI, fGd], axis=0), weights, f_wa=fWa, pathlength_wa=100, plot=True, mats_dom=["Water", "Iodine"], mats_range=["Iodine", "Gadolinium"])

print(f"PE/PVC-> Wa/I: A = {pepvc_to_waio.A}, Wa/I -> I/Gd: A = {wai_to_igd.A}, PE/PVC -> I/Gd: A = {pepvc_to_igd.A}, PE/PVC -> I/Gd (via Wa/I)= {pepvc_to_waio.A @ wai_to_igd.A}, Wa/I -> I/Gd (inverse) = {igd_to_waio.Ainv}")


# Gammex

# convert torch tensors to numpy arrays

gammex_data = {k: v.numpy() for k, v in gammex_data.items()}

PE = gammex_data["PE"]
PVC = gammex_data["PVC"]
I_gt = gammex_data["I"] 
Gd_gt = gammex_data["Gd"]
mask = (I_gt > 1e-4) | (Gd_gt > 1e-5)

plt.clf()
plt.imshow(mask)
plt.title("Mask")
plt.savefig(PLOT_DIR / "gammex_mask.png", dpi=300)

# Get mask components

mask_comps = cv2.connectedComponents(mask.astype(np.uint8))

print(len(mask_comps), mask_comps[0], mask_comps[1].shape)

# plot

fig, ax = plt.subplots(2, 1, figsize=(6, 12))

im = ax[0].imshow(PE, cmap='gray')
ax[0].set_title("PE Image")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

im = ax[1].imshow(PVC, cmap='gray')
ax[1].set_title("PVC Image")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
plt.savefig(PLOT_DIR / "gammex_images_pepvc.png", dpi=300)

# Convert PE/PVC to Water/Iodine

aPEPVC = np.stack([PE, PVC], axis=0)

aWaI = pepvc_to_waio.apply(aPEPVC)

Wa = aWaI[0, :]
I = aWaI[1, :]

# plot

fig, ax = plt.subplots(2, 1, figsize=(6, 12))

im = ax[0].imshow(Wa, cmap='gray')
ax[0].set_title("Water Image")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)



im = ax[1].imshow(I, cmap='gray')
ax[1].set_title("Iodine Image")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)


plt.savefig(PLOT_DIR / "gammex_images_waio.png", dpi=300)

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

# Plot background and signal

Ilim = [0, 0.004]

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
im = ax[0, 0].imshow(WaB, cmap='gray')
ax[0, 0].set_title("Water Background")
fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04) 

im = ax[0, 1].imshow(IB, cmap='gray')
ax[0, 1].set_title("Iodine Background")
fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)

im = ax[1, 0].imshow(WaR, cmap='gray')
ax[1, 0].set_title("Water Signal")
fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)

im = ax[1, 1].imshow(IR, cmap='gray')
ax[1, 1].set_title("Iodine Signal")
fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
plt.savefig(PLOT_DIR / "gammex_images_waio_signal_background.png", dpi=300)


aC_pred_linear = igd_to_waio.apply_inverse(aR)  # [2 x H x W]

aI_pred_linear = aC_pred_linear[0, ...]
aGd_pred_linear = aC_pred_linear[1, ...]
# post proc

for val in range(1, mask_comps[0]):
    
    cc = (mask_comps[1] == val).astype(np.uint8)  # Component mask

    # Compute predicted values for the component
    
    aI_pred_linear[cc == 1] = np.mean(aI_pred_linear[cc == 1])
    aGd_pred_linear[cc == 1] = np.mean(aGd_pred_linear[cc == 1])




# plot



rhoW = 1000
rhoI = 4930
rhoGd = 7400

aC_gt[0, :] = aC_gt[0, :] * rhoI
aC_gt[1, :] = aC_gt[1, :] * rhoGd
aC_pred_linear[0, :] = aC_pred_linear[0, :] * rhoI
aC_pred_linear[1, :] = aC_pred_linear[1, :] * rhoGd 

diff = np.abs(aC_pred_linear - aC_gt)

Ilim = [0, 20]
Gdlim = [0, 5]

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
im = ax[0, 0].imshow(aC_gt[0, :], vmin=Ilim[0], vmax=Ilim[1], cmap='gray')
ax[0, 0].set_title("Iodine Ground Truth")
fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)

im = ax[0, 1].imshow(aC_gt[1, :], vmin=Gdlim[0], vmax=Gdlim[1], cmap='gray')
ax[0, 1].set_title("Gadolinium Ground Truth")
fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)

im = ax[1, 0].imshow(aC_pred_linear[0, :], vmin=Ilim[0], vmax=Ilim[1], cmap='gray')
ax[1, 0].set_title("Iodine Predicted (Linear)")
fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)

im = ax[1, 1].imshow(aC_pred_linear[1, :], vmin=Gdlim[0], vmax=Gdlim[1], cmap='gray')
ax[1, 1].set_title("Gadolinium Predicted (Linear)")
fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)

im = ax[2, 0].imshow(diff[0, :], vmin=Ilim[0], vmax=Ilim[1], cmap='gray')
ax[2, 0].set_title("Iodine Difference")
fig.colorbar(im, ax=ax[2, 0], fraction=0.046, pad=0.04)

im = ax[2, 1].imshow(diff[1, :], vmin=Gdlim[0], vmax=Gdlim[1], cmap='gray')
ax[2, 1].set_title("Gadolinium Difference")
fig.colorbar(im, ax=ax[2, 1], fraction=0.046, pad=0.04)

plt.savefig(PLOT_DIR / "gammex_images_waio_to_igd_linear.png", dpi=300)

# Perform nonlinear optimization using Newton's method

calibrated_model = CalibrationModel.from_artifact("shallow-net", 456)

ray_trafo = RayTrafoModule()

AB = ray_trafo.forward(torch.from_numpy(aB))
AR = ray_trafo.forward(torch.from_numpy(aR))

sino_sz = AB.shape[1:]

AB = AB.view(-1, 2)  # [P x 2]
AR = AR.view(-1, 2)  # [P x 2]

out = calibrated_model.quick_eval(AB=AB, AR=AR, return_dict=True)
AC_pred_nonlinear = out["AC_cal"]

AC_pred_nonlinear = AC_pred_nonlinear.view(-1, *sino_sz).numpy()  # [2 x H x W]

aC_pred_nonlinear = ray_trafo.backward(AC_pred_nonlinear).numpy()  # [2 x H x W]
# Plot nonlinear results

aC_pred_nonlinear[0, :] = aC_pred_nonlinear[0, :] * rhoI
aC_pred_nonlinear[1, :] = aC_pred_nonlinear[1, :] * rhoGd

plt.clf()
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
im = ax[0, 0].imshow(aC_gt[0, :], vmin=Ilim[0], vmax=Ilim[1], cmap='gray')
ax[0, 0].set_title("Iodine Ground Truth")
fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04) 
ax[0, 1].imshow(aC_gt[1, :], vmin=Gdlim[0], vmax=Gdlim[1], cmap='gray')
ax[0, 1].set_title("Gadolinium Ground Truth")
fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)
ax[1, 0].imshow(aC_pred_nonlinear[0, :], vmin=Ilim[0], vmax=Ilim[1], cmap='gray')
ax[1, 0].set_title("Iodine Predicted (Nonlinear)")
fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
ax[1, 1].imshow(aC_pred_nonlinear[1, :], vmin=Gdlim[0], vmax=Gdlim[1], cmap='gray')
ax[1, 1].set_title("Gadolinium Predicted (Nonlinear)")
fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
diff_nonlinear = np.abs(aC_pred_nonlinear - aC_gt)
im = ax[2, 0].imshow(diff_nonlinear[0, :], vmin=Ilim[0], vmax=Ilim[1], cmap='gray')
ax[2, 0].set_title("Iodine Difference (Nonlinear)")
fig.colorbar(im, ax=ax[2, 0], fraction=0.046, pad=0.04)
im = ax[2, 1].imshow(diff_nonlinear[1, :], vmin=Gdlim[0], vmax=Gdlim[1], cmap='gray')
ax[2, 1].set_title("Gadolinium Difference (Nonlinear)")
fig.colorbar(im, ax=ax[2, 1], fraction=0.046, pad=0.04)

plt.savefig(PLOT_DIR / "gammex_images_waio_to_igd_nonlinear.png", dpi=300)