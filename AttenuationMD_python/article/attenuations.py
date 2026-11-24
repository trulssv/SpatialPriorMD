from config import *
import torch
from matplotlib import pyplot as plt
import matplotlib
from data_processing.utils import load_materials, load_weights, load_energies
from optim.solvers import NewtonSolver
import numpy as np
from scipy.optimize import fmin, minimize


PEPVC_TO_WAIO = torch.tensor([[ 1.2465975,  -0.4603571 ], [-0.00212777,  0.02189077]])

# PEPVC_TO_WAIO = torch.tensor([[0.83204985,   0.08087475], [17.497787,    47.382126]])


class LinearMaterialDecomposition():
    """Class for performing linear material decomposition using a change of basis matrix. 
    It uses the Polychromatic Lambert-Beer law in PCCT to derive a change of basis matrix from the attenuation coefficients of the materials to the basis functions. 
    The change of basis matrix is derived using a weighted least squares approach, where the weights are derived from the energy spectrum and the pathlength of the materials. 
    The change of basis matrix can then be used to decompose the measured attenuation coefficients into the basis functions.
    """

    def __init__(self, f_domain: np.ndarray, f_range: np.ndarray, weights: np.ndarray, f_wa: np.ndarray = None, pathlength_wa: float = None, plot: bool = False, mats_dom: list[str] = None, mats_range: list[str] = None):

        assert f_domain.shape[0] == f_range.shape[0], "f_domain and f_range must have the same number of rows"
        assert f_domain.shape[1] == weights.shape[1], "f_domain and weights must have the same number of columns"
        assert f_range.shape[1] == weights.shape[1], "f_range and weights must have the same number of columns"

        self.f_domain = f_domain
        self.f_range = f_range
        self.weights = weights


        self.mats_dom = mats_dom
        self.mats_range = mats_range

        # Attenuate spectrum if effective attenuation in water is provided

        if f_wa is not None and pathlength_wa is not None:
            att = np.exp(-pathlength_wa * f_wa)  # [nE]
            self.weights = weights * att  # [nBins x nE]
        
        # normalize weights to a probability distribution over energies for each bin 


        self.weights = self.weights / np.max(self.weights, axis=1, keepdims=True) # [nBins x nE]
        self.A, self.Ainv = self.compute_change_of_basis_matrix()  # [2 x 2], map from f_domain to f_range

        
        if plot:
            self.plot_change_of_basis_matrix()
    



    def compute_change_of_basis_matrix(self):
        
        w = np.einsum('be, me->bm', self.weights, np.exp(-self.f_range))  # [nBins x nMat]

        # print(w, self.mats_dom, self.mats_range)

        def objective_func_1(A):
            """Objective function to minimize the difference between the predicted and actual attenuation coefficients."""

            att = np.exp(-A @ self.f_domain)  # [1 x nE]
            w1 = np.einsum('be, e->b', self.weights, att)  # [nBins]

            obj = np.sum((w1 - w[:, 0]) ** 2)
            # print(f"Objective function value: {obj}, {self.mats_range[0]} -> {self.mats_dom[0], self.mats_dom[1]}: {A1}, {self.mats_range[1]} -> {self.mats_dom[0], self.mats_dom[1]}: {A2}")
            return obj
        
        def objective_func_2(A):
            """Objective function to minimize the difference between the predicted and actual attenuation coefficients."""

            att = np.exp(-A @ self.f_domain)  # [1 x nE]
            w2 = np.einsum('be, e->b', self.weights, att)  # [nBins]

            obj = np.sum((w2 - w[:, 1]) ** 2)
            # print(f"Objective function value: {obj}, {self.mats_range[0]} -> {self.mats_dom[0], self.mats_dom[1]}: {A1}, {self.mats_range[1]} -> {self.mats_dom[0], self.mats_dom[1]}: {A2}")
            return obj



        A0 = np.zeros((2, ), dtype=np.float32)  # Initial guess for the optimization
        A1 = minimize(objective_func_1, A0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False}).x
        A2 = minimize(objective_func_2, A0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False}).x

        Ainv = np.stack([A1, A2], axis=0).T  # [2 x 2], map from f_range to f_domain
        A = np.linalg.inv(Ainv)  # [2 x 2], map from f_range to f_domain

        return A, Ainv
    
    def apply(self, a: np.ndarray) -> np.ndarray:
        """Apply the change of basis matrix to a  stack of basis maps to obtain the corresponding stack of basis maps in the new basis.
        Args:
            a (np.ndarray): Stack of basis maps in the original basis. Shape: [nMats x nPixels] or [nMats x H x W]
        Returns:
            np.ndarray: Stack of basis maps in the new basis. Shape: [nBasis x nPixels] or [nBasis x H x W]
        """
        assert a.shape[0] == self.f_domain.shape[0], "Input attenuation coefficients must have the same number of rows as f_domain"
        return np.einsum('ij,j...->i...', self.A, a)  # [nBasis x nPixels] or [nBasis x H x W]

    def apply_inverse(self, a: np.ndarray) -> np.ndarray:
        """Apply the inverse change of basis matrix to a stack of basis maps to obtain the corresponding stack of basis maps in the original basis.
        Args:
            a (np.ndarray): Stack of basis maps in the new basis. Shape: [nBasis x nPixels] or [nBasis x H x W]
        Returns:
            np.ndarray: Stack of basis maps in the original basis. Shape: [nMats x nPixels] or [nMats x H x W]
        """
        assert a.shape[0] == self.f_range.shape[0], "Input attenuation coefficients must have the same number of rows as f_range"
        return np.einsum('ij,j...->i...', self.Ainv, a)  # [nMats x nPixels] or [nMats x H x W]


    def plot_change_of_basis_matrix(self):
        """Plot the change of basis matrix and its inverse."""
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))

        f1 = self.f_range[0, :]  # [nE]
        f2 = self.f_range[1, :]  # [nE]

        f1_pred = self.Ainv[:, 0] @ self.f_domain  # [nE]
        f2_pred = self.Ainv[: , 1] @ self.f_domain  # [nE]

        ax[0].plot(f1, label=f"{self.mats_range[0] if self.mats_range is not None else 'f1'} (true)")
        ax[0].plot(f1_pred, label=f"{self.mats_range[0] if self.mats_range is not None else 'f1'} (predicted)  = {self.Ainv[0, 0]:.2f} * {self.mats_dom[0] if self.mats_dom is not None else 'f1'} + {self.Ainv[1, 0]:.2f} * {self.mats_dom[1] if self.mats_dom is not None else 'f2'}")
        ax[0].set_xlabel("Energy (keV)")
        ax[0].set_ylabel("Attenuation (mm$^{-1}$)")
        ax[0].legend()  

        ax[1].plot(f2, label=f"{self.mats_range[1] if self.mats_range is not None else 'f2'} (true)")
        ax[1].plot(f2_pred, label=f"{self.mats_range[1] if self.mats_range is not None else 'f2'} (predicted)= {self.Ainv[0, 1]:.2f} * {self.mats_dom[0] if self.mats_dom is not None else 'f1'} + {self.Ainv[1, 1]:.2f} * {self.mats_dom[1] if self.mats_dom is not None else 'f2'}")
        ax[1].set_xlabel("Energy (keV)")
        ax[1].set_ylabel("Attenuation (mm$^{-1}$)")
        ax[1].legend()

        plt.savefig(REPO_ROOT / "AttenuationMD_python" / "article" /"figs" / "plots" / f"attenuations_predicted_{self.mats_dom[0]}_{self.mats_dom[1]}_to_{self.mats_range[0]}_{self.mats_range[1]}.png", dpi=300)


def main():
    # Load materials and weights
    fB, fR, fC = load_materials()
    weights = load_weights()

    # Newton solver
    solver = NewtonSolver(fB=fB, fR=fR, fC=fC, weights=weights)

    # Load energies
    E = load_energies()  # [nE]
    E = E.numpy()  # Convert to numpy array for plotting

    # Plot materials
    fWa = fB[0, :].numpy()  # [nE]
    fI = fC[0, :].numpy()
    fGd = fC[1, :].numpy()

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    ax[0].plot(E, fWa, label="Water")
    ax[0].set_xlabel("Energy (keV)")
    ax[0].set_ylabel("Attenuation (mm$^{-1}$)")
    ax[0].legend()

    ax[1].plot(E, fI, label="Iodine")
    ax[1].plot(E, fGd, label="Gadolinium")
    ax[1].set_xlabel("Energy (keV)")
    ax[1].set_ylabel("Attenuation (mm$^{-1}$)")
    ax[1].legend()

    plt.savefig(REPO_ROOT / "AttenuationMD_python" / "article" /"figs" / "plots" / "attenuations.png", dpi=300)

    # Load materials and weights

    E = load_energies()  # [nE]
    E = E.numpy()  # Convert to numpy array for plotting
    weights = load_weights()  # nBins x nE
    weights = torch.softmax(weights, dim=1).numpy()  # Convert to numpy array for plotting

    fB, _, fC = load_materials()

    fWa = fB[0, :].numpy()  # [nE]
    fI = fC[0, :].numpy()
    fGd = fC[1, :].numpy()

    fB, _, _ = load_materials(usePePvc=True)

    fPE = fB[0, :].numpy()  # [nE]
    fPVC  = fB[1, :].numpy()  # [nE]

    # plot materials

    print(E.shape, fWa.shape, fI.shape, fGd.shape)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    ax[0].plot(E, fWa, label="Water")
    ax[0].plot(E, fPE, label="PE")
    ax[0].plot(E, fPVC, label="PVC")
    ax[0].set_xlabel("Energy (keV)")
    ax[0].set_ylabel("Attenuation (mm$^{-1}$)")
    ax[0].legend()

    ax[1].plot(E, fI, label="Iodine")
    ax[1].plot(E, fGd, label="Gadolinium")
    ax[1].set_xlabel("Energy (keV)")
    ax[1].set_ylabel("Attenuation (mm$^{-1}$)")
    ax[1].legend()

    plt.savefig(REPO_ROOT / "AttenuationMD_python" / "article" /"figs" / "plots" / "attenuations.png", dpi=300)

    # ---- Derive change of basis matrix from materials to basis functions

    # Naive approach

    f1 = np.stack([fPE, fPVC], axis=0)  # [2 x nE]

    c1 = np.linalg.lstsq(f1.T, fWa, rcond=None)[0]  # [2]
    c2 = np.linalg.lstsq(f1.T, fI, rcond=None)[0]  # [2]

    C = np.stack([c1, c2], axis=0)  # [2 x 2]
    Cinv = np.linalg.inv(C)

    print(f"Change of basis matrix (naive): {Cinv}")


    # Weighted approach

    water_pathlength = 0 # mm
    att = np.exp(-water_pathlength * fWa)  # [nE]
    print(att.shape, weights.shape)
    attenuated_weights = weights * att  # [nBins x nE]

    wW = np.sum(attenuated_weights * np.exp(-fWa), axis=1)  # [nBins]
    wI = np.sum(attenuated_weights * np.exp(-fI), axis=1)  # [nBins]

    def objective_func(A):
        """Objective function to minimize the difference between the predicted and actual water attenuation."""
        A1 = A[0:2].reshape(1, 2)  # [1 x 2]
        A2 = A[2:4].reshape(1, 2)  # [1 x 2]          # [2 x nE]
        w1 = np.sum(attenuated_weights * np.exp(-A1 @ f1), axis=1)  # [nBins]
        w2 = np.sum(attenuated_weights * np.exp(-A2 @ f1), axis=1)  # [nBins]

        return np.sum((w1 - wW) ** 2) + np.sum((w2 - wI) ** 2)

    A0 = np.zeros((4, 1), dtype=np.float32)  # Initial guess for the optimization


    Amin = fmin(objective_func, A0, disp=True)

    Amin = Amin.reshape(2, 2).T # Scale iodine coefficients to match units of water
    Ainv = np.linalg.inv(Amin)

    print(f"Change of basis matrix (weighted): {Amin}")
    print(f"Inverse change of basis matrix (weighted): {Ainv}")

    # plot the estimated basis functions

    fIest = Amin[:, 1] @ f1  # [nE]
    fWaest = Amin[:, 0] @ f1  # [nE]

    fIest_naive = C[1, :] @ f1  # [nE]
    fWaest_naive = C[0, :] @ f1  # [nE]

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    ax[0].plot(E, fWa, label="Water (true)")
    ax[0].plot(E, fWaest, label="Water (estimated lambert beer)")
    ax[0].plot(E, fWaest_naive, label="Water (estimated least squares)")
    ax[0].set_xlabel("Energy (keV)")
    ax[0].set_ylabel("Attenuation (mm$^{-1}$)")
    ax[0].legend()

    ax[1].plot(E, fI, label="Iodine (true)")
    ax[1].plot(E, fIest, label="Iodine (estimated lambert beer)")
    ax[1].plot(E, fIest_naive, label="Iodine (estimated least squares)")
    ax[1].set_xlabel("Energy (keV)")
    ax[1].set_ylabel("Attenuation (mm$^{-1}$)")
    ax[1].legend()

    plt.savefig(REPO_ROOT / "AttenuationMD_python" / "article" /"figs" / "plots" / "attenuations_estimated.png", dpi=300)

    rhoWa = 1.0  # g/mL
    rhoI = 4.93  # g/mL
    rhoPE = 0.94  # g/mL
    rhoPVC = 1.38  # g/mL

    # Rescale matrix from mass fraction to mg/ml


    # Karins 

    pepvc_to_waio = np.array([[9.867838176015617e+02,1.204811667136770e+03],[-18.1968, 295.4176]])

if __name__ == "__main__":
    # main()

    
    # Load materials and weights

    E = load_energies()  # [nE]
    E = E.numpy()  # Convert to numpy array for plotting
    weights = load_weights()  # nBins x nE
    # weights = torch.softmax(weights, dim=1).numpy()  # Convert to numpy array for plotting
    weights = weights.numpy()  # Convert to numpy array for plotting
    fB, _, fC = load_materials()

    fWa = fB[0, :].numpy()  # [nE]
    fI = fC[0, :].numpy()
    fGd = fC[1, :].numpy()

    fB, _, _ = load_materials(usePePvc=True)

    fPE = fB[0, :].numpy()  # [nE]
    fPVC  = fB[1, :].numpy()  # [nE]
    
    
    atts = [50 * j for j in range(1, 20)]  # mm

    for att in atts:
        print(f"Pathlength: {att} mm")

        MD = LinearMaterialDecomposition(f_domain=np.stack([fPE, fPVC], axis=0), f_range=np.stack([fWa, fI], axis=0), weights=weights, f_wa=fWa, pathlength_wa=att)

        print(f"Change of basis matrix (weighted): {MD.A}")
