"""
newton_solver.py
----------------
Unrolled, batched Newton solver for spectral CT material decomposition.

Differentiable w.r.t. weights_upd (and therefore w.r.t. the weight network
parameters) because all operations are native PyTorch.

Key design choices
------------------
* **Unrolled iterations** — each Newton step is an explicit PyTorch op, so
  autograd traces through the full unrolled loop.  This is simpler than
  implicit differentiation and works well for ~20-30 unroll steps.
* **Gauss-Newton Hessian** — the curvature term is dropped; this has no effect
  on the fixed-point (solver accuracy) and avoids second-order autograd.
* **LM damping** — diagonal regularisation keeps the linear system SPD even
  when the Hessian is poorly conditioned.
* **Armijo line-search** — used during validation/inference (torch.no_grad)
  for robustness; disabled during training to keep the graph simple and the
  unroll depth manageable.

Usage
-----
    from newton_solver import newton_solve, newton_solve_inference

    # During training (gradient flows back through solver):
    AC_pred = newton_solve(lambda_true, weights_upd, fC, AC_init)

    # During inference / validation (no gradient, more iterations, robust):
    with torch.no_grad():
        AC_pred = newton_solve_inference(lambda_true, weights_upd, fC, AC_init)
"""


import torch
from config import DEBUG_DIR

class NewtonSolver:

    def __init__(
            self, 
            fB: torch.Tensor, 
            fR: torch.Tensor, 
            fC: torch.Tensor, 
            weights: torch.Tensor, 
            E: torch.Tensor | None = None,
            dE: float = 1.0, 
            floor: float = 1e-12, 
            normalize_weights: bool = True,
            LM_damping: bool = True,
            Armijo_backtracking: bool = True,
            mu_reg: torch.Tensor = torch.tensor(1e-8), 
            verbose: bool = False,
            objective: str = "KL"
        ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else fB.device)

        print(f"Initializing NewtonSolver on device: {self.device}, with objective: {objective}...\n")


                
        self.fB = fB.to(device=self.device)
        self.fR = fR.to(device=self.device)
        self.fC = fC.to(device=self.device)
        self.weights = weights.to(device=self.device)
        self.objective = objective
        self.E = E.to(device=self.device) if E is not None else None

        self.dE = dE
        self.floor = floor
        self.mu_reg = mu_reg

        # Settings


        self.normalize_weights = normalize_weights
        self.LM_damping = LM_damping
        self.Armijo_backtracking = Armijo_backtracking
        self.verbose = verbose



        self.grad = False  # Flag to indicate whether to compute gradient w.r.t. weight network parameters
        self.AC_gt, self.w_grad = None, None

 
        self.nE = fB.shape[1]
        self.nBins = weights.shape[0]
        self.nMat = fC.shape[0]

        assert fB.shape == fR.shape == (2, self.nE), \
            "fB and fR should be [2 x nE]"
        assert fC.shape == (self.nMat, self.nE), \
            "fC should be [nMat x nE]"

    def _setup_gradient_computation(self, AC_gt: torch.Tensor, w_grad: torch.Tensor):
        """Set up the gradient computation by storing the ground truth AC and the gradient of the weights w.r.t. the weight network parameters."""
        if self.verbose:
            print(f"Setting up gradient computation with AC_gt shape: {AC_gt.shape}, w_grad shape: {w_grad.shape}")

        assert (AC_gt.ndim == 2) or (AC_gt.ndim == 3 and AC_gt.shape[-1] == self.nMat), "AC_gt should be [P x nMat] or [nAngles x nDet x nMat]"

        if AC_gt.ndim == 3:
            AC_gt = AC_gt.reshape(-1, self.nMat)  # [P x nMat]
        weights = self._normalize_weights(w_grad.reshape(-1, self.nBins, self.nE))  # [P x nBins x nE]
        self.grad = True
        self.AC_gt = AC_gt
        self.w_grad = weights

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalize the weights to [0, 1] across bins."""
        if self.normalize_weights:
            if self.verbose:
                print("Normalizing weights to [0, 1] across bins. If you do not want this behavior, set normalize_weights=False...\n")
            return weights / weights.max(dim=1, keepdim=True).values.clamp(min=self.floor).to(device=self.device)  # [nBins x nE]
        return weights

    def _set_materials(self, fB: torch.Tensor, fR: torch.Tensor, fC: torch.Tensor):
        """Set the material basis functions for the solver."""
        assert fB.shape == fR.shape == (2, self.nE), "fB and fR should be [2 x nE]"
        assert fC.shape == (self.nMat, self.nE), "fC should be [nMat x nE]"

        self.fB = fB.to(device=self.device)
        self.fR = fR.to(device=self.device)
        self.fC = fC.to(device=self.device)

    def _validate_residual_curve(self, residual_curve: torch.Tensor | None, P: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        """Validate an optional residual attenuation curve with shape [P x nE]."""
        if residual_curve is None:
            return None
        if residual_curve.ndim != 2 or residual_curve.shape != (P, self.nE):
            raise ValueError(f"Expected residual_curve to have shape [{P}, {self.nE}], got {residual_curve.shape}")
        return residual_curve.to(device=device, dtype=dtype)

    def _filter_weights_near_k_edge(self, weights: torch.Tensor, E: torch.Tensor, plot:bool=True) -> torch.Tensor:

        def filter(x, a, b, p):


            # Smooth filter around the k-edge of gadolinium (Egd) with a width of delta

            mask = torch.ones_like(x)  # Initialize mask to ones

            gaussian = torch.exp(-((x - a) ** 2) / (2 * b ** 2))  # Gaussian function

            mask = mask * (1 - gaussian).pow(p)  # Apply the Gaussian filter with power p

            # mask = mask / mask.mean()  # Normalize the mask to have a mean of 1

            return mask



        """Apply a filter to the weights near the k-edge of gadolinium to reduce the influence of the residual."""
        assert weights.ndim in (2, 3), "weights should be [nBins x nE] or [P x nBins x nE]"
        assert E.ndim == 1 and E.shape[0] == self.nE, "E should be [nE]"

        # Create a smooth mask around the k-edge of gadolinium (Egd) with a width of delta

        mask = filter(E, a=50.2, b=15, p=2)  # [nE]

        if weights.ndim == 2:
            weights_filt = weights * mask.unsqueeze(0)  # [nBins x nE]
        else:
            weights_filt = weights * mask.unsqueeze(0).unsqueeze(1)  # [P x nBins x nE]

        # Debug: plot the weights before and after filtering

        if plot:

            weights_norm = weights / weights.sum(dim=-1, keepdim=True).clamp(min=self.floor)
            weights_filt_norm = weights_filt / weights_filt.sum(dim=-1, keepdim=True).clamp(min=self.floor)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))

            # plot all the weights before filtering

            for b in range(weights.shape[0]):
                ax[0].plot(E.cpu().numpy(), weights_norm[b].cpu().numpy(), label=f"Bin {b+1}")
            ax[0].legend()
            ax[0].set_title("Weights Before Filtering")
            ax[0].set_xlabel("Energy (keV)")
            ax[0].set_ylabel("Weight")
            ax[0].grid(True)

            # plot all the weights after filtering
            for b in range(weights_filt.shape[0]):
                ax[1].plot(E.cpu().numpy(), weights_filt_norm[b].cpu().numpy(), label=f"Bin {b+1}")
            ax[1].legend()
            ax[1].set_title("Weights After Filtering")
            ax[1].set_xlabel("Energy (keV)")
            ax[1].set_ylabel("Weight")
            ax[1].grid(True)

            # plot the mask

            ax[2].plot(E.cpu().numpy(), mask.cpu().numpy(), color="black")

            plt.savefig("weights_filter_debug.png")

            
        return weights_filt

    def solve_from_counts(self, n, AC_init, max_iter=20, gtol=1e-8, ftol=1e-8):
        """Solve for AC given counts n, weights, and basis functions fC."""
        # Compute predicted lambda from counts n


        # move to device and convert to float

        AC_init = AC_init.to(device=self.device, dtype=torch.float32) if AC_init is not None else torch.zeros((n.shape[0], self.nMat), device=self.device, dtype=torch.float32)
        n = n.to(device=self.device, dtype=torch.float32)  # [P x nBins]

        lam = n.float()  # Assuming n is already in the correct shape [P x nBins]

        self.P = n.shape[0]
        weights = self.weights

        assert isinstance(n, torch.Tensor), f"Input n must be a torch.Tensor but got {type(n)}"
        assert isinstance(weights, torch.Tensor), f"Input weights must be a torch.Tensor but got {type(weights)}"
        assert isinstance(AC_init, torch.Tensor), f"Input AC_init must be a torch.Tensor but got {type(AC_init)}"
        assert isinstance(self.fC, torch.Tensor), f"fC must be a torch.Tensor but got {type(self.fC)}"

        attenuated_weights = weights.unsqueeze(0).expand(n.shape[0], -1, -1)  # extend from nBins x nE to P x nBins x nE

        f_prev = torch.full((self.P,), float('inf'), device=n.device, dtype=n.dtype)  # [P]

        mu = self.mu_reg * torch.ones(self.P, device=n.device, dtype=n.dtype)  if self.LM_damping else None  # [P]


        # Initialize solver
      
        AC = AC_init.clone()
        if self.verbose:
            print(f"Starting Newton solver with max_iter={max_iter}, gtol={gtol}, ftol={ftol}, P={self.P}, nMat={self.nMat}, nBins={self.nBins}, nE={self.nE} and initialization AC_init={AC_init is not None}")

        # Initialize logging dictionary to store iteration metrics

        self.log = {
            "Objective": [],
            "Gradient Norm": [],
            "Maximum Gradient Norm": [],
            "Function Difference Norm": [],
            "Mu": [],
            "AC": [AC.clone()],
        }

        for iter in range(max_iter):
            AC, grad, hessian_reg, f, mu = self.newton_step(AC, self.fC, attenuated_weights, lam, f_prev, mu)

            diff = torch.abs(f_prev - f)
            ray_grad_norm = torch.norm(grad, dim=1)
            grad_norm = ray_grad_norm.mean().item()
            max_grad_norm = ray_grad_norm.max().item()
            diff_norm = torch.norm(diff).mean().item()

            # Log the metrics for this iteration

            self.log["Objective"].append(f.clone())
            self.log["Gradient Norm"].append(grad_norm)
            self.log["Maximum Gradient Norm"].append(max_grad_norm)
            self.log["Function Difference Norm"].append(diff_norm)
            self.log["Mu"].append(mu.clone() if mu is not None else None)
            self.log["AC"].append(AC.clone())


            if self.verbose:
                print(f"Iteration {iter+1}: mean objective={float(f.mean()):.3e}, diff norm={diff_norm:.3e}, max grad norm={max_grad_norm:.3e}")

                # component-wise convergence check

                print(f"AC1: {AC.mean(dim=0)[0]}, AC2: {AC.mean(dim=0)[1]}, grad1: {grad.mean(dim=0)[0]}, grad2: {grad.mean(dim=0)[1]}")


            # The rays are independent, so a batch-average objective change can
            # become small while an individual ray is still far from stationary.
            # Stop only when every accepted post-step state satisfies the
            # stationarity tolerance.
            if max_grad_norm < gtol:
                    
                if self.verbose:
                    print(
                        f"Newton solver converged within per-ray gradient tolerance "
                        f"{gtol} after {iter+1} iterations"
                    )
                
                return {"AC": AC, "log": self.log}
            f_prev = f
        
        if self.verbose:
            print(f"Warning: Newton solver did not converge within {max_iter} iterations")
        return {"AC": AC, "log": self.log}  

    def newton_solve(
            self, 
            AB: torch.Tensor, 
            AR: torch.Tensor, 
            AC_init: torch.Tensor | None = None, 
            weights: torch.Tensor | None = None, 
            residual_curve: torch.Tensor | None = None,
            max_iter: int = 20, 
            gtol: float = 1e-6,
            ftol: float = 1e-6,
            )->dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        
        if weights is None:
            weights = self.weights

 
        # optional: normalize weights to sum to 1 across bins

        # self._plot_weights(weights, title="original weights", output_dir=DEBUG_DIR)
        if self.normalize_weights:
            weights = self._normalize_weights(weights)  # [nBins x nE]

        # DEBUG plot weights before and after normalization

        #self._plot_weights(weights, title="normalized weights", output_dir=DEBUG_DIR)


        self.P = AB.shape[0]
        # residual_curve = self._validate_residual_curve(residual_curve, self.P, AB.device, AB.dtype)

        # Step 0: move to device and convert to float

        AB = AB.to(device=self.device, dtype=torch.float32)  # [P x nE]
        AR = AR.to(device=self.device, dtype=torch.float32)  # [P x nE]
        AC_init = AC_init.to(device=self.device, dtype=torch.float32) if AC_init is not None else None

        # Step 1: compute attenuations and weigthts

        attenuation_background = self.compute_attenuation(AB, self.fB)
        attenuation_signal = self.compute_attenuation(AR, self.fR)
        attenuated_weights = self.update_weights(weights, attenuation_background)

        # Optional: normalize weights to sum to 1 accros bins:

        # attenuated_weights = attenuated_weights / attenuated_weights.sum(dim=1, keepdim=True).clamp(min=self.floor)  # [P x nBins x nE]

        # Step 2: compute initial lambda, objective, and LM damping

        lam = self.compute_lambda(attenuated_weights, attenuation_signal) # [P x nBins]
        f_prev = torch.full((self.P,), float('inf'), device=AB.device, dtype=AB.dtype)  # [P]

        mu = self.mu_reg * torch.ones(self.P, device=AB.device, dtype=AB.dtype)  if self.LM_damping else None  # [P]

        

        # Initialize solver

        if AC_init is None:
            AC = torch.zeros_like(AB)  # [P x nMat]
        else:
            AC = AC_init.clone()
        if self.verbose:
            print(f"Starting Newton solver with max_iter={max_iter}, gtol={gtol}, ftol={ftol}, P={self.P}, nMat={self.nMat}, nBins={self.nBins}, nE={self.nE} and initialization AC_init={AC_init is not None}")

        # Initialize logging dictionary to store iteration metrics

        self.log = {
            "Objective": [],
            "Gradient Norm": [],
            "Maximum Gradient Norm": [],
            "Function Difference Norm": [],
            "Mu": [],
            "AC": [AC.clone()],
        }

        for iter in range(max_iter):
            AC, grad, hessian_reg, f, mu = self.newton_step(AC, self.fC, attenuated_weights, lam, f_prev, mu)

            diff = torch.abs(f_prev - f)
            ray_grad_norm = torch.norm(grad, dim=1)
            grad_norm = ray_grad_norm.mean().item()
            max_grad_norm = ray_grad_norm.max().item()
            diff_norm = torch.norm(diff).mean().item()

            # Log the metrics for this iteration

            self.log["Objective"].append(f.clone())
            self.log["Gradient Norm"].append(grad_norm)
            self.log["Maximum Gradient Norm"].append(max_grad_norm)
            self.log["Function Difference Norm"].append(diff_norm)
            self.log["Mu"].append(mu.clone() if mu is not None else None)
            self.log["AC"].append(AC.clone())


            if self.verbose:
                print(f"Iteration {iter+1}: mean objective={float(f.mean()):.3e}, diff norm={diff_norm:.3e}, max grad norm={max_grad_norm:.3e}")

                # component-wise convergence check

                print(f"AC1: {AC.mean(dim=0)[0]}, AC2: {AC.mean(dim=0)[1]}, grad1: {grad.mean(dim=0)[0]}, grad2: {grad.mean(dim=0)[1]}")


            # The rays are independent, so a batch-average objective change can
            # become small while an individual ray is still far from stationary.
            # Stop only when every accepted post-step state satisfies the
            # stationarity tolerance.
            if max_grad_norm < gtol:
                    
                if self.verbose:
                    print(
                        f"Newton solver converged within per-ray gradient tolerance "
                        f"{gtol} after {iter+1} iterations"
                    )
                
                return {"AC": AC, "log": self.log}
            f_prev = f
        
        if self.verbose:
            print(f"Warning: Newton solver did not converge within {max_iter} iterations")
        return {"AC": AC, "log": self.log}

    def newton_step(self, AC: torch.Tensor, fC: torch.Tensor, attenuated_weights: torch.Tensor, lam: torch.Tensor, f_prev: torch.Tensor, mu: torch.Tensor | None)->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        
        ## Compute contrast attenuation

        attenuation_contrast = self.compute_attenuation(AC, fC)                            # [P x nE]

        
        nu = self.compute_lambda(attenuated_weights, attenuation_contrast)                            # [P x nBins]
        f_current, dfdnu, df2dnu2, df2dnudlambda = self.compute_objective(lam, nu)                                       # [P]

        ## Calculate derivatives

        J = self.compute_jacobian_lambda(attenuated_weights, fC, attenuation_contrast)                                     # [P x nBins x nMat]
        gA = self.compute_gradient(J, dfdnu)                                    # [P x nMat]
        H = self.compute_hessian_lambda(attenuated_weights, fC, attenuation_contrast)         # [P x nMat x nMat]        
        HA = self.compute_hessian(J, H, dfdnu, df2dnu2)                                 # [P x nMat x nMat]


        # Regularise Hessian

        

        HA = self.compute_regularised_hessian(HA, mu) if (self.LM_damping and mu is not None) else HA                                     # [P x nMat x nMat]


        # Compute Newton direction and line-search step size

        p_step = self.compute_newton_direction(HA, gA)                                          # [P x nMat]

        # Newton directions can be non-descent if HA is locally indefinite;
        # fall back to steepest descent for those rays.
        gp = torch.sum(gA * p_step, dim=1)
        non_descent = gp >= 0
        if torch.any(non_descent):
            p_step = p_step.clone()
            p_step[non_descent] = -gA[non_descent]

        alpha = self.armijo_backtracking(p_step, gA, f_current, lam, AC, attenuated_weights, fC) if self.Armijo_backtracking else torch.ones_like(f_current)  # [P]
        # alpha = torch.ones_like(f_current) # For training, we can skip line search and use full step to keep the graph simple
        # Update AC and LM damping

        AC_new = AC + alpha.unsqueeze(-1) * p_step
        attenuation_contrast_new = self.compute_attenuation(AC_new, fC)
        nu_new = self.compute_lambda(attenuated_weights, attenuation_contrast_new)
        f_new, dfdnu_new, df2dnu2_new, _ = self.compute_objective(lam, nu_new)
        J_new = self.compute_jacobian_lambda(attenuated_weights, fC, attenuation_contrast_new)
        gA_new = self.compute_gradient(J_new, dfdnu_new)
        H_new = self.compute_hessian_lambda(attenuated_weights, fC, attenuation_contrast_new)
        HA_new = self.compute_hessian(J_new, H_new, dfdnu_new, df2dnu2_new)
        HA_new = self.compute_regularised_hessian(HA_new, mu) if (self.LM_damping and mu is not None) else HA_new
        
        
        mu_new = self.levenberg_marquardt_update(mu, alpha, f_current, f_new, p_step, gA, HA_new) if (self.LM_damping and mu is not None) else mu  # [P]
        # mu_new = 1e-6 * torch.ones_like(mu)  # For training, we can skip LM update to keep the graph simple

        if self.verbose:

            # Hessian diagnostics

            det_reg = torch.mean(torch.linalg.det(HA)) if (self.LM_damping and mu is not None) else None
            det_unreg = torch.mean(torch.linalg.det(HA_new))


            eigenvalues_reg = torch.linalg.eigvals(HA) if (self.LM_damping and mu is not None) else None
            min_eig_reg = torch.min(eigenvalues_reg.real, dim=1).values if (eigenvalues_reg is not None) else None
            max_eig_reg = torch.max(eigenvalues_reg.real, dim=1).values if (eigenvalues_reg is not None) else None

            eigenvalues_unreg = torch.linalg.eigvals(HA_new)
            min_eig_unreg = torch.min(eigenvalues_unreg.real, dim=1).values
            max_eig_unreg = torch.max(eigenvalues_unreg.real, dim=1).values

            print(f"Hessian diagnostics:")
            if self.LM_damping:
                print(f"Average determinant of regularised Hessian: {det_reg}")
                if min_eig_reg is not None and max_eig_reg is not None:
                    print(f"Minimum eigenvalue of regularised Hessian: {torch.mean(min_eig_reg)}")
                    print(f"Maximum eigenvalue of regularised Hessian: {torch.mean(max_eig_reg)}")
            print(f"Average determinant of unregularised Hessian: {det_unreg}")
            print(f"Minimum eigenvalue of unregularised Hessian: {torch.mean(min_eig_unreg)}")
            print(f"Maximum eigenvalue of unregularised Hessian: {torch.mean(max_eig_unreg)}")
            if self.LM_damping and mu is not None:
                print(f"Unregularised Determinant / mu: {det_unreg / mu.mean()}")
            print()
        

        return AC_new, gA_new, HA_new, f_new, mu_new

    def grad_step(self, AB: torch.Tensor, AR: torch.Tensor, AC: torch.Tensor, AC_gt: torch.Tensor, attenuated_weights: torch.Tensor, lam: torch.Tensor, w_grad: torch.Tensor)->torch.Tensor:
        
        ## Compute contrast attenuation

        attenuation_contrast = self.compute_attenuation(AC, self.fC)                            # [P x nE]
        attenuation_background = self.compute_attenuation(AB, self.fB)                            # [P x nE]
        attenuation_signal = self.compute_attenuation(AR, self.fR)                            # [P x nE]

        
        nu = self.compute_lambda(attenuated_weights, attenuation_contrast)                            # [P x nBins]
        f, dfdnu, df2dnu2, df2dnudlambda = self.compute_objective(lam, nu)                                       # [P]

        ## Calculate derivatives

        J = self.compute_jacobian_lambda(attenuated_weights, self.fC, attenuation_contrast)                                     # [P x nBins x nMat]
        
        
        H = self.compute_hessian_lambda(attenuated_weights, self.fC, attenuation_contrast)         # [P x nMat x nMat]        
        HA = self.compute_hessian(J, H, dfdnu, df2dnu2)                                 # [P x nMat x nMat]


        # Regularise Hessian

        # Compute gradient w.r.t. weight network parameters

        gDtheta = self.compute_Jacobian_gradient(w_grad, attenuation_background, attenuation_signal, attenuation_contrast, J, dfdnu, df2dnu2, df2dnudlambda)  # [nParams x P x nMat]
        loss_grad = self.compute_objective_gradient(AC, AC_gt, HA, gDtheta)  # [nParams]

        return loss_grad
        
    def compute_attenuation(self, A, f) -> torch.Tensor:
        """Compute attenuation for given AC and basis functions."""


        assert A.ndim == 2 and f.ndim == 2 and A.shape[1] == f.shape[0], \
            f"A should be [P x nMat] and f should be [nMat x nE] but got A: {A.shape}, f: {f.shape}"

        return torch.exp(-(A @ f))  # [P x nE]

    def update_weights(self, w: torch.Tensor, Attenuation: torch.Tensor) -> torch.Tensor:
        """Updates the weights based on the current attenuation from the background AB."""

        assert w.ndim in (2, 3) and Attenuation.ndim == 2, \
            "w should be [P x nBins x nE] or [nBins x nE] and Attenuation should be [P x nE]"

        if w.ndim == 2:
            # Broadcast Attenuation to match w's shape: [P x 1 x nE]
            return w.unsqueeze(0) * Attenuation.unsqueeze(1)  # [P x nBins x nE]
        else:
            # Broadcast Attenuation to match w's shape: [P x 1 x nE]
            return w * Attenuation.unsqueeze(1)  # [P x nBins x nE]

    def compute_lambda(self, w, attenuation, residual_curve=None) -> torch.Tensor:
        """Compute the predicted detector rates (lambda) for given weights and attenuation."""

        assert w.ndim == 3 and attenuation.ndim == 2 and w.shape[0] == attenuation.shape[0], \
            "w should be [P x nBins x nE] and attenuation should be [P x nE]"
        
        if residual_curve is not None:
            residual_curve = self._validate_residual_curve(residual_curve, attenuation.shape[0], attenuation.device, attenuation.dtype)
            assert residual_curve is not None, "residual_curve validation failed"
            return self.dE * torch.einsum('pbe,pe, pe->pb', w, attenuation, torch.exp(-residual_curve))  # [P x nBins]
        else:
            return self.dE * torch.einsum('pbe,pe->pb', w, attenuation)  # [P x nBins]

    def compute_jacobian_lambda(self, w:torch.Tensor, f:torch.Tensor, attenuation:torch.Tensor) -> torch.Tensor:

        """Compute the Jacobian of lambda w.r.t. AC (A) for given weights and basis functions: the formula for the jacobian is:
        
        J_pbm = -int w_b(e) * f_m(e) * exp(-(AC_p * fC_m)) dE

        where w_b(e) are the attenuated weights:

        w_b(e) = weights[b,e] * exp(-(sum_{m}(AB_p * fB_m(e))))
        
        """
        assert w.ndim == 3 and f.ndim == 2 and attenuation.ndim == 2, \
            "w should be [P x nBins x nE], f should be [nMat x nE], and attenuation should be [P x nE]"

        return -torch.einsum('pbe,me,pe->pbm', w, f, attenuation) * self.dE  # [P x nBins x nMat]
    
    def compute_hessian_lambda(self, w:torch.Tensor, f:torch.Tensor, attenuation:torch.Tensor)->torch.Tensor:
        """Compute the Hessian of lambda w.r.t. AC (A) for given weights and basis functions: the formula for the hessian is:
        
        H_pbmn = int w_b(e) * f_m(e) * f_n(e) * exp(-(AC_p * fC_m)) dE

        where w_b(e) are the attenuated weights:

        w_b(e) = weights[b,e] * exp(-(sum_{m}(AB_p * fB_m(e))))
        
        """
        
        assert w.ndim == 3 and f.ndim == 2 and attenuation.ndim == 2, \
            "w should be [P x nBins x nE], f should be [nMat x nE], and attenuation should be [P x nE]"

        return torch.einsum('pbe,me,ne,pe->pbmn', w, f, f, attenuation) * self.dE  # [P x nBins x nMat x nMat]
  
    def compute_objective(self, lam:torch.Tensor, nu:torch.Tensor, alpha_lasso: float = 1.0)->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the objective function value for given predicted and true lambda: the formula for the objective is:

        f = sum_{bins} (lambda_contrast_b - lambda0_b + lambda0_b * log(lambda0_b / lambda_contrast_b))

        where lambda0_b is the predicted lambda and lambda_contrast_b is the true lambda.
        """

        assert lam.ndim == 2 and nu.ndim == 2, \
            "lam and nu should be [P x nBins]"
        

        def KLsym(lam, nu) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q = lam / nu.clamp(min=self.floor)             # [P x nBins]
            f = 0.5 * ((lam  - nu) * torch.log(q)).sum(-1)                                                # [P] 
            dfdnu = 0.5 * (1 - q - torch.log(q))                                                                     # [P, nBins]
            df2dnu2 = 0.5 * ((lam + nu) / (nu.clamp(min=self.floor)).pow(2))         # [P, nBins]
            df2dnudlambda = -0.5 * ((nu + lam) / (nu * lam).clamp(min=self.floor))        # [P, nBins]
            return f, dfdnu, df2dnu2, df2dnudlambda

        def KL(lam, nu) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q = lam / nu.clamp(min=self.floor)             # [P x nBins]
            f = (torch.xlogy(lam, q) + nu - lam).sum(-1)  # [P]
            dfdnu = 1 - q  # [P, nBins]
            df2dnu2 = lam / (nu.clamp(min=self.floor)).pow(2)  # [P, nBins]
            df2dnudlambda = -1 / (nu.clamp(min=self.floor))  # [P, nBins]
            return f, dfdnu, df2dnu2, df2dnudlambda

        def L2(lam, nu) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            f = 0.5 * ((lam - nu).pow(2)).sum(-1)  # [P]
            dfdnu = nu - lam  # [P, nBins]
            df2dnu2 = torch.ones_like(nu)  # [P, nBins]
            df2dnudlambda = torch.zeros_like(nu)  # [P, nBins]
            return f, dfdnu, df2dnu2, df2dnudlambda
        
        def L1(lam, nu) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            f = torch.abs(lam - nu).sum(-1)  # [P]
            dfdnu = torch.sign(nu - lam)  # [P, nBins]
            df2dnu2 = torch.zeros_like(nu)  # [P, nBins]
            df2dnudlambda = torch.zeros_like(nu)  # [P, nBins]
            return f, dfdnu, df2dnu2, df2dnudlambda

        if self.objective == "KLsym":
            return KLsym(lam, nu)
        elif self.objective == "KL":
            return KL(lam, nu)

        elif self.objective == "L2":
            return L2(lam, nu)
        
        elif self.objective == "L1":
            return L1(lam, nu)
        elif self.objective == "L1smooth":
            # Smooth L1 loss (Huber loss) with delta=1.0
            delta = 1.0
            diff = lam - nu
            abs_diff = torch.abs(diff)
            f = torch.where(abs_diff < delta, 0.5 * diff.pow(2), delta * (abs_diff - 0.5 * delta)).sum(-1)  # [P]
            dfdnu = torch.where(abs_diff < delta, -diff, -delta * torch.sign(diff))  # [P, nBins]
            df2dnu2 = torch.where(abs_diff < delta, torch.ones_like(nu), torch.zeros_like(nu))  # [P, nBins]
            df2dnudlambda = torch.zeros_like(nu)  # [P, nBins]
            return f, dfdnu, df2dnu2, df2dnudlambda
        elif self.objective == "Lasso":
            L2out = L2(lam, nu)
            L1out = L1(lam, nu)
            
            return tuple(L2out[i] + alpha_lasso * L1out[i] for i in range(len(L2out)))  # type: ignore


        else:
            raise NotImplementedError(f"Objective '{self.objective}' not implemented yet")

    def compute_gradient(self, J:torch.Tensor, dfdnu:torch.Tensor)->torch.Tensor:
        """Compute the gradient of the objective w.r.t. AC using the Jacobian and the contrast between predicted and true lambda. The gradient is given by:

        gA = 1/2 sum_{bins} r_b * J_pbm

        where r_b = 1 - q_b - log(q_b) and q_b = lambda0_b / lambda_contrast_b
        """
        
        assert J.ndim == 3 and J.shape[1] == self.nBins and J.shape[2] == self.nMat, \
            f"J should be [{self.P} x {self.nBins} x {self.nMat}], but got {J.shape}"
        assert dfdnu.ndim == 2 and dfdnu.shape[0] == J.shape[0] and dfdnu.shape[1] == self.nBins, \
            f"dfdnu should be [{self.P} x {self.nBins}], but got {dfdnu.shape}"

        gA = torch.einsum('pb, pbm -> pm', dfdnu, J)  # [P x nMat]
        return gA

    def compute_hessian(self, J:torch.Tensor, H:torch.Tensor, dfdnu:torch.Tensor, df2dnu2:torch.Tensor)->torch.Tensor:
        """Compute the Hessian of the objective w.r.t. AC using the Jacobian, the Hessian of lambda, and the contrast between predicted and true lambda. The Hessian is given by:

        HA_pmn = 1/2 sum_{bins} (r_b * H_pbmn + s_b * J_pbm * J_pbn)

        where r_b = 1 - q_b - log(q_b) and s_b = q_b - 1
        """

        HA =  torch.einsum('pb, pbmn -> pmn', dfdnu, H) + torch.einsum('pb, pbm, pbn -> pmn', df2dnu2, J, J)  # [P x nMat x nMat]

        return HA

    def compute_regularised_hessian(self, HA:torch.Tensor, mu:torch.Tensor)->torch.Tensor:
        """Compute the regularised Hessian by adding LM damping to the diagonal."""


        assert (HA.ndim == 3 and HA.shape[1] == HA.shape[2]) and (HA.shape[1] == self.nMat), \
            "HA should be [P x nMat x nMat]"

        if isinstance(mu, float) or (isinstance(mu, torch.Tensor) and mu.ndim == 0):
            return HA + mu * torch.eye(self.nMat, device=HA.device, dtype=HA.dtype)  # [P x nMat x nMat]
        elif mu.ndim == 1 and mu.shape[0] == self.P:
            return HA + mu.unsqueeze(-1).unsqueeze(-1) * torch.eye(self.nMat, device=HA.device, dtype=HA.dtype)  # [P x nMat x nMat]
        else:
            raise ValueError("mu should be either a scalar or a vector of shape [P]")

    def compute_newton_direction(self, H:torch.Tensor, g:torch.Tensor)->torch.Tensor:
        """Compute the Newton direction by solving the linear system H @ p = -g."""

        assert H.ndim == 3 and H.shape[1] == H.shape[2] and H.shape[1] == self.nMat, \
            "H should be [P x nMat x nMat]"
        assert g.ndim == 2 and g.shape[1] == self.nMat, \
            "g should be [P x nMat]"

        return torch.linalg.solve(H, -g.unsqueeze(-1)).squeeze(-1)  # [P x nMat]

    def compute_objective_gradient(self, AC: torch.Tensor, AC_gt: torch.Tensor, H:torch.Tensor, gDtheta:torch.Tensor) -> torch.Tensor:
        
        """Compute the gradient of the objective w.r.t. the weight network parameters theta using the chain rule:

        gDtheta = sum_{bins} (r_b * H_pbmn * J_pbm * dnu/dtheta + s_b * J_pbm * J_pbn * dnu/dtheta)

        where r_b = 1 - q_b - log(q_b) and s_b = q_b - 1
        """

        assert AC.ndim == 2 and AC_gt.ndim == 2 and AC.shape[1] == AC_gt.shape[1] and AC.shape[0] == AC_gt.shape[0], \
            "AC and AC_gt should be [P x nMat] and have the same shape"
        assert gDtheta.ndim == 3 and gDtheta.shape[-1] == self.nMat, \
            "gDtheta should be [nParams x P x nMat]"
        assert H.ndim == 3 and H.shape[-2] == H.shape[-1] and H.shape[-1] == self.nMat, \
            "H should be [P x nMat x nMat]"

        dAtheta = -torch.linalg.solve(H, gDtheta.permute(1, 2, 0)).permute(2, 0, 1)  # [nParams x P x nMat]
        diff = AC - AC_gt  # [P x nMat]
        return torch.einsum('tpm, pm -> t', dAtheta, diff)

    def compute_Jacobian_gradient(self, w_grad:torch.Tensor, attenuation_background, attenuation_signal, attenuation_contrast, J, dfdnu, df2dnu2, df2dnudlambda)->torch.Tensor:
        """Compute the gradient of the objective with respect to the parameters theta of the weight network using the chain rule:"""


        attenuated_gradient_wegihts = torch.einsum('tbe, pe -> tpbe', w_grad, attenuation_background)  # [nParams x P x nBins]
        # Chain rule for g_i = sum_b alpha_b J_bi:
        #   d g_i / d theta = sum_b [
        #       J_bi * (d alpha_b / d lambda_b) * d lambda_b / d theta
        #     + J_bi * (d alpha_b / d nu_b)     * d nu_b     / d theta
        #     + alpha_b * d J_bi / d theta
        #   ]
        # Here,
        #   d lambda / d theta  ~ exp(-AB·fB) exp(-AR·fR) d omega / d theta
        #   d nu     / d theta  ~ exp(-AB·fB) exp(-AC·fC) d omega / d theta
        #   d J / d theta       ~ -f_i exp(-AB·fB) exp(-AC·fC) d omega / d theta
        term_lambda = torch.einsum(
            'pbm, pb, pe, tpbe -> tpm',
            J,
            df2dnudlambda,
            attenuation_signal,
            attenuated_gradient_wegihts,
        )
        term_nu = torch.einsum(
            'pbm, pb, pe, tpbe -> tpm',
            J,
            df2dnu2,
            attenuation_contrast,
            attenuated_gradient_wegihts,
        )
        term_jacobian = -torch.einsum(
            'pb, me, pe, tpbe -> tpm',
            dfdnu,
            self.fC,
            attenuation_contrast,
            attenuated_gradient_wegihts,
        )
        gDtheta = (term_lambda + term_nu + term_jacobian) * self.dE  # [nParams x P x nMat]
        
        return gDtheta
        
    def armijo_backtracking(
            self, p_step: torch.Tensor, 
            gA: torch.Tensor, 
            Psi: torch.Tensor, 
            lambda0: torch.Tensor, 
            AC: torch.Tensor, 
            attenuated_weights: torch.Tensor, 
            fC: torch.Tensor, 
            armijo_c1=1e-2,
            max_backtracks=20
        )->torch.Tensor:
        """Per-ray Armijo backtracking for decoupled line-wise objectives."""

        alpha = torch.ones(Psi.shape[0], device=AC.device, dtype=AC.dtype)  # [P]
        gp = torch.sum(gA * p_step, dim=1)  # [P]

        # Only rays with descent directions are eligible for Armijo updates.
        descent = gp < 0
        accepted = ~descent

        for i in range(max_backtracks):
            AC_trial = AC + alpha.unsqueeze(-1) * p_step

            attenuation_contrast = self.compute_attenuation(AC_trial, fC)                   # [P x nE]
            lambda_contrast = self.compute_lambda(attenuated_weights, attenuation_contrast) # [P x nBins]
            Psi_trial, _, _, _ = self.compute_objective(lambda0, lambda_contrast)

            armijo_rhs = Psi + armijo_c1 * alpha * gp
            newly_accepted = (Psi_trial <= armijo_rhs) & descent & (~accepted)
            accepted = accepted | newly_accepted

            if bool(torch.all(accepted)):
                break

            alpha = torch.where(accepted, alpha, alpha * 0.5)

        # Keep alpha=0 only for truly failed rays (rare once descent fallback is active).
        alpha = torch.where(accepted, alpha, torch.zeros_like(alpha))

        if self.verbose:
            accepted_ratio = float(accepted.float().mean())
            mean_alpha = float(alpha.mean())
            print(
                f"Armijo: accepted={accepted_ratio:.3f}, mean_alpha={mean_alpha:.3e}, "
                f"backtracks={i+1}"
            )

        return alpha

    def levenberg_marquardt_update(self, mu: torch.Tensor, alpha:torch.Tensor, f_current:torch.Tensor, f_new:torch.Tensor, p_step:torch.Tensor, gA:torch.Tensor, HA: torch.Tensor)->torch.Tensor:
        """Update the LM damping parameter based on the ratio of actual to predicted reduction."""



        assert f_current.ndim == 1 and f_current.shape[0] == self.P, \
            "f_current should be [P]"
        assert f_new.ndim == 1 and f_new.shape[0] == self.P, \
            "f_new should be [P]"


        actual_reduction = f_current - f_new  # [P]

        s = p_step * alpha.unsqueeze(-1)
        predicted_reduction = -torch.einsum('pm,pm->p', gA, s) - 0.5 * torch.einsum('pm,pmn,pn->p', s, HA, s)  # [P]


        rho = actual_reduction / predicted_reduction.clamp(min=self.floor)  # [P]


        mu = torch.where(rho > 0.75, mu * 0.1, mu)
        mu = torch.where(rho < 0.25, mu * 10.0, mu)
        mu = mu.clamp(1e-16, 1e16)

        if self.verbose:
            print(f"Mean Actual reduction: {float(actual_reduction.mean()):.3e}, Mean Predicted reduction: {float(predicted_reduction.mean()):.3e}")
            print(f"LM update: mean_rho={float(rho.mean()):.3e}, mean_mu={float(mu.mean()):.3e}")

        return mu

    def plot_metrics(self, output_dir: str | None = None):
        """Plot the logged metrics from the Newton solver iterations."""
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot Objective
        axs[0, 0].plot(self.log["Objective"].cpu().detach().numpy().mean(axis=1))
        axs[0, 0].set_title("Objective Function Value")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Objective Value")
        axs[0, 0].set_yscale('log')

        # Plot Gradient Norm
        axs[0, 1].plot(np.array(self.log["Gradient Norm"].cpu().detach().numpy()))
        axs[0, 1].set_title("Gradient Norm")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Norm of Gradient")
        axs[0, 1].set_yscale('log')

        # Plot Function Difference Norm
        axs[1, 0].plot(np.array(self.log["Function Difference Norm"].cpu().detach().numpy()))
        axs[1, 0].set_title("Function Difference Norm")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Norm of Function Difference")
        axs[1, 0].set_yscale('log')

        # Plot Mu
        axs[1, 1].plot(np.array(self.log["Mu"].cpu().detach().numpy()).mean(axis=1))
        axs[1, 1].set_title("Levenberg-Marquardt Damping Parameter (Mu)")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Mu Value")
        axs[1, 1].set_yscale('log')

        plt.tight_layout()
        if output_dir:
            import os
            print(f"Saving Newton solver log plot to {output_dir}/newton_solver_log_{self.objective}.png")
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/newton_solver_log_{self.objective}.png") if output_dir else plt.show()
        plt.close(fig)

    def plot_AC_optim(self, AB: torch.Tensor, AR: torch.Tensor, output_dir: str | None = None) -> None:
        """Plot the evolution of AC during the Newton solver iterations."""
        import matplotlib.pyplot as plt

        AC_history = torch.stack(self.log["AC"], dim=0)  # [iterations x P x nMat]
        iterations = AC_history.shape[0]


        # Pick a few rays to visualize
        num_rays_to_plot = min(16, self.P) 

        assert num_rays_to_plot % 4 == 0, "Number of rays to plot should be a multiple of 4 for subplot arrangement."

        rays_to_plot = torch.linspace(0, self.P - 1, num_rays_to_plot, dtype=torch.long)

        fig, axs = plt.subplots(num_rays_to_plot // 4, 4, figsize=(4 * (num_rays_to_plot // 4), 12), sharex=True, sharey=True)

        # Create grid of AC values to get colormap for objective for each ray:
        N = 100

        MinAC = AC_history.view(-1, AC_history.shape[-1]).min(dim=0).values  # [nMat]
        MaxAC = AC_history.view(-1, AC_history.shape[-1]).max(dim=0).values  # [nMat]
        

        ACtrials = torch.meshgrid(
            torch.linspace(MinAC[0], MaxAC[0], N),
            torch.linspace(MinAC[1], MaxAC[1], N)
        ) # [N x N]
        ACtrials = torch.stack(ACtrials, dim=-1)  # [N x N x 2]
        

        f = torch.zeros((N, N), dtype=torch.float32)

        if self.normalize_weights:
            weights = self._normalize_weights(self.weights)  # [nBins x nE]
        else:
            weights = self.weights  # [nBins x nE]

        attenuation_background = self.compute_attenuation(AB, self.fB)
        attenuation_signal = self.compute_attenuation(AR, self.fR)
        attenuated_weights = self.update_weights(weights, attenuation_background)
        lam = self.compute_lambda(attenuated_weights, attenuation_signal) 


        for idx, ray in enumerate(rays_to_plot):

            lam_ray = lam[ray, :].unsqueeze(0)  # [1 x nBins]
            weights_ray = attenuated_weights[ray, :, :].unsqueeze(0)  # [1 x nBins x nE]
            for i in range(N):
                for j in range(N):
                    AC = ACtrials[i, j, :].unsqueeze(0)  # [1 x 2]

                    attenuation_contrast = self.compute_attenuation(AC, self.fC)
                    nu = self.compute_lambda(weights_ray, attenuation_contrast)
                    out = self.compute_objective(lam_ray, nu)

                    f[i, j] = torch.log(out[0])


            ax = axs[idx // 4, idx % 4]
            c = ax.contourf(ACtrials[..., 0].cpu().numpy(), ACtrials[..., 1].cpu().numpy(), f.numpy(), levels=10, cmap='viridis')
            
            # add path

            path = AC_history[:, ray, :].detach().cpu().numpy()  # [iterations x 2]

            ax.plot(path[:, 0], path[:, 1], 'w-o', linewidth=0.4, markersize=3)
            ax.scatter(path[0, 0], path[0, 1], c='red', s=40, label='start')
            ax.scatter(path[-1, 0], path[-1, 1], c='cyan', s=40, label='end')
            ax.legend()
                        
            
            fig.colorbar(c, ax=ax, label='log(Objective)')
            ax.set_title(f"Ray {ray.item()}: Objective Evolution")
            ax.set_xlabel("AC1")
            ax.set_ylabel("AC2")

        plt.tight_layout()
        if output_dir:
            import os
            print(f"Saving AC optimization plot to {output_dir}/AC_optimization_{self.objective}.png")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/AC_optimization_{self.objective}.png")
        else:
            plt.show()
        plt.close(fig)

    def plot_AC_optim_with_curves(self, AB, AR, fR, fC, output_dir=None):
        """Works the same as plot_AC_optim but only picks one ray and then plot the corresponding attenuation curves for the signal 
        and predicted contrast for the optimal AC values."""
        fs = 26
        import matplotlib.pyplot as plt

        AC_history = torch.stack(self.log["AC"], dim=0)  # [iterations x P x nMat]
        iterations = AC_history.shape[0]

        # Pick a single ray to visualize
        ray = 0

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 15), sharex=False)
    
        # Create grid of AC values to get colormap for objective for the selected ray:
        N = 100

        MinAC = AC_history.view(-1, AC_history.shape[-1]).min().item()
        MaxAC = AC_history.view(-1, AC_history.shape[-1]).max().item()

        # Add some padding to the min and max values for better visualization
        padding = 0.5 * (MaxAC - MinAC)
        MinAC -= padding
        MaxAC += padding

        # Build the grid on the same device/dtype as the solver tensors (e.g. CUDA)
        ACtrials = torch.meshgrid(
            torch.linspace(MinAC, MaxAC, N, device=AB.device, dtype=AB.dtype),
            torch.linspace(MinAC, MaxAC, N, device=AB.device, dtype=AB.dtype),
            indexing="ij",
        )  # [N x N]
        ACtrials = torch.stack(ACtrials, dim=-1)  # [N x N x 2]

        f = torch.zeros((N, N), dtype=torch.float32)

        if self.normalize_weights:
            weights = self._normalize_weights(self.weights)  # [nBins x nE]
        else:
            weights = self.weights  # [nBins x nE]

        attenuation_background = self.compute_attenuation(AB, self.fB)
        attenuation_signal = self.compute_attenuation(AR, self.fR)
        attenuated_weights = self.update_weights(weights, attenuation_background)
        lam = self.compute_lambda(attenuated_weights, attenuation_signal)

        lam_ray = lam[ray, :].unsqueeze(0)  # [1 x nBins]
        weights_ray = attenuated_weights[ray, :, :].unsqueeze(0)  # [1 x nBins x nE]
        for i in range(N):
            for j in range(N):
                AC = ACtrials[i, j, :].unsqueeze(0)  # [1 x 2]

                attenuation_contrast = self.compute_attenuation(AC, self.fC)
                nu = self.compute_lambda(weights_ray, attenuation_contrast)
                out = self.compute_objective(lam_ray, nu)
                # print(f"AC: {AC}, Objective: {out[0].item()}")
                f[i, j] = torch.log(out[0].clamp_min(self.floor)).item()

        # Plot the contour of the objective function for the selected ray
        ax = axs[0]
        c = ax.contourf(ACtrials[..., 0].cpu().numpy(), ACtrials[..., 1].cpu().numpy(), f.numpy(), levels=10, cmap='viridis')
        path = AC_history[:, ray, :].detach().cpu().numpy()
        ax.plot(path[:, 0], path[:, 1], 'w-o', linewidth=2, linestyle='--', markersize=8)
        ax.scatter(path[0, 0], path[0, 1], c='red', s=200, label='start')
        ax.scatter(path[-1, 0], path[-1, 1], c='cyan', s=200, label='end')
        ax.legend(loc='upper right', fontsize=20)
        cbar = fig.colorbar(c, ax=ax,)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(rf'$\log(\mathsf{{\Psi}})$', fontsize=fs)
        ax.set_title(
            f"Optimization trajectory with\n"
            rf"$A_{{B}}$: ({AB[ray, 0].detach().cpu().numpy():.2f}, {AB[ray, 1].detach().cpu().numpy():.2f}) and "
            rf"$A_{{S}}$: ({AR[ray, 0].detach().cpu().numpy():.2f}, {AR[ray, 1].detach().cpu().numpy():.2f})", 
            fontsize=fs
        )
        ax.set_xlabel("Normalized Iodine", fontsize=20)
        ax.set_ylabel("Normalized Gadolinium", fontsize=20)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=14)
        # Plot the attenuation curves for the signal and predicted contrast for the optimal AC values
        optimal_AC = AC_history[-1, ray, :].unsqueeze(0)  # [1 x 2]

        att_signal = torch.einsum('pm,me->pe', AR[ray, :].unsqueeze(0), fR.to(AR.device))  # [1 x nE]
        att_contrast = torch.einsum('pm,me->pe', optimal_AC.to(AR.device), fC.to(AR.device))  # [1 x nE]

        energies = torch.linspace(15, 120, self.nE)

        axs[1].plot(energies.detach().cpu().numpy(), att_signal.squeeze().detach().cpu().numpy(), label=rf'Signal: $\hat{{f}}_{{S}}$= {AR[ray, 0].item():.2f} $\hat{{f}}_{{Wa}}$ + {AR[ray, 1].item():.2f} $\hat{{f}}_{{I}}$', 
                    color='k', linestyle='-.', linewidth=1.0)
        axs[1].plot(energies.detach().cpu().numpy(), att_contrast.squeeze().detach().cpu().numpy(), label=rf'Predicted Contrast: $\hat{{f}}_{{K}}$= {optimal_AC[0, 0].item():.2f} $\hat{{f}}_{{I}}$ + {optimal_AC[0, 1].item():.2f} $\hat{{f}}_{{Gd}}$', 
                    color='r', linestyle='--', linewidth=1.0)
        axs[1].set_title(f"Attenuation Curves", fontsize=fs)
        axs[1].set_xlabel("Energy [keV]", fontsize=fs)
        axs[1].set_ylabel("Normalized attenuation", fontsize=fs)
        axs[1].legend(loc = "upper right", fontsize=20)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        if output_dir:
            import os
            print(f"Saving AC optimization with curves plot to {output_dir}/AC_optimization_with_curves_{self.objective}.png")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/AC_optimization_with_curves_{self.objective}.png")
        else:
            plt.show()
        plt.close(fig)

        

    def _plot_weights(self, weights: torch.Tensor, title: str, output_dir: str | None = None):
        """Plot the weights for each energy bin."""
        import matplotlib.pyplot as plt

        assert weights.ndim == 2 and weights.shape[0] == self.nBins and weights.shape[1] == self.nE, \
            "weights should be [nBins x nE]"

        plt.figure(figsize=(10, 6))
        for b in range(self.nBins):
            plt.plot(weights[b, :].detach().cpu().numpy(), label=f'Bin {b}')
        plt.title(title)
        plt.xlabel('Energy Index')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid()
        if output_dir:
            import os
            print(f"Saving weights plot to {output_dir}/{title.replace(' ', '_')}.png")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
        else:
            plt.show()
        plt.close()
    

class ResidualSolver(NewtonSolver):
    """A Newton solver that includes an optional residual attenuation curve in the weight update step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_residual(self, AR, fR, AC, fC):

        # use linear approximation to compute residual attenuation curve

        residual_curve = torch.einsum('pm,me->pe', AC, fC) - torch.einsum('pm,me->pe', AR, fR)

        return residual_curve

    def outer_update(self, residual, AB, AR, AC, alpha=0.5):
        """Update the weights based on the current attenuation from the background AB and the residual curve."""
        assert residual.ndim == 2 and residual.shape[0] == AB.shape[0], \
            "residual should be [P x nE] and match the number of rays"

        sol = self.newton_solve(AB, AR, AC_init=AC, residual_curve=residual)
        AC = sol["AC"]
        residual = alpha * (self.calculate_residual(AR, self.fR, AC, self.fC) - residual) + residual  # Update residual with a weighted average to stabilize convergence


        return AC, residual

    def residual_solve(self, AB, AR, AC_init=None, max_outer_iter=5, tol=1e-3):
        """Solve for AC and the residual curve iteratively."""
        if AC_init is None:
            AC_init = torch.zeros((AB.shape[0], self.nMat), device=AB.device, dtype=AB.dtype)

        AC = AC_init
        residual = torch.zeros((AB.shape[0], self.nE), device=AB.device, dtype=AB.dtype)

        for outer_iter in range(max_outer_iter):
            AC_prev = AC.clone()
            AC, residual = self.outer_update(residual, AB, AR, AC)

            # Check convergence based on change in AC
            change_norm = torch.norm(AC - AC_prev)
            if change_norm < tol:
                if self.verbose:
                    print(f"Residual solver converged after {outer_iter + 1} outer iterations with change norm {change_norm:.3e}")
                break
        else:
            if self.verbose:
                print(f"Warning: Residual solver did not converge within {max_outer_iter} outer iterations")

        return {"AC": AC, "residual": residual}

