from email import generator

import torch
import odl
from optim.solvers import NewtonSolver


class MDOperator(NewtonSolver):
    def __init__(
            self, 
            fB: torch.Tensor, 
            fR: torch.Tensor, 
            fC: torch.Tensor, 
            weights: torch.Tensor, 
            dE: float=1, 
            floor: float = 1e-6) -> None:
        super().__init__(fB, fR, fC, weights, dE, floor)
    
    

    def __call__(self, A):

        assert A.shape[1] == 4 and A.dim() == 2, "Input should be [P x 4]"

        self.AB = A[:, 0:2]
        self.AR = A[:, 2:4]

        # Solve

        sol = self.newton_solve(self.AB, self.AR, AC_init=None, weights=self.weights)

        self.AC = sol["AC"]
        self.hess = sol["hess"]


        return sol
        

class ParameterizedMDOperator(MDOperator):
    def __init__(
            self, 
            WeightNetwork: torch.nn.Module, 
            fB: torch.Tensor, 
            fR: torch.Tensor, 
            fC: torch.Tensor, 
            dE: float=1, 
            floor: float = 1e-6,
            batch_size: int = 2048,
            AR_threshold: float = 1e-1,
            ):
        
        weights = WeightNetwork.w_ref  # [nBins x nE]

        super().__init__(fB, fR, fC, weights, dE, floor)
        self.weight_net = WeightNetwork
        self.batch_size = batch_size
        self.AR_threshold = AR_threshold

    def get_weights(self):
        x = torch.linspace(0, 1, self.nE)  # [nE]
        weights = self.weight_net(x)  # [nBins x nE]
        return weights

    def _set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def create_generator(self, A):
        P = A.shape[0]  # number of pixels/rays

        # Create generator

        n_batches = (P + self.batch_size - 1) // self.batch_size
       
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, P)
            yield A[start:end, :]  # Yield a batch of rays [4 x batch_size]
    
    def mask_sinogram(self, AR):
        AR_mean = torch.mean(AR, dim=0)

        mask1 = AR[:, 0] > AR_mean[0] * 1e-2
        mask2 = AR[:, 1] > AR_mean[1] * 1e-2

        mask = mask1 | mask2



        return mask

    def forward(self, A):

        assert A.dim() == 2 and A.shape[1] in (4, 6), "Input should be [P x 4] or [P x 6]"

        # get weights

        weights = self.get_weights()  # [nBins x nE]

        AB = A[:, 0:2]
        AR = A[:, 2:4]

        # Solve

        sol = self.newton_solve(AB, AR, AC_init=None, weights=weights)
        AC = sol["AC"]

        return AC




    def __call__(self, A, implicit_gradient=False):
        """Forward pass through the parameterized MD operator.  This will compute the predicted AC values and store intermediate results for gradient computation."""

        self.grad = False
        self.AC_gt = None
        self.w_grad = None

        assert A.dim() == 3 and A.shape[-1] in (4, 6), "Input should be [nAngles x nDet x 4] or [nAngles x nDet x 6]"

        gt_included = A.shape[-1] == 6  # Check if ground truth AC is included in the input
        if implicit_gradient:
            assert gt_included, "Ground-truth AC must be included when implicit_gradient=True"

        weights = self.get_weights()  # [nBins x nE]

        # Step 1: flatten images. A is [nAngles x nDet x 4], we want to reshape to [4 x nRays]

        sz = A.shape
        A = A.view(-1, 4 if not gt_included else 6)  # [P x 4] or [P x 6], where P = nAngles * nDet is the number of rays
        P = A.shape[0]

            
        self.AB = A[:, 0:2]
        self.AR = A[:, 2:4]

        # Mask out rays with negligible signal to speed up computation (these will have near-zero gradient anyway, so we can skip them without affecting optimization)

        mask = self.mask_sinogram(A)  # Only compute for rays with significant signal
        # mask = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)  # For now, we will not mask any rays. This can be changed later.

        A = A[mask, :]  # [P' x 4], where P' is the number of rays that pass the threshold
        P_masked = A.shape[0]  # Update number of rays after filtering
        AC_pred_masked = torch.zeros((P_masked, self.nMat), device=A.device)  # [P' x nMat], for storing results of the processed rays
        hess_masked = torch.zeros((P_masked, self.nMat, self.nMat), device=A.device)  # [P' x nMat x nMat], for storing Hessians of the processed rays
        grad = None

        if implicit_gradient:
            w_grad = self.weight_net.get_gradient_weights().detach()  # [nParams x nBins x nE]
            reg_grad = self.weight_net.reg_grad().detach()  # [nParams]
            data_grad_total = torch.zeros_like(reg_grad)

        g = self.create_generator(A)  # Generator that yields batches of rays [4 x batch_size]


        offset = 0
        for i, A_batch in enumerate(g):
            # print(f"Processing batch {i+1}/{(P_masked + self.batch_size - 1) // self.batch_size} with shape {A_batch.shape}...")
            # Here you would typically compute the forward pass for this batch and store results for gradient computation
            # For demonstration, we will just print the shape of the batch


            AB = A_batch[:, 0:2]
            AR = A_batch[:, 2:4]

            if implicit_gradient:
                AC_gt = A_batch[:, 4:6] if gt_included else None  # Ground truth AC if available
                self._setup_gradient_computation(AC_gt, w_grad)  # Setup gradient computation for this batch

            # Solve

            sol = self.newton_solve(AB, AR, AC_init=None, weights=weights)
            AC = sol["AC"]
            hess = sol["hess"]  # [batch_size x nMat x nMat]

            if self.verbose:
                objective = torch.mean(sol["f"])  # Objective function value for this batch
                data_loss = torch.nn.MSELoss(reduction='mean')(AC, AC_gt) if implicit_gradient and AC_gt is not None else None
                if data_loss is None:
                    print(f"Batch {i+1}: Objective = {objective.item()}")
                else:
                    print(f"Batch {i+1}: Objective = {objective.item()}, Data Loss = {data_loss.item()}")

            # Store the results for this batch


            if implicit_gradient:
                data_grad = sol["loss_grad"].detach()  # [nParams], gradient of loss w.r.t. weight network parameters
                data_grad_total += data_grad

            batch_len = A_batch.shape[0]
            AC_pred_masked[offset:offset + batch_len, :] = AC
            hess_masked[offset:offset + batch_len, :, :] = hess
            offset += batch_len

        if implicit_gradient:
            data_grad_total = data_grad_total / max(P_masked * self.nMat, 1)
            grad = (data_grad_total + reg_grad).detach()


        # Set unprocessed rays to zero and reshape back to [nAngles x nDet x nMat]

        AC_pred = torch.zeros((P, self.nMat), device=A.device)  # [nRays x nMat]
        AC_pred[mask, :] = AC_pred_masked  # Fill in the predicted AC values for rays that passed the threshold
        hess = torch.zeros((P, self.nMat, self.nMat), device=A.device)  # [nRays x nMat x nMat]
        hess[mask, :, :] = hess_masked  # Fill in the Hessian values for rays that passed the threshold

        # Reshape

        AC_pred = AC_pred.view(sz[0], sz[1], self.nMat)  # [nAngles x nDet x nMat] 
        hess = hess.view(sz[0], sz[1], self.nMat, self.nMat)  # [nAngles x nDet x nMat x nMat]
        self.hess = hess  # Store Hessian for later use in gradient computation

        return {"AC": AC_pred, "hessian_reg": hess, "grad": grad, "mask": mask.view(sz[0], sz[1])}

