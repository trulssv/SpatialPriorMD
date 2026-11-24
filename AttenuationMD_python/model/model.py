
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, functional_call


# ---------------------------------------------------------------------------
# Weight network
# ---------------------------------------------------------------------------

class WeightNetwork(nn.Module):
    """
    MLP that maps a normalised energy grid [nE] to weight functions [nBins x nE].

    Parameters
    ----------
    nE     : number of energy points (default 106)
    nBins  : number of detector energy bins (default 8)
    hidden : hidden layer width (default 128)
    """

    def __init__(
        self,
        w_ref: torch.Tensor,  # [nBins x nE] reference weights (default None → uniform)
        nE:     int = 106,
        nBins:  int = 8,
        hidden: int = 32,
        lambda_reg: float = 0.01,
    ) -> None:
        super().__init__()
        self.w_ref = w_ref # [nBins x nE]
        self.nE    = nE
        self.nBins = nBins
        self.lambda_reg = lambda_reg

        # Shared backbone: processes each energy point independently
        # Input is a single scalar (normalised energy)
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Head: maps backbone features to nBins weight values
        self.head = nn.Linear(hidden, nBins * 2)

        self._init_weights()

    def get_energy_grid(
        self, 
    ) -> torch.Tensor:
        """
        Return normalised energy values in [0, 1] for the range [Emin, Emax].

        The grid has nE = int((Emax - Emin) / dE) + 1 points (inclusive).
        Normalisation: E_norm = (E - Emin) / (Emax - Emin).
        """
        E_norm = torch.linspace(0, 1, self.nE)  # [nE]  
        return E_norm   # [nE]

    def _init_weights(self) -> None:
        """Kaiming init for backbone; zero-init head (delta=0 → W=W_ref at start)."""
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Zero head: network outputs delta=0 everywhere, so W = W_ref exactly at init.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


    def get_gradient_weights(self):
        params = dict(self.named_parameters())
        x = self.get_energy_grid()  # [nE x 1], input to the network for autograd

        def f(params):
            return functional_call(self, params, x)

        jac = jacrev(f)(params)  # dict: param_name -> [nBins x nE x *param_shape]

        # Flatten param dims, then concatenate over all params -> [nBins x nE x nParams]
        J = torch.cat(
            [v.flatten(start_dim=2) for v in jac.values()],
            dim=2
        )  # [nBins x nE x nParams]

        return J.permute(2, 0, 1)  # -> [nParams x nBins x nE], matching your original convention 

    def manual_gradient_step(self, loss_grad, lr=1e-3):
        """
        loss_grad: [nParams] tensor, gradient of loss w.r.t. theta
        lr: learning rate
        """
        params = dict(self.named_parameters())
        
        # Track offset into the flat [nParams] gradient vector
        offset = 0
        with torch.no_grad():
            for name, p in params.items():
                numel = p.numel()
                # Slice out the gradient chunk corresponding to this parameter
                grad_chunk = loss_grad[offset: offset + numel].reshape(p.shape)
                p.copy_(p - lr * grad_chunk)
                offset += numel


    def reg_grad(self):

        E = self.get_energy_grid()  # [nE]
        weights = self.forward(E)  # [nBins x nE]
        gradient_weights = self.get_gradient_weights()  # [nParams x nBins x nE]

        grad = torch.einsum('be,tbe->t', weights - self.w_ref, gradient_weights)  # [nParams]
        return self.lambda_reg * grad
    

    def forward(self, E_norm: torch.Tensor | None) -> torch.Tensor:
        """
        Parameters
        ----------
        E_norm : [nE]   normalised energy values in [0, 1]

        Returns
        -------
        W : [nBins x nE]  positive weight functions  W = W_ref * exp(delta)
        """

        if E_norm is None:
            E_norm = self.get_energy_grid()  # [nE]
        
        x     = E_norm.unsqueeze(-1)        # [nE x 1]
        h     = self.backbone(x)            # [nE x hidden]
        out = self.head(h)                   # [nE x nBins*2]
        a = out[:, :self.nBins].T                  # [nBins x nE]
        b = out[:, self.nBins:].T                  # [nBins x nE]
        return self.w_ref * torch.exp(a) + torch.exp(b) # [nBins x nE]    
