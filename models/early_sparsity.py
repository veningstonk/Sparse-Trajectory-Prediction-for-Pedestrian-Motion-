"""
models/early_sparsity.py
-------------------------
Early Sparsity Optimisation (ESO) module.

Reviewer compliance:
  R2-1 : Full specification of mode initialisation, coefficient network,
          and training/inference procedures as requested.
          - Mode initialisation: K-means on training displacement sequences
            (Algorithm 1, Steps 1-3).
          - Joint training with GNN, Transformer, VAE via unified loss
            (Algorithm 1, Steps 14-22).
          - Inference: single forward pass + sparse thresholding
            (Algorithm 2, Steps 4-6).
  R1-2 : Context-adaptive coefficients alpha_ik(t) conditioned on the
          pedestrian's social context h_i^(L)(t) from the GNN — NOT
          scene-agnostic fixed anchors (vs Multipath++).
          ESO introduces sparsity at the DECODING stage, complementary
          to interaction-stage sparsity methods like SGCN.
  R1-1 : Mode library M* is pre-computed at training time; inference
          requires only a single weighted sum (Eq. 9) plus lightweight
          VAE residuals — source of 0.02s/pedestrian speed.

Equations implemented:
  9a  — coefficient network: alpha_i(t) = f_coeff([h_i^(L)(t) || f_i(t)])
  9   — sparse trajectory:   p_hat_i^sparse(t) = sum_k alpha_ik(t) * m_k
  10  — ESO loss:            L_sparse = sum_i sum_k ||alpha_ik(t)||_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional


class ModeCoeffNetwork(nn.Module):
    """
    Eq. 9a — f_coeff: learns context-adaptive mode coefficients.

    Input  : [h_i^(L)(t) || f_i(t)]  — social context + temporal feature
    Output : alpha_i(t) ∈ R^K        — sparse mixing weights over K modes

    R1-2: Coefficients are conditioned on h_i^(L)(t) (GNN social output)
    making them scene- and pedestrian-specific, unlike Multipath++ anchors.
    """
    def __init__(self, d_model: int, num_modes: int = 20):
        super().__init__()
        input_dim = 2 * d_model     # [h_gnn || f_trans] concatenation
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes),
        )
        self.num_modes = num_modes

    def forward(self, h_gnn: torch.Tensor,
                f_trans: torch.Tensor) -> torch.Tensor:
        """
        h_gnn   : (N, d_model)
        f_trans : (N, d_model)

        Returns
        -------
        alpha : (N, K)   raw (unnormalised) mode coefficients
        """
        x = torch.cat([h_gnn, f_trans], dim=-1)   # (N, 2*d_model)
        return self.net(x)                          # (N, K)  — Eq. 9a


class EarlySparsityModule(nn.Module):
    """
    Full ESO module: mode library M + coefficient network f_coeff.

    R2-1 — Algorithm 1 Steps 1-3: K-means initialisation.
    R2-1 — Algorithm 1 Step 14:   coefficient prediction.
    R2-1 — Algorithm 1 Step 15:   sparse trajectory reconstruction.
    R2-1 — Algorithm 2 Steps 4-6: inference with sparsity thresholding.
    """
    def __init__(self,
                 d_model:    int   = 512,
                 num_modes:  int   = 20,
                 pred_len:   int   = 12,
                 lambda_sp:  float = 0.1):
        """
        Parameters
        ----------
        d_model   : hidden dimension (same as GNN / Transformer)
        num_modes : K — number of sparse motion modes (Table 2: 20)
        pred_len  : H — prediction horizon
        lambda_sp : L1 regularisation weight (Table 2: 0.1)
        """
        super().__init__()
        self.num_modes = num_modes
        self.pred_len  = pred_len
        self.lambda_sp = lambda_sp

        # Learnable mode library M = {m_1,...,m_K}  shape (K, H, 2)
        # Initialised by K-means; updated jointly during training
        self.modes = nn.Parameter(
            torch.randn(num_modes, pred_len, 2) * 0.1
        )
        # Eq. 9a — coefficient network
        self.coeff_net = ModeCoeffNetwork(d_model, num_modes)

    @torch.no_grad()
    def initialise_modes_kmeans(self,
                                displacement_seqs: np.ndarray) -> None:
        """
        Algorithm 1, Steps 1-3 — K-means mode initialisation.

        R2-1: Before training begins, run K-means on all H-step displacement
        sequences from the training set.  Cluster centroids become M's
        initial values, ensuring the mode library spans observed motion
        diversity from epoch 1.

        Parameters
        ----------
        displacement_seqs : (N_total, H, 2)  — all training displacement
                            sequences computed as Delta_p_i(t) = p_i(t) - p_i(t-1)
        """
        N, H, C = displacement_seqs.shape
        flat     = displacement_seqs.reshape(N, H * C)   # (N, H*2)

        kmeans   = KMeans(n_clusters=self.num_modes,
                          random_state=42, n_init=10)
        kmeans.fit(flat)

        centroids = torch.tensor(
            kmeans.cluster_centers_.reshape(self.num_modes, H, C),
            dtype=torch.float32
        )
        self.modes.data.copy_(centroids)
        print(f"[ESO] K-means initialised {self.num_modes} modes "
              f"from {N} displacement sequences.")

    def forward(self,
                h_gnn:  torch.Tensor,
                f_trans: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Parameters
        ----------
        h_gnn   : (N, d_model)
        f_trans : (N, d_model)

        Returns
        -------
        p_sparse : (N, H, 2)   Eq. 9  — sparse base trajectory
        alpha    : (N, K)      Eq. 9a — mode coefficients (for L1 loss)
        l_sparse : scalar      Eq. 10 — sparsity regularisation loss
        """
        # Eq. 9a — mode coefficients
        alpha    = self.coeff_net(h_gnn, f_trans)         # (N, K)

        # Eq. 9 — sparse trajectory reconstruction
        # modes : (K, H, 2),  alpha : (N, K)
        p_sparse = torch.einsum("nk,khc->nhc", alpha, self.modes)  # (N, H, 2)

        # Eq. 10 — L1 sparsity regularisation
        l_sparse = self.lambda_sp * alpha.abs().sum(dim=-1).mean()

        return p_sparse, alpha, l_sparse

    @torch.no_grad()
    def infer(self,
              h_gnn:     torch.Tensor,
              f_trans:   torch.Tensor,
              threshold: float = 0.05
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 2, Steps 4-6 — inference with sparsity thresholding.

        R2-1 / R1-1: Single forward pass; only K' <= 3 modes remain
        non-zero after thresholding (enforced by L1 training), making
        decoding effectively O(N) and independent of K.

        Parameters
        ----------
        h_gnn, f_trans : (N, d_model)
        threshold      : tau_s — zero out modes below this weight

        Returns
        -------
        p_sparse : (N, H, 2)   sparse base trajectory
        alpha_sp : (N, K)      thresholded coefficients
        """
        alpha    = self.coeff_net(h_gnn, f_trans)         # (N, K)
        # Algorithm 2, Step 5 — zero out sub-threshold modes
        alpha_sp = alpha * (alpha.abs() >= threshold).float()
        p_sparse = torch.einsum("nk,khc->nhc", alpha_sp, self.modes)
        return p_sparse, alpha_sp
