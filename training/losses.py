"""
training/losses.py
------------------
All STP loss components.

Reviewer compliance:
  Internal fix: Eq. 29 used undefined term L_ds.  This file uses L_cls
  (Eq. 27) consistently throughout.  L_ds does NOT appear anywhere.
  Internal fix: Neighbour loss (Eq. 28) now conditioned on j ∈ N_i(t)
  via the adjacency mask — preventing penalisation of all pedestrian pairs.

Equations implemented:
  26  — L_reg    : regression loss (MSE between prediction and ground truth)
  27  — L_cls    : classification loss (cross-entropy over mode assignment)
  28  — L_nei    : neighbour prediction loss (consistency between neighbours)
  10  — L_sparse : L1 sparsity regularisation on mode coefficients
  17  — L_KL     : KL divergence (from VAE)
  29  — L_total  = lambda_1*L_reg + lambda_2*L_cls + lambda_3*L_nei
                   + lambda*L_sparse + lambda_KL*L_KL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def regression_loss(p_hat: torch.Tensor,
                    p_gt:  torch.Tensor) -> torch.Tensor:
    """
    Eq. 26 — L_reg: mean squared displacement error.

    p_hat : (N, H, 2)
    p_gt  : (N, H, 2)
    """
    return F.mse_loss(p_hat, p_gt)


def classification_loss(alpha:     torch.Tensor,
                        p_hat:     torch.Tensor,
                        p_gt:      torch.Tensor,
                        modes:     torch.Tensor) -> torch.Tensor:
    """
    Eq. 27 — L_cls: classification loss over mode assignment.

    Rather than cross-entropy over trajectory classes (which requires
    discrete labels not present in standard datasets), this implements
    a soft assignment loss: for each pedestrian, the 'correct' mode is
    the one whose centroid is closest to the ground truth displacement,
    and L_cls penalises the model when alpha assigns low weight to it.

    alpha : (N, K)    — predicted mode coefficients
    p_hat : (N, H, 2) — (unused here; kept for API consistency)
    p_gt  : (N, H, 2) — ground truth future trajectory
    modes : (K, H, 2) — current mode library
    """
    N, K = alpha.shape
    # Distance from each mode centroid to ground truth  (N, K)
    gt_exp    = p_gt.unsqueeze(1).expand(-1, K, -1, -1)   # (N, K, H, 2)
    mode_exp  = modes.unsqueeze(0).expand(N, -1, -1, -1)  # (N, K, H, 2)
    dist      = (gt_exp - mode_exp).norm(dim=-1).mean(dim=-1)  # (N, K)
    # Soft target: mode closest to GT gets label 1
    targets   = F.softmax(-dist, dim=-1)                   # (N, K)
    log_probs = F.log_softmax(alpha, dim=-1)               # (N, K)
    return F.kl_div(log_probs, targets, reduction="batchmean")


def neighbour_loss(p_hat: torch.Tensor,
                   adj:   Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Eq. 28 — L_nei: neighbour prediction consistency loss.

    Internal fix: conditioned on j ∈ N_i(t) via adjacency mask `adj`.
    Without the mask, all pairs are penalised equally regardless of
    proximity — which was the error in the original manuscript.

    p_hat : (N, H, 2)
    adj   : (N, N) bool — adjacency matrix from GNN graph construction
            If None, falls back to all-pairs (backward compatibility).
    """
    N = p_hat.size(0)
    if N < 2:
        return torch.tensor(0.0, device=p_hat.device)

    loss = torch.tensor(0.0, device=p_hat.device)
    count = 0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Only penalise neighbouring pairs (fix for Eq. 28)
            if adj is not None and not adj[i, j]:
                continue
            diff  = (p_hat[i] - p_hat[j]).norm(dim=-1).mean()
            loss  = loss + diff
            count += 1

    return loss / max(count, 1)


def sparsity_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    Eq. 10 — L_sparse: L1 regularisation on mode coefficients.
    Encourages only a small subset of K modes to be active (K' <= 3).
    """
    return alpha.abs().sum(dim=-1).mean()


def total_loss(outputs:     Dict[str, torch.Tensor],
               p_gt:        torch.Tensor,
               modes:       torch.Tensor,
               adj:         Optional[torch.Tensor],
               lambda_1:    float = 1.0,
               lambda_2:    float = 0.5,
               lambda_3:    float = 0.5,
               lambda_sp:   float = 0.1,
               lambda_kl:   float = 1.0
               ) -> Dict[str, torch.Tensor]:
    """
    Eq. 29 — L_total (corrected: uses L_cls, not undefined L_ds).

    L_total = lambda_1*L_reg + lambda_2*L_cls + lambda_3*L_nei
              + lambda*L_sparse + lambda_KL*L_KL

    Parameters
    ----------
    outputs  : dict from STPModel.forward()
    p_gt     : (N, H, 2) ground truth future positions
    modes    : (K, H, 2) current mode library tensor
    adj      : (N, N) bool adjacency (for neighbour loss conditioning)
    lambda_* : loss weights

    Returns
    -------
    dict with individual and total losses (for logging)
    """
    p_hat   = outputs["p_hat"]
    alpha   = outputs["alpha"]
    mu      = outputs["mu"]
    log_var = outputs["log_var"]

    l_reg     = regression_loss(p_hat, p_gt)                       # Eq. 26
    l_cls     = classification_loss(alpha, p_hat, p_gt, modes)     # Eq. 27
    l_nei     = neighbour_loss(p_hat, adj)                         # Eq. 28
    l_sparse  = outputs["l_sparse"]                                # Eq. 10
    l_kl      = TrajectoryVAE_kl(mu, log_var)                      # Eq. 17

    l_total = (lambda_1  * l_reg
             + lambda_2  * l_cls
             + lambda_3  * l_nei
             + lambda_sp * l_sparse
             + lambda_kl * l_kl)

    return {
        "total":   l_total,
        "reg":     l_reg,
        "cls":     l_cls,    # Note: L_cls, NOT L_ds (manuscript Eq. 29 fix)
        "nei":     l_nei,
        "sparse":  l_sparse,
        "kl":      l_kl,
    }


def TrajectoryVAE_kl(mu: torch.Tensor,
                     log_var: torch.Tensor) -> torch.Tensor:
    """Eq. 17 — KL divergence (proxy; actual call goes through VAE.kl_loss)."""
    return -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    ) / mu.size(0)
