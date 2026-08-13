"""
evaluation/metrics.py
---------------------
All evaluation metrics for trajectory prediction.

Reviewer compliance:
  R2-3 : minADE@K and minFDE@K use oracle selection over K HYPOTHESES
          (Eq. 31c), not over time steps.
  R2-4 : Mean ADE/FDE use deployment-time model-selected hypothesis (Eq. 31).
          All metrics documented with units (metres) and K=20 default.
  R2-5 : Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE)
          evaluate VAE uncertainty calibration (Eqs. 36-37).

Equations implemented:
  32  — ADE  (deployment: model-selected hypothesis)
  33  — FDE  (deployment: model-selected hypothesis)
  36  — minADE@K (oracle best-of-K)
  37  — minFDE@K (oracle best-of-K)
  36* — NLL  (negative log-likelihood of Gaussian prediction)
  37* — ECE  (expected calibration error via 1-sigma coverage)
"""

import math
import numpy as np
import torch
from typing import Dict, Optional


# ── Displacement errors ───────────────────────────────────────────────────────

def ade(p_hat: torch.Tensor, p_gt: torch.Tensor) -> float:
    """
    Eq. 32 — Average Displacement Error (metres).
    Uses the model-selected hypothesis (Eq. 31) — deployment metric.

    p_hat : (N, H, 2)
    p_gt  : (N, H, 2)
    """
    return (p_hat - p_gt).norm(dim=-1).mean().item()


def fde(p_hat: torch.Tensor, p_gt: torch.Tensor) -> float:
    """
    Eq. 33 — Final Displacement Error (metres).
    Error at the last predicted time step T+H.

    p_hat : (N, H, 2)
    p_gt  : (N, H, 2)
    """
    return (p_hat[:, -1] - p_gt[:, -1]).norm(dim=-1).mean().item()


def min_ade_k(p_all: torch.Tensor, p_gt: torch.Tensor) -> float:
    """
    Eq. 36 — minADE@K: oracle best-of-K average displacement error.

    R2-3: argmin over k ∈ {1,...,K} — selects best hypothesis per pedestrian.
    R2-4: K=20 is the standard benchmark value.

    p_all : (N, K, H, 2)
    p_gt  : (N, H, 2)
    """
    N, K, H, _ = p_all.shape
    gt_exp = p_gt.unsqueeze(1).expand(-1, K, -1, -1)          # (N, K, H, 2)
    # Per-hypothesis ADE  (N, K)
    per_hyp_ade = (p_all - gt_exp).norm(dim=-1).mean(dim=-1)  # (N, K)
    # Oracle selection: min over K hypotheses (R2-3)
    best_ade    = per_hyp_ade.min(dim=1).values                # (N,)
    return best_ade.mean().item()


def min_fde_k(p_all: torch.Tensor, p_gt: torch.Tensor) -> float:
    """
    Eq. 37 — minFDE@K: oracle best-of-K final displacement error.

    p_all : (N, K, H, 2)
    p_gt  : (N, H, 2)
    """
    N, K, H, _ = p_all.shape
    gt_final  = p_gt[:, -1].unsqueeze(1).expand(-1, K, -1)   # (N, K, 2)
    per_hyp   = (p_all[:, :, -1] - gt_final).norm(dim=-1)    # (N, K)
    best_fde  = per_hyp.min(dim=1).values                     # (N,)
    return best_fde.mean().item()


# ── Probabilistic calibration (R2-5) ─────────────────────────────────────────

def negative_log_likelihood(p_hat:  torch.Tensor,
                             p_gt:   torch.Tensor,
                             sigma:  torch.Tensor) -> float:
    """
    Eq. 36 (Table 5) — NLL under a Gaussian decoder.

    For each pedestrian i and time step t, evaluates:
      NLL = 0.5 * ||p_hat - p_gt||^2 / sigma^2
            + log(sigma) + 0.5*log(2*pi)

    R2-5: Lower NLL indicates the predicted distribution places higher
    probability mass on actual ground-truth positions.

    p_hat : (N, H, 2)
    p_gt  : (N, H, 2)
    sigma : (N, H) or scalar — predicted standard deviation
    """
    diff_sq = (p_hat - p_gt).norm(dim=-1) ** 2           # (N, H)

    if sigma.dim() == 0:
        sig = sigma.expand_as(diff_sq)
    else:
        sig = sigma

    nll = 0.5 * diff_sq / (sig ** 2 + 1e-8) \
        + sig.log() \
        + 0.5 * math.log(2 * math.pi)

    return nll.mean().item()


def expected_calibration_error(p_hat:  torch.Tensor,
                                p_gt:   torch.Tensor,
                                sigma:  torch.Tensor,
                                n_bins: int = 10) -> float:
    """
    Eq. 37 (Table 5) — Expected Calibration Error (ECE).

    R2-5: Groups predictions into n_bins bins by predicted sigma;
    within each bin computes empirical 1-sigma coverage and compares
    to the expected Gaussian 68.3% coverage.

    ECE = sum_b (|B_b| / N_total) * |Coverage(B_b) - 0.683|

    p_hat : (N, H, 2)
    p_gt  : (N, H, 2)
    sigma : (N, H) or scalar
    """
    dist  = (p_hat - p_gt).norm(dim=-1)                  # (N, H) residuals
    if sigma.dim() == 0:
        sig = sigma.expand_as(dist)
    else:
        sig = sigma

    # Normalised residual: within 1-sigma if < 1.0
    norm_res  = dist / (sig + 1e-8)                      # (N, H)
    within    = (norm_res < 1.0).float()                  # 1 if inside 1σ

    flat_sig  = sig.reshape(-1).detach().cpu().numpy()
    flat_wit  = within.reshape(-1).detach().cpu().numpy()

    bins      = np.linspace(flat_sig.min(), flat_sig.max() + 1e-8, n_bins + 1)
    ece       = 0.0
    n_total   = len(flat_sig)

    for b in range(n_bins):
        mask = (flat_sig >= bins[b]) & (flat_sig < bins[b + 1])
        if mask.sum() == 0:
            continue
        coverage = flat_wit[mask].mean()
        weight   = mask.sum() / n_total
        ece     += weight * abs(coverage - 0.683)

    return float(ece)


# ── Brier scores (Table 9 column labels — corrected from original) ────────────

def brier_ade(p_hat: torch.Tensor, p_gt: torch.Tensor) -> float:
    """
    Brier-ADE: squared displacement error (uncertainty-weighted ADE).
    Eq. 34.

    Note: in Table 9 these ARE ADE/FDE values, not Brier scores.
    Column header corrected to minADE@20 / minFDE@20 per R2-3 revision.
    """
    return (p_hat - p_gt).norm(dim=-1).pow(2).mean().item()


def brier_fde(p_hat: torch.Tensor, p_gt: torch.Tensor) -> float:
    """Brier-FDE. Eq. 35."""
    return (p_hat[:, -1] - p_gt[:, -1]).norm(dim=-1).pow(2).mean().item()


# ── Aggregate ─────────────────────────────────────────────────────────────────

def compute_all_metrics(p_all:      torch.Tensor,
                         p_selected: torch.Tensor,
                         p_gt:       torch.Tensor,
                         mu:         torch.Tensor,
                         sigma:      torch.Tensor
                         ) -> Dict[str, float]:
    """
    Compute all metrics for one scene.

    p_all      : (N, K, H, 2)  — K hypotheses
    p_selected : (N, H, 2)     — model-selected hypothesis (Eq. 31)
    p_gt       : (N, H, 2)     — ground truth
    mu         : (N, latent)   — VAE posterior mean
    sigma      : (N, latent)   — VAE posterior std
    """
    # Expand sigma to match (N, H) shape for NLL/ECE
    sigma_h = sigma.mean(dim=-1, keepdim=True).expand(
        -1, p_gt.size(1)
    )                                                     # (N, H)

    return {
        "ADE":      ade(p_selected, p_gt),               # Eq. 32
        "FDE":      fde(p_selected, p_gt),               # Eq. 33
        "minADE20": min_ade_k(p_all, p_gt),              # Eq. 36
        "minFDE20": min_fde_k(p_all, p_gt),              # Eq. 37
        "NLL":      negative_log_likelihood(
                        p_selected, p_gt, sigma_h),       # Eq. 36 (Table 5)
        "ECE":      expected_calibration_error(
                        p_selected, p_gt, sigma_h),       # Eq. 37 (Table 5)
        "BrierADE": brier_ade(p_selected, p_gt),         # Eq. 34
        "BrierFDE": brier_fde(p_selected, p_gt),         # Eq. 35
    }
