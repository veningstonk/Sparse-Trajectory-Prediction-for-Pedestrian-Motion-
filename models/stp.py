"""
models/stp.py
-------------
Sparse Trajectory Prediction (STP) — main model.

Reviewer compliance:
  R1-1 : Architectural novelty paragraph — the three components interact as:
          GNN (social context h_i) → Transformer (temporal features f_i)
          → [ESO coefficient net + VAE encoder] (both conditioned on h_i + f_i)
          The Transformer receives GNN hidden states, not raw positions, so
          self-attention scores carry social context.  ESO applies sparsity
          at the decoding stage; VAE samples only lightweight residuals.
  R2-1 : Complete data flow matching Algorithm 1 (training) and
          Algorithm 2 (inference).
  R2-3 : Inference hypothesis selection uses argmin over K hypotheses
          (Eq. 31), NOT over time steps.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .gnn            import SocialGNNEncoder
from .transformer    import TemporalTransformerEncoder
from .vae            import TrajectoryVAE
from .early_sparsity import EarlySparsityModule


class STPModel(nn.Module):
    """
    Sparse Trajectory Prediction model.

    Architecture (Algorithm 1 data flow):
      1. GNN social encoder       → h_i^(L)(t)          Eqs. 3-8
      2. Transformer encoder      → f_i(t)
      3. Mode coefficient network → alpha_i(t)           Eq. 9a
      4. Sparse base trajectory   → p_hat_i^sparse(t)   Eq. 9
      5. VAE encoder              → mu, log_var          Eq. 12
      6. Reparameterisation       → z                    Eq. 13
      7. VAE decoder (learned)    → delta_i(t)           Eq. 15
      8. Final prediction         → p_hat_i = p_sparse + delta
    """
    def __init__(self,
                 d_model:      int   = 512,
                 gnn_layers:   int   = 2,
                 trans_layers: int   = 3,
                 num_heads:    int   = 8,
                 latent_dim:   int   = 64,
                 num_modes:    int   = 20,
                 pred_len:     int   = 12,
                 obs_len:      int   = 8,
                 delta:        float = 2.0,
                 dropout:      float = 0.1,
                 lambda_sp:    float = 0.1):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len

        # Component 1 — GNN social encoder (Eqs. 3-8, R2-2)
        self.gnn = SocialGNNEncoder(
            d_model=d_model, num_layers=gnn_layers,
            num_heads=num_heads, delta=delta, dropout=dropout
        )
        # Component 2 — Transformer temporal encoder (R1-1)
        self.transformer = TemporalTransformerEncoder(
            d_model=d_model, num_heads=num_heads,
            num_layers=trans_layers, dropout=dropout
        )
        # Component 3 — Early Sparsity Optimisation (Eqs. 9, 9a, 10)
        self.eso = EarlySparsityModule(
            d_model=d_model, num_modes=num_modes,
            pred_len=pred_len, lambda_sp=lambda_sp
        )
        # Component 4 — Variational Autoencoder (Eqs. 11-18)
        self.vae = TrajectoryVAE(
            d_model=d_model, latent_dim=latent_dim,
            pred_len=pred_len, dropout=dropout
        )

    def _encode(self, obs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """
        Shared encoding path used by both training and inference.

        obs : (N, T, 2)  observed positions

        Returns
        -------
        h_gnn   : (N, d_model)  GNN output at final observation step
        f_trans : (N, d_model)  Transformer output at final observation step
        h_seq   : (N, T, d_model)  full GNN sequence (for future use)
        f_seq   : (N, T, d_model)  full Transformer sequence
        """
        h_seq   = self.gnn(obs)                     # (N, T, d_model)  Eqs. 3-8
        f_seq   = self.transformer(h_seq)           # (N, T, d_model)
        h_gnn   = h_seq[:, -1, :]                   # last step context
        f_trans = f_seq[:, -1, :]
        return h_gnn, f_trans, h_seq, f_seq

    def forward(self, obs: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass — Algorithm 1, Steps 7-16.

        obs : (N, T, 2)

        Returns dict with all outputs needed for loss computation.
        """
        # Steps 7-10 — encode
        h_gnn, f_trans, _, _ = self._encode(obs)

        # Step 14 — mode coefficient prediction (Eq. 9a)
        # Step 15 — sparse base trajectory (Eq. 9) + sparsity loss (Eq. 10)
        p_sparse, alpha, l_sparse = self.eso(h_gnn, f_trans)

        # Steps 11-13 — VAE encode + reparameterise (Eqs. 12-13)
        # Step 16 — VAE decode residual correction (Eq. 15)
        delta, sigma, mu, log_var = self.vae(h_gnn, f_trans)

        # Final prediction: sparse base + learned residual
        p_hat = p_sparse + delta                    # (N, H, 2)

        return {
            "p_hat":    p_hat,        # (N, H, 2) — primary prediction
            "p_sparse": p_sparse,     # (N, H, 2) — ESO base
            "delta":    delta,        # (N, H, 2) — VAE residual
            "sigma":    sigma,        # scalar    — output std (for NLL)
            "mu":       mu,           # (N, latent_dim)
            "log_var":  log_var,      # (N, latent_dim)
            "alpha":    alpha,        # (N, K)    — for L1 loss
            "l_sparse": l_sparse,     # scalar    — Eq. 10
        }

    @torch.no_grad()
    def predict(self,
                obs:       torch.Tensor,
                k_samples: int   = 20,
                threshold: float = 0.05
                ) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass — Algorithm 2.

        R2-3: Returns K complete trajectory hypotheses; selection is
        over hypotheses k ∈ {1,...,K}, NOT over time steps.

        obs : (N, T, 2)

        Returns
        -------
        p_selected : (N, H, 2)   best hypothesis (Eq. 31)
        p_all      : (N, K, H, 2) all K hypotheses (for minADE@K)
        mu, sigma  : VAE distribution parameters (for NLL / ECE, R2-5)
        """
        self.eval()
        h_gnn, f_trans, _, _ = self._encode(obs)

        # Algorithm 2, Steps 4-6 — sparse base (single forward pass)
        p_sparse, _ = self.eso.infer(h_gnn, f_trans, threshold)  # (N, H, 2)

        # Algorithm 2, Steps 7-12 — K VAE residual samples
        x         = torch.cat([h_gnn, f_trans], dim=-1)
        mu, log_var = self.vae.encoder(x)
        sigma     = torch.exp(0.5 * log_var)

        hypotheses = []
        for _ in range(k_samples):
            eps   = torch.randn_like(mu)
            z     = mu + sigma * eps
            delta, _ = self.vae.decoder(z)
            hypotheses.append(p_sparse + delta)               # (N, H, 2)

        p_all = torch.stack(hypotheses, dim=1)                # (N, K, H, 2)

        # Algorithm 2, Steps 13-16 — best-of-K selection (Eq. 31)
        # R2-3: argmin over K hypotheses (not time steps)
        mean_pred = p_sparse + self.vae.decoder(mu)[0]       # (N, H, 2)
        dists     = (p_all - mean_pred.unsqueeze(1)).norm(
            dim=-1).mean(dim=-1)                             # (N, K)
        k_star    = dists.argmin(dim=1)                      # (N,)
        p_selected = p_all[torch.arange(p_all.size(0)), k_star]  # (N, H, 2)

        return {
            "p_selected": p_selected,   # (N, H, 2)  Eq. 31
            "p_all":      p_all,        # (N, K, H, 2) for minADE@K
            "mu":         mu,           # for NLL / ECE (R2-5)
            "sigma":      sigma,        # for NLL / ECE (R2-5)
        }
