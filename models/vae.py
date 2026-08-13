"""
models/vae.py
-------------
Variational Autoencoder for multimodal trajectory prediction.

Reviewer compliance:
  R1-1 : VAE decoder applies a LEARNED nonlinear transformation f_decoder(z)
          to map the latent variable z to the output space.  The original
          manuscript wrote p(p_hat | z) = N(p_hat; z, sigma^2), making z
          directly the output mean — bypassing any learned decoding.
          This implementation corrects that: output = f_decoder(z) where
          f_decoder is a multi-layer MLP (Eq. 15).
  R2-1 : VAE encoder and decoder are jointly trained with GNN, Transformer,
          and ESO via Algorithm 1 (training/trainer.py).
  R2-5 : The encoder outputs (mu, log_var) are used for NLL and ECE
          calibration metrics in evaluation/metrics.py.

Equations implemented:
  11  — posterior:  q(z | P_obs) = N(mu(P_obs), sigma^2(P_obs))
  12  — encoder:    mu, sigma = f_mu(P_obs), f_sigma(P_obs)
  13  — reparameterisation: z = mu + sigma * epsilon, epsilon ~ N(0,I)
  14  — decoder distribution (corrected): p(p_hat | z) = N(f_decoder(z), sigma^2)
  15  — decoder: p_hat_i(t) = f_decoder(z)
  16  — reconstruction loss
  17  — KL divergence loss
  18  — total VAE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """
    Eq. 11–12.
    Maps the concatenated [GNN context || Transformer feature] into
    (mu, log_var) of the posterior Gaussian q(z | P_obs).
    """
    def __init__(self, input_dim: int, latent_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # f_mu and f_sigma (Eq. 12)
        self.fc_mu      = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (N, input_dim)  — concatenated social + temporal features

        Returns
        -------
        mu      : (N, latent_dim)
        log_var : (N, latent_dim)
        """
        h       = self.net(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    Eqs. 14–15 (corrected).

    f_decoder(z) : latent_dim → H × 2
    The decoder is a multi-layer MLP that learns a nonlinear mapping
    from the latent variable z to the predicted trajectory residual.

    R1-1 (internal): This corrects the original manuscript formulation
    where z was used directly as the output mean.  Here z is transformed
    through learned layers before producing position predictions.
    """
    def __init__(self, latent_dim: int = 64, pred_len: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        # f_decoder (Eq. 15)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, pred_len * 2),   # H × 2 output
        )
        # Learnable log-variance for output distribution (Eq. 14)
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z : (N, latent_dim)

        Returns
        -------
        delta   : (N, H, 2)   — predicted position residual (Eq. 15)
        sigma   : scalar      — output standard deviation (Eq. 14)
        """
        out   = self.net(z)                               # (N, H*2)
        delta = out.view(z.size(0), self.pred_len, 2)    # (N, H, 2)
        sigma = torch.exp(0.5 * self.log_sigma)
        return delta, sigma


class TrajectoryVAE(nn.Module):
    """
    Full VAE: encoder + reparameterisation + decoder.

    Eqs. 11–18.
    """
    def __init__(self,
                 d_model:    int   = 512,
                 latent_dim: int   = 64,
                 pred_len:   int   = 12,
                 dropout:    float = 0.1):
        super().__init__()
        # Input = [h_gnn ‖ f_trans] at final observation step → 2 * d_model
        self.encoder = VAEEncoder(2 * d_model, latent_dim, dropout)
        self.decoder = VAEDecoder(latent_dim,  pred_len,   dropout)

    def reparameterise(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        Eq. 13 — z = mu + sigma * epsilon, epsilon ~ N(0, I)
        Reparameterisation trick enables gradient flow through sampling.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps                             # Eq. 13

    def forward(self,
                h_gnn:  torch.Tensor,
                f_trans: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_gnn   : (N, d_model)  GNN output at last observation step
        f_trans : (N, d_model)  Transformer output at last observation step

        Returns
        -------
        delta   : (N, H, 2)   trajectory residual correction (Eq. 15)
        sigma   : scalar      output std
        mu      : (N, latent_dim)
        log_var : (N, latent_dim)
        """
        x           = torch.cat([h_gnn, f_trans], dim=-1)  # (N, 2*d_model)
        mu, log_var = self.encoder(x)                       # Eq. 12
        z           = self.reparameterise(mu, log_var)      # Eq. 13
        delta, sigma = self.decoder(z)                      # Eq. 14–15
        return delta, sigma, mu, log_var

    def sample(self, h_gnn: torch.Tensor, f_trans: torch.Tensor,
               k: int = 20) -> torch.Tensor:
        """
        Draw K independent samples for minADE@K evaluation (R2-3).

        Returns : (N, K, H, 2)  — K trajectory residuals
        """
        N = h_gnn.size(0)
        x = torch.cat([h_gnn, f_trans], dim=-1)            # (N, 2*d_model)
        mu, log_var = self.encoder(x)                       # (N, latent_dim)

        samples = []
        for _ in range(k):
            z, _  = self.reparameterise(mu, log_var), None
            delta, _ = self.decoder(z)
            samples.append(delta)                           # (N, H, 2)
        return torch.stack(samples, dim=1)                  # (N, K, H, 2)

    @staticmethod
    def kl_loss(mu: torch.Tensor,
                log_var: torch.Tensor) -> torch.Tensor:
        """
        Eq. 17 — KL divergence: D_KL(q(z|P_obs) || p(z))
        p(z) = N(0, I)
        """
        return -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp()
        ) / mu.size(0)
