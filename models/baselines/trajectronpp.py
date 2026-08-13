"""
Trajectron++ (Salzmann et al., ECCV 2020).

R1-1: Key baseline for architectural novelty comparison.
      Trajectron++ combines GNN + CVAE (similar components to STP) but:
      - Uses LSTM recurrent encoders (not Transformers) → cannot capture
        long-range temporal dependencies as efficiently.
      - Dense O(N^2) pairwise attention.
      - No inference-time sparsity: K samples each require a full CVAE
        forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import OBS_LEN, PRED_LEN


class TrajectronPP(nn.Module):
    """Simplified Trajectron++ for comparison. Reports minADE@20."""
    def __init__(self, obs_len: int = OBS_LEN, pred_len: int = PRED_LEN,
                 hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.obs_len   = obs_len
        self.pred_len  = pred_len
        self.encoder   = nn.LSTM(2, hidden_dim, batch_first=True)
        self.edge_enc  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_var    = nn.Linear(hidden_dim, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 256), nn.ReLU(),
            nn.Linear(256, pred_len * 2),
        )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        N = obs.size(0)
        _, (h, _) = self.encoder(obs)
        h = h.squeeze(0)                                   # (N, hidden_dim)

        # Dense pairwise edge encoding (O(N^2) — contrast with STP O(N|N_i|))
        h_agg = []
        for i in range(N):
            h_j = torch.stack([
                self.edge_enc(torch.cat([h[i], h[j]], dim=-1))
                for j in range(N) if j != i
            ] or [torch.zeros(self.hidden_dim, device=h.device)], dim=0)
            h_agg.append(h_j.mean(dim=0))
        h_social = torch.stack(h_agg, dim=0)              # (N, hidden_dim)

        mu  = self.fc_mu(h_social)
        var = self.fc_var(h_social)
        std = torch.exp(0.5 * var)

        samples = []
        for _ in range(k):
            z    = mu + std * torch.randn_like(std)
            inp  = torch.cat([h_social, z], dim=-1)
            pred = self.decoder(inp).view(N, self.pred_len, 2)
            samples.append(pred)
        return torch.stack(samples, dim=1)                 # (N, K, H, 2)
