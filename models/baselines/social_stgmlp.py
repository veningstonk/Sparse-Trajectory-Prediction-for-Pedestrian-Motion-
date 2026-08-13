"""
Social-STGMLP (Meng et al., Information 2024).

Reference:
  Meng, D., Zhao, G., & Yan, F. (2024). Social-STGMLP: A Social
  Spatio-Temporal Graph Multi-Layer Perceptron for Pedestrian Trajectory
  Prediction. Information, 15(6), 341.
"""
import torch
import torch.nn as nn
from data.dataset import OBS_LEN, PRED_LEN


class SpatioTemporalMLP(nn.Module):
    """Lightweight MLP replacing graph attention in spatiotemporal graph."""
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.spatial  = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,   hidden_dim),
        )
        self.temporal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h   : (N, T, node_dim)
        adj : (N, N) — adjacency
        Returns : (N, T, hidden_dim)
        """
        N, T, D = h.shape
        # Spatial aggregation via MLP (not attention)
        h_spatial = torch.zeros(N, T, D * 2, device=h.device)
        for i in range(N):
            neighbours = h[adj[i]]                       # (|N_i|, T, D)
            if neighbours.numel() == 0:
                agg = torch.zeros(T, D, device=h.device)
            else:
                agg = neighbours.mean(dim=0)             # (T, D)
            h_spatial[i] = torch.cat([h[i], agg], dim=-1)
        h_sp = self.spatial(h_spatial)                   # (N, T, hidden_dim)
        # Temporal aggregation
        h_t  = self.temporal(h_sp)                       # (N, T, hidden_dim)
        return self.norm(h_t + h_sp)


class SocialSTGMLP(nn.Module):
    """
    Social-STGMLP — graph MLP for pedestrian trajectory prediction.
    R1-3: included in Table 6 comparison.
    """
    def __init__(self, obs_len: int = OBS_LEN, pred_len: int = PRED_LEN,
                 node_dim: int = 32, hidden_dim: int = 128,
                 delta: float = 2.0, num_layers: int = 2):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.delta    = delta
        self.embed    = nn.Linear(2, node_dim)
        self.stg_layers = nn.ModuleList([
            SpatioTemporalMLP(node_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.decoder  = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, pred_len * 2),
        )

    def _adjacency(self, pos: torch.Tensor) -> torch.Tensor:
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
        return dist <= self.delta

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        """
        obs : (N, T, 2)
        Returns : (N, K, H, 2)  — K identical deterministic predictions
        """
        N    = obs.size(0)
        adj  = self._adjacency(obs[:, -1])                # (N, N)
        h    = self.embed(obs)                            # (N, T, node_dim)
        for layer in self.stg_layers:
            h = layer(h, adj)                             # (N, T, hidden_dim)
        h_last = h[:, -1]                                 # (N, hidden_dim)
        pred   = self.decoder(h_last).view(N, self.pred_len, 2)  # (N, H, 2)
        return pred.unsqueeze(1).expand(-1, k, -1, -1)   # (N, K, H, 2)
