"""
D-STGCN (Sighencea et al., Electronics 2023).

Reference:
  Sighencea, B.I., Stanciu, I.R., & Căleanu, C.D. (2023). D-STGCN:
  Dynamic Pedestrian Trajectory Prediction Using Spatio-Temporal Graph
  Convolutional Networks. Electronics, 12(3), 611.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import OBS_LEN, PRED_LEN


class DynamicGraphConv(nn.Module):
    """
    Dynamic GCN layer: topology updated at each time step.
    D-STGCN's key distinction from static-graph GCNs.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h   : (N, in_dim)
        adj : (N, N) float  — normalised adjacency
        Returns : (N, out_dim)
        """
        # Symmetric normalisation  D^{-1/2} A D^{-1/2}
        deg  = adj.sum(dim=1, keepdim=True).clamp(min=1)
        a_n  = adj / deg.sqrt() / deg.sqrt().T
        agg  = a_n @ h                                    # (N, in_dim)
        out  = F.relu(self.W(agg))
        return self.norm(out)


class TemporalConv(nn.Module):
    """1-D temporal convolution across T time steps."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad        = kernel_size // 2
        self.conv  = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm  = nn.BatchNorm1d(channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (N, T, C) → (N, T, C)"""
        x = h.permute(0, 2, 1)                            # (N, C, T)
        x = F.relu(self.norm(self.conv(x)))
        return x.permute(0, 2, 1)                         # (N, T, C)


class DSTGCN(nn.Module):
    """
    Dynamic STGCN for pedestrian trajectory prediction.
    R1-3: included in Table 6 comparison.
    """
    def __init__(self, obs_len: int = OBS_LEN, pred_len: int = PRED_LEN,
                 in_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2,
                 delta: float = 2.0):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.delta    = delta

        self.embed    = nn.Linear(in_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([
            DynamicGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.tcn_layers = nn.ModuleList([
            TemporalConv(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, pred_len * 2),
        )

    def _dynamic_adj(self, pos: torch.Tensor) -> torch.Tensor:
        """Build soft dynamic adjacency from current positions."""
        diff  = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)  # (N, N)
        adj   = torch.exp(-diff / self.delta)
        adj   = adj * (diff <= self.delta * 2).float()
        return adj

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        """
        obs : (N, T, 2)
        Returns : (N, K, H, 2)
        """
        N, T, _  = obs.shape
        h        = self.embed(obs)                        # (N, T, hidden_dim)

        for gcn, tcn in zip(self.gcn_layers, self.tcn_layers):
            # Dynamic graph per time step
            h_spatial = torch.zeros_like(h)
            for t in range(T):
                adj_t       = self._dynamic_adj(obs[:, t])
                h_spatial[:, t] = gcn(h[:, t], adj_t)
            h = tcn(h_spatial)                            # temporal mixing

        h_last = h[:, -1]                                 # (N, hidden_dim)
        pred   = self.decoder(h_last).view(N, self.pred_len, 2)
        return pred.unsqueeze(1).expand(-1, k, -1, -1)   # (N, K, H, 2)
