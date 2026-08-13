"""
Social LSTM (Alahi et al., CVPR 2016).

R2-4: Baseline for ETH-UCY leave-one-scene-out evaluation.
      Reports single-sample ADE/FDE (no multimodal sampling).
      Observation T=8, prediction H=12.
"""
import torch
import torch.nn as nn
from data.dataset import OBS_LEN, PRED_LEN


class SocialPooling(nn.Module):
    def __init__(self, hidden_dim: int = 64, grid_size: int = 4,
                 neighbourhood_size: float = 2.0):
        super().__init__()
        self.grid_size = grid_size
        self.pool_size = grid_size * grid_size * hidden_dim
        self.hidden_dim = hidden_dim
        self.neighbourhood_size = neighbourhood_size
        self.mlp = nn.Sequential(
            nn.Linear(self.pool_size, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, hidden: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        """
        hidden : (N, hidden_dim)
        pos    : (N, 2)
        Returns: (N, hidden_dim)  social pooling vector
        """
        N = pos.size(0)
        pool = torch.zeros(N, self.pool_size, device=pos.device)
        for i in range(N):
            diff = pos - pos[i]
            mask = diff.norm(dim=-1) < self.neighbourhood_size
            mask[i] = False
            if mask.sum() > 0:
                h_j = hidden[mask]
                pool[i, :h_j.numel()] = h_j.flatten()[:self.pool_size]
        return self.mlp(pool)


class SocialLSTM(nn.Module):
    """
    Social LSTM — Alahi et al. CVPR 2016.
    Deterministic single-sample prediction (no multimodal output).
    """
    def __init__(self, obs_len: int = OBS_LEN, pred_len: int = PRED_LEN,
                 hidden_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.embed    = nn.Linear(2, embed_dim)
        self.lstm     = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.pool     = SocialPooling(hidden_dim)
        self.decoder  = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : (N, T, 2)
        Returns : (N, H, 2)  single trajectory prediction
        """
        N = obs.size(0)
        h = torch.zeros(N, self.hidden_dim, device=obs.device)
        c = torch.zeros(N, self.hidden_dim, device=obs.device)

        for t in range(self.obs_len):
            e  = torch.relu(self.embed(obs[:, t]))         # (N, embed_dim)
            sp = self.pool(h, obs[:, t])                   # (N, hidden_dim)
            h, c = self.lstm(torch.cat([e, sp], dim=-1), (h, c))

        preds = []
        pos   = obs[:, -1]
        for _ in range(self.pred_len):
            e  = torch.relu(self.embed(pos))
            sp = self.pool(h, pos)
            h, c = self.lstm(torch.cat([e, sp], dim=-1), (h, c))
            pos  = self.decoder(h)
            preds.append(pos)

        return torch.stack(preds, dim=1)                   # (N, H, 2)
