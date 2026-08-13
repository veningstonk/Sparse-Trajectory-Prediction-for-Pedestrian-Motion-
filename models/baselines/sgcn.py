"""
SGCN — Sparse Graph Convolution Network (Shi et al., CVPR 2021).

R1-2: Included as interaction-stage sparsity baseline.
      SGCN reduces O(N^2) attention cost by building a sparse directed
      graph during ENCODING.  Key limitation: dense decoding still
      requires K separate forward passes, unlike STP's single-pass ESO.

Reference:
  Shi, L., Wang, L., Long, C., Zhou, S., Zhou, M., Niu, Z., & Hua, G.
  (2021). SGCN: Sparse graph convolution network for pedestrian trajectory
  prediction. CVPR 2021.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import OBS_LEN, PRED_LEN


class SparseDirGraph(nn.Module):
    """Learnable sparse directed graph — SGCN's interaction encoder."""
    def __init__(self, in_dim: int, out_dim: int, top_k: int = 3):
        super().__init__()
        self.top_k  = top_k
        self.W      = nn.Linear(in_dim * 2, 1)
        self.msg    = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (N, in_dim) → (N, out_dim)"""
        N   = h.size(0)
        # Edge scores for all pairs
        hi  = h.unsqueeze(1).expand(-1, N, -1)           # (N, N, D)
        hj  = h.unsqueeze(0).expand(N, -1, -1)           # (N, N, D)
        e   = self.W(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # (N, N)
        # Sparse: keep only top-K neighbours
        topk_vals, topk_idx = e.topk(min(self.top_k, N), dim=-1)
        mask = torch.full_like(e, float("-inf"))
        mask.scatter_(1, topk_idx, topk_vals)
        alpha = F.softmax(mask, dim=-1)                   # (N, N) sparse
        agg   = alpha @ self.msg(h)                       # (N, out_dim)
        return F.relu(agg)


class SGCN(nn.Module):
    """SGCN baseline. R1-2: interaction-stage sparsity comparison."""
    def __init__(self, obs_len: int = OBS_LEN, pred_len: int = PRED_LEN,
                 hidden_dim: int = 64, noise_dim: int = 16, top_k: int = 3):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.embed    = nn.Linear(2, hidden_dim)
        self.sgcn     = SparseDirGraph(hidden_dim, hidden_dim, top_k)
        self.encoder  = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder  = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, 256), nn.ReLU(),
            nn.Linear(256, pred_len * 2),
        )
        self.noise_dim = noise_dim

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        N     = obs.size(0)
        h_emb = self.embed(obs)                           # (N, T, hidden_dim)
        h_sp  = torch.stack([
            self.sgcn(h_emb[:, t]) for t in range(self.obs_len)
        ], dim=1)                                          # (N, T, hidden_dim)
        _, h  = self.encoder(h_sp)
        h     = h.squeeze(0)                              # (N, hidden_dim)

        samples = []
        for _ in range(k):
            z    = torch.randn(N, self.noise_dim, device=obs.device)
            pred = self.decoder(torch.cat([h, z], dim=-1))
            samples.append(pred.view(N, self.pred_len, 2))
        return torch.stack(samples, dim=1)                 # (N, K, H, 2)
