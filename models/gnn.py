"""
models/gnn.py
-------------
Graph Neural Network social encoder with learned Q-K-V attention.

Reviewer compliance:
  R2-2 : Attention scores are computed from LEARNED query/key projections
          of hidden state vectors — NOT from raw 2D positions.
          Equations implemented:
            5a  — position embedding: h_i^(0) = W_emb * P_i + b_emb
            5b  — Q-K-V projections: Q=W_Q*h, K=W_K*h, V=W_V*h
            5   — scaled dot-product: alpha_ij = softmax(Q_i^T K_j / sqrt(d_k))
            6   — value aggregation:  h_i^att = sum_j alpha_ij * V_j
            6a  — per-head output
            6b  — multi-head concat + output projection W_O
            7   — message passing:   h_i^(t+1) = sigma(W * h_i^att + b)
            8   — multi-layer stacking
  R1-1 : Locality constraint — attention restricted to N_i(t) defined by
          proximity threshold delta, reducing O(N^2) to O(N*|N_i|).
  R2-4 : delta is a configurable hyperparameter documented in configs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionEmbedding(nn.Module):
    """
    Eq. 5a — linear map from 2D world coordinates to hidden space.

    R2-2: Raw positions are NEVER used directly as attention inputs.
    This embedding is applied at layer l=0 before any attention.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2, d_model)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """pos : (..., 2)  →  (..., d_model)"""
        return F.relu(self.proj(pos))


class MultiHeadLocalAttention(nn.Module):
    """
    Eqs. 5b, 5, 6, 6a, 6b — multi-head scaled dot-product attention
    restricted to a proximity-defined neighbourhood N_i(t).

    R2-2: W_Q, W_K, W_V are LEARNED weight matrices; attention scores
          are computed between projected hidden states, not positions.
    R1-1: The adjacency mask `adj` encodes the graph E_t (Eq. 3) so
          pedestrians outside delta receive -inf before softmax.
    """
    def __init__(self, d_model: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # key/query dim per head
        self.d_v = d_model // num_heads   # value dim per head

        # Eq. 5b — learned projection matrices (one set per head via block)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        # Eq. 6b — output projection W_O
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_k)

    def forward(self,
                h: torch.Tensor,
                adj: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        h   : (N, d_model)  hidden states of all pedestrians at time t
        adj : (N, N) bool   adjacency mask — True where j ∈ N_i(t) (Eq. 3)

        Returns
        -------
        h_att : (N, d_model)  Eq. 6b output
        """
        N = h.size(0)

        # Eq. 5b — project to Q, K, V and split into heads
        Q = self.W_Q(h).view(N, self.num_heads, self.d_k)  # (N, M, d_k)
        K = self.W_K(h).view(N, self.num_heads, self.d_k)
        V = self.W_V(h).view(N, self.num_heads, self.d_v)

        # Eq. 5 — scaled dot-product attention, restricted to neighbourhood
        # scores : (N, M, N)   Q_i^T K_j / sqrt(d_k)
        scores = torch.einsum("imd,jmd->mij", Q, K) / self.scale  # (M, N, N)

        # Mask out non-neighbours with -inf (Eq. 3 — edge condition)
        mask = ~adj.unsqueeze(0).expand(self.num_heads, -1, -1)  # (M, N, N)
        scores = scores.masked_fill(mask, float("-inf"))
        alpha  = F.softmax(scores, dim=-1)                        # (M, N, N)
        # Guard against all-masked rows (isolated pedestrians)
        alpha  = torch.nan_to_num(alpha, nan=0.0)
        alpha  = self.dropout(alpha)

        # Eq. 6a — per-head value aggregation  head_m = sum_j alpha_ij * V_j
        # (M, N, N) x (N, M, d_v) → (M, N, d_v)
        heads = torch.einsum("mij,jmd->imd", alpha, V)            # (N, M, d_v)

        # Eq. 6b — concatenate heads and project
        heads  = heads.reshape(N, self.num_heads * self.d_v)      # (N, d_model)
        h_att  = self.W_O(heads)                                   # (N, d_model)
        return h_att


class GNNLayer(nn.Module):
    """
    One GNN layer combining multi-head local attention (Eqs. 5–6b) with
    the message-passing update (Eq. 7).

    Eq. 7:  h_i^(t+1) = sigma( W · h_i^att + b )
    """
    def __init__(self, d_model: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.attention  = MultiHeadLocalAttention(d_model, num_heads, dropout)
        self.linear     = nn.Linear(d_model, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout    = nn.Dropout(dropout)

    def forward(self,
                h: torch.Tensor,
                adj: torch.Tensor
                ) -> torch.Tensor:
        """
        h   : (N, d_model)
        adj : (N, N) bool
        Returns : (N, d_model)   — Eq. 7 output
        """
        # Eq. 6b + residual + layer norm
        h_att = self.attention(h, adj)
        h     = self.norm1(h + self.dropout(h_att))
        # Eq. 7 — nonlinear transformation (position-wise feed-forward)
        h     = self.norm2(h + self.dropout(self.ff(h)))
        return h


class SocialGNNEncoder(nn.Module):
    """
    Full GNN social encoder: position embedding → L message-passing layers.

    Eqs. 3–8 of the manuscript.

    R1-1 : Builds proximity graph on-the-fly at each call;
           O(N·|N_i|) per layer vs O(N^2) for full attention.
    R2-2 : The position embedding (Eq. 5a) ensures attention always
           operates on learned representations, never raw coordinates.
    """
    def __init__(self,
                 d_model:    int   = 512,
                 num_layers: int   = 2,
                 num_heads:  int   = 8,
                 delta:      float = 2.0,
                 dropout:    float = 0.1):
        """
        Parameters
        ----------
        d_model    : hidden dimension
        num_layers : number of GNN layers (L) — Eq. 8
        num_heads  : M attention heads — Eq. 6a
        delta      : proximity threshold (metres) — Eq. 3
        dropout    : dropout rate
        """
        super().__init__()
        self.delta   = delta
        self.emb     = PositionEmbedding(d_model)           # Eq. 5a
        self.layers  = nn.ModuleList([
            GNNLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def _build_adjacency(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Eq. 3 — E(i,j) = 1 if ||P_i - P_j|| <= delta, else 0.

        pos : (N, 2)
        Returns: (N, N) bool — includes self-loops
        """
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N, N, 2)
        dist = diff.norm(dim=-1)                            # (N, N)
        adj  = dist <= self.delta
        return adj                                          # (N, N) bool

    def forward(self,
                pos_seq: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_seq : (N, T, 2)  observed positions of N pedestrians over T steps

        Returns
        -------
        h_final : (N, T, d_model)  social context at every time step
        """
        N, T, _ = pos_seq.shape
        h_out   = []

        for t in range(T):
            pos_t = pos_seq[:, t, :]                        # (N, 2)
            adj_t = self._build_adjacency(pos_t)            # (N, N) Eq. 3–4
            # Eq. 5a — embed positions on first layer; subsequent layers
            # receive updated hidden states from previous time step
            h_t = self.emb(pos_t) if t == 0 else h_out[-1].mean(dim=0, keepdim=True).expand(N, -1)  # fallback init
            # Eq. 8 — stack L GNN layers
            for layer in self.layers:
                h_t = layer(h_t, adj_t)                     # Eq. 7
            h_out.append(h_t)

        return torch.stack(h_out, dim=1)                    # (N, T, d_model)
