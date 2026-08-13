"""
models/transformer.py
---------------------
Transformer temporal encoder for sequential trajectory modelling.

Reviewer compliance:
  R1-1 : Operates on hidden state vectors h_i^(L)(t) produced by the GNN
          social encoder, NOT on raw positions.  This means self-attention
          scores already carry social context from prior message-passing.
  R2-2 : Input to transformer is always a learned representation (GNN output),
          never raw 2D coordinates — maintaining the Q-K-V correctness
          requirement throughout the full pipeline.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence position."""
    def __init__(self, d_model: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder.

    Processes the sequence of GNN social context vectors
    {h_i^(L)(1), ..., h_i^(L)(T)} and produces temporal feature
    vectors f_i(t) used by both the VAE and the mode coefficient
    network (Eq. 9a).

    Architecture:
      - Positional encoding
      - num_layers standard TransformerEncoderLayer blocks
      - 8 attention heads (Table 2)
      - d_model = 512, d_ff = 1024 (2 × d_model)
    """
    def __init__(self,
                 d_model:    int = 512,
                 num_heads:  int = 8,
                 num_layers: int = 3,
                 d_ff:       int = 1024,
                 dropout:    float = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer,
                                              num_layers=num_layers)
        self.d_model  = d_model

    def forward(self,
                h_gnn: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_gnn : (N, T, d_model)  GNN social context over T time steps
        src_key_padding_mask : optional (N, T) bool mask

        Returns
        -------
        f : (N, T, d_model)  temporal feature vectors f_i(t)
        """
        x = self.pos_enc(h_gnn)                    # add positional info
        f = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return f                                   # (N, T, d_model)
