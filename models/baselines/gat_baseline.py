"""GAT baseline — Graph Attention Network for trajectory prediction."""
import torch, torch.nn as nn, torch.nn.functional as F
from data.dataset import OBS_LEN, PRED_LEN

class GATBaseline(nn.Module):
    def __init__(self, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 hidden_dim=64, num_heads=4, noise_dim=16):
        super().__init__()
        self.pred_len  = pred_len
        self.noise_dim = noise_dim
        self.embed     = nn.Linear(2, hidden_dim)
        self.gat_q     = nn.Linear(hidden_dim, hidden_dim)
        self.gat_k     = nn.Linear(hidden_dim, hidden_dim)
        self.gat_v     = nn.Linear(hidden_dim, hidden_dim)
        self.encoder   = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder   = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, 256), nn.ReLU(),
            nn.Linear(256, pred_len * 2))

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        N    = obs.size(0)
        h    = self.embed(obs)
        _, h = self.encoder(h); h = h.squeeze(0)
        Q, K, V = self.gat_q(h), self.gat_k(h), self.gat_v(h)
        attn = F.softmax(Q @ K.T / (Q.size(-1) ** 0.5), dim=-1)
        h    = attn @ V
        samples = []
        for _ in range(k):
            z = torch.randn(N, self.noise_dim, device=obs.device)
            samples.append(
                self.decoder(torch.cat([h, z], -1)).view(N, self.pred_len, 2))
        return torch.stack(samples, dim=1)
