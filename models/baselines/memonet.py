"""MemoNet baseline (simplified)."""
import torch, torch.nn as nn
from data.dataset import OBS_LEN, PRED_LEN

class MemoNet(nn.Module):
    def __init__(self, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 hidden_dim=128, mem_size=512, noise_dim=16):
        super().__init__()
        self.pred_len  = pred_len
        self.noise_dim = noise_dim
        self.encoder   = nn.LSTM(2, hidden_dim, batch_first=True)
        self.memory    = nn.Embedding(mem_size, hidden_dim)
        self.attn      = nn.Linear(hidden_dim, mem_size)
        self.decoder   = nn.Sequential(
            nn.Linear(hidden_dim * 2 + noise_dim, 256), nn.ReLU(),
            nn.Linear(256, pred_len * 2))

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        N     = obs.size(0)
        _, (h, _) = self.encoder(obs); h = h.squeeze(0)
        a     = torch.softmax(self.attn(h), dim=-1)       # (N, mem_size)
        mem_r = a @ self.memory.weight                    # (N, hidden_dim)
        samples = []
        for _ in range(k):
            z = torch.randn(N, self.noise_dim, device=obs.device)
            samples.append(
                self.decoder(torch.cat([h, mem_r, z], -1)).view(N, self.pred_len, 2))
        return torch.stack(samples, dim=1)
