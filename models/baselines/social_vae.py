"""SocialVAE-FPC baseline (simplified)."""
import torch, torch.nn as nn
from data.dataset import OBS_LEN, PRED_LEN

class SocialVAEBaseline(nn.Module):
    def __init__(self, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 hidden_dim=128, latent_dim=32):
        super().__init__()
        self.pred_len   = pred_len
        self.encoder    = nn.LSTM(2, hidden_dim, batch_first=True)
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim, latent_dim)
        self.decoder    = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 256), nn.ReLU(),
            nn.Linear(256, pred_len * 2))

    def forward(self, obs: torch.Tensor, k: int = 20) -> torch.Tensor:
        N = obs.size(0)
        _, (h, _) = self.encoder(obs); h = h.squeeze(0)
        mu  = self.fc_mu(h)
        std = torch.exp(0.5 * self.fc_logvar(h))
        samples = []
        for _ in range(k):
            z = mu + std * torch.randn_like(std)
            samples.append(
                self.decoder(torch.cat([h, z], -1)).view(N, self.pred_len, 2))
        return torch.stack(samples, dim=1)
