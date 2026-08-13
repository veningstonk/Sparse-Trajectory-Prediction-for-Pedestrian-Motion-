"""
Social GAN (Gupta et al., CVPR 2018).

R2-4: Reports single-sample ADE/FDE.
      Multimodal via GAN noise; K samples drawn at inference.
"""
import torch
import torch.nn as nn
from data.dataset import OBS_LEN, PRED_LEN


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x: torch.Tensor):
        _, (h, c) = self.lstm(x)
        return h.squeeze(0), c.squeeze(0)


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 64, noise_dim: int = 16,
                 pred_len: int = PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.lstm     = nn.LSTM(2 + noise_dim, hidden_dim, batch_first=True)
        self.linear   = nn.Linear(hidden_dim, 2)

    def forward(self, last_pos: torch.Tensor, h: torch.Tensor,
                c: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        noise_rep = noise.unsqueeze(1).expand(-1, self.pred_len, -1)
        last_rep  = last_pos.unsqueeze(1).expand(-1, self.pred_len, -1)
        inp, _    = self.lstm(torch.cat([last_rep, noise_rep], dim=-1),
                              (h.unsqueeze(0), c.unsqueeze(0)))
        return self.linear(inp)


class SocialGAN(nn.Module):
    def __init__(self, obs_len: int = OBS_LEN, pred_len: int = PRED_LEN,
                 hidden_dim: int = 64, noise_dim: int = 16):
        super().__init__()
        self.obs_len   = obs_len
        self.pred_len  = pred_len
        self.noise_dim = noise_dim
        self.encoder   = EncoderLSTM(2, hidden_dim)
        self.decoder   = DecoderLSTM(hidden_dim, noise_dim, pred_len)

    def forward(self, obs: torch.Tensor,
                k: int = 20) -> torch.Tensor:
        """
        obs : (N, T, 2)
        Returns : (N, K, H, 2)  for minADE@K evaluation
        """
        N        = obs.size(0)
        h, c     = self.encoder(obs)
        last_pos = obs[:, -1]
        samples  = []
        for _ in range(k):
            noise  = torch.randn(N, self.noise_dim, device=obs.device)
            traj   = self.decoder(last_pos, h, c, noise)  # (N, H, 2)
            samples.append(traj)
        return torch.stack(samples, dim=1)                 # (N, K, H, 2)
