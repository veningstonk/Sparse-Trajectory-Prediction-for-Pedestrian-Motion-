"""
training/trainer.py
-------------------
STP training loop — implements Algorithm 1 from the manuscript.


"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Dict

from models.stp          import STPModel
from training.losses     import total_loss
from evaluation.metrics  import compute_all_metrics
from data.dataset        import OBS_LEN, PRED_LEN


def set_seed(seed: int) -> None:
    """R2-4: fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_displacement_seqs(loader: DataLoader,
                               pred_len: int = PRED_LEN
                               ) -> np.ndarray:
    """
    Algorithm 1, Steps 1-2: compute H-step displacement sequences
    Delta_p_i(t) = p_i(t) - p_i(t-1) for all pedestrians in training set.
    Used for K-means mode initialisation.
    """
    seqs = []
    for batch in loader:
        for scene in batch["scenes"]:
            pred = scene["pred"].numpy()                # (N, H, 2)
            for n in range(pred.shape[0]):
                traj = pred[n]                          # (H, 2)
                disp = np.diff(traj, axis=0,
                               prepend=traj[:1])        # (H, 2) displacements
                seqs.append(disp)
    return np.array(seqs)                              # (N_total, H, 2)


class STPTrainer:
    """
    End-to-end trainer for the STP model.

    Implements Algorithm 1 with all five loss components,
    AdamW optimiser, and TensorBoard logging.
    """
    def __init__(self,
                 model:       STPModel,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 epochs:       int   = 100,
                 lr:           float = 1e-3,
                 weight_decay: float = 1e-2,
                 dropout:      float = 0.3,
                 lambda_1:     float = 1.0,
                 lambda_2:     float = 0.5,
                 lambda_3:     float = 0.5,
                 lambda_sp:    float = 0.1,
                 lambda_kl:    float = 1.0,
                 device:       str   = "cuda",
                 log_dir:      str   = "runs/stp",
                 ckpt_dir:     str   = "checkpoints",
                 seed:         int   = 42):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.epochs       = epochs
        self.device       = device
        self.ckpt_dir     = ckpt_dir
        self.seed         = seed

        self.loss_weights = dict(lambda_1=lambda_1, lambda_2=lambda_2,
                                 lambda_3=lambda_3, lambda_sp=lambda_sp,
                                 lambda_kl=lambda_kl)

        # Algorithm 1, Step 24 — AdamW with LR and weight decay (Table 2)
        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Table 2 — linear warm-up for 10% of training epochs
        warmup_steps = max(1, int(0.1 * epochs))
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimiser,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimiser, start_factor=0.1, end_factor=1.0,
                    total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimiser, T_max=epochs - warmup_steps),
            ],
            milestones=[warmup_steps]
        )

        self.writer = SummaryWriter(log_dir=log_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

    def initialise_modes(self) -> None:
        """
        Algorithm 1, Steps 1-3: K-means mode initialisation.
        Must be called BEFORE the first training epoch.
        """
        print("[Trainer] Collecting displacement sequences for K-means…")
        disp_seqs = collect_displacement_seqs(self.train_loader)
        self.model.eso.initialise_modes_kmeans(disp_seqs)

    def _scene_to_device(self, scene: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in scene.items()}

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Algorithm 1, Steps 5-25 — one epoch."""
        self.model.train()
        totals = {k: 0.0 for k in
                  ["total", "reg", "cls", "nei", "sparse", "kl"]}
        n_batches = 0

        for batch in self.train_loader:
            batch_loss = {k: 0.0 for k in totals}
            n_scenes   = 0

            for scene in batch["scenes"]:
                scene  = self._scene_to_device(scene)
                obs    = scene["obs"]        # (N, T, 2)
                p_gt   = scene["pred"]       # (N, H, 2)
                N      = obs.size(0)
                if N < 1:
                    continue

                # Build adjacency for neighbour loss (Eq. 28 fix)
                pos_last = obs[:, -1, :]
                diff     = (pos_last.unsqueeze(0) -
                            pos_last.unsqueeze(1)).norm(dim=-1)
                adj      = diff <= self.model.gnn.delta

                # Algorithm 1, Steps 7-16 — forward pass
                outputs  = self.model(obs)

                # Algorithm 1, Steps 17-22 — loss computation
                losses   = total_loss(
                    outputs, p_gt,
                    modes=self.model.eso.modes,
                    adj=adj,
                    **self.loss_weights
                )
                for k in batch_loss:
                    batch_loss[k] += losses[k].item()
                n_scenes += 1

                # Algorithm 1, Steps 23-24 — backprop + AdamW update
                self.optimiser.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimiser.step()

            if n_scenes > 0:
                for k in totals:
                    totals[k] += batch_loss[k] / n_scenes
                n_batches += 1

        self.scheduler.step()

        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        for k, v in avg.items():
            self.writer.add_scalar(f"train/{k}_loss", v, epoch)
        return avg

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run minADE@20 / minFDE@20 on validation set (R2-3, R2-4)."""
        self.model.eval()
        all_metrics: List[Dict] = []

        with torch.no_grad():
            for batch in self.val_loader:
                for scene in batch["scenes"]:
                    scene = self._scene_to_device(scene)
                    obs   = scene["obs"]
                    p_gt  = scene["pred"]
                    out   = self.model.predict(obs, k_samples=20)
                    m     = compute_all_metrics(
                        out["p_all"], out["p_selected"],
                        p_gt, out["mu"], out["sigma"]
                    )
                    all_metrics.append(m)

        avg = {}
        for key in all_metrics[0]:
            avg[key] = float(np.mean([m[key] for m in all_metrics]))
        for k, v in avg.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        return avg

    def train(self, init_modes: bool = True) -> None:
        """
        Full training procedure — Algorithm 1.

        Parameters
        ----------
        init_modes : run K-means initialisation before epoch 1 (recommended)
        """
        set_seed(self.seed)

        if init_modes:
            self.initialise_modes()   # Algorithm 1, Steps 1-3

        best_ade = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0     = time.time()
            train_m = self.train_one_epoch(epoch)
            val_m   = self.validate(epoch)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"loss {train_m['total']:.4f} | "
                f"minADE@20 {val_m.get('minADE20', float('nan')):.4f} | "
                f"minFDE@20 {val_m.get('minFDE20', float('nan')):.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_m.get("minADE20", float("inf")) < best_ade:
                best_ade = val_m["minADE20"]
                path = os.path.join(self.ckpt_dir,
                                    f"stp_best_seed{self.seed}.pth")
                torch.save({
                    "epoch":       epoch,
                    "state_dict":  self.model.state_dict(),
                    "optimiser":   self.optimiser.state_dict(),
                    "val_metrics": val_m,
                }, path)

        self.writer.close()
        print(f"[Trainer] Training complete. Best minADE@20: {best_ade:.4f}")


def run_with_seeds(model_cls,
                   model_kwargs:  Dict,
                   trainer_kwargs: Dict,
                   seeds:         List[int] = [42, 123, 456]) -> Dict:
    """
    R2-4: Run training with multiple seeds; report mean ± std.

    Returns aggregated metrics dict.
    """
    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        model   = model_cls(**model_kwargs)
        trainer = STPTrainer(model, seed=seed, **trainer_kwargs)
        trainer.train()
        # Load best checkpoint and evaluate
        ckpt    = torch.load(
            os.path.join(trainer.ckpt_dir, f"stp_best_seed{seed}.pth"),
            map_location="cpu"
        )
        all_results.append(ckpt["val_metrics"])

    # R2-4: aggregate across seeds
    final = {}
    for key in all_results[0]:
        vals       = [r[key] for r in all_results]
        final[key] = {"mean": float(np.mean(vals)),
                      "std":  float(np.std(vals))}
        print(f"  {key}: {final[key]['mean']:.4f} ± {final[key]['std']:.4f}")
    return final
