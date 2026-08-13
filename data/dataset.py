"""
data/dataset.py
---------------
ETH-UCY and Stanford Drone Dataset loaders.

Reviewer compliance:
  R2-4 : Implements the standard leave-one-scene-out (LOSO) protocol for
          ETH-UCY and the fixed half-split for SDD.  Observation horizon
          T=8, prediction horizon H=12 are enforced here, matching every
          baseline in Table 6.
  R2-4 : Coordinate normalisation (zero-mean / unit-std per training fold)
          is applied inside the dataset; statistics are computed on the
          training split only and reused for validation/test without
          recomputation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

# ── Constants (R2-4) ──────────────────────────────────────────────────────────
OBS_LEN   = 8   # T  — observation steps
PRED_LEN  = 12  # H  — prediction steps
SEQ_LEN   = OBS_LEN + PRED_LEN

# ETH-UCY scene names used in leave-one-scene-out protocol
ETH_UCY_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

# SDD scene names
SDD_SCENES = [
    "bookstore", "coupa", "deathCircle", "gates",
    "hyang", "little", "nexus", "quad"
]


# ── Coordinate normalisation (R2-4) ──────────────────────────────────────────
class Normalizer:
    """
    Zero-mean / unit-std normaliser.

    R2-4: Statistics computed on the training split only; the same
    statistics are applied to val/test without recomputation.
    """
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, coords: np.ndarray) -> "Normalizer":
        """coords : (N, 2)  — all observed positions in the training split."""
        self.mean = coords.mean(axis=0)
        self.std  = coords.std(axis=0).clip(min=1e-6)
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self.mean) / self.std

    def inverse_transform(self, coords: np.ndarray) -> np.ndarray:
        return coords * self.std + self.mean


# ── Raw trajectory reader ────────────────────────────────────────────────────
def read_file(path: str, delimiter: str = "\t") -> np.ndarray:
    """
    Read a trajectory text file.

    Expected columns: frame_id  pedestrian_id  x  y
    Returns: (N_rows, 4)
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) < 4:
                parts = line.split()
            data.append([float(p) for p in parts[:4]])
    return np.array(data)


def extract_sequences(raw: np.ndarray,
                      obs_len: int = OBS_LEN,
                      pred_len: int = PRED_LEN,
                      skip: int = 1) -> List[np.ndarray]:
    """
    Slide a window of length obs_len+pred_len over each pedestrian's track.

    Returns a list of arrays, each shaped (num_peds, seq_len, 2), containing
    only pedestrians present for the full window.
    """
    seq_len  = obs_len + pred_len
    frames   = np.unique(raw[:, 0]).tolist()
    frame_data = {f: raw[raw[:, 0] == f] for f in frames}
    sequences: List[np.ndarray] = []

    for i in range(0, len(frames) - seq_len + 1, skip):
        window_frames = frames[i: i + seq_len]
        # pedestrians present in every frame of the window
        peds_in_all = set(frame_data[window_frames[0]][:, 1])
        for fr in window_frames[1:]:
            peds_in_all &= set(frame_data[fr][:, 1])
        if not peds_in_all:
            continue

        peds = sorted(peds_in_all)
        seq = np.zeros((len(peds), seq_len, 2))
        for t, fr in enumerate(window_frames):
            fd = frame_data[fr]
            for pi, ped in enumerate(peds):
                row = fd[fd[:, 1] == ped]
                seq[pi, t] = row[0, 2:4]
        sequences.append(seq)

    return sequences


# ── Base trajectory dataset ───────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    """
    Holds a list of sequence arrays and exposes them as tensors.

    R2-4 : Normalisation is applied here using a pre-fitted Normalizer so
           test statistics are never contaminated by training data.
    """
    def __init__(self,
                 sequences: List[np.ndarray],
                 normalizer: Optional[Normalizer] = None,
                 fit_normalizer: bool = False):
        self.obs_len  = OBS_LEN
        self.pred_len = PRED_LEN

        all_coords = np.concatenate(
            [s.reshape(-1, 2) for s in sequences], axis=0
        )

        if fit_normalizer:
            normalizer = Normalizer().fit(all_coords)
        self.normalizer = normalizer

        self.sequences: List[np.ndarray] = []
        for seq in sequences:
            shape = seq.shape          # (num_peds, seq_len, 2)
            if self.normalizer is not None:
                seq = self.normalizer.transform(seq.reshape(-1, 2)).reshape(shape)
            self.sequences.append(seq.astype(np.float32))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]                         # (P, T+H, 2)
        obs  = torch.tensor(seq[:, :self.obs_len])        # (P, T, 2)
        pred = torch.tensor(seq[:, self.obs_len:])        # (P, H, 2)
        return {"obs": obs, "pred": pred,
                "num_peds": torch.tensor(seq.shape[0])}


# ── ETH-UCY  — leave-one-scene-out (R2-4) ────────────────────────────────────
def load_eth_ucy_fold(root: str,
                      test_scene: str,
                      batch_size: int = 128,
                      skip: int = 1
                      ) -> Tuple[DataLoader, DataLoader, Normalizer]:
    """
    Standard ETH-UCY leave-one-scene-out protocol.

    R2-4: The model is trained on four scenes and tested on the held-out
    fifth. This is repeated for every scene, and results are averaged
    across all five folds.

    Parameters
    ----------
    root       : path to ETH-UCY data directory
    test_scene : one of ETH_UCY_SCENES — the held-out scene
    batch_size : mini-batch size (128 per Table 2)
    skip       : frame stride

    Returns
    -------
    train_loader, test_loader, normalizer
    """
    assert test_scene in ETH_UCY_SCENES, f"Unknown scene: {test_scene}"

    train_seqs, test_seqs = [], []

    for scene in ETH_UCY_SCENES:
        path = os.path.join(root, scene, "pixel_pos_interpolate.csv")
        if not os.path.exists(path):
            path = os.path.join(root, f"{scene}_test.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found for scene '{scene}' under {root}"
            )
        raw  = read_file(path)
        seqs = extract_sequences(raw, skip=skip)
        if scene == test_scene:
            test_seqs.extend(seqs)
        else:
            train_seqs.extend(seqs)

    train_ds = TrajectoryDataset(train_seqs, fit_normalizer=True)
    test_ds  = TrajectoryDataset(test_seqs,
                                 normalizer=train_ds.normalizer,
                                 fit_normalizer=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, train_ds.normalizer


# ── SDD — fixed half-split (R2-4) ────────────────────────────────────────────
def load_sdd(root: str,
             batch_size: int = 128,
             skip: int = 1
             ) -> Tuple[DataLoader, DataLoader, Normalizer]:
    """
    Stanford Drone Dataset with the standard fixed half-split.

    R2-4: First half of each scene's trajectories → training;
          second half → test.  No scene-level cross-validation.
    """
    train_seqs, test_seqs = [], []

    for scene in SDD_SCENES:
        scene_dir = os.path.join(root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for video in sorted(os.listdir(scene_dir)):
            ann_path = os.path.join(scene_dir, video,
                                    "annotations.txt")
            if not os.path.exists(ann_path):
                continue
            raw  = read_file(ann_path, delimiter=" ")
            seqs = extract_sequences(raw, skip=skip)
            mid  = len(seqs) // 2
            train_seqs.extend(seqs[:mid])
            test_seqs.extend(seqs[mid:])

    train_ds = TrajectoryDataset(train_seqs, fit_normalizer=True)
    test_ds  = TrajectoryDataset(test_seqs,
                                 normalizer=train_ds.normalizer,
                                 fit_normalizer=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, train_ds.normalizer


# ── Collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """
    Variable-pedestrian-count batching: each item keeps its own tensor;
    the batch is a list of dicts rather than a stacked tensor.
    """
    return {"scenes": batch}
