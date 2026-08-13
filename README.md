# STP — Sparse Trajectory Prediction

End-to-end implementation of the STP framework: Transformer + GNN +
Variational Inference + Early Sparsity Optimisation for real-time
pedestrian trajectory prediction.

---

## Reviewer Compliance Index

Every reviewer comment addressed in the revision is reflected directly
in the code. The table below maps each comment to the implementing file
and the specific functions/classes involved.

| Comment | File | Symbol |
|---------|------|--------|
| **R1-1** Architectural novelty vs Trajectron++/AgentFormer | `models/stp.py` | `STPModel` docstring; `models/baselines/trajectronpp.py` inline comment |
| **R1-2** Comparison with motion primitive methods | `models/baselines/sgcn.py` | `SGCN`; `evaluation/metrics.py`; `scripts/evaluate.py` |
| **R1-3** Social-STGMLP added | `models/baselines/social_stgmlp.py` | `SocialSTGMLP` |
| **R1-3** D-STGCN added | `models/baselines/d_stgcn.py` | `DSTGCN` |
| **R2-1** Mode initialisation (K-means) | `models/early_sparsity.py` | `EarlySparsityModule.initialise_modes_kmeans` |
| **R2-1** Algorithm 1 (joint training) | `training/trainer.py` | `STPTrainer.train_one_epoch` |
| **R2-1** Algorithm 2 (inference) | `models/stp.py` | `STPModel.predict`; `models/early_sparsity.py` `EarlySparsityModule.infer` |
| **R2-2** Q-K-V attention (not raw positions) | `models/gnn.py` | `MultiHeadLocalAttention`, `PositionEmbedding` |
| **R2-2** Eq. 5a position embedding | `models/gnn.py` | `PositionEmbedding` |
| **R2-2** Eq. 23 self-dot-product fix | `models/gnn.py` | `MultiHeadLocalAttention.forward` |
| **R2-3** argmin over hypotheses (Eq. 31) | `models/stp.py` | `STPModel.predict` (k_star selection); `evaluation/metrics.py` `min_ade_k` |
| **R2-3** minADE@K / minFDE@K metrics | `evaluation/metrics.py` | `min_ade_k`, `min_fde_k` |
| **R2-4** ETH-UCY leave-one-scene-out | `data/dataset.py` | `load_eth_ucy_fold`; `scripts/evaluate.py` `run_eth_ucy_loso` |
| **R2-4** SDD fixed half-split | `data/dataset.py` | `load_sdd` |
| **R2-4** T=8, H=12 | `data/dataset.py` | `OBS_LEN=8`, `PRED_LEN=12` |
| **R2-4** Coordinate normalisation | `data/dataset.py` | `Normalizer` |
| **R2-4** 3 random seeds, mean±std | `training/trainer.py` | `run_with_seeds`; `set_seed` |
| **R2-4** Baseline result sourcing note | `scripts/evaluate.py` | Table 6 footer comment |
| **R2-5** NLL calibration metric | `evaluation/metrics.py` | `negative_log_likelihood` |
| **R2-5** ECE calibration metric | `evaluation/metrics.py` | `expected_calibration_error` |
| **Internal** Eq. 29 L_ds → L_cls | `training/losses.py` | `total_loss` comment |
| **Internal** Neighbour loss conditioned on N_i | `training/losses.py` | `neighbour_loss(adj=...)` |
| **Internal** VAE decoder f_decoder(z) not z directly | `models/vae.py` | `VAEDecoder.forward` |

---

## Project Structure

```
stp/
├── configs/
│   ├── eth_ucy.yaml        # Table 2 hyperparameters for ETH-UCY
│   └── sdd.yaml            # Table 2 hyperparameters for SDD
├── data/
│   └── dataset.py          # ETH-UCY LOSO, SDD split, normalisation (R2-4)
├── models/
│   ├── stp.py              # Main STP model (Algorithms 1 & 2)
│   ├── gnn.py              # GNN encoder with Q-K-V attention (R2-2)
│   ├── transformer.py      # Transformer temporal encoder
│   ├── vae.py              # VAE with learned decoder (not z-as-mean)
│   ├── early_sparsity.py   # ESO: K-means init + coefficient net (R2-1)
│   └── baselines/
│       ├── social_lstm.py
│       ├── social_gan.py
│       ├── trajectronpp.py
│       ├── social_stgmlp.py  ← R1-3
│       ├── d_stgcn.py        ← R1-3
│       ├── sgcn.py           ← R1-2 (motion primitive comparison)
│       ├── gat_baseline.py
│       ├── memonet.py
│       └── social_vae.py
├── training/
│   ├── trainer.py          # Algorithm 1 (R2-1)
│   └── losses.py           # All loss functions (L_cls fix, neighbour fix)
├── evaluation/
│   └── metrics.py          # ADE, FDE, minADE@K, minFDE@K, NLL, ECE
└── scripts/
    ├── train.py            # Training entry point
    └── evaluate.py         # Full evaluation: Table 6 + Table 5
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train STP — ETH-UCY, test scene = eth
python scripts/train.py --scene eth --dataset eth_ucy --config configs/eth_ucy.yaml

# Train with 3 seeds (R2-4: mean ± std)
python scripts/train.py --scene eth --dataset eth_ucy --multi_seed

# Full evaluation (Table 6 + calibration Table 5)
python scripts/evaluate.py \
    --dataset eth_ucy \
    --data_root data/raw \
    --ckpt checkpoints/eth_ucy/stp_best_seed42.pth \
    --k_samples 20
```

---

## Key Design Decisions (Reviewer-Driven)

### R2-2 — Q-K-V Attention
Raw 2D positions are *never* used as attention inputs. The pipeline is:

```
P_i(t) ∈ R^2  →  W_emb * P_i  →  h_i^(0) ∈ R^d_model   [Eq. 5a]
h_i^(l)       →  W_Q * h_i,  W_K * h_j,  W_V * h_j       [Eq. 5b]
alpha_ij = softmax( Q_i^T K_j / sqrt(d_k) )               [Eq. 5]
h_i^att  = sum_j alpha_ij * V_j                            [Eq. 6]
```

### R2-3 — Hypothesis Selection
Inference selects the best of K=20 hypotheses by argmin *over k*,
not over time steps:

```python
k_star = dists.argmin(dim=1)      # (N,)  — over K hypotheses
p_selected = p_all[arange(N), k_star]   # (N, H, 2)
```

### R2-1 — Mode Initialisation
K-means runs on all training displacement sequences before epoch 1:

```python
trainer.initialise_modes()   # Algorithm 1, Steps 1-3
trainer.train()              # Algorithm 1, Steps 5-26
```

---

## Datasets

Download and place under `data/raw/`:
- **ETH-UCY**: https://www.kaggle.com/datasets/menhari/ethucyjaad-data-set-with-labels
- **SDD**: https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset
