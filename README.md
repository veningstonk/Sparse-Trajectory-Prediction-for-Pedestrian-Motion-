# STP — Sparse Trajectory Prediction

End-to-end implementation of the STP framework: Transformer + GNN +
Variational Inference + Early Sparsity Optimisation for real-time
pedestrian trajectory prediction.

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
