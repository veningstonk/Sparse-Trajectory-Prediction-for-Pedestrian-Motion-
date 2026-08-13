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

## Datasets

Download and place under `data/raw/`:
- **ETH-UCY**: https://www.kaggle.com/datasets/menhari/ethucyjaad-data-set-with-labels
- **SDD**: https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset
