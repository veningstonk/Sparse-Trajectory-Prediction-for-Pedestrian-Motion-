"""
scripts/train.py
----------------
End-to-end training entry point.

Usage
-----
  # Single seed, ETH-UCY leave-one-scene-out fold:
  python scripts/train.py --scene eth --dataset eth_ucy

  # Multi-seed run (R2-4: 3 seeds, mean ± std reported):
  python scripts/train.py --scene eth --dataset eth_ucy --multi_seed

  # SDD:
  python scripts/train.py --dataset sdd
"""

import argparse
import yaml
import torch

from data.dataset   import (load_eth_ucy_fold, load_sdd, ETH_UCY_SCENES)
from models.stp     import STPModel
from training.trainer import STPTrainer, run_with_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/eth_ucy.yaml")
    parser.add_argument("--scene",      default="eth",
                        choices=ETH_UCY_SCENES)
    parser.add_argument("--dataset",    default="eth_ucy",
                        choices=["eth_ucy", "sdd"])
    parser.add_argument("--data_root",  default="data/raw")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true",
                        help="Run 3 seeds and report mean±std (R2-4)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"
    model_cfg   = cfg.get("model",   {})
    trainer_cfg = cfg.get("trainer", {})
    trainer_cfg.update({"device": device,
                         "log_dir": f"runs/stp_{args.dataset}_{args.scene}"})

    if args.dataset == "eth_ucy":
        train_loader, val_loader, _ = load_eth_ucy_fold(
            args.data_root, args.scene,
            batch_size=trainer_cfg.pop("batch_size", 128)
        )
    else:
        train_loader, val_loader, _ = load_sdd(
            args.data_root,
            batch_size=trainer_cfg.pop("batch_size", 128)
        )

    if args.multi_seed:
        # R2-4: multi-seed run
        run_with_seeds(
            STPModel, model_cfg,
            dict(train_loader=train_loader,
                 val_loader=val_loader, **trainer_cfg),
            seeds=[42, 123, 456]
        )
    else:
        model   = STPModel(**model_cfg).to(device)
        trainer = STPTrainer(model, train_loader, val_loader,
                             seed=args.seed, **trainer_cfg)
        trainer.train(init_modes=True)


if __name__ == "__main__":
    main()
