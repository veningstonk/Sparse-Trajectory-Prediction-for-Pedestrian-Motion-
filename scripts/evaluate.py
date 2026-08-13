"""
scripts/evaluate.py
-------------------
Full evaluation pipeline: STP + all baselines on ETH-UCY and SDD.


"""

import argparse
import numpy as np
import torch
import yaml
from typing import Dict, List

from data.dataset        import (load_eth_ucy_fold, load_sdd,
                                  ETH_UCY_SCENES)
from models.stp          import STPModel
from models.baselines    import BASELINE_REGISTRY
from evaluation.metrics  import compute_all_metrics


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, device: str, k: int = 20,
                   is_stp: bool = False) -> Dict[str, float]:
    """Run evaluation on a DataLoader; return averaged metrics."""
    model.eval()
    all_m: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            for scene in batch["scenes"]:
                obs  = scene["obs"].to(device)
                p_gt = scene["pred"].to(device)

                if is_stp:
                    out       = model.predict(obs, k_samples=k)
                    p_all     = out["p_all"]
                    p_sel     = out["p_selected"]
                    mu, sigma = out["mu"], out["sigma"]
                    sigma_h   = sigma.mean(-1, keepdim=True).expand(
                        -1, p_gt.size(1))
                else:
                    p_all = model(obs, k=k)                # (N, K, H, 2)
                    # Use best-of-K as selected (oracle for baselines)
                    gt_exp   = p_gt.unsqueeze(1).expand_as(p_all)
                    ade_each = (p_all - gt_exp).norm(dim=-1).mean(-1)
                    k_star   = ade_each.argmin(dim=1)
                    p_sel    = p_all[torch.arange(p_all.size(0)), k_star]
                    mu       = p_sel[:, -1].mean(-1, keepdim=True)
                    sigma_h  = torch.ones_like(p_gt[:, :, 0])

                m = compute_all_metrics(p_all, p_sel, p_gt, mu, sigma_h)
                all_m.append(m)

    return {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}


def run_eth_ucy_loso(model, is_stp: bool, data_root: str,
                     device: str, k: int = 20) -> Dict[str, float]:
    """
    R2-4: Full leave-one-scene-out protocol.
    Train on 4 scenes, test on 1, repeat 5 times, average results.
    """
    fold_results: List[Dict] = []
    for scene in ETH_UCY_SCENES:
        print(f"  [LOSO] test scene = {scene}")
        _, test_loader, _ = load_eth_ucy_fold(data_root, scene)
        m = evaluate_model(model, test_loader, device, k, is_stp)
        fold_results.append(m)
        print(f"         minADE@{k}: {m['minADE20']:.4f}  "
              f"minFDE@{k}: {m['minFDE20']:.4f}")

    return {key: float(np.mean([r[key] for r in fold_results]))
            for key in fold_results[0]}


def run_sdd(model, is_stp: bool, data_root: str,
            device: str, k: int = 20) -> Dict[str, float]:
    """R2-4: SDD fixed half-split evaluation."""
    _, test_loader, _ = load_sdd(data_root)
    return evaluate_model(model, test_loader, device, k, is_stp)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STP Full Evaluation")
    parser.add_argument("--config",     default="configs/eth_ucy.yaml")
    parser.add_argument("--ckpt",       default="checkpoints/stp_best_seed42.pth")
    parser.add_argument("--dataset",    choices=["eth_ucy", "sdd"],
                        default="eth_ucy")
    parser.add_argument("--data_root",  default="data/raw")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--k_samples",  type=int, default=20)
    parser.add_argument("--baselines",  nargs="+",
                        default=list(BASELINE_REGISTRY.keys()))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Evaluate STP ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STP MODEL")
    print("="*60)
    stp = STPModel(**cfg.get("model", {})).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    stp.load_state_dict(ckpt["state_dict"])

    if args.dataset == "eth_ucy":
        stp_metrics = run_eth_ucy_loso(
            stp, True, args.data_root, device, args.k_samples)
    else:
        stp_metrics = run_sdd(stp, True, args.data_root, device, args.k_samples)

    print(f"\nSTP results (avg):")
    for k, v in stp_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # ── Evaluate baselines ──────────────────────────────────────────────────
    results = {"STP": stp_metrics}

    for name in args.baselines:
        if name not in BASELINE_REGISTRY:
            continue
        print(f"\n{'='*60}\nBaseline: {name}\n{'='*60}")
        baseline = BASELINE_REGISTRY[name]().to(device)

        if args.dataset == "eth_ucy":
            m = run_eth_ucy_loso(
                baseline, False, args.data_root, device, args.k_samples)
        else:
            m = run_sdd(baseline, False, args.data_root, device, args.k_samples)

        results[name] = m
        print(f"  minADE@{args.k_samples}: {m['minADE20']:.4f}  "
              f"minFDE@{args.k_samples}: {m['minFDE20']:.4f}")

    # ── Print Table 6 / Table 5 ─────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"TABLE 6 — {args.dataset.upper()} (T=8, H=12, K={args.k_samples})")
    print("="*70)
    print(f"{'Method':<22} {'minADE@20':>10} {'minFDE@20':>10} "
          f"{'NLL':>8} {'ECE':>8}")
    print("-"*70)
    for method, m in results.items():
        print(f"{method:<22} {m['minADE20']:>10.4f} {m['minFDE20']:>10.4f} "
              f"{m.get('NLL', float('nan')):>8.4f} "
              f"{m.get('ECE', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
