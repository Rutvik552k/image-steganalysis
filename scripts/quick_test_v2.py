"""Quick A/B test: v1 (current) vs v2 (fixed) on 500 samples, 5 epochs each.

v1 config (baseline):
  - max_grad_norm=1.0
  - Kendall uncertainty weighting
  - warmup=1

v2 config (fixed):
  - max_grad_norm=5.0
  - Fixed loss weights: binary=1.0, algo_class=0.3, algo_id=0.1, payload=0.1
  - warmup=1

Both use: 500 train images, 150 val, 5 epochs, batch 32, lr=1e-3, AMP.
"""
import sys
import os
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.data.dataset import create_dataloaders
from src.models.unisteg import UniStegLite
from src.training.train_loop import train


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_test_splits(splits_dir: str, out_dir: str, n_train: int = 500, n_val: int = 150):
    """Sample small splits for quick testing."""
    os.makedirs(out_dir, exist_ok=True)
    for name, n in [("train", n_train), ("val", n_val), ("test", n_val)]:
        df = pd.read_csv(os.path.join(splits_dir, f"{name}.csv"), low_memory=False)
        sampled = df.sample(n=min(n, len(df)), random_state=42)
        sampled.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        print(f"  {name}: {len(sampled)} samples")
    shutil.copy(
        os.path.join(splits_dir, "label_maps.json"),
        os.path.join(out_dir, "label_maps.json"),
    )


def run_experiment(name: str, train_cfg: dict, splits_dir: str, output_dir: str):
    """Run one training experiment and return results."""
    set_seed(42)

    with open(os.path.join(splits_dir, "label_maps.json")) as f:
        label_maps = json.load(f)
    num_algo_classes = len(label_maps["algorithm_class"])
    num_algorithms = len(label_maps["algorithm"])

    model = UniStegLite(
        num_experts=5,
        num_algo_classes=num_algo_classes,
        num_algorithms=num_algorithms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Device: {device}")
    print(f"  max_grad_norm: {train_cfg.get('max_grad_norm', 1.0)}")
    print(f"  loss_mode: {train_cfg.get('loss_mode', 'kendall')}")
    if train_cfg.get("fixed_loss_weights"):
        print(f"  fixed_weights: {train_cfg['fixed_loss_weights']}")
    print(f"{'='*60}\n")

    train_loader, val_loader, _ = create_dataloaders(
        splits_dir=splits_dir,
        batch_size=32,
        target_size=256,
        apply_srm=False,
        num_workers=2,
        pin_memory=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    best_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        output_dir=output_dir,
    )

    # Load training log for summary
    log_path = os.path.join(output_dir, "training_log.json")
    history = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)

    return best_acc, history


def main():
    splits_dir = os.environ.get("SPLITS_DIR", "data/splits")
    print("Creating test splits...")
    test_splits = "data/splits_ab_test"
    create_test_splits(splits_dir, test_splits, n_train=500, n_val=150)

    # Common config
    base_cfg = {
        "lr": 1e-3,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "epochs": 5,
        "warmup_epochs": 1,
        "accumulation_steps": 2,
        "use_amp": True,
        "patience": 100,  # no early stopping for fair comparison
        "plateau_patience": 100,
        "save_every": 10,
        "save_every_steps": 0,
        "keep_last_checkpoints": 1,
    }

    # --- v1: baseline (current settings) ---
    v1_cfg = {**base_cfg, "max_grad_norm": 1.0, "loss_mode": "kendall"}
    v1_acc, v1_hist = run_experiment("v1 (baseline)", v1_cfg, test_splits, "checkpoints_v1_test")

    # --- v2: fixed settings ---
    v2_cfg = {
        **base_cfg,
        "max_grad_norm": 5.0,
        "loss_mode": "fixed",
        "fixed_loss_weights": [1.0, 0.3, 0.1, 0.1],
    }
    v2_acc, v2_hist = run_experiment("v2 (fixed)", v2_cfg, test_splits, "checkpoints_v2_test")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'v1 (baseline)':>15} {'v2 (fixed)':>15} {'Delta':>10}")
    print(f"{'-'*60}")

    def fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    print(f"{'Best balanced_acc':<20} {fmt(v1_acc):>15} {fmt(v2_acc):>15} {fmt(v2_acc - v1_acc):>10}")

    # Per-epoch comparison
    if v1_hist and v2_hist:
        metrics_to_compare = [
            ("val_balanced_acc", "Val balanced_acc"),
            ("val_min_p_e", "Val min_P_E"),
            ("val_f1", "Val F1"),
            ("val_auc_roc", "Val AUC-ROC"),
            ("grad_norm_avg", "Avg grad norm"),
        ]
        print(f"\n{'Epoch-by-epoch comparison':}")
        print(f"{'Epoch':<8} {'v1 bal_acc':>12} {'v2 bal_acc':>12} {'v1 min_PE':>12} {'v2 min_PE':>12} {'v1 grad':>10} {'v2 grad':>10}")
        print(f"{'-'*68}")
        n_epochs = min(len(v1_hist.get("epochs", [])), len(v2_hist.get("epochs", [])))
        for i in range(n_epochs):
            v1_ba = v1_hist.get("val_balanced_acc", [0])[i]
            v2_ba = v2_hist.get("val_balanced_acc", [0])[i]
            v1_pe = v1_hist.get("val_min_p_e", [0.5])[i]
            v2_pe = v2_hist.get("val_min_p_e", [0.5])[i]
            v1_gn = v1_hist.get("grad_norm_avg", [0])[i]
            v2_gn = v2_hist.get("grad_norm_avg", [0])[i]
            print(f"{i+1:<8} {v1_ba:>12.4f} {v2_ba:>12.4f} {v1_pe:>12.4f} {v2_pe:>12.4f} {v1_gn:>10.2f} {v2_gn:>10.2f}")

        # Final epoch summary
        print(f"\nFinal epoch metrics:")
        for key, label in metrics_to_compare:
            v1_vals = v1_hist.get(key, [])
            v2_vals = v2_hist.get(key, [])
            if v1_vals and v2_vals:
                v1_final = v1_vals[-1]
                v2_final = v2_vals[-1]
                delta = v2_final - v1_final
                better = "v2" if (delta > 0 if "p_e" not in key else delta < 0) else "v1"
                print(f"  {label:<20} v1={v1_final:.4f}  v2={v2_final:.4f}  delta={delta:+.4f}  ({better} better)")

    print(f"\n{'='*60}")
    print(f"  v1 best: {v1_acc:.4f}  |  v2 best: {v2_acc:.4f}  |  improvement: {v2_acc - v1_acc:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
