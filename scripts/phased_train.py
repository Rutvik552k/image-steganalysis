"""
Phased Training for UniSteg

Splits training data into non-overlapping phases and trains sequentially,
resuming from previous phase's best checkpoint.

Phases:
  1: 50,000 images
  2: 100,000 images
  3: 100,000 images
  4: 100,000 images (adjusted to available data)
  5: remaining images

Usage:
    python scripts/phased_train.py --splits-dir data/splits --output-dir checkpoints --epochs-per-phase 20
    python scripts/phased_train.py --phase 3 --resume checkpoints/phase_2/best.pt  # resume from phase 3
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Phase sizes (images per phase)
PHASE_SIZES = [50_000, 100_000, 100_000, 100_000]  # phase 5 = remainder


def split_into_phases(
    train_csv: str,
    output_dir: str,
    seed: int = 42,
) -> list[dict]:
    """Split train.csv into non-overlapping phase CSVs.

    Returns list of phase info dicts with keys: phase, csv_path, num_images.
    """
    df = pd.read_csv(train_csv, low_memory=False)
    total = len(df)
    print(f"Total training images: {total:,}")

    # Deterministic shuffle
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total)

    os.makedirs(output_dir, exist_ok=True)
    phases = []
    offset = 0

    for i, size in enumerate(PHASE_SIZES):
        phase_num = i + 1
        actual_size = min(size, total - offset)
        if actual_size <= 0:
            print(f"Phase {phase_num}: skipped (no images left)")
            continue

        phase_idx = indices[offset:offset + actual_size]
        phase_df = df.iloc[phase_idx]
        csv_path = os.path.join(output_dir, f"phase_{phase_num}_train.csv")
        phase_df.to_csv(csv_path, index=False)

        phases.append({
            "phase": phase_num,
            "csv_path": csv_path,
            "num_images": actual_size,
        })
        print(f"Phase {phase_num}: {actual_size:,} images -> {csv_path}")
        offset += actual_size

    # Phase 5: remainder
    remaining = total - offset
    if remaining > 0:
        phase_idx = indices[offset:]
        phase_df = df.iloc[phase_idx]
        csv_path = os.path.join(output_dir, "phase_5_train.csv")
        phase_df.to_csv(csv_path, index=False)
        phases.append({
            "phase": 5,
            "csv_path": csv_path,
            "num_images": remaining,
        })
        print(f"Phase 5: {remaining:,} images -> {csv_path}")
    else:
        print("Phase 5: skipped (no remaining images)")

    # Save phase manifest
    manifest_path = os.path.join(output_dir, "phases.json")
    with open(manifest_path, "w") as f:
        json.dump(phases, f, indent=2)
    print(f"\nPhase manifest: {manifest_path}")

    return phases


def train_phase(
    phase_num: int,
    phase_csv: str,
    val_csv: str,
    label_maps_path: str,
    output_dir: str,
    epochs: int,
    resume_path: str | None,
    batch_size: int,
    lr: float,
    seed: int,
    num_workers: int = 4,
):
    """Train one phase."""
    import torch
    import yaml
    from src.data.dataset import SteganalysisDataset, BayarConv2d
    from src.models.unisteg import UniStegLite
    from src.training.train_loop import train
    from torch.utils.data import DataLoader, WeightedRandomSampler

    # Seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")

    # Label maps
    with open(label_maps_path) as f:
        label_maps = json.load(f)
    num_algo_classes = len(label_maps.get("algorithm_class", {}))
    num_algorithms = len(label_maps.get("algorithm", {}))
    print(f"Label maps: {num_algo_classes} algo classes, {num_algorithms} algorithms")

    # Model
    model = UniStegLite(
        num_experts=5,
        num_algo_classes=num_algo_classes,
        num_algorithms=num_algorithms,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UniStegLite — {total_params:,} params")

    # Phase dataset
    train_count = sum(1 for _ in open(phase_csv)) - 1
    use_preload = train_count <= 220000
    print(f"Phase {phase_num} train: {train_count:,} images (preload={use_preload})")

    train_ds = SteganalysisDataset(
        csv_path=phase_csv,
        label_maps_path=label_maps_path,
        target_size=256,
        apply_srm=False,
        augment=True,
        preload=use_preload,
    )

    val_ds = SteganalysisDataset(
        csv_path=val_csv,
        label_maps_path=label_maps_path,
        target_size=256,
        apply_srm=False,
        augment=False,
    )

    # Balanced sampler
    binary_labels = train_ds.df["is_stego"].values
    class_counts = np.bincount(binary_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[binary_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Phase output dir
    phase_out = os.path.join(output_dir, f"phase_{phase_num}")
    os.makedirs(phase_out, exist_ok=True)

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(phase_out, "runs")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard: {tb_dir}/")
    except ImportError:
        print("TensorBoard not installed, skipping")

    # Training config
    accum = max(1, 64 // batch_size)  # target effective batch ~64
    train_cfg = {
        "lr": lr,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "warmup_epochs": min(3, epochs // 4),
        "accumulation_steps": accum,
        "max_grad_norm": 1.0,
        "patience": max(10, epochs // 2),
        "save_every": max(1, epochs // 5),
        "save_every_steps": 500,
        "keep_last_checkpoints": 3,
        "plateau_patience": 5,
        "use_amp": True,
    }

    print(f"Config: {epochs} epochs, batch {batch_size}x{accum}={batch_size*accum} eff, lr={lr}")
    if resume_path:
        print(f"Resume: {resume_path}")

    best_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        output_dir=phase_out,
        resume_path=resume_path,
        writer=writer,
    )

    if writer:
        writer.close()

    print(f"\nPhase {phase_num} complete. Best balanced_acc: {best_acc:.4f}")
    print(f"Checkpoints: {phase_out}/best.pt, {phase_out}/last.pt")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Phased UniSteg Training")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs-per-phase", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    # Resume from specific phase
    parser.add_argument("--phase", type=int, default=None,
                        help="Start from this phase (1-5). Skips earlier phases.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from (for --phase)")
    args = parser.parse_args()

    splits_dir = args.splits_dir
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    label_maps_path = os.path.join(splits_dir, "label_maps.json")

    # Verify files exist
    for f in [train_csv, val_csv, label_maps_path]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found")
            sys.exit(1)

    # Split into phases
    phase_dir = os.path.join(args.output_dir, "phases")
    phases = split_into_phases(train_csv, phase_dir, seed=args.seed)

    # Determine start phase
    start_phase = 1
    resume_path = args.resume
    if args.phase is not None:
        start_phase = args.phase
        if not resume_path and start_phase > 1:
            # Auto-find previous phase's best checkpoint
            prev_best = os.path.join(args.output_dir, f"phase_{start_phase - 1}", "best.pt")
            if os.path.exists(prev_best):
                resume_path = prev_best
                print(f"Auto-resume from phase {start_phase - 1}: {prev_best}")

    # Run phases
    results = {}
    for phase_info in phases:
        phase_num = phase_info["phase"]
        if phase_num < start_phase:
            print(f"\nSkipping phase {phase_num} (starting from phase {start_phase})")
            continue

        best_acc = train_phase(
            phase_num=phase_num,
            phase_csv=phase_info["csv_path"],
            val_csv=val_csv,
            label_maps_path=label_maps_path,
            output_dir=args.output_dir,
            epochs=args.epochs_per_phase,
            resume_path=resume_path,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + phase_num,  # different seed per phase
            num_workers=args.num_workers,
        )
        results[phase_num] = best_acc

        # Next phase resumes from this phase's best
        resume_path = os.path.join(args.output_dir, f"phase_{phase_num}", "best.pt")

    # Summary
    print(f"\n{'='*60}")
    print("PHASED TRAINING COMPLETE")
    print(f"{'='*60}")
    for p, acc in results.items():
        print(f"  Phase {p}: best balanced_acc = {acc:.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "phased_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "phases": [{"phase": p, "best_balanced_acc": float(a)} for p, a in results.items()],
            "config": {
                "epochs_per_phase": args.epochs_per_phase,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            },
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
