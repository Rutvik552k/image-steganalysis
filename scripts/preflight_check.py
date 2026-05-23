"""
Pre-training validation suite.

Runs comprehensive checks BEFORE committing to a full training run:
  1. Model forward pass — correct output shapes
  2. Backward pass — gradients flow to all trainable params
  3. Loss convergence — 20 steps on 1 batch, loss should decrease
  4. BayarConv constraint — weights satisfy center=-1, rest sums to 1
  5. Checkpoint round-trip — save, load, verify state matches
  6. Dataloader integrity — shapes, label ranges, no NaN/Inf
  7. Label distribution — class balance sanity check
  8. LR schedule — verify warmup ramp and cosine decay shape
  9. Gradient norm — verify clipping works, no explosion

Usage:
    python scripts/preflight_check.py --splits data/splits --device cpu
    python scripts/preflight_check.py --splits data/dry_run/splits --device cpu
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BayarConv2d, create_dataloaders
from src.models.unisteg import UniStegLite
from src.training.losses import UniStegLoss
from src.training.train_loop import (
    build_optimizer, build_scheduler, project_bayar_weights, compute_grad_norm,
)
from src.training.checkpoint import save_checkpoint, load_checkpoint


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
failed = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name} -- {detail}")
        failed.append(name)


# ============================================================
# 1. Forward pass
# ============================================================

def test_forward(model, device):
    print("\n--- 1. Forward Pass ---")
    x = torch.randn(4, 1, 256, 256, device=device) * 127 + 128
    try:
        out = model(x)
        check("Output keys", set(out.keys()) == {"binary", "algo_class", "algorithm", "payload_rate"})
        check("Binary shape", out["binary"].shape == (4, 2), f"got {out['binary'].shape}")
        check("Payload shape", out["payload_rate"].shape == (4,), f"got {out['payload_rate'].shape}")
        check("No NaN in output", not any(torch.isnan(v).any() for v in out.values()))
        check("No Inf in output", not any(torch.isinf(v).any() for v in out.values()))
        check("Payload in [0,1]",
              out["payload_rate"].min() >= 0 and out["payload_rate"].max() <= 1,
              f"range [{out['payload_rate'].min():.3f}, {out['payload_rate'].max():.3f}]")
    except Exception as e:
        check("Forward pass", False, str(e))


# ============================================================
# 2. Backward pass — gradient flow
# ============================================================

def test_backward(model, device):
    print("\n--- 2. Backward Pass (gradient flow) ---")
    x = torch.randn(4, 1, 256, 256, device=device) * 127 + 128
    criterion = UniStegLoss().to(device)
    labels = {
        "binary": torch.randint(0, 2, (4,), device=device),
        "algorithm_class": torch.randint(0, 3, (4,), device=device),
        "algorithm": torch.randint(0, 5, (4,), device=device),
        "payload_rate": torch.rand(4, device=device) * 0.5,
    }

    model.zero_grad()
    out = model(x)
    loss, _ = criterion(out, labels)
    loss.backward()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_grad = [(n, p) for n, p in trainable if p.grad is None]
    zero_grad = [(n, p) for n, p in trainable if p.grad is not None and p.grad.abs().max() == 0]

    check("Loss is finite", torch.isfinite(loss), f"loss={loss.item()}")
    check("All trainable params have gradients",
          len(no_grad) == 0,
          f"{len(no_grad)} params with no gradient: {[n for n,_ in no_grad[:5]]}")
    check("No zero gradients",
          len(zero_grad) == 0,
          f"{len(zero_grad)} params with zero gradient: {[n for n,_ in zero_grad[:5]]}")

    grad_norm = compute_grad_norm(model)
    check("Gradient norm is finite", grad_norm < 1e6, f"norm={grad_norm:.2e}")
    print(f"    Total grad norm: {grad_norm:.4f}")
    print(f"    Loss value: {loss.item():.4f}")


# ============================================================
# 3. Loss convergence on 1 batch
# ============================================================

def test_overfit_one_batch(model, device, steps=20):
    print(f"\n--- 3. Overfit 1 Batch ({steps} steps) ---")
    x = torch.randn(8, 1, 256, 256, device=device) * 127 + 128
    criterion = UniStegLoss().to(device)
    labels = {
        "binary": torch.randint(0, 2, (8,), device=device),
        "algorithm_class": torch.randint(0, 3, (8,), device=device),
        "algorithm": torch.randint(0, 5, (8,), device=device),
        "payload_rate": torch.rand(8, device=device) * 0.5,
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    model.train()
    for i in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss, _ = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        project_bayar_weights(model)
        losses.append(loss.item())

    check("Loss decreased", losses[-1] < losses[0],
          f"start={losses[0]:.4f} end={losses[-1]:.4f}")
    check("Loss decreased >20%",
          losses[-1] < losses[0] * 0.8,
          f"ratio={losses[-1]/losses[0]:.3f}")
    check("No NaN in losses", all(not (l != l) for l in losses))
    print(f"    Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({(1-losses[-1]/losses[0])*100:.1f}% decrease)")


# ============================================================
# 4. BayarConv constraint
# ============================================================

def test_bayar_constraint(model):
    print("\n--- 4. BayarConv Constraint ---")
    for name, m in model.named_modules():
        if isinstance(m, BayarConv2d):
            m.project_weights()
            w = m.weight.data
            center = m.kernel_size // 2

            center_vals = w[:, :, center, center]
            check(f"Center=-1 ({name})",
                  torch.allclose(center_vals, torch.tensor(-1.0), atol=1e-5),
                  f"center values: {center_vals.flatten()[:3].tolist()}")

            rest_sum = w.sum(dim=(2, 3)) - center_vals
            check(f"Rest sums to 1 ({name})",
                  torch.allclose(rest_sum, torch.ones_like(rest_sum), atol=1e-4),
                  f"sums: {rest_sum.flatten()[:3].tolist()}")


# ============================================================
# 5. Checkpoint round-trip
# ============================================================

def test_checkpoint_roundtrip(model, device):
    print("\n--- 5. Checkpoint Round-Trip ---")
    criterion = UniStegLoss().to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, 5, 100, 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pt")

        # Save
        save_checkpoint(model, optimizer, scheduler, criterion, 42,
                        {"binary_acc": 0.85}, path, global_step=1000, best_val_acc=0.85)
        check("Save succeeded", os.path.exists(path))
        check("No tmp file left", not os.path.exists(path + ".tmp"))

        # Load into new model
        model2 = UniStegLite(num_algo_classes=model.model.heads.algo_class_head.out_features,
                             num_algorithms=model.model.heads.algorithm_head.out_features).to(device)
        ckpt = load_checkpoint(path, model2, map_location=str(device))

        check("Epoch restored", ckpt["epoch"] == 42)
        check("Global step restored", ckpt["global_step"] == 1000)
        check("Best val acc restored", abs(ckpt["best_val_acc"] - 0.85) < 1e-6)
        check("Timestamp present", "timestamp" in ckpt)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if not torch.equal(p1.data, p2.data):
                check(f"Weight match {n1}", False, "mismatch")
                break
        else:
            check("All weights match after load", True)


# ============================================================
# 6. Dataloader integrity
# ============================================================

def test_dataloader(splits_dir, device):
    print("\n--- 6. Dataloader Integrity ---")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            splits_dir=splits_dir, batch_size=8, target_size=256,
            apply_srm=False, num_workers=0, pin_memory=False,
        )

        batch = next(iter(train_loader))
        images = batch["image"]
        labels = batch["labels"]

        check("Image shape", images.shape[1:] == (1, 256, 256),
              f"got {images.shape}")
        check("Pixel range [0,255]",
              images.min() >= 0 and images.max() <= 255,
              f"range [{images.min():.0f}, {images.max():.0f}]")
        check("No NaN in images", not torch.isnan(images).any())
        check("Binary labels in {0,1}",
              set(labels["binary"].tolist()).issubset({0, 1}),
              f"values: {labels['binary'].unique().tolist()}")
        check("Payload rate in [0, 0.5]",
              labels["payload_rate"].min() >= 0 and labels["payload_rate"].max() <= 0.5,
              f"range [{labels['payload_rate'].min():.2f}, {labels['payload_rate'].max():.2f}]")
        check("No NaN in labels",
              not any(torch.isnan(v.float()).any() for v in labels.values()))

        # Check all splits have data
        check("Train not empty", len(train_loader.dataset) > 0,
              f"size={len(train_loader.dataset)}")
        check("Val not empty", len(val_loader.dataset) > 0,
              f"size={len(val_loader.dataset)}")
        check("Test not empty", len(test_loader.dataset) > 0,
              f"size={len(test_loader.dataset)}")

        print(f"    Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    except Exception as e:
        check("Dataloader creation", False, str(e))


# ============================================================
# 7. Label distribution
# ============================================================

def test_label_distribution(splits_dir):
    print("\n--- 7. Label Distribution ---")
    import json
    import pandas as pd

    label_maps_path = os.path.join(splits_dir, "label_maps.json")
    if not os.path.exists(label_maps_path):
        check("label_maps.json exists", False)
        return

    with open(label_maps_path) as f:
        label_maps = json.load(f)

    train_csv = os.path.join(splits_dir, "train.csv")
    if not os.path.exists(train_csv):
        check("train.csv exists", False)
        return

    df = pd.read_csv(train_csv)
    n_cover = (df["is_stego"] == 0).sum()
    n_stego = (df["is_stego"] == 1).sum()
    ratio = n_cover / max(n_stego, 1)

    check("Has covers and stego", n_cover > 0 and n_stego > 0,
          f"cover={n_cover}, stego={n_stego}")
    check("Cover/stego ratio reasonable (0.05-20x)",
          0.05 < ratio < 20,
          f"ratio={ratio:.2f}")

    n_algos = df["algorithm"].nunique()
    n_classes = df["algorithm_class"].nunique()
    check("Multiple algorithms present", n_algos >= 2, f"found {n_algos}")
    check("Multiple algorithm classes", n_classes >= 2, f"found {n_classes}")

    print(f"    Cover: {n_cover}, Stego: {n_stego} (ratio {ratio:.2f})")
    print(f"    Algorithms: {n_algos}, Classes: {n_classes}")
    print(f"    Label maps: {len(label_maps['algorithm_class'])} classes, {len(label_maps['algorithm'])} algos")


# ============================================================
# 8. LR schedule shape
# ============================================================

def test_lr_schedule(device):
    print("\n--- 8. LR Schedule ---")
    model = UniStegLite(num_algo_classes=3, num_algorithms=5).to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, warmup_epochs=5, total_epochs=100,
                                steps_per_epoch=50, min_lr=1e-6)

    lrs = []
    for step in range(5000):  # 100 epochs x 50 steps
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    warmup_end_lr = lrs[250]  # step 250 = end of 5 warmup epochs
    mid_lr = lrs[2500]
    final_lr = lrs[-1]

    check("Warmup: LR increases", lrs[1] > lrs[0], f"lr[0]={lrs[0]:.6f}, lr[1]={lrs[1]:.6f}")
    check("Warmup: reaches near base LR", warmup_end_lr > 5e-4,
          f"warmup_end={warmup_end_lr:.6f}")
    check("Cosine: LR decreases after warmup", mid_lr < warmup_end_lr,
          f"mid={mid_lr:.6f}")
    check("Final: LR > 0", final_lr > 0, f"final={final_lr:.8f}")

    print(f"    Warmup end (step 250): {warmup_end_lr:.6f}")
    print(f"    Mid (step 2500): {mid_lr:.6f}")
    print(f"    Final (step 5000): {final_lr:.8f}")


# ============================================================
# 9. Gradient clipping
# ============================================================

def test_gradient_clipping(model, device):
    print("\n--- 9. Gradient Clipping ---")
    x = torch.randn(4, 1, 256, 256, device=device) * 127 + 128
    criterion = UniStegLoss().to(device)
    labels = {
        "binary": torch.randint(0, 2, (4,), device=device),
        "algorithm_class": torch.randint(0, 3, (4,), device=device),
        "algorithm": torch.randint(0, 5, (4,), device=device),
        "payload_rate": torch.rand(4, device=device) * 0.5,
    }

    model.zero_grad()
    out = model(x)
    loss, _ = criterion(out, labels)
    loss.backward()

    norm_before = compute_grad_norm(model)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    norm_after = compute_grad_norm(model)

    check("Gradient norm computed", norm_before > 0, f"norm={norm_before:.4f}")
    check("Clipping effective (norm <= 1.0 + eps)", norm_after <= 1.01,
          f"before={norm_before:.4f}, after={norm_after:.4f}")
    print(f"    Before clip: {norm_before:.4f}, After: {norm_after:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-training validation")
    parser.add_argument("--splits", type=str, default="data/dry_run/splits")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-algo-classes", type=int, default=3)
    parser.add_argument("--num-algorithms", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"=" * 60)
    print("PRE-TRAINING VALIDATION SUITE")
    print(f"=" * 60)

    model = UniStegLite(
        num_algo_classes=args.num_algo_classes,
        num_algorithms=args.num_algorithms,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: UniStegLite ({trainable:,} trainable / {total:,} total)")

    test_forward(model, device)
    test_backward(model, device)
    test_overfit_one_batch(model, device, steps=20)
    test_bayar_constraint(model)
    test_checkpoint_roundtrip(model, device)

    if os.path.exists(args.splits):
        test_dataloader(args.splits, device)
        test_label_distribution(args.splits)
    else:
        print(f"\n  [SKIP] Dataloader tests (splits not found: {args.splits})")

    test_lr_schedule(device)
    test_gradient_clipping(model, device)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {len(failed)} checks")
        for f in failed:
            print(f"  - {f}")
        print(f"{'=' * 60}")
        sys.exit(1)
    else:
        print("ALL PRE-FLIGHT CHECKS PASSED")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
