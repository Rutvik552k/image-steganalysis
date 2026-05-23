"""
Dry-run: synthetic data -> stego -> splits -> dataloader -> 2-epoch train.

Validates the FULL pipeline end-to-end without real datasets or conseal.
Uses LSB replacement (no external deps) on synthetic grayscale images.

Usage:
    python scripts/dry_run.py
    python scripts/dry_run.py --num-covers 200 --epochs 3
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# STEP 1: Generate synthetic cover images
# ============================================================

def generate_covers(output_dir: str, num_images: int = 100, size: int = 256):
    """Generate synthetic grayscale covers with natural-image-like statistics.

    Uses random smooth gradients + noise to simulate natural image textures.
    Not realistic, but exercises the same pixel value range [0, 255].
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i in tqdm(range(num_images), desc="Generating covers"):
        # Base: smooth random gradient
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)

        # Random frequency/phase
        rng = np.random.RandomState(i)
        freq = rng.uniform(1, 5, size=4)
        phase = rng.uniform(0, 2 * np.pi, size=4)

        img = (
            np.sin(freq[0] * xx + phase[0]) * np.cos(freq[1] * yy + phase[1])
            + np.sin(freq[2] * (xx + yy) + phase[2]) * 0.5
            + np.cos(freq[3] * (xx - yy) + phase[3]) * 0.3
        )

        # Normalize to [0, 255] and add Gaussian noise
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 200 + 25
        noise = rng.normal(0, 8, (size, size))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        path = os.path.join(output_dir, f"cover_{i:05d}.png")
        Image.fromarray(img).save(path)
        paths.append(path)

    print(f"  Created {len(paths)} covers in {output_dir}")
    return paths


# ============================================================
# STEP 2: Generate LSB stego + metadata
# ============================================================

ALGORITHMS = {
    "lsb_replacement": "class_a_direct",
    "lsb_matching": "class_a_direct",
    "s_uniward_sim": "class_b_stc_spatial",
    "hill_sim": "class_b_stc_spatial",
}

RATES = [0.1, 0.2, 0.4]


def lsb_replace(cover: np.ndarray, payload: bytes) -> np.ndarray:
    """LSB replacement embedding."""
    flat = cover.flatten().copy()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    n = min(len(bits), len(flat))
    flat[:n] = (flat[:n] & 0xFE) | bits[:n]
    return flat.reshape(cover.shape)


def lsb_match(cover: np.ndarray, payload: bytes, rng: np.random.RandomState) -> np.ndarray:
    """LSB matching (+-1 embedding)."""
    flat = cover.flatten().astype(np.int16).copy()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    n = min(len(bits), len(flat))
    for i in range(n):
        if (flat[i] & 1) != bits[i]:
            if flat[i] == 0:
                flat[i] += 1
            elif flat[i] == 255:
                flat[i] -= 1
            else:
                flat[i] += rng.choice([-1, 1])
    return np.clip(flat, 0, 255).astype(np.uint8).reshape(cover.shape)


def simulated_adaptive(cover: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
    """Simulate adaptive stego by adding +-1 noise to rate fraction of pixels.

    Not cryptographically correct STC, but mimics the statistical footprint
    enough to validate the pipeline. Real training uses conseal.
    """
    flat = cover.flatten().astype(np.int16).copy()
    n_change = int(len(flat) * rate)
    indices = rng.choice(len(flat), n_change, replace=False)
    changes = rng.choice([-1, 1], n_change)
    flat[indices] += changes
    return np.clip(flat, 0, 255).astype(np.uint8).reshape(cover.shape)


def assign_split(cover_path: str) -> str:
    """Same SHA-256 hash split as generate_stego.py."""
    h = hashlib.sha256(cover_path.encode("utf-8")).hexdigest()
    ratio = int(h[:8], 16) / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    elif ratio < 0.85:
        return "val"
    else:
        return "test"


def generate_stego(
    cover_paths: list[str],
    output_dir: str,
    metadata_path: str,
):
    """Generate stego for all covers x algorithms x rates."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    fieldnames = [
        "image_id", "split", "is_stego", "algorithm", "algorithm_class",
        "domain", "cover_source", "cover_path", "stego_path",
        "payload_rate_bpp", "payload_type", "payload_bytes",
        "payload_hash", "n_changes", "change_rate",
        "image_width", "image_height", "format", "quality_factor",
    ]

    for idx, cover_path in enumerate(tqdm(cover_paths, desc="Generating stego")):
        cover = np.array(Image.open(cover_path))
        h, w = cover.shape
        split = assign_split(cover_path)

        # Register cover
        records.append({
            "image_id": f"cover_{idx:05d}",
            "split": split,
            "is_stego": 0,
            "algorithm": "none",
            "algorithm_class": "cover",
            "domain": "none",
            "cover_source": "synthetic",
            "cover_path": cover_path,
            "stego_path": "",
            "payload_rate_bpp": 0.0,
            "payload_type": "none",
            "payload_bytes": 0,
            "payload_hash": "",
            "n_changes": 0,
            "change_rate": 0.0,
            "image_width": w,
            "image_height": h,
            "format": "png",
            "quality_factor": 0,
        })

        for algo, algo_class in ALGORITHMS.items():
            for rate in RATES:
                rng = np.random.RandomState(idx * 1000 + hash(algo) % 1000)
                n_pixels = h * w
                payload_bytes = max(int(n_pixels * rate) // 8, 1)
                payload = os.urandom(payload_bytes)
                phash = hashlib.sha256(payload).hexdigest()[:16]

                # Embed
                if algo == "lsb_replacement":
                    stego = lsb_replace(cover, payload)
                elif algo == "lsb_matching":
                    stego = lsb_match(cover, payload, rng)
                else:
                    stego = simulated_adaptive(cover, rate, rng)

                # Save
                algo_dir = os.path.join(output_dir, algo, f"rate_{rate}")
                os.makedirs(algo_dir, exist_ok=True)
                out_name = f"stego_{idx:05d}_{algo}_{rate}.png"
                out_path = os.path.join(algo_dir, out_name)
                Image.fromarray(stego).save(out_path)

                n_changes = int(np.sum(cover != stego))

                records.append({
                    "image_id": f"{algo}_{idx:05d}_{rate}",
                    "split": split,
                    "is_stego": 1,
                    "algorithm": algo,
                    "algorithm_class": algo_class,
                    "domain": "spatial",
                    "cover_source": "synthetic",
                    "cover_path": cover_path,
                    "stego_path": out_path,
                    "payload_rate_bpp": rate,
                    "payload_type": "random_binary",
                    "payload_bytes": payload_bytes,
                    "payload_hash": phash,
                    "n_changes": n_changes,
                    "change_rate": n_changes / n_pixels,
                    "image_width": w,
                    "image_height": h,
                    "format": "png",
                    "quality_factor": 0,
                })

    # Write metadata CSV
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    covers = sum(1 for r in records if r["is_stego"] == 0)
    stegos = sum(1 for r in records if r["is_stego"] == 1)
    print(f"  Metadata: {len(records)} records ({covers} covers, {stegos} stego)")
    print(f"  Saved to {metadata_path}")
    return records


# ============================================================
# STEP 3: Build splits
# ============================================================

def build_splits(metadata_path: str, output_dir: str):
    """Build train/val/test CSVs + label_maps.json from metadata."""
    import pandas as pd

    df = pd.read_csv(metadata_path)
    os.makedirs(output_dir, exist_ok=True)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Leakage check
    train_covers = set(train_df["cover_path"].unique())
    val_covers = set(val_df["cover_path"].unique())
    test_covers = set(test_df["cover_path"].unique())

    leak = (train_covers & val_covers) | (train_covers & test_covers) | (val_covers & test_covers)
    if leak:
        print(f"  LEAKAGE DETECTED: {len(leak)} covers in multiple splits!")
        sys.exit(1)
    print(f"  Leakage check: PASSED")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        n_cover = (split_df["is_stego"] == 0).sum()
        n_stego = (split_df["is_stego"] == 1).sum()
        print(f"  {name}: {len(split_df)} rows ({n_cover} cover, {n_stego} stego)")

    # Label maps
    label_maps = {
        "binary": {"cover": 0, "stego": 1},
        "algorithm_class": {
            cls: i for i, cls in enumerate(sorted(df["algorithm_class"].unique()))
        },
        "algorithm": {
            algo: i for i, algo in enumerate(sorted(df["algorithm"].unique()))
        },
    }
    label_path = os.path.join(output_dir, "label_maps.json")
    with open(label_path, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"  Label maps: {label_path}")
    print(f"    algorithm_class: {label_maps['algorithm_class']}")
    print(f"    algorithm: {label_maps['algorithm']}")

    return label_maps


# ============================================================
# STEP 4: Validate dataloader
# ============================================================

def validate_dataloader(splits_dir: str):
    """Load one batch from train split, verify shapes."""
    from src.data.dataset import create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        splits_dir=splits_dir,
        batch_size=8,
        target_size=256,
        apply_srm=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["labels"]

    print(f"\n  Batch shapes:")
    print(f"    images: {images.shape} (expected [8, 1, 256, 256])")
    print(f"    binary: {labels['binary'].shape}")
    print(f"    algorithm_class: {labels['algorithm_class'].shape}")
    print(f"    algorithm: {labels['algorithm'].shape}")
    print(f"    payload_rate: {labels['payload_rate'].shape}")
    print(f"    pixel range: [{images.min():.0f}, {images.max():.0f}]")
    print(f"    binary labels: {labels['binary'].tolist()}")

    assert images.shape == (8, 1, 256, 256), f"Bad image shape: {images.shape}"
    assert images.min() >= 0 and images.max() <= 255, "Pixel range outside [0,255]"
    print(f"  Dataloader: PASSED")

    return train_loader, val_loader


# ============================================================
# STEP 5: Training dry-run
# ============================================================

def training_dry_run(train_loader, val_loader, epochs: int = 2, device_str: str = "cpu"):
    """Run 2-epoch training on synthetic data."""
    import torch
    from src.models.unisteg import UniStegLite
    from src.training.train_loop import train

    device = torch.device(device_str)

    model = UniStegLite(
        num_experts=5,
        num_algo_classes=len(set(
            r for r in [
                "cover", "class_a_direct", "class_b_stc_spatial",
            ]
        )),
        num_algorithms=len(set(
            r for r in [
                "none", "lsb_replacement", "lsb_matching",
                "s_uniward_sim", "hill_sim",
            ]
        )),
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: UniStegLite ({trainable:,} trainable params)")

    config = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "warmup_epochs": 1,
        "accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "patience": 100,  # no early stopping in dry-run
        "save_every": 1,
    }

    output_dir = str(ROOT / "data" / "dry_run" / "checkpoints")

    best_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    # Verify checkpoints exist
    assert os.path.exists(os.path.join(output_dir, "best.pt")), "best.pt not created"
    assert os.path.exists(os.path.join(output_dir, "last.pt")), "last.pt not created"
    print(f"\n  Checkpoints: PASSED")
    print(f"  Best binary_acc: {best_acc:.4f}")

    return best_acc


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline dry-run with synthetic data")
    parser.add_argument("--num-covers", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    base_dir = str(ROOT / "data" / "dry_run")

    print("=" * 60)
    print("PIPELINE DRY-RUN (synthetic data)")
    print("=" * 60)

    # Step 1
    print(f"\n--- Step 1: Generate {args.num_covers} synthetic covers ---")
    covers_dir = os.path.join(base_dir, "covers")
    cover_paths = generate_covers(covers_dir, args.num_covers)

    # Step 2
    print(f"\n--- Step 2: Generate stego images ---")
    stego_dir = os.path.join(base_dir, "stego")
    metadata_path = os.path.join(base_dir, "metadata.csv")
    generate_stego(cover_paths, stego_dir, metadata_path)

    # Step 3
    print(f"\n--- Step 3: Build splits ---")
    splits_dir = os.path.join(base_dir, "splits")
    build_splits(metadata_path, splits_dir)

    # Step 4
    print(f"\n--- Step 4: Validate dataloader ---")
    train_loader, val_loader = validate_dataloader(splits_dir)

    # Step 5
    print(f"\n--- Step 5: Training dry-run ({args.epochs} epochs) ---")
    training_dry_run(train_loader, val_loader, args.epochs, args.device)

    print("\n" + "=" * 60)
    print("ALL PIPELINE STAGES PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
