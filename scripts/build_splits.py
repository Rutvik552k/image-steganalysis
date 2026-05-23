"""
Build train/val/test splits from generated metadata.

Reads metadata.csv produced by generate_stego.py and creates split files
for the PyTorch dataset loader.

Splits:
  - train (70%): learning
  - val (15%): hyperparameter tuning
  - test_in_domain (15%): same-distribution evaluation
  - test_cross_dataset: separate datasets (BOWS2, IStego100K, StegoAppDB)
  - test_unseen_algo: algorithms held out from training

Usage:
    python scripts/build_splits.py --metadata data/processed/metadata.csv
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_splits(metadata_path: str, output_dir: str, seed: int = 42):
    """Build stratified train/val/test splits."""
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"  Total records: {len(df)}")
    print(f"  Covers: {(df['is_stego'] == 0).sum()}")
    print(f"  Stego: {(df['is_stego'] == 1).sum()}")
    print(f"  Algorithms: {df['algorithm'].nunique()}")

    os.makedirs(output_dir, exist_ok=True)

    # Use pre-assigned splits from generate_stego.py
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
    else:
        # ANTI-LEAKAGE: Split by COVER IMAGE, not by row.
        # All stego variants of one cover must be in the same split.
        # Group by cover_path, split the GROUPS, then expand back to rows.
        cover_paths = df["cover_path"].unique()
        cover_df = pd.DataFrame({"cover_path": cover_paths})

        train_covers, temp_covers = train_test_split(
            cover_df, test_size=0.30, random_state=seed
        )
        val_covers, test_covers = train_test_split(
            temp_covers, test_size=0.50, random_state=seed
        )

        train_set = set(train_covers["cover_path"])
        val_set = set(val_covers["cover_path"])
        test_set = set(test_covers["cover_path"])

        train_df = df[df["cover_path"].isin(train_set)]
        val_df = df[df["cover_path"].isin(val_set)]
        test_df = df[df["cover_path"].isin(test_set)]

    # LEAKAGE CHECK: verify no cover_path appears in multiple splits
    train_covers = set(train_df["cover_path"].unique())
    val_covers = set(val_df["cover_path"].unique())
    test_covers = set(test_df["cover_path"].unique())

    train_val_leak = train_covers & val_covers
    train_test_leak = train_covers & test_covers
    val_test_leak = val_covers & test_covers

    if train_val_leak or train_test_leak or val_test_leak:
        print(f"\n  [LEAKAGE DETECTED]")
        print(f"    Train-Val overlap: {len(train_val_leak)} covers")
        print(f"    Train-Test overlap: {len(train_test_leak)} covers")
        print(f"    Val-Test overlap: {len(val_test_leak)} covers")
        print(f"    ABORTING — fix split assignment before proceeding!")
        sys.exit(1)
    else:
        print(f"\n  [LEAKAGE CHECK] PASSED — no cover overlap across splits")

    # Save split files
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(split_path, index=False)
        print(f"  {name}: {len(split_df)} records -> {split_path}")

    # Build label mappings
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
    print(f"  Label maps -> {label_path}")

    # Print distribution summary
    print(f"\n=== Split Distribution ===")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n  [{name}] {len(split_df)} images")
        print(f"    Cover/Stego: {(split_df['is_stego']==0).sum()} / {(split_df['is_stego']==1).sum()}")
        algo_counts = split_df["algorithm"].value_counts()
        for algo, count in algo_counts.items():
            print(f"    {algo}: {count}")

    # Stats file
    stats = {
        "total_images": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_algorithms": df["algorithm"].nunique(),
        "algorithms": sorted(df["algorithm"].unique().tolist()),
        "algorithm_classes": sorted(df["algorithm_class"].unique().tolist()),
        "payload_rates": sorted(df["payload_rate_bpp"].unique().tolist()),
        "payload_types": sorted(df["payload_type"].unique().tolist()),
        "cover_sources": sorted(df["cover_source"].unique().tolist()),
    }
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats -> {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset splits")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(args.metadata, args.output_dir, args.seed)
