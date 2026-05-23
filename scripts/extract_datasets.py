"""
Extract downloaded dataset archives.

Usage:
    python scripts/extract_datasets.py                     # extract all
    python scripts/extract_datasets.py --dataset bossbase  # extract one
"""

import argparse
import os
import sys
import zipfile
import tarfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def extract_bossbase():
    zip_path = DATA_DIR / "bossbase" / "BOSSbase_1.01.zip"
    out_dir = DATA_DIR / "bossbase"

    if not zip_path.exists():
        print(f"[SKIP] BOSSbase zip not found: {zip_path}")
        return

    # Check if already extracted
    pgm_count = len(list(out_dir.glob("*.pgm")))
    if pgm_count >= 10000:
        print(f"[SKIP] BOSSbase already extracted ({pgm_count} PGM files)")
        return

    print(f"[EXTRACT] BOSSbase ({zip_path.stat().st_size / 1e9:.1f} GB)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Move files from subdirectory if needed
    for sub in out_dir.iterdir():
        if sub.is_dir() and sub.name != "__MACOSX":
            for f in sub.glob("*.pgm"):
                f.rename(out_dir / f.name)

    pgm_count = len(list(out_dir.glob("*.pgm")))
    print(f"[OK] BOSSbase: {pgm_count} PGM images")

    # Remove zip to save space
    if pgm_count >= 9000:
        zip_path.unlink()
        print(f"  Removed zip to save space")


def extract_div2k():
    zip_path = DATA_DIR / "div2k" / "DIV2K_train_HR.zip"
    out_dir = DATA_DIR / "div2k"

    if not zip_path.exists():
        print(f"[SKIP] DIV2K zip not found: {zip_path}")
        return

    png_count = len(list(out_dir.rglob("*.png")))
    if png_count >= 800:
        print(f"[SKIP] DIV2K already extracted ({png_count} PNG files)")
        return

    print(f"[EXTRACT] DIV2K ({zip_path.stat().st_size / 1e9:.1f} GB)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Move from subdirectory
    hr_dir = out_dir / "DIV2K_train_HR"
    if hr_dir.exists():
        for f in hr_dir.glob("*.png"):
            f.rename(out_dir / f.name)
        hr_dir.rmdir()

    png_count = len(list(out_dir.glob("*.png")))
    print(f"[OK] DIV2K: {png_count} PNG images")

    if png_count >= 800:
        zip_path.unlink()
        print(f"  Removed zip to save space")


def extract_alaska2():
    zip_path = DATA_DIR / "alaska2" / "alaska2-image-steganalysis.zip"
    out_dir = DATA_DIR / "alaska2"

    if not zip_path.exists():
        print(f"[SKIP] ALASKA2 zip not found: {zip_path}")
        return

    cover_dir = out_dir / "Cover"
    if cover_dir.exists() and len(list(cover_dir.glob("*.jpg"))) >= 10000:
        print(f"[SKIP] ALASKA2 already extracted")
        return

    print(f"[EXTRACT] ALASKA2 ({zip_path.stat().st_size / 1e9:.1f} GB)...")
    print(f"  This will take a while (~50GB compressed)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    for subdir in ["Cover", "JMiPOD", "JUNIWARD", "UERD"]:
        d = out_dir / subdir
        if d.exists():
            count = len(list(d.glob("*.jpg")))
            print(f"  {subdir}: {count} images")

    # Keep zip (too expensive to re-download)
    print(f"[OK] ALASKA2 extracted (keeping zip)")


def main():
    parser = argparse.ArgumentParser(description="Extract dataset archives")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "bossbase", "div2k", "alaska2"])
    args = parser.parse_args()

    print("=== Dataset Extraction ===\n")

    if args.dataset in ("all", "bossbase"):
        extract_bossbase()
    if args.dataset in ("all", "div2k"):
        extract_div2k()
    if args.dataset in ("all", "alaska2"):
        extract_alaska2()

    print("\nDone.")


if __name__ == "__main__":
    main()
