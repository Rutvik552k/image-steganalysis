"""
Fast parallel stego generation using multiprocessing.

Generates stego images across spatial + registers ALASKA2 JPEG.
Uses multiprocessing.Pool for ~4x speedup on 4 vCPUs.

Usage:
    python scripts/generate_stego_fast.py --workers 4 --data-dir ~/data/raw --output-dir ~/data/processed
"""

import argparse
import csv
import hashlib
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# SPLIT ASSIGNMENT (same as generate_stego.py — must match)
# ============================================================

def assign_split(cover_path: str) -> str:
    """Deterministic split via SHA-256 hash of cover path."""
    h = hashlib.sha256(cover_path.encode("utf-8")).hexdigest()
    ratio = int(h[:8], 16) / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    elif ratio < 0.85:
        return "val"
    else:
        return "test"


def get_algo_class(algo: str) -> str:
    mapping = {
        "lsb_replacement": "class_a_direct",
        "lsb_matching": "class_a_direct",
        "hugo": "class_b_stc_spatial",
        "wow": "class_b_stc_spatial",
        "s_uniward": "class_b_stc_spatial",
        "hill": "class_b_stc_spatial",
        "mipod": "class_b_stc_spatial",
        "j_uniward": "class_d_stc_jpeg",
        "jmipod": "class_d_stc_jpeg",
        "uerd": "class_d_stc_jpeg",
    }
    return mapping.get(algo, "unknown")


# ============================================================
# WORKER FUNCTION
# ============================================================

def process_one_cover(args):
    """Process one cover image — generate all stego variants.

    Runs in a worker process. Imports conseal per-worker to avoid
    pickle issues with numba-compiled functions.
    """
    idx, cover_path, algorithms, rates, output_dir = args

    import conseal as cl

    cover_str = str(cover_path)
    split = assign_split(cover_str)

    img = Image.open(cover_path)
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    cover = np.array(img, dtype=np.uint8)
    img.close()

    records = []

    # Register cover
    records.append({
        "image_id": f"spatial_cover_{idx:06d}",
        "split": split,
        "is_stego": 0,
        "algorithm": "none",
        "algorithm_class": "cover",
        "domain": "none",
        "cover_source": "bossbase",
        "cover_path": cover_str,
        "stego_path": "",
        "payload_rate_bpp": 0.0,
        "payload_type": "none",
        "n_changes": 0,
        "change_rate": 0.0,
        "image_width": w,
        "image_height": h,
        "format": "pgm",
    })

    for algo in algorithms:
        for rate in rates:
            # Deterministic seed from cover path + algo + rate
            seed_str = f"{cover_str}_{algo}_{rate}"
            seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)

            out_name = f"{Path(cover_path).stem}_{algo}_{rate}.png"
            out_path = os.path.join(output_dir, algo, f"rate_{rate}", out_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                if algo == "s_uniward":
                    stego = cl.suniward.simulate_single_channel(x0=cover, alpha=rate, seed=seed)
                elif algo == "hill":
                    stego = cl.hill.simulate_single_channel(x0=cover, alpha=rate, seed=seed)
                elif algo == "hugo":
                    stego = cl.hugo.simulate_single_channel(x0=cover, alpha=rate, seed=seed)
                elif algo == "wow":
                    stego = cl.wow.simulate_single_channel(x0=cover, alpha=rate, seed=seed)
                elif algo == "mipod":
                    stego = cl.mipod.simulate_single_channel(x0=cover, alpha=rate, seed=seed)
                elif algo == "lsb_matching":
                    stego = cl.lsb.simulate(cover=cover, alpha=rate, modify=cl.lsb.Change.LSB_MATCHING, seed=seed)
                elif algo == "lsb_replacement":
                    stego = cl.lsb.simulate(cover=cover, alpha=rate, modify=cl.lsb.Change.LSB_REPLACEMENT, seed=seed)
                else:
                    continue

                Image.fromarray(stego).save(out_path)

                n_changes = int(np.sum(cover != stego))
                n_pixels = w * h

                records.append({
                    "image_id": f"spatial_{algo}_{idx:06d}_{rate}",
                    "split": split,
                    "is_stego": 1,
                    "algorithm": algo,
                    "algorithm_class": get_algo_class(algo),
                    "domain": "spatial",
                    "cover_source": "bossbase",
                    "cover_path": cover_str,
                    "stego_path": out_path,
                    "payload_rate_bpp": rate,
                    "payload_type": "random_binary",
                    "n_changes": n_changes,
                    "change_rate": n_changes / n_pixels,
                    "image_width": w,
                    "image_height": h,
                    "format": "png",
                })
            except Exception as e:
                print(f"  [ERROR] {algo} rate={rate} on {Path(cover_path).name}: {e}")

    return records


# ============================================================
# ALASKA2 REGISTRATION (single-threaded, I/O only)
# ============================================================

def register_alaska2(alaska2_dir: str, max_images=None):
    """Register ALASKA2 pre-generated stego. No embedding needed."""
    print(f"\n=== Registering ALASKA2 Pre-Generated Stego ===")

    cover_dir = os.path.join(alaska2_dir, "Cover")
    cover_files = sorted(Path(cover_dir).glob("*.jpg"))
    if max_images:
        cover_files = cover_files[:max_images]

    print(f"  Covers: {len(cover_files)}")

    algo_dirs = {
        "JMiPOD": "jmipod",
        "JUNIWARD": "j_uniward",
        "UERD": "uerd",
    }

    records = []
    for idx, cover_path in enumerate(tqdm(cover_files, desc="ALASKA2")):
        cover_str = str(cover_path)
        split = assign_split(cover_str)
        img = Image.open(cover_path)
        w, h = img.size
        img.close()

        # Cover record
        records.append({
            "image_id": f"alaska2_cover_{idx:06d}",
            "split": split,
            "is_stego": 0,
            "algorithm": "none",
            "algorithm_class": "cover",
            "domain": "none",
            "cover_source": "alaska2",
            "cover_path": cover_str,
            "stego_path": "",
            "payload_rate_bpp": 0.0,
            "payload_type": "none",
            "n_changes": 0,
            "change_rate": 0.0,
            "image_width": w,
            "image_height": h,
            "format": "jpeg",
        })

        # Stego records
        for dir_name, algo_name in algo_dirs.items():
            stego_path = os.path.join(alaska2_dir, dir_name, cover_path.name)
            if os.path.exists(stego_path):
                records.append({
                    "image_id": f"alaska2_{algo_name}_{idx:06d}",
                    "split": split,
                    "is_stego": 1,
                    "algorithm": algo_name,
                    "algorithm_class": "class_d_stc_jpeg",
                    "domain": "jpeg",
                    "cover_source": "alaska2",
                    "cover_path": cover_str,
                    "stego_path": stego_path,
                    "payload_rate_bpp": 0.4,
                    "payload_type": "random_binary",
                    "n_changes": 0,
                    "change_rate": 0.0,
                    "image_width": w,
                    "image_height": h,
                    "format": "jpeg",
                })

    return records


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fast parallel stego generation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--algorithms", type=str, nargs="+",
                        default=["s_uniward", "hill"])
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[0.2, 0.4])
    parser.add_argument("--skip-alaska2", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_records = []

    # 1. Register ALASKA2
    if not args.skip_alaska2:
        alaska2_dir = os.path.join(args.data_dir, "alaska2")
        if os.path.isdir(alaska2_dir):
            alaska2_records = register_alaska2(alaska2_dir, args.max_images)
            all_records.extend(alaska2_records)
            print(f"  ALASKA2: {len(alaska2_records)} records")

    # 2. Parallel spatial stego generation
    bossbase_dir = os.path.join(args.data_dir, "bossbase")
    cover_files = sorted(Path(bossbase_dir).glob("*.pgm"))
    if args.max_images:
        cover_files = cover_files[:args.max_images]

    print(f"\n=== Spatial Stego Generation (parallel, {args.workers} workers) ===")
    print(f"  Covers: {len(cover_files)}")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Rates: {args.rates}")
    print(f"  Total embeds: {len(cover_files) * len(args.algorithms) * len(args.rates)}")

    spatial_out = os.path.join(args.output_dir, "spatial")

    # Build work items
    work_items = [
        (idx, str(cf), args.algorithms, args.rates, spatial_out)
        for idx, cf in enumerate(cover_files)
    ]

    # Warm up conseal JIT with first image (numba compilation)
    print("  Warming up conseal JIT...")
    warmup_result = process_one_cover(work_items[0])
    all_records.extend(warmup_result)
    work_items = work_items[1:]
    print("  JIT warm-up done.")

    # Parallel execution
    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one_cover, work_items),
            total=len(work_items),
            desc="Spatial stego",
        ):
            all_records.extend(result)

    # 3. Save metadata
    fieldnames = [
        "image_id", "split", "is_stego", "algorithm", "algorithm_class",
        "domain", "cover_source", "cover_path", "stego_path",
        "payload_rate_bpp", "payload_type", "n_changes", "change_rate",
        "image_width", "image_height", "format",
    ]

    meta_path = os.path.join(args.output_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # Summary
    covers = sum(1 for r in all_records if r["is_stego"] == 0)
    stegos = sum(1 for r in all_records if r["is_stego"] == 1)
    algos = set(r["algorithm"] for r in all_records if r["is_stego"] == 1)

    print(f"\n=== Generation Complete ===")
    print(f"  Total records: {len(all_records)}")
    print(f"  Covers: {covers}")
    print(f"  Stego: {stegos}")
    print(f"  Algorithms: {sorted(algos)}")
    print(f"  Metadata: {meta_path}")

    # Split distribution
    from collections import Counter
    splits = Counter(r["split"] for r in all_records)
    print(f"  Splits: {dict(splits)}")


if __name__ == "__main__":
    main()
