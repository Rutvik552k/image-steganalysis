"""
Neural/Diffusion Stego Generation Pipeline

Generates stego images from:
  Class E: Neural encoder-decoder (SteganoGAN)
  Class F: Diffusion model (DiffStega)

SteganoGAN API (verified from github.com/DAI-Lab/SteganoGAN):
  pip install steganogan
  from steganogan import SteganoGAN
  model = SteganoGAN.load(architecture='dense')
  model.encode(input_path, output_path, message)

DiffStega API (verified from github.com/evtricks/DiffStega):
  CLI-only: python main.py --image_path ... --prompt1 "" --prompt2 "..." --save_path ...
  Requires: Stable Diffusion 1.5, IP-Adapter models
  Password-based coverless steganography

Usage:
    python scripts/generate_neural_diffusion_stego.py --phase steganogan
    python scripts/generate_neural_diffusion_stego.py --phase diffstega
    python scripts/generate_neural_diffusion_stego.py --phase all
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# SteganoGAN GENERATOR (Class E)
# ============================================================

class SteganoGANGenerator:
    """Generate stego using SteganoGAN pretrained models.

    Verified API (github.com/DAI-Lab/SteganoGAN):
      from steganogan import SteganoGAN
      model = SteganoGAN.load(architecture='dense')
      model.encode(input_path, output_path, message_string)
      decoded = model.decode(stego_path)
    """

    def __init__(self, architecture: str = "dense"):
        try:
            from steganogan import SteganoGAN
            self.model = SteganoGAN.load(architecture=architecture)
            self.architecture = architecture
        except ImportError:
            print("ERROR: steganogan not installed. Install with: pip install steganogan")
            sys.exit(1)

    def embed(
        self,
        cover_path: str,
        output_path: str,
        message: str,
    ) -> dict:
        """Embed text message into cover image via SteganoGAN.

        Args:
            cover_path: Path to cover image (PNG)
            output_path: Path to save stego image (PNG)
            message: Text message to hide
        """
        try:
            self.model.encode(cover_path, output_path, message)
            success = os.path.exists(output_path)

            return {
                "algorithm": "steganogan",
                "domain": "neural",
                "cover": cover_path,
                "stego": output_path if success else None,
                "success": success,
                "message_len": len(message),
                "architecture": self.architecture,
            }
        except Exception as e:
            return {
                "algorithm": "steganogan",
                "domain": "neural",
                "cover": cover_path,
                "stego": None,
                "success": False,
                "error": str(e),
            }

    def verify(self, stego_path: str, original_message: str) -> bool:
        """Verify message can be decoded from stego image."""
        try:
            decoded = self.model.decode(stego_path)
            return decoded == original_message
        except Exception:
            return False


# ============================================================
# DiffStega GENERATOR (Class F)
# ============================================================

class DiffStegaGenerator:
    """Generate stego using DiffStega (training-free diffusion steganography).

    Verified API (github.com/evtricks/DiffStega):
      CLI: python main.py --image_path <path> --prompt1 "" --prompt2 <prompt>
           --save_path <dir> --pw <password> --edit_strength <float>
      Requires: SD 1.5, IP-Adapter models in ./pretrained_models/

    DiffStega generates NEW images (coverless) — no cover modification.
    The password acts as the secret key.
    """

    def __init__(self, diffstega_dir: str, device: str = "cuda"):
        self.diffstega_dir = diffstega_dir
        self.main_script = os.path.join(diffstega_dir, "main.py")
        self.device = device

        if not os.path.exists(self.main_script):
            print(f"ERROR: DiffStega not found at {self.main_script}")
            print("  Clone: git clone https://github.com/evtricks/DiffStega.git")
            print("  Download pretrained models per README instructions")
            sys.exit(1)

    def generate(
        self,
        reference_image_path: str,
        output_dir: str,
        password: int = 9000,
        prompt: str = "",
        edit_strength: float = 0.6,
        num_steps: int = 50,
        noise_flip_scale: float = 0.05,
    ) -> dict:
        """Generate stego image via DiffStega CLI.

        Args:
            reference_image_path: Reference image (acts as visual key)
            output_dir: Directory to save generated stego image
            password: Integer password (secret key)
            prompt: Text prompt for image generation (optional)
            edit_strength: How much to modify from reference (0.6-0.7)
            num_steps: Diffusion sampling steps
            noise_flip_scale: Noise flip encryption strength
        """
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, self.main_script,
            "--image_path", reference_image_path,
            "--prompt1", "",  # null-text primary prompt
            "--prompt2", prompt if prompt else "a photograph",
            "--save_path", output_dir,
            "--pw", str(password),
            "--edit_strength", str(edit_strength),
            "--num_steps", str(num_steps),
            "--noise_flip_scale", str(noise_flip_scale),
            "--single_model",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.diffstega_dir,
                timeout=300,  # 5 min timeout per image
            )
            success = result.returncode == 0

            # Find generated output file
            output_files = list(Path(output_dir).glob("*.png")) + \
                           list(Path(output_dir).glob("*.jpg"))

            return {
                "algorithm": "diffstega",
                "domain": "diffusion",
                "cover": reference_image_path,
                "stego": str(output_files[-1]) if output_files else None,
                "success": success,
                "password": password,
                "prompt": prompt,
                "error": result.stderr if not success else None,
            }
        except subprocess.TimeoutExpired:
            return {
                "algorithm": "diffstega",
                "domain": "diffusion",
                "cover": reference_image_path,
                "stego": None,
                "success": False,
                "error": "Timeout (>5 min)",
            }
        except Exception as e:
            return {
                "algorithm": "diffstega",
                "domain": "diffusion",
                "cover": reference_image_path,
                "stego": None,
                "success": False,
                "error": str(e),
            }


# ============================================================
# METADATA APPENDER
# ============================================================

class MetadataAppender:
    """Append neural/diffusion stego records to existing metadata.csv."""

    FIELDNAMES = [
        "image_id", "split", "is_stego", "algorithm", "algorithm_class",
        "domain", "cover_source", "cover_path", "stego_path",
        "payload_rate_bpp", "payload_type", "payload_bytes",
        "payload_hash", "n_changes", "change_rate",
        "image_width", "image_height", "format", "quality_factor",
    ]

    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.new_records = []

    def add_record(
        self,
        image_id: str,
        split: str,
        result: dict,
        source: str,
        algo_class: str,
        payload_type: str,
        payload_bytes: int,
        fmt: str,
        width: int,
        height: int,
    ):
        self.new_records.append({
            "image_id": image_id,
            "split": split,
            "is_stego": 1,
            "algorithm": result.get("algorithm", "unknown"),
            "algorithm_class": algo_class,
            "domain": result.get("domain", "unknown"),
            "cover_source": source,
            "cover_path": result.get("cover", ""),
            "stego_path": result.get("stego", ""),
            "payload_rate_bpp": 0.0,  # neural/diffusion don't use bpp
            "payload_type": payload_type,
            "payload_bytes": payload_bytes,
            "payload_hash": "",
            "n_changes": 0,
            "change_rate": 0.0,
            "image_width": width,
            "image_height": height,
            "format": fmt,
            "quality_factor": 0,
        })

    def save(self):
        """Append new records to existing metadata CSV."""
        file_exists = os.path.exists(self.metadata_path)
        mode = "a" if file_exists else "w"

        with open(self.metadata_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.new_records)

        print(f"[METADATA] Appended {len(self.new_records)} records to {self.metadata_path}")


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def assign_split(cover_path: str) -> str:
    """Deterministic split by cover image path hash. Same logic as generate_stego.py.

    ANTI-LEAKAGE: Uses SHA-256 hash of cover path for consistent assignment
    across all scripts — prevents same cover ending up in different splits.
    """
    import hashlib
    h = hashlib.sha256(cover_path.encode("utf-8")).hexdigest()
    ratio = int(h[:8], 16) / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    elif ratio < 0.85:
        return "val"
    else:
        return "test"


# ============================================================
# GENERATION PIPELINES
# ============================================================

def generate_steganogan_stego(
    covers_dir: str,
    output_dir: str,
    metadata_path: str,
    text_corpus_path: str,
    max_images: Optional[int] = None,
    architecture: str = "dense",
):
    """Generate SteganoGAN stego images from cover images.

    SteganoGAN embeds text messages into PNG images via an encoder-decoder network.
    """
    print(f"\n=== SteganoGAN Stego Generation ===")
    print(f"  Covers: {covers_dir}")
    print(f"  Architecture: {architecture}")

    # SteganoGAN works best with PNG
    cover_files = sorted(Path(covers_dir).glob("*.png"))
    if not cover_files:
        # Convert from other formats
        cover_files = sorted(Path(covers_dir).glob("*.jpg"))
    if not cover_files:
        cover_files = sorted(Path(covers_dir).glob("*.pgm"))

    if max_images:
        cover_files = cover_files[:max_images]

    if not cover_files:
        print(f"  [ERROR] No cover images found in {covers_dir}")
        return

    print(f"  Cover images: {len(cover_files)}")

    # Load text corpus for messages
    with open(text_corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus = f.read()
    # Split into sentences for messages
    sentences = [s.strip() for s in corpus.split(".") if len(s.strip()) > 20]
    print(f"  Text sentences available: {len(sentences)}")

    gen = SteganoGANGenerator(architecture=architecture)
    appender = MetadataAppender(metadata_path)

    os.makedirs(output_dir, exist_ok=True)
    total = len(cover_files)
    success_count = 0

    for idx, cover_path in enumerate(tqdm(cover_files, desc="SteganoGAN")):
        split = assign_split(str(cover_path))

        # Select message from corpus
        message = sentences[idx % len(sentences)]

        # Prepare cover (SteganoGAN needs PNG)
        cover_str = str(cover_path)
        if not cover_str.endswith(".png"):
            img = Image.open(cover_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            png_path = os.path.join(output_dir, f"_tmp_cover_{idx:06d}.png")
            img.save(png_path)
            cover_str = png_path

        out_name = f"steganogan_{idx:06d}.png"
        out_path = os.path.join(output_dir, out_name)

        result = gen.embed(cover_str, out_path, message)

        if result["success"] and result["stego"]:
            success_count += 1
            img = Image.open(result["stego"])
            w, h = img.size
            img.close()

            appender.add_record(
                image_id=f"steganogan_{idx:06d}",
                split=split,
                result=result,
                source="div2k" if "div2k" in covers_dir.lower() else "other",
                algo_class="class_e_neural",
                payload_type="english_text",
                payload_bytes=len(message.encode("utf-8")),
                fmt="png",
                width=w,
                height=h,
            )

    appender.save()
    print(f"  Success: {success_count}/{total}")


def generate_diffstega_stego(
    reference_dir: str,
    output_dir: str,
    metadata_path: str,
    diffstega_dir: str,
    max_images: Optional[int] = None,
    prompts: Optional[list[str]] = None,
):
    """Generate DiffStega stego images (coverless diffusion steganography).

    DiffStega generates new images from scratch using Stable Diffusion.
    The password-dependent noise flip encodes the secret.
    """
    print(f"\n=== DiffStega Stego Generation ===")
    print(f"  Reference images: {reference_dir}")
    print(f"  DiffStega repo: {diffstega_dir}")

    ref_files = sorted(Path(reference_dir).glob("*.png"))
    if not ref_files:
        ref_files = sorted(Path(reference_dir).glob("*.jpg"))
    if max_images:
        ref_files = ref_files[:max_images]

    if not ref_files:
        print(f"  [ERROR] No reference images found in {reference_dir}")
        return

    print(f"  Reference images: {len(ref_files)}")

    if prompts is None:
        prompts = [
            "a photograph",
            "a landscape photograph",
            "a portrait photograph",
            "a street scene",
            "a nature photograph",
        ]

    gen = DiffStegaGenerator(diffstega_dir)
    appender = MetadataAppender(metadata_path)

    os.makedirs(output_dir, exist_ok=True)
    total = len(ref_files)
    success_count = 0

    for idx, ref_path in enumerate(tqdm(ref_files, desc="DiffStega")):
        split = assign_split(str(ref_path))

        # Rotate through prompts and passwords
        prompt = prompts[idx % len(prompts)]
        password = 9000 + idx  # unique password per image

        img_output_dir = os.path.join(output_dir, f"diffstega_{idx:06d}")

        result = gen.generate(
            reference_image_path=str(ref_path),
            output_dir=img_output_dir,
            password=password,
            prompt=prompt,
        )

        if result["success"] and result["stego"]:
            success_count += 1
            img = Image.open(result["stego"])
            w, h = img.size
            img.close()

            appender.add_record(
                image_id=f"diffstega_{idx:06d}",
                split=split,
                result=result,
                source="ffhq" if "ffhq" in reference_dir.lower() else "other",
                algo_class="class_f_diffusion",
                payload_type="password_encoded",
                payload_bytes=0,
                fmt="png",
                width=w,
                height=h,
            )

    appender.save()
    print(f"  Success: {success_count}/{total}")


def register_cover_images(
    covers_dir: str,
    metadata_path: str,
    source_name: str,
    domain: str,
    max_images: Optional[int] = None,
):
    """Register cover images for neural/diffusion datasets."""
    print(f"\n=== Registering {source_name} covers ===")

    extensions = ["*.png", "*.jpg", "*.jpeg", "*.pgm"]
    cover_files = []
    for ext in extensions:
        cover_files.extend(Path(covers_dir).glob(ext))
    cover_files = sorted(cover_files)

    if max_images:
        cover_files = cover_files[:max_images]

    if not cover_files:
        print(f"  [ERROR] No images found in {covers_dir}")
        return

    appender = MetadataAppender(metadata_path)
    total = len(cover_files)

    for idx, cover_path in enumerate(tqdm(cover_files, desc=f"{source_name} covers")):
        split = assign_split(str(cover_path))
        img = Image.open(cover_path)
        w, h = img.size
        fmt = cover_path.suffix.lstrip(".")
        img.close()

        appender.new_records.append({
            "image_id": f"{source_name}_cover_{idx:06d}",
            "split": split,
            "is_stego": 0,
            "algorithm": "none",
            "algorithm_class": "cover",
            "domain": "none",
            "cover_source": source_name,
            "cover_path": str(cover_path),
            "stego_path": "",
            "payload_rate_bpp": 0.0,
            "payload_type": "none",
            "payload_bytes": 0,
            "payload_hash": "",
            "n_changes": 0,
            "change_rate": 0.0,
            "image_width": w,
            "image_height": h,
            "format": fmt,
            "quality_factor": 0,
        })

    appender.save()
    print(f"  Registered: {total} cover images")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neural/Diffusion Stego Generation Pipeline"
    )
    parser.add_argument(
        "--phase", type=str, default="steganogan",
        choices=["steganogan", "diffstega", "covers", "all"],
        help="Which pipeline to run",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.csv")
    parser.add_argument(
        "--text-corpus", type=str,
        default="data/raw/text/english_corpus.txt",
    )
    parser.add_argument(
        "--diffstega-dir", type=str,
        default="external/DiffStega",
        help="Path to cloned DiffStega repository",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit number of images (for testing)",
    )
    args = parser.parse_args()

    phases = [args.phase] if args.phase != "all" else ["covers", "steganogan", "diffstega"]

    for phase in phases:
        if phase == "covers":
            # Register DIV2K covers (for SteganoGAN)
            div2k_dir = os.path.join(args.data_dir, "div2k")
            if os.path.isdir(div2k_dir):
                register_cover_images(
                    div2k_dir, args.metadata, "div2k", "neural",
                    max_images=args.max_images,
                )

            # Register FFHQ covers (for DiffStega)
            ffhq_dir = os.path.join(args.data_dir, "ffhq")
            if os.path.isdir(ffhq_dir):
                register_cover_images(
                    ffhq_dir, args.metadata, "ffhq", "diffusion",
                    max_images=args.max_images,
                )

        elif phase == "steganogan":
            # DIV2K covers for SteganoGAN
            div2k_dir = os.path.join(args.data_dir, "div2k")
            if not os.path.isdir(div2k_dir):
                # Fallback: use any available PNG covers
                div2k_dir = os.path.join(args.data_dir, "bossbase")

            generate_steganogan_stego(
                covers_dir=div2k_dir,
                output_dir=os.path.join(args.output_dir, "steganogan"),
                metadata_path=args.metadata,
                text_corpus_path=args.text_corpus,
                max_images=args.max_images,
            )

        elif phase == "diffstega":
            # FFHQ reference images for DiffStega
            ffhq_dir = os.path.join(args.data_dir, "ffhq")
            if not os.path.isdir(ffhq_dir):
                print("[SKIP] FFHQ not found. Download first via scripts/download_datasets.sh")
                continue

            generate_diffstega_stego(
                reference_dir=ffhq_dir,
                output_dir=os.path.join(args.output_dir, "diffstega"),
                metadata_path=args.metadata,
                diffstega_dir=args.diffstega_dir,
                max_images=args.max_images,
            )

    print("\n=== Neural/Diffusion Pipeline Complete ===")


if __name__ == "__main__":
    main()
