"""
Global Steganalysis - Stego Image Generation Pipeline

Generates stego images across ALL algorithm classes:
  Class A: Direct bit embedding (LSB-R, LSB-M, F5, nsF5)
  Class B: STC + hand-crafted costs (HUGO, WOW, S-UNIWARD, HILL, MiPOD)
  Class D: JPEG adaptive (J-UNIWARD, JMiPOD, UERD) -- ALASKA2 pre-generated
  Class E: Neural encoder-decoder (SteganoGAN, HiDDeN)
  Class F: Diffusion model (DiffStega, Diffusion-Stego)
  Class H: Real-world tools (Steghide, OpenStego)

Usage:
    python scripts/generate_stego.py --phase 1     # minimal (S-UNIWARD, HILL, J-UNIWARD)
    python scripts/generate_stego.py --phase 2     # full (all algorithms)
    python scripts/generate_stego.py --algo hugo   # single algorithm
"""

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# PAYLOAD GENERATION
# ============================================================

class PayloadGenerator:
    """Generate text and random payloads of exact bit lengths."""

    def __init__(self, text_corpus_path: str):
        self.corpus_path = text_corpus_path
        self._text_data: Optional[bytes] = None

    @property
    def text_data(self) -> bytes:
        if self._text_data is None:
            with open(self.corpus_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            # Clean: remove gutenberg headers, normalize whitespace
            lines = raw.split("\n")
            cleaned = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("***"):
                    cleaned.append(line)
            self._text_data = " ".join(cleaned).encode("utf-8")
        return self._text_data

    def get_text_payload(self, num_bytes: int, offset: int = 0) -> bytes:
        """Extract text payload of exact byte length from corpus."""
        corpus = self.text_data
        # Wrap around if offset exceeds corpus length
        start = offset % len(corpus)
        payload = bytearray()
        while len(payload) < num_bytes:
            remaining = num_bytes - len(payload)
            chunk = corpus[start : start + remaining]
            payload.extend(chunk)
            start = (start + len(chunk)) % len(corpus)
        return bytes(payload[:num_bytes])

    def get_random_payload(self, num_bytes: int) -> bytes:
        """Generate cryptographically random payload."""
        return os.urandom(num_bytes)

    def payload_to_file(self, payload: bytes, path: str) -> str:
        """Write payload to temporary file, return path."""
        with open(path, "wb") as f:
            f.write(payload)
        return path


# ============================================================
# EMBEDDING RATE CALCULATOR
# ============================================================

def calc_payload_bytes(image_path: str, bpp: float, domain: str = "spatial") -> int:
    """Calculate payload size in bytes for given bpp and image."""
    img = Image.open(image_path)
    w, h = img.size

    if domain == "spatial":
        # bpp = bits per pixel (grayscale) or bits per sample (color)
        total_bits = int(w * h * bpp)
    elif domain == "jpeg":
        # bpp = bits per non-zero AC coefficient
        # Approximate: ~63 AC coefficients per 8x8 block, ~60% non-zero
        num_blocks = (w // 8) * (h // 8)
        nz_ac = int(num_blocks * 63 * 0.6)  # approximate
        total_bits = int(nz_ac * bpp)
    else:
        total_bits = int(w * h * bpp)

    return max(total_bits // 8, 1)


# ============================================================
# SPATIAL DOMAIN GENERATORS (Class A + B)
# ============================================================

class ConsealGenerator:
    """Generate stego using conseal library.

    Verified API (github.com/uibk-uncover/conseal README):
      Spatial: x1 = cl.<algo>.simulate_single_channel(x0=cover, alpha=rate, seed=s)
      JPEG:    y1 = cl.<algo>.simulate_single_channel(y0=dct, qt=qt, alpha=rate, seed=s)
      J-UNIWARD needs x0 (spatial) + y0 (dct) + qt.
      UERD uses embedding_rate= instead of alpha=.
      LSB spatial: cl.lsb.simulate(x0=cover, alpha=rate, seed=s)
    """

    SPATIAL_ALGOS = ["hugo", "wow", "s_uniward", "hill", "mipod", "lsb_matching"]
    JPEG_ALGOS = ["j_uniward", "uerd", "nsf5", "f5"]

    def __init__(self):
        try:
            import conseal as cl
            self.cl = cl
        except ImportError:
            print("ERROR: conseal not installed. Install with: pip install conseal")
            print("  See: https://github.com/uibk-uncover/conseal")
            sys.exit(1)

    def embed_spatial(
        self,
        cover_path: str,
        output_path: str,
        algorithm: str,
        payload: bytes,
        alpha: float,
    ) -> dict:
        """Embed into spatial-domain cover image via conseal simulate_single_channel."""
        cover = np.array(Image.open(cover_path))
        if cover.ndim == 3:
            cover = cover[:, :, 0]  # grayscale

        n_pixels = cover.shape[0] * cover.shape[1]

        # Derive deterministic seed from payload content
        seed = int.from_bytes(hashlib.sha256(payload[:32]).digest()[:4], "big")

        # Each algo provides simulate_single_channel(x0=, alpha=, seed=)
        if algorithm == "hugo":
            stego = self.cl.hugo.simulate_single_channel(x0=cover, alpha=alpha, seed=seed)
        elif algorithm == "wow":
            stego = self.cl.wow.simulate_single_channel(x0=cover, alpha=alpha, seed=seed)
        elif algorithm == "s_uniward":
            stego = self.cl.suniward.simulate_single_channel(x0=cover, alpha=alpha, seed=seed)
        elif algorithm == "hill":
            stego = self.cl.hill.simulate_single_channel(x0=cover, alpha=alpha, seed=seed)
        elif algorithm == "mipod":
            stego = self.cl.mipod.simulate_single_channel(x0=cover, alpha=alpha, seed=seed)
        elif algorithm == "lsb_matching":
            stego = self.cl.lsb.simulate(x0=cover, alpha=alpha, seed=seed)
        else:
            raise ValueError(f"Unknown spatial algorithm: {algorithm}")

        Image.fromarray(stego).save(output_path)

        n_changes = int(np.sum(cover != stego))
        return {
            "algorithm": algorithm,
            "domain": "spatial",
            "cover": cover_path,
            "stego": output_path,
            "alpha": alpha,
            "n_changes": n_changes,
            "change_rate": n_changes / n_pixels,
        }

    def embed_jpeg(
        self,
        cover_path: str,
        output_path: str,
        algorithm: str,
        payload: bytes,
        alpha: float,
    ) -> dict:
        """Embed into JPEG-domain cover image via conseal simulate_single_channel.

        J-UNIWARD needs spatial pixels (x0) + DCT (y0) + quantization table (qt).
        UERD/EBS need DCT (y0) + qt. UERD uses embedding_rate= param.
        nsF5/F5 need only DCT (y0).
        """
        import jpeglib

        jpeg = jpeglib.read_dct(cover_path)
        dct_coeffs_orig = jpeg.Y.copy()

        seed = int.from_bytes(hashlib.sha256(payload[:32]).digest()[:4], "big")

        if algorithm == "j_uniward":
            # J-UNIWARD needs spatial pixels too
            im0 = jpeglib.read_spatial(cover_path, jpeglib.JCS_GRAYSCALE)
            jpeg.Y = self.cl.juniward.simulate_single_channel(
                x0=im0.spatial[..., 0],
                y0=jpeg.Y,
                qt=jpeg.qt[0],
                alpha=alpha,
                seed=seed,
            )
        elif algorithm == "uerd":
            # UERD uses embedding_rate= instead of alpha=
            jpeg.Y = self.cl.uerd.simulate_single_channel(
                y0=jpeg.Y,
                qt=jpeg.qt[0],
                embedding_rate=alpha,
                seed=seed,
            )
        elif algorithm == "nsf5":
            jpeg.Y = self.cl.nsF5.simulate_single_channel(
                y0=jpeg.Y,
                alpha=alpha,
                seed=seed,
            )
        elif algorithm == "f5":
            jpeg.Y = self.cl.F5.simulate_single_channel(
                y0=jpeg.Y,
                alpha=alpha,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown JPEG algorithm: {algorithm}")

        jpeg.write_dct(output_path)

        n_changes = int(np.sum(dct_coeffs_orig != jpeg.Y))
        return {
            "algorithm": algorithm,
            "domain": "jpeg",
            "cover": cover_path,
            "stego": output_path,
            "alpha": alpha,
            "n_changes": n_changes,
        }


class LSBReplacementGenerator:
    """Simple LSB replacement (Class A)."""

    def embed(
        self,
        cover_path: str,
        output_path: str,
        payload: bytes,
    ) -> dict:
        cover = np.array(Image.open(cover_path))
        if cover.ndim == 3:
            cover = cover[:, :, 0]

        flat = cover.flatten().copy()
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        n_embed = min(len(bits), len(flat))
        flat[:n_embed] = (flat[:n_embed] & 0xFE) | bits[:n_embed]

        stego = flat.reshape(cover.shape)
        Image.fromarray(stego.astype(np.uint8)).save(output_path)

        return {
            "algorithm": "lsb_replacement",
            "domain": "spatial",
            "cover": cover_path,
            "stego": output_path,
            "n_bits_embedded": n_embed,
        }


class SteghideGenerator:
    """Steghide CLI wrapper — real-world JPEG stego with text payloads."""

    def embed(
        self,
        cover_path: str,
        output_path: str,
        payload_file: str,
        passphrase: str = "global_steg_benchmark",
    ) -> dict:
        cmd = [
            "steghide", "embed",
            "-cf", cover_path,
            "-ef", payload_file,
            "-sf", output_path,
            "-p", passphrase,
            "-f",  # force overwrite
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0

        return {
            "algorithm": "steghide",
            "domain": "jpeg",
            "cover": cover_path,
            "stego": output_path if success else None,
            "success": success,
            "error": result.stderr if not success else None,
        }


class OpenStegoGenerator:
    """OpenStego CLI wrapper — spatial LSB with text payloads."""

    def embed(
        self,
        cover_path: str,
        output_path: str,
        payload_file: str,
    ) -> dict:
        cmd = [
            "openstego", "embed",
            "-mf", payload_file,
            "-cf", cover_path,
            "-sf", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0

        return {
            "algorithm": "openstego",
            "domain": "spatial",
            "cover": cover_path,
            "stego": output_path if success else None,
            "success": success,
            "error": result.stderr if not success else None,
        }


# ============================================================
# METADATA TRACKER
# ============================================================

class MetadataTracker:
    """Track all generated stego images with full metadata."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.records = []
        self.fieldnames = [
            "image_id", "split", "is_stego", "algorithm", "algorithm_class",
            "domain", "cover_source", "cover_path", "stego_path",
            "payload_rate_bpp", "payload_type", "payload_bytes",
            "payload_hash", "n_changes", "change_rate",
            "image_width", "image_height", "format", "quality_factor",
        ]

    def add_cover(self, image_id: str, split: str, cover_path: str,
                  source: str, fmt: str, width: int, height: int,
                  qf: Optional[int] = None):
        self.records.append({
            "image_id": image_id,
            "split": split,
            "is_stego": 0,
            "algorithm": "none",
            "algorithm_class": "cover",
            "domain": "none",
            "cover_source": source,
            "cover_path": cover_path,
            "stego_path": "",
            "payload_rate_bpp": 0.0,
            "payload_type": "none",
            "payload_bytes": 0,
            "payload_hash": "",
            "n_changes": 0,
            "change_rate": 0.0,
            "image_width": width,
            "image_height": height,
            "format": fmt,
            "quality_factor": qf or 0,
        })

    def add_stego(self, image_id: str, split: str, result: dict,
                  source: str, algo_class: str, payload_type: str,
                  payload_rate: float, payload_hash: str,
                  payload_bytes: int, fmt: str,
                  width: int, height: int, qf: Optional[int] = None):
        self.records.append({
            "image_id": image_id,
            "split": split,
            "is_stego": 1,
            "algorithm": result.get("algorithm", "unknown"),
            "algorithm_class": algo_class,
            "domain": result.get("domain", "unknown"),
            "cover_source": source,
            "cover_path": result.get("cover", ""),
            "stego_path": result.get("stego", result.get("output", "")),
            "payload_rate_bpp": payload_rate,
            "payload_type": payload_type,
            "payload_bytes": payload_bytes,
            "payload_hash": payload_hash,
            "n_changes": result.get("n_changes", 0),
            "change_rate": result.get("change_rate", 0.0),
            "image_width": width,
            "image_height": height,
            "format": fmt,
            "quality_factor": qf or 0,
        })

    def save(self):
        """Save metadata to CSV."""
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[METADATA] Saved {len(self.records)} records to {self.output_path}")

    def save_json(self):
        """Save metadata as JSON (for PyTorch dataset)."""
        json_path = self.output_path.replace(".csv", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        print(f"[METADATA] Saved JSON to {json_path}")


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def assign_split(cover_path: str) -> str:
    """Deterministic split assignment based on COVER IMAGE HASH.

    ANTI-LEAKAGE: Split is determined by the cover image path hash,
    NOT by index. This guarantees:
      1. Same cover always maps to same split regardless of script/run order
      2. All stego variants of one cover land in the same split
      3. Consistent across generate_stego.py and generate_neural_diffusion_stego.py
      4. Independent of total image count or max_images setting
    """
    h = hashlib.sha256(cover_path.encode("utf-8")).hexdigest()
    # Use first 8 hex chars as uniform hash → deterministic float in [0, 1)
    ratio = int(h[:8], 16) / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    elif ratio < 0.85:
        return "val"
    else:
        return "test"


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_spatial_stego(
    covers_dir: str,
    output_dir: str,
    algorithms: list[str],
    rates: list[float],
    payload_gen: PayloadGenerator,
    tracker: MetadataTracker,
    payload_types: list[str] = ["english_text", "random_binary"],
    max_images: Optional[int] = None,
):
    """Generate spatial-domain stego images."""
    print(f"\n=== Spatial Domain Stego Generation ===")
    print(f"  Covers: {covers_dir}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Rates: {rates}")
    print(f"  Payload types: {payload_types}")

    cover_files = sorted(Path(covers_dir).glob("*.pgm"))
    if not cover_files:
        cover_files = sorted(Path(covers_dir).glob("*.png"))
    if max_images:
        cover_files = cover_files[:max_images]

    print(f"  Cover images: {len(cover_files)}")

    gen = ConsealGenerator()
    lsb_gen = LSBReplacementGenerator()

    total = len(cover_files)
    for idx, cover_path in enumerate(tqdm(cover_files, desc="Spatial stego")):
        cover_str = str(cover_path)
        split = assign_split(cover_str)
        img = Image.open(cover_path)
        w, h = img.size
        img.close()

        # Register cover
        cover_id = f"spatial_cover_{idx:06d}"
        tracker.add_cover(
            image_id=cover_id, split=split, cover_path=cover_str,
            source="bossbase", fmt="pgm", width=w, height=h,
        )

        for algo in algorithms:
            for rate in rates:
                payload_bytes = calc_payload_bytes(cover_str, rate, "spatial")

                for ptype in payload_types:
                    # Generate payload
                    if ptype == "english_text":
                        payload = payload_gen.get_text_payload(
                            payload_bytes, offset=idx * 10000
                        )
                    elif ptype == "random_binary":
                        payload = payload_gen.get_random_payload(payload_bytes)
                    else:
                        continue

                    phash = hashlib.sha256(payload).hexdigest()[:16]

                    # Output path
                    out_name = f"{cover_path.stem}_{algo}_{rate}_{ptype}.png"
                    out_path = os.path.join(output_dir, algo, f"rate_{rate}", out_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    try:
                        if algo == "lsb_replacement":
                            result = lsb_gen.embed(cover_str, out_path, payload)
                        else:
                            result = gen.embed_spatial(
                                cover_str, out_path, algo, payload, rate
                            )

                        stego_id = f"spatial_{algo}_{idx:06d}_{rate}_{ptype}"
                        tracker.add_stego(
                            image_id=stego_id, split=split, result=result,
                            source="bossbase", algo_class=get_algo_class(algo),
                            payload_type=ptype, payload_rate=rate,
                            payload_hash=phash, payload_bytes=payload_bytes,
                            fmt="png", width=w, height=h,
                        )
                    except Exception as e:
                        print(f"\n  [ERROR] {algo} on {cover_path.name}: {e}")


def generate_jpeg_stego(
    covers_dir: str,
    output_dir: str,
    algorithms: list[str],
    rates: list[float],
    payload_gen: PayloadGenerator,
    tracker: MetadataTracker,
    payload_types: list[str] = ["english_text", "random_binary"],
    max_images: Optional[int] = None,
):
    """Generate JPEG-domain stego images (beyond ALASKA2 pre-generated)."""
    print(f"\n=== JPEG Domain Stego Generation ===")
    print(f"  Covers: {covers_dir}")
    print(f"  Algorithms: {algorithms}")

    cover_files = sorted(Path(covers_dir).glob("*.jpg"))
    if not cover_files:
        cover_files = sorted(Path(covers_dir).glob("*.jpeg"))
    if max_images:
        cover_files = cover_files[:max_images]

    print(f"  Cover images: {len(cover_files)}")

    gen = ConsealGenerator()
    steghide_gen = SteghideGenerator()

    total = len(cover_files)
    tmp_payload = os.path.join(output_dir, "_tmp_payload.bin")

    for idx, cover_path in enumerate(tqdm(cover_files, desc="JPEG stego")):
        cover_str = str(cover_path)
        split = assign_split(cover_str)
        img = Image.open(cover_path)
        w, h = img.size
        img.close()

        # Only register cover if not already registered (avoids duplicate with
        # register_alaska2_pregenerated which also registers ALASKA2 covers)
        cover_id = f"jpeg_cover_{idx:06d}"
        already_registered = any(
            r["cover_path"] == cover_str and r["is_stego"] == 0
            for r in tracker.records
        )
        if not already_registered:
            tracker.add_cover(
                image_id=cover_id, split=split, cover_path=cover_str,
                source="alaska2", fmt="jpeg", width=w, height=h,
            )

        for algo in algorithms:
            for rate in rates:
                payload_bytes = calc_payload_bytes(cover_str, rate, "jpeg")

                for ptype in payload_types:
                    if ptype == "english_text":
                        payload = payload_gen.get_text_payload(
                            payload_bytes, offset=idx * 10000
                        )
                    elif ptype == "random_binary":
                        payload = payload_gen.get_random_payload(payload_bytes)
                    else:
                        continue

                    phash = hashlib.sha256(payload).hexdigest()[:16]

                    out_name = f"{cover_path.stem}_{algo}_{rate}_{ptype}.jpg"
                    out_path = os.path.join(output_dir, algo, f"rate_{rate}", out_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    try:
                        if algo == "steghide":
                            payload_gen.payload_to_file(payload, tmp_payload)
                            result = steghide_gen.embed(
                                cover_str, out_path, tmp_payload
                            )
                        elif algo in ConsealGenerator.JPEG_ALGOS:
                            result = gen.embed_jpeg(
                                cover_str, out_path, algo, payload, rate
                            )
                        else:
                            continue

                        stego_id = f"jpeg_{algo}_{idx:06d}_{rate}_{ptype}"
                        tracker.add_stego(
                            image_id=stego_id, split=split, result=result,
                            source="alaska2", algo_class=get_algo_class(algo),
                            payload_type=ptype, payload_rate=rate,
                            payload_hash=phash, payload_bytes=payload_bytes,
                            fmt="jpeg", width=w, height=h,
                        )
                    except Exception as e:
                        print(f"\n  [ERROR] {algo} on {cover_path.name}: {e}")

    # Cleanup
    if os.path.exists(tmp_payload):
        os.remove(tmp_payload)


def register_alaska2_pregenerated(
    alaska2_dir: str,
    tracker: MetadataTracker,
    max_images: Optional[int] = None,
):
    """Register ALASKA2 pre-generated stego (JMiPOD, J-UNIWARD, UERD)."""
    print(f"\n=== Registering ALASKA2 Pre-Generated Stego ===")

    algo_dirs = {
        "JMiPOD": ("jmipod", "class_d_stc_jpeg"),
        "JUNIWARD": ("j_uniward", "class_d_stc_jpeg"),
        "UERD": ("uerd", "class_d_stc_jpeg"),
    }

    cover_dir = os.path.join(alaska2_dir, "Cover")
    if not os.path.isdir(cover_dir):
        print(f"  [ERROR] ALASKA2 Cover dir not found: {cover_dir}")
        return

    cover_files = sorted(Path(cover_dir).glob("*.jpg"))
    if max_images:
        cover_files = cover_files[:max_images]
    total = len(cover_files)

    print(f"  Covers: {total}")

    for idx, cover_path in enumerate(tqdm(cover_files, desc="ALASKA2 registry")):
        split = assign_split(str(cover_path))
        img = Image.open(cover_path)
        w, h = img.size
        img.close()

        cover_id = f"alaska2_cover_{idx:06d}"
        tracker.add_cover(
            image_id=cover_id, split=split, cover_path=str(cover_path),
            source="alaska2", fmt="jpeg", width=w, height=h,
        )

        for dir_name, (algo_name, algo_class) in algo_dirs.items():
            stego_path = os.path.join(alaska2_dir, dir_name, cover_path.name)
            if os.path.exists(stego_path):
                stego_id = f"alaska2_{algo_name}_{idx:06d}"
                tracker.add_stego(
                    image_id=stego_id, split=split,
                    result={
                        "algorithm": algo_name,
                        "domain": "jpeg",
                        "cover": str(cover_path),
                        "stego": stego_path,
                    },
                    source="alaska2", algo_class=algo_class,
                    payload_type="random_binary",  # ALASKA2 uses random
                    payload_rate=0.4,  # standard ALASKA2 rate
                    payload_hash="alaska2_pregenerated",
                    payload_bytes=0,  # unknown exact
                    fmt="jpeg", width=w, height=h,
                )


def get_algo_class(algo: str) -> str:
    """Map algorithm name to class label."""
    mapping = {
        "lsb_replacement": "class_a_direct",
        "lsb_matching": "class_a_direct",
        "f5": "class_a_direct",
        "outguess": "class_a_direct",
        "nsf5": "class_a_direct",
        "hugo": "class_b_stc_spatial",
        "wow": "class_b_stc_spatial",
        "s_uniward": "class_b_stc_spatial",
        "hill": "class_b_stc_spatial",
        "mipod": "class_b_stc_spatial",
        "j_uniward": "class_d_stc_jpeg",
        "jmipod": "class_d_stc_jpeg",
        "uerd": "class_d_stc_jpeg",
        "steganogan": "class_e_neural",
        "hidden": "class_e_neural",
        "diffstega": "class_f_diffusion",
        "diffusion_stego": "class_f_diffusion",
        "steghide": "class_h_realworld",
        "openstego": "class_h_realworld",
    }
    return mapping.get(algo, "unknown")


# ============================================================
# PHASE DEFINITIONS
# ============================================================

PHASE_1 = {
    "spatial_algorithms": ["s_uniward", "hill"],
    "jpeg_algorithms": [],  # use ALASKA2 pre-generated
    "rates": [0.2, 0.4],
    "payload_types": ["english_text", "random_binary"],
    "max_spatial_images": 10000,
    "max_jpeg_images": None,  # register all ALASKA2
    "description": "Minimal: validate hypothesis with 2 spatial algos + ALASKA2 JPEG",
}

PHASE_2 = {
    "spatial_algorithms": [
        "lsb_replacement", "lsb_matching",
        "hugo", "wow", "s_uniward", "hill", "mipod",
    ],
    "jpeg_algorithms": ["nsf5", "steghide"],
    "rates": [0.1, 0.2, 0.3, 0.4, 0.5],
    "payload_types": ["english_text", "random_binary"],
    "max_spatial_images": 10000,
    "max_jpeg_images": 10000,
    "description": "Full: all spatial + JPEG algos, all rates",
}


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Global Steganalysis Dataset Generator")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Generation phase (1=minimal, 2=full)")
    parser.add_argument("--algo", type=str, default=None,
                        help="Generate for single algorithm only")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Raw data directory")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for generated stego")
    parser.add_argument("--text-corpus", type=str,
                        default="data/raw/text/english_corpus.txt",
                        help="Path to text corpus for payloads")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of cover images (for testing)")
    args = parser.parse_args()

    # Select phase config
    config = PHASE_1 if args.phase == 1 else PHASE_2
    print(f"=== Global Steganalysis Dataset Generator ===")
    print(f"Phase: {args.phase} — {config['description']}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    # Initialize
    os.makedirs(args.output_dir, exist_ok=True)
    payload_gen = PayloadGenerator(args.text_corpus)
    tracker = MetadataTracker(os.path.join(args.output_dir, "metadata.csv"))

    max_img = args.max_images

    # Override for single algorithm
    if args.algo:
        spatial = [args.algo] if args.algo in PHASE_2["spatial_algorithms"] else []
        jpeg = [args.algo] if args.algo in PHASE_2["jpeg_algorithms"] else []
        config["spatial_algorithms"] = spatial
        config["jpeg_algorithms"] = jpeg

    # 1. Register ALASKA2 pre-generated
    alaska2_dir = os.path.join(args.data_dir, "alaska2")
    if os.path.isdir(alaska2_dir):
        register_alaska2_pregenerated(
            alaska2_dir, tracker,
            max_images=max_img or config.get("max_jpeg_images"),
        )

    # 2. Generate spatial stego
    bossbase_dir = os.path.join(args.data_dir, "bossbase")
    if config["spatial_algorithms"] and os.path.isdir(bossbase_dir):
        generate_spatial_stego(
            covers_dir=bossbase_dir,
            output_dir=os.path.join(args.output_dir, "spatial"),
            algorithms=config["spatial_algorithms"],
            rates=config["rates"],
            payload_gen=payload_gen,
            tracker=tracker,
            payload_types=config["payload_types"],
            max_images=max_img or config.get("max_spatial_images"),
        )

    # 3. Generate additional JPEG stego (beyond ALASKA2)
    alaska2_covers = os.path.join(args.data_dir, "alaska2", "Cover")
    if config["jpeg_algorithms"] and os.path.isdir(alaska2_covers):
        generate_jpeg_stego(
            covers_dir=alaska2_covers,
            output_dir=os.path.join(args.output_dir, "jpeg"),
            algorithms=config["jpeg_algorithms"],
            rates=config["rates"],
            payload_gen=payload_gen,
            tracker=tracker,
            payload_types=config["payload_types"],
            max_images=max_img or config.get("max_jpeg_images"),
        )

    # 4. Save metadata
    tracker.save()
    tracker.save_json()

    # Summary
    print(f"\n=== Generation Complete ===")
    print(f"Total records: {len(tracker.records)}")
    covers = sum(1 for r in tracker.records if r["is_stego"] == 0)
    stegos = sum(1 for r in tracker.records if r["is_stego"] == 1)
    print(f"  Covers: {covers}")
    print(f"  Stego: {stegos}")

    algos = set(r["algorithm"] for r in tracker.records if r["is_stego"] == 1)
    print(f"  Algorithms: {sorted(algos)}")
    print(f"  Metadata: {args.output_dir}/metadata.csv")


if __name__ == "__main__":
    main()
