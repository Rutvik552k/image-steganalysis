"""
Embed text into a cover image using multiple stego algorithms,
then produce a side-by-side comparison with amplified difference maps.
"""

import numpy as np
from PIL import Image
import conseal as cl
import os

# ── Config ──────────────────────────────────────────────────────────────
COVER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dry_run", "covers", "cover_00000.png")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "stego_comparison")
SECRET_TEXT = "This is a secret message hidden inside an image using steganography for our research paper."
BPP = 0.4  # bits per pixel — moderate payload
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load cover ──────────────────────────────────────────────────────────
cover_img = Image.open(COVER_PATH)
cover = np.array(cover_img, dtype=np.float64)
h, w = cover.shape[:2]
print(f"Cover: {COVER_PATH}")
print(f"Size : {w}x{h}, dtype={cover.dtype}, range=[{cover.min()}, {cover.max()}]")
print(f"Text : \"{SECRET_TEXT}\"")
print(f"Rate : {BPP} bpp\n")

# ── Helper: text to binary payload ──────────────────────────────────────
def text_to_payload_bytes(text: str, num_bytes: int) -> bytes:
    """Convert text to bytes, repeat/truncate to fill num_bytes."""
    text_bytes = text.encode("utf-8")
    repeats = (num_bytes // len(text_bytes)) + 1
    return (text_bytes * repeats)[:num_bytes]

# ── Helper: LSB replacement ─────────────────────────────────────────────
def lsb_replace(cover_arr: np.ndarray, payload: bytes, seed: int) -> np.ndarray:
    """Simple LSB replacement embedding."""
    flat = cover_arr.flatten().astype(np.uint8).copy()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    n_bits = len(bits)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(flat))[:n_bits]

    for i, bit in zip(indices, bits):
        flat[i] = (flat[i] & 0xFE) | bit  # clear LSB, set to payload bit

    return flat.reshape(cover_arr.shape).astype(np.float64)

# ── Embed with each algorithm ──────────────────────────────────────────
total_bits = int(h * w * BPP)
payload_bytes = total_bits // 8
payload = text_to_payload_bytes(SECRET_TEXT, payload_bytes)
print(f"Payload: {payload_bytes} bytes ({total_bits} bits)\n")

results = {}

# Single-channel cover for conseal (expects 2D float64)
cover_ch = cover.astype(np.float64) if cover.ndim == 2 else cover[:,:,0].astype(np.float64)

# 1) LSB Replacement
print("Embedding: LSB Replacement...")
stego_lsb = lsb_replace(cover_ch, payload, SEED)
results["LSB Replacement"] = stego_lsb
print(f"  Changed pixels: {np.sum(stego_lsb != cover_ch)}")

# 2) S-UNIWARD (adaptive, hard to detect)
print("Embedding: S-UNIWARD...")
stego_uniward = cl.suniward.simulate_single_channel(
    x0=cover_ch,
    alpha=BPP,
    seed=SEED,
)
results["S-UNIWARD"] = stego_uniward
print(f"  Changed pixels: {np.sum(stego_uniward != cover_ch)}")

# 3) HILL
print("Embedding: HILL...")
stego_hill = cl.hill.simulate_single_channel(
    x0=cover_ch,
    alpha=BPP,
    seed=SEED,
)
results["HILL"] = stego_hill
print(f"  Changed pixels: {np.sum(stego_hill != cover_ch)}")

# ── Build comparison figure ─────────────────────────────────────────────
print("\nGenerating comparison figure...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cover_display = cover if cover.ndim == 2 else cover[:,:,0]
n_algos = len(results)
fig, axes = plt.subplots(n_algos + 1, 3, figsize=(15, 5 * (n_algos + 1)))

# Row 0: Cover image (show in all 3 columns for context)
axes[0, 0].imshow(cover_display, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original Cover Image", fontsize=13, fontweight="bold")
axes[0, 0].axis("off")

# Show histogram of cover
axes[0, 1].hist(cover_display.flatten(), bins=256, range=(0, 255), color="steelblue", alpha=0.8)
axes[0, 1].set_title("Cover Histogram", fontsize=13)
axes[0, 1].set_xlabel("Pixel Value")
axes[0, 1].set_ylabel("Count")

# Text info
info_text = (
    f"Image: {w}×{h} grayscale\n"
    f"Payload: {BPP} bpp ({payload_bytes} bytes)\n"
    f"Text: \"{SECRET_TEXT[:60]}...\""
)
axes[0, 2].text(0.1, 0.5, info_text, fontsize=11, verticalalignment="center",
                fontfamily="monospace", transform=axes[0, 2].transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
axes[0, 2].set_title("Embedding Parameters", fontsize=13)
axes[0, 2].axis("off")

# Rows 1+: Each algorithm
for i, (name, stego) in enumerate(results.items(), start=1):
    stego_display = stego.astype(np.float64)
    cover_ch_display = cover_display.astype(np.float64)

    # Col 0: Stego image
    axes[i, 0].imshow(stego_display, cmap="gray", vmin=0, vmax=255)
    axes[i, 0].set_title(f"{name} — Stego Image", fontsize=13, fontweight="bold")
    axes[i, 0].axis("off")

    # Col 1: Difference map (amplified 50x)
    diff = np.abs(stego_display - cover_ch_display)
    n_changed = np.sum(diff > 0)
    change_rate = n_changed / diff.size * 100

    amplified = np.clip(diff * 50, 0, 255)
    axes[i, 1].imshow(amplified, cmap="hot", vmin=0, vmax=255)
    axes[i, 1].set_title(f"Difference ×50 — {n_changed} pixels changed ({change_rate:.1f}%)", fontsize=11)
    axes[i, 1].axis("off")

    # Col 2: Stats
    stats_text = (
        f"Algorithm: {name}\n"
        f"Changed pixels: {n_changed:,} / {diff.size:,}\n"
        f"Change rate: {change_rate:.2f}%\n"
        f"Max diff: {diff.max():.0f}\n"
        f"Mean diff (changed): {diff[diff > 0].mean():.2f}\n"
        f"PSNR: {10 * np.log10(255**2 / np.mean((stego_display - cover_ch_display)**2)):.1f} dB"
    )
    axes[i, 2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment="center",
                    fontfamily="monospace", transform=axes[i, 2].transAxes,
                    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
    axes[i, 2].set_title(f"{name} — Statistics", fontsize=13)
    axes[i, 2].axis("off")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "stego_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nComparison saved: {out_path}")

# ── Save individual images ──────────────────────────────────────────────
cover_out = os.path.join(OUT_DIR, "cover_original.png")
Image.fromarray(cover_display.astype(np.uint8)).save(cover_out)
print(f"Cover saved: {cover_out}")

for name, stego in results.items():
    fname = name.lower().replace(" ", "_").replace("-", "_")
    stego_out = os.path.join(OUT_DIR, f"stego_{fname}.png")
    Image.fromarray(stego.astype(np.uint8)).save(stego_out)
    print(f"Stego saved: {stego_out}")

print("\nDone! Open stego_comparison.png to see side-by-side comparison.")
