"""
Global Steganalysis Dataset - PyTorch DataLoader

Feature engineering based on verified research (2024-2026):
  - SRM 30-filter bank: exact kernels from PENet/Ye-Net (with proper normalization)
  - BayarConv: constrained prediction-error filters (Bayar & Stamm, IEEE TIFS 2018)
  - TLU activation: truncation to [-T, T] after filtering (standard T=3)
  - NO ImageNet normalization (raw [0,255] → SRM → TLU → BN pipeline)
  - NO resize (destroys stego signal) — only center crop to target size
  - Safe augmentation only: D4 group (flips + 90-degree rotations)

Usage:
    from src.data.dataset import SteganalysisDataset, create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        splits_dir="data/splits",
        batch_size=32,
        target_size=256,
    )
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ============================================================
# SRM FILTER BANK — Verified from PENet (TIFS 2024) / Ye-Net
# ============================================================

def build_srm_filters() -> np.ndarray:
    """Build the 30 SRM high-pass filter kernels.

    Exact kernels verified from PENet/Ye-Net implementations:
      - 8 first-order  (3x3, embedded in 5x5, no normalization)
      - 4 second-order  (3x3, embedded in 5x5, normalized /2)
      - 8 third-order  (5x5, normalized /3)
      - 4 EDGE 3x3     (3x3, embedded in 5x5, normalized /4)
      - 4 EDGE 5x5     (5x5, normalized /12)
      - 1 SQUARE 3x3   (3x3, embedded in 5x5, normalized /4)
      - 1 SQUARE 5x5   (5x5, normalized /12)

    All kernels are zero-padded to 5x5 for uniform conv2d.
    Returns shape (30, 1, 5, 5).

    Reference: Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital
    Images," IEEE TIFS, Vol. 7, No. 3, 2012.
    """
    filters = []

    def pad_3x3(k):
        """Embed a 3x3 kernel into 5x5 with zero padding."""
        f = np.zeros((5, 5), dtype=np.float32)
        f[1:4, 1:4] = k
        return f

    # ------ Class 1: 1st-order differences (8 filters, unnormalized) ------
    # 8 directions from center pixel to each neighbor
    first_order_3x3 = [
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),   # right
        np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32),   # left
        np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),   # up
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),   # down
        np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32),   # upper-right
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),   # upper-left
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),   # lower-right
        np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32),   # lower-left
    ]
    for k in first_order_3x3:
        filters.append(pad_3x3(k))

    # ------ Class 2: 2nd-order differences (4 filters, /2) ------
    second_order_3x3 = [
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),   # horizontal
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),   # vertical
        np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32),   # diagonal
        np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32),   # anti-diagonal
    ]
    for k in second_order_3x3:
        filters.append(pad_3x3(k / 2.0))

    # ------ Class 3: 3rd-order differences (8 filters, /3) ------
    # Directional 3rd-order finite differences in 8 directions (5x5)
    third_order_base = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)  # horizontal
    for k in range(4):
        filters.append(np.rot90(third_order_base, k) / 3.0)

    third_order_diag = np.array([
        [0, 0, 0, 0, -1],
        [0, 0, 0, 3, 0],
        [0, 0, -3, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)  # diagonal
    for k in range(4):
        filters.append(np.rot90(third_order_diag, k) / 3.0)

    # ------ EDGE 3x3: directional Laplacian (4 filters, /4) ------
    edge3_base = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [0, 0, 0],
    ], dtype=np.float32)
    for k in range(4):
        filters.append(pad_3x3(np.rot90(edge3_base, k) / 4.0))

    # ------ SQUARE 3x3: isotropic Laplacian (1 filter, /4) ------
    sq3 = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1],
    ], dtype=np.float32)
    filters.append(pad_3x3(sq3 / 4.0))

    # ------ SQUARE 5x5: extended isotropic Laplacian (1 filter, /12) ------
    sq5 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32)
    filters.append(sq5 / 12.0)

    # Total: 8 + 4 + 8 + 4 + 1 + 1 = 26. Pad to 30 with EDGE 5x5 (4 filters, /12)
    edge5_base = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)
    for k in range(4):
        filters.append(np.rot90(edge5_base, k) / 12.0)

    # Total: 26 + 4 = 30 filters
    assert len(filters) == 30, f"Expected 30 filters, got {len(filters)}"

    return np.array(filters, dtype=np.float32).reshape(30, 1, 5, 5)


# ============================================================
# BayarConv — Constrained Prediction-Error Filter
# ============================================================

class BayarConv2d(nn.Module):
    """Constrained convolutional layer for forensic feature extraction.

    Forces filters to compute prediction errors: center weight = -1,
    remaining weights sum to 1. This makes the output = (predicted value) - (actual value).

    Reference: Bayar & Stamm, "Constrained Convolutional Neural Networks:
    A New Approach Towards General Purpose Image Manipulation Detection,"
    IEEE TIFS, 2018.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_normal_(self.weight)

        # Mask: 1 at center, 0 elsewhere
        center = kernel_size // 2
        self.register_buffer("center_mask", torch.zeros(kernel_size, kernel_size))
        self.center_mask[center, center] = 1.0

    @torch.no_grad()
    def project_weights(self):
        """Project weights into feasible set: center=-1, rest sums to 1.

        Call AFTER optimizer.step() in the training loop, NOT during forward().
        In-place mutation during forward() is unsafe with autograd.

        Usage in training loop:
            optimizer.step()
            for m in model.modules():
                if isinstance(m, BayarConv2d):
                    m.project_weights()
        """
        # Zero out center
        self.weight.data *= (1.0 - self.center_mask)
        # Normalize remaining weights per filter to sum to 1
        rest_sum = self.weight.data.sum(dim=(2, 3), keepdim=True)
        self.weight.data /= (rest_sum + 1e-8)
        # Set center to -1
        self.weight.data += (-1.0) * self.center_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.conv2d(x, self.weight, padding=pad)


# ============================================================
# TLU (Truncated Linear Unit) Activation
# ============================================================

class TLU(nn.Module):
    """Truncated Linear Unit — clips residuals to [-T, T].

    Standard in steganalysis preprocessing (Ye-Net, Yedroudj-Net).
    Default T=3 as per Yedroudj-Net.
    """

    def __init__(self, threshold: float = 3.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -self.threshold, self.threshold)


# ============================================================
# D4 GROUP AUGMENTATION (Safe for Steganalysis)
# ============================================================

def d4_augment(x: np.ndarray) -> np.ndarray:
    """Apply random D4 group transformation (8 symmetries).

    SAFE augmentations only — preserves exact pixel values:
      - 4 rotations (0, 90, 180, 270)
      - 2 flips (horizontal, vertical)
      - Total: 8 unique transformations

    NO resize, blur, noise, crop, color jitter, JPEG recompression.
    These destroy the stego signal.
    """
    choice = np.random.randint(0, 8)
    if choice == 0:
        return x  # identity
    elif choice == 1:
        return np.rot90(x, 1).copy()
    elif choice == 2:
        return np.rot90(x, 2).copy()
    elif choice == 3:
        return np.rot90(x, 3).copy()
    elif choice == 4:
        return np.fliplr(x).copy()
    elif choice == 5:
        return np.flipud(x).copy()
    elif choice == 6:
        return np.fliplr(np.rot90(x, 1)).copy()
    else:
        return np.flipud(np.rot90(x, 1)).copy()


# ============================================================
# DATASET CLASS
# ============================================================

class SteganalysisDataset(Dataset):
    """PyTorch dataset for global steganalysis.

    Feature engineering pipeline (verified from literature):
      1. Load image as raw [0, 255] — NO ImageNet normalization
         (Tabares-Soto et al. 2021: normalization causes 12% accuracy drop)
      2. Center crop to target_size (NO resize — interpolation destroys stego)
      3. D4 augmentation (flips + 90-degree rotations only)
      4. SRM 30-filter preprocessing (fixed, non-trainable in dataloader)
      5. TLU truncation to [-3, 3]
      6. Return raw image [0,255] + SRM residuals [-3,3] as separate tensors
    """

    def __init__(
        self,
        csv_path: str,
        label_maps_path: str,
        target_size: int = 256,
        apply_srm: bool = True,
        augment: bool = False,
        tlu_threshold: float = 3.0,
    ):
        self.df = pd.read_csv(csv_path)
        with open(label_maps_path, "r") as f:
            self.label_maps = json.load(f)

        self.target_size = target_size
        self.apply_srm = apply_srm
        self.augment = augment
        self.tlu_threshold = tlu_threshold

        # Pre-build SRM filters as torch tensor (non-trainable in dataloader;
        # the model can have its own trainable copy)
        if self.apply_srm:
            self.srm_kernels = torch.from_numpy(build_srm_filters()).float()

    def __len__(self) -> int:
        return len(self.df)

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        """Center crop to target_size WITHOUT resize.

        Resize uses interpolation which alters pixel values and destroys
        the stego signal. Center crop preserves exact pixel values.
        If image is smaller than target, zero-pad instead.
        """
        h, w = img.shape[:2]
        th, tw = self.target_size, self.target_size

        if h >= th and w >= tw:
            # Center crop
            top = (h - th) // 2
            left = (w - tw) // 2
            return img[top:top + th, left:left + tw]
        else:
            # Pad if smaller (rare — most stego datasets are 512x512)
            out = np.zeros((th, tw), dtype=img.dtype)
            ph = min(h, th)
            pw = min(w, tw)
            top = (th - ph) // 2
            left = (tw - pw) // 2
            out[top:top + ph, left:left + pw] = img[:ph, :pw]
            return out

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Determine which image path to use
        if row["is_stego"] == 1 and row["stego_path"]:
            img_path = row["stego_path"]
        else:
            img_path = row["cover_path"]

        # Load image as raw pixels [0, 255]
        # NO normalization — stego signal is at sub-pixel level
        img = Image.open(img_path)
        if img.mode != "L":
            img = img.convert("L")  # grayscale

        x = np.array(img, dtype=np.float32)  # [0, 255] range

        # Center crop (NO resize — interpolation destroys stego)
        x = self._center_crop(x)

        # D4 augmentation (safe: preserves exact pixel values)
        if self.augment:
            x = d4_augment(x)

        # To tensor: (1, H, W), keep raw [0, 255]
        x_raw = torch.from_numpy(x.copy()).unsqueeze(0)

        # Apply SRM filtering on raw pixels → residuals
        if self.apply_srm:
            x_padded = F.pad(x_raw.unsqueeze(0), (2, 2, 2, 2), mode="reflect")
            srm_out = F.conv2d(x_padded, self.srm_kernels)
            srm_out = srm_out.squeeze(0)  # (30, H, W)

            # TLU: truncate residuals to [-T, T]
            srm_out = torch.clamp(srm_out, -self.tlu_threshold, self.tlu_threshold)

            # Output: raw pixel channel + 30 SRM residual channels = 31 channels
            # Raw channel: [0, 255], SRM channels: [-3, 3]
            x_out = torch.cat([x_raw, srm_out], dim=0)
        else:
            x_out = x_raw

        # Labels
        labels = {
            "binary": torch.tensor(int(row["is_stego"]), dtype=torch.long),
            "algorithm_class": torch.tensor(
                self.label_maps["algorithm_class"].get(row["algorithm_class"], 0),
                dtype=torch.long,
            ),
            "algorithm": torch.tensor(
                self.label_maps["algorithm"].get(row["algorithm"], 0),
                dtype=torch.long,
            ),
            "payload_rate": torch.tensor(
                float(row["payload_rate_bpp"]), dtype=torch.float32
            ),
        }

        return {"image": x_out, "labels": labels, "path": img_path}


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloaders(
    splits_dir: str,
    batch_size: int = 32,
    target_size: int = 256,
    apply_srm: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv, label_maps.json
        batch_size: Batch size
        target_size: Image crop size (square). Uses center crop, NOT resize.
        apply_srm: Apply SRM in DataLoader. Default False because the model's
            ForensicPreprocessor does its own SRM+BayarConv. Set True only
            if using a model that expects pre-computed SRM channels.
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer

    Returns:
        (train_loader, val_loader, test_loader)
    """
    label_maps_path = os.path.join(splits_dir, "label_maps.json")

    train_ds = SteganalysisDataset(
        csv_path=os.path.join(splits_dir, "train.csv"),
        label_maps_path=label_maps_path,
        target_size=target_size,
        apply_srm=apply_srm,
        augment=True,  # D4 group augmentation during training
    )

    val_ds = SteganalysisDataset(
        csv_path=os.path.join(splits_dir, "val.csv"),
        label_maps_path=label_maps_path,
        target_size=target_size,
        apply_srm=apply_srm,
        augment=False,
    )

    test_ds = SteganalysisDataset(
        csv_path=os.path.join(splits_dir, "test.csv"),
        label_maps_path=label_maps_path,
        target_size=target_size,
        apply_srm=apply_srm,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    channels = "31 (1 raw [0,255] + 30 SRM [-3,3])" if apply_srm else "1 (raw [0,255])"
    print(f"Dataloaders created:")
    print(f"  Train: {len(train_ds)} images, {len(train_loader)} batches (D4 augment)")
    print(f"  Val:   {len(val_ds)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds)} images, {len(test_loader)} batches")
    print(f"  Channels: {channels}")
    print(f"  Size: {target_size}x{target_size} (center crop, NO resize)")
    print(f"  Normalization: raw pixels [0,255] — NO ImageNet normalization")

    return train_loader, val_loader, test_loader
