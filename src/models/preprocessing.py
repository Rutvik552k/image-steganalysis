"""
UniSteg Preprocessing Module

Multi-branch forensic preprocessing:
  - Branch A: Fixed SRM 30 filters (non-trainable)
  - Branch B: Learnable SRM 30 filters (initialized from SRM, fine-tuned)
  - Branch C: BayarConv constrained filters (3 channels)
  - Branch D: Gabor filter bank (24 filters: 8 orientations x 3 scales)
  - Branch E: Local entropy maps (2 channels: 8x8 and 16x16 windows)
  - TLU truncation on residual branches

Input: raw grayscale image [0, 255], shape (B, 1, H, W)
Output: 92-channel feature tensor, shape (B, 92, H, W)
  63 residual + 24 Gabor + 3 entropy + 2 entropy maps = 92

References:
  - ESNet (IEEE TIFS 2024): dual-branch fixed+learnable SRM
  - CVTStego-Net (2024): bifurcated SRM + Gabor preprocessing
  - Bayar & Stamm (IEEE TIFS 2018): constrained prediction-error filters
  - Song et al. (2015): GFR — Gabor Filter Residual features
  - Entropy-driven DNN steganalysis (Signal Processing, 2024)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import BayarConv2d, TLU, build_srm_filters


# ============================================================
# Gabor Filter Bank
# ============================================================

def build_gabor_filters(
    num_orientations: int = 8,
    num_scales: int = 3,
    kernel_size: int = 9,
) -> np.ndarray:
    """Build 2D Gabor filter bank for texture/stego artifact detection.

    Gabor filters provide optimal joint localization in spatial and
    frequency domains. Multi-orientation captures directional embedding
    artifacts that DWT's 3 fixed orientations miss.

    g(x,y) = exp(-(x'^2 + gamma^2*y'^2)/(2*sigma^2)) * cos(2*pi*x'/lambda)

    where x' = x*cos(theta) + y*sin(theta),
          y' = -x*sin(theta) + y*cos(theta)

    Args:
        num_orientations: number of theta values (default 8 = 22.5 deg steps)
        num_scales: number of lambda/sigma scales
        kernel_size: spatial extent of each filter

    Returns:
        shape (num_orientations * num_scales, 1, kernel_size, kernel_size)
    """
    filters = []
    center = kernel_size // 2

    # Scale parameters: wavelengths from fine to coarse
    lambdas = [3.0, 5.0, 8.0][:num_scales]
    gamma = 0.5  # spatial aspect ratio (elongation)

    for lam in lambdas:
        sigma = 0.56 * lam  # bandwidth ~1 octave
        for k in range(num_orientations):
            theta = k * math.pi / num_orientations

            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            for i in range(kernel_size):
                for j in range(kernel_size):
                    x = j - center
                    y = i - center
                    xp = x * cos_t + y * sin_t
                    yp = -x * sin_t + y * cos_t
                    envelope = math.exp(-(xp**2 + gamma**2 * yp**2) / (2 * sigma**2))
                    kernel[i, j] = envelope * math.cos(2 * math.pi * xp / lam)

            # Zero-mean normalization (makes it a proper bandpass filter)
            kernel -= kernel.mean()
            # Unit energy
            norm = np.sqrt((kernel**2).sum())
            if norm > 1e-8:
                kernel /= norm

            filters.append(kernel)

    return np.array(filters, dtype=np.float32).reshape(-1, 1, kernel_size, kernel_size)


# ============================================================
# Local Entropy Feature Map
# ============================================================

class LocalEntropyMap(nn.Module):
    """Compute local Shannon entropy of residuals over sliding windows.

    Text payloads have lower entropy than random binary (~1-4.5 bits/char
    vs 8 bits/byte). This creates detectable local entropy variations
    in the stego residual domain.

    H_block = -sum(p_k * log2(p_k))

    Uses soft histogram via Gaussian kernel binning (differentiable).
    """

    def __init__(self, window_sizes: tuple[int, ...] = (8, 16), num_bins: int = 16):
        super().__init__()
        self.window_sizes = window_sizes
        self.num_bins = num_bins
        # Bin centers for soft histogram in [-3, 3] (TLU range)
        bin_centers = torch.linspace(-3.0, 3.0, num_bins)
        self.register_buffer("bin_centers", bin_centers)
        # Bandwidth for Gaussian kernel binning
        self.bandwidth = 6.0 / num_bins  # adaptive to bin spacing

    @torch.autocast(device_type="cuda", enabled=False)
    def _soft_entropy(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Compute local entropy using average-pooled soft histograms.

        Force float32 — exp()/log2() overflow in float16 under AMP.

        Args:
            x: (B, C, H, W) residuals
            window: pooling window size

        Returns:
            (B, 1, H, W) entropy map (same spatial size via padding)
        """
        x = x.float()
        B, C, H, W = x.shape
        # Average over channels to get single residual map
        x_avg = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Soft histogram: compute distance to each bin center
        # x_avg: (B, 1, H, W) -> (B, 1, H, W, 1)
        # bin_centers: (num_bins,) -> (1, 1, 1, 1, num_bins)
        x_exp = x_avg.unsqueeze(-1)
        bins = self.bin_centers.reshape(1, 1, 1, 1, -1)
        # Gaussian kernel assignment
        weights = torch.exp(-0.5 * ((x_exp - bins) / self.bandwidth) ** 2)
        # Normalize to probability per pixel
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Average pool the soft histogram over local windows
        # Reshape: (B, num_bins, H, W)
        weights = weights.squeeze(1).permute(0, 3, 1, 2)  # (B, num_bins, H, W)

        # Pad so output matches input spatial size after avg_pool
        # For kernel K, stride 1: need total pad = K-1 -> each side = (K-1)//2
        pad_l = (window - 1) // 2
        pad_r = window - 1 - pad_l
        weights_padded = F.pad(weights, (pad_l, pad_r, pad_l, pad_r), mode="reflect")
        p = F.avg_pool2d(weights_padded, kernel_size=window, stride=1)

        # Shannon entropy: H = -sum(p * log2(p))
        p_clamped = p.clamp(min=1e-8)
        entropy = -(p_clamped * torch.log2(p_clamped)).sum(dim=1, keepdim=True)

        return entropy  # (B, 1, H, W)

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale entropy maps.

        Args:
            residuals: (B, C, H, W) from SRM/BayarConv branches

        Returns:
            (B, len(window_sizes), H, W) entropy feature maps
        """
        maps = []
        for ws in self.window_sizes:
            maps.append(self._soft_entropy(residuals, ws))
        return torch.cat(maps, dim=1)


# ============================================================
# ForensicPreprocessor (enhanced)
# ============================================================

class ForensicPreprocessor(nn.Module):
    """Multi-branch forensic preprocessing for universal steganalysis.

    Produces 92-channel output:
      - 30 channels: fixed SRM filters
      - 30 channels: learnable SRM filters
      -  3 channels: BayarConv constrained filters
      - 24 channels: Gabor filter bank (8 orientations x 3 scales)
      -  3 channels: learnable Gabor-initialized filters
      -  2 channels: local entropy maps (8x8 and 16x16 windows)
    All residual branches passed through TLU truncation.

    Total: 30 + 30 + 3 + 24 + 3 + 2 = 92 channels
    """

    def __init__(self, tlu_threshold: float = 3.0):
        super().__init__()

        srm_np = build_srm_filters()  # (30, 1, 5, 5)
        srm_tensor = torch.from_numpy(srm_np)

        # Fixed SRM filters (non-trainable)
        self.register_buffer("srm_fixed", srm_tensor.clone())

        # Learnable SRM filters (initialized from SRM, trainable)
        self.srm_learnable = nn.Parameter(srm_tensor.clone())

        # BayarConv: 3 constrained prediction-error filters
        self.bayar_conv = BayarConv2d(
            in_channels=1, out_channels=3, kernel_size=5
        )

        # Gabor filter bank: 24 fixed filters (8 orientations x 3 scales)
        gabor_np = build_gabor_filters(
            num_orientations=8, num_scales=3, kernel_size=9,
        )  # (24, 1, 9, 9)
        self.register_buffer("gabor_fixed", torch.from_numpy(gabor_np))
        self.gabor_pad = 9 // 2  # = 4

        # 3 additional learnable Gabor-initialized filters
        self.gabor_learnable = nn.Parameter(
            torch.from_numpy(gabor_np[:3].copy())
        )

        # Local entropy maps (differentiable)
        self.entropy_map = LocalEntropyMap(window_sizes=(8, 16), num_bins=16)

        # TLU truncation
        self.tlu = TLU(threshold=tlu_threshold)

        # Batch normalization per branch
        self.bn_fixed = nn.BatchNorm2d(30)
        self.bn_learnable = nn.BatchNorm2d(30)
        self.bn_bayar = nn.BatchNorm2d(3)
        self.bn_gabor = nn.BatchNorm2d(24)
        self.bn_gabor_learn = nn.BatchNorm2d(3)
        self.bn_entropy = nn.BatchNorm2d(2)

        # Total output channels
        self.out_channels = 92

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw grayscale image, shape (B, 1, H, W), values in [0, 255]

        Returns:
            Feature tensor, shape (B, 92, H, W)
        """
        # === Residual branches (SRM + BayarConv) ===
        x_srm_padded = F.pad(x, (2, 2, 2, 2), mode="reflect")

        # Fixed SRM
        fixed_out = F.conv2d(x_srm_padded, self.srm_fixed)
        fixed_out = self.tlu(fixed_out)
        fixed_out = self.bn_fixed(fixed_out)

        # Learnable SRM
        learn_out = F.conv2d(x_srm_padded, self.srm_learnable)
        learn_out = self.tlu(learn_out)
        learn_out = self.bn_learnable(learn_out)

        # BayarConv
        bayar_out = self.bayar_conv(x)
        bayar_out = self.tlu(bayar_out)
        bayar_out = self.bn_bayar(bayar_out)

        # === Gabor branches ===
        x_gabor_padded = F.pad(x, (self.gabor_pad,) * 4, mode="reflect")

        # Fixed Gabor (24 channels)
        gabor_out = F.conv2d(x_gabor_padded, self.gabor_fixed)
        gabor_out = self.tlu(gabor_out)
        gabor_out = self.bn_gabor(gabor_out)

        # Learnable Gabor (3 channels)
        gabor_learn_out = F.conv2d(x_gabor_padded, self.gabor_learnable)
        gabor_learn_out = self.tlu(gabor_learn_out)
        gabor_learn_out = self.bn_gabor_learn(gabor_learn_out)

        # === Entropy branch ===
        # Compute on the SRM residuals (before BN, after TLU)
        srm_residuals = self.tlu(F.conv2d(x_srm_padded, self.srm_fixed))
        entropy_out = self.entropy_map(srm_residuals)  # (B, 2, H, W)
        entropy_out = self.bn_entropy(entropy_out)

        # Concatenate all: (B, 92, H, W)
        return torch.cat([
            fixed_out, learn_out, bayar_out,
            gabor_out, gabor_learn_out,
            entropy_out,
        ], dim=1)
