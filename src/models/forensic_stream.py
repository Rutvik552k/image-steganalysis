"""
UniSteg Forensic Stream (Stream 1)

Multi-branch noise analysis backbone:
  - Branch A: Spatial CNN (SRNet-style residual blocks on noise residuals)
  - Branch B: Frequency analysis (DWT subbands)
  - Branch C: Statistical pooling (global covariance + histogram moments)

Input: preprocessed residuals from ForensicPreprocessor, shape (B, 63, H, W)
Output: forensic feature vector, shape (B, forensic_dim)

References:
  - SRNet (IEEE TIFS 2018): residual blocks without pooling in early layers
  - WaReCo (SocialSec 2025): DWT subbands for steganalysis
  - US-CovNet (JCST 2022): global covariance pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """SRNet-style residual block with no downsampling in early layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DownBlock(nn.Module):
    """Residual block with channel expansion and spatial downsampling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pool(self.shortcut(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        return F.relu(out + residual)


class SpatialBranch(nn.Module):
    """Branch A: CNN backbone on noise residuals.

    Architecture inspired by SRNet: residual blocks without pooling
    in early layers (preserves weak stego signal), then gradually
    downsample with channel expansion.
    """

    def __init__(self, in_channels: int = 63, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Early layers: no pooling (preserve stego signal)
        self.res_blocks = nn.Sequential(
            ResidualBlock(c),
            ResidualBlock(c),
            ResidualBlock(c),
        )

        # Downsample stages
        self.down1 = DownBlock(c, c * 2)       # H/2
        self.down2 = DownBlock(c * 2, c * 4)   # H/4
        self.down3 = DownBlock(c * 4, c * 8)   # H/8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, out_dim) feature vector."""
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.gap(x).flatten(1)
        return x


class FrequencyBranch(nn.Module):
    """Branch B: DWT-based frequency analysis with cross-subband correlation.

    Uses Haar wavelet with 2-level decomposition. Processes detail subbands
    (LH, HL, HH) and their pairwise correlations.

    Cross-subband correlation rationale: natural images have predictable
    inter-subband dependencies (e.g., edges produce correlated LH/HL
    responses). Steganographic embedding breaks these correlations because
    modifications are driven by cost functions, not image content.

    Three cross-subband correlation features per channel:
      - corr(LH, HL): horizontal-vertical edge correlation
      - corr(LH, HH): horizontal-diagonal correlation
      - corr(HL, HH): vertical-diagonal correlation

    Implements DWT via fixed convolution kernels (no pywt dependency).
    """

    def __init__(self, in_channels: int = 92, feat_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels

        # Haar wavelet 2D filters
        ll = torch.tensor([[ 1,  1], [ 1,  1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[-1, -1], [ 1,  1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[-1,  1], [-1,  1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[ 1, -1], [-1,  1]], dtype=torch.float32) / 2.0

        band_filters = torch.stack([ll, lh, hl, hh])  # (4, 2, 2)
        per_channel = band_filters.unsqueeze(0).repeat(in_channels, 1, 1, 1)
        per_channel = per_channel.reshape(4 * in_channels, 1, 2, 2)
        self.register_buffer("dwt_filters", per_channel)

        # Detail subbands CNN: 3 * C channels
        detail_ch = 3 * in_channels

        self.detail_conv = nn.Sequential(
            nn.Conv2d(detail_ch, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Cross-subband correlation MLP
        # 3 correlation pairs, reduced to 16 channels for efficiency
        self.corr_reduce = nn.Conv2d(in_channels, 16, 1, bias=False)
        corr_feat_dim = 3 * 16  # 3 pairs x 16 reduced channels
        self.corr_fc = nn.Sequential(
            nn.Linear(corr_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        self.out_dim = feat_dim + 64  # detail CNN + correlation features

    def _haar_dwt(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Single-level Haar DWT via grouped convolution."""
        B, C, H, W = x.shape
        out = F.conv2d(x, self.dwt_filters, stride=2, groups=C)
        out = out.reshape(B, C, 4, H // 2, W // 2)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def _cross_subband_correlation(
        self, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise Pearson correlation between detail subbands.

        For each channel, compute correlation coefficient between subband pairs.
        Embedding modifies pixel values based on cost functions (not content),
        which disrupts the natural correlation structure.

        Returns: (B, 3*C_reduced) correlation features
        """
        # Reduce channels: (B, C, H, W) -> (B, 16, H, W)
        lh_r = self.corr_reduce(lh)
        hl_r = self.corr_reduce(hl)
        hh_r = self.corr_reduce(hh)

        B, C, H, W = lh_r.shape
        N = H * W

        def _pearson(a, b):
            """Batched per-channel Pearson correlation."""
            a_flat = a.reshape(B, C, N)
            b_flat = b.reshape(B, C, N)
            a_mean = a_flat.mean(dim=2, keepdim=True)
            b_mean = b_flat.mean(dim=2, keepdim=True)
            a_c = a_flat - a_mean
            b_c = b_flat - b_mean
            cov = (a_c * b_c).sum(dim=2)
            std_a = a_c.pow(2).sum(dim=2).sqrt().clamp(min=1e-8)
            std_b = b_c.pow(2).sum(dim=2).sqrt().clamp(min=1e-8)
            return cov / (std_a * std_b)  # (B, C)

        corr_lh_hl = _pearson(lh_r, hl_r)  # (B, 16)
        corr_lh_hh = _pearson(lh_r, hh_r)
        corr_hl_hh = _pearson(hl_r, hh_r)

        return torch.cat([corr_lh_hl, corr_lh_hh, corr_hl_hh], dim=1)  # (B, 48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, out_dim) feature vector from detail subbands + correlations."""
        ll, lh, hl, hh = self._haar_dwt(x)

        # Detail subband CNN features
        details = torch.cat([lh, hl, hh], dim=1)
        detail_feat = self.detail_conv(details).flatten(1)  # (B, feat_dim)

        # Cross-subband correlation features
        corr_raw = self._cross_subband_correlation(lh, hl, hh)  # (B, 48)
        corr_feat = self.corr_fc(corr_raw)  # (B, 64)

        return torch.cat([detail_feat, corr_feat], dim=1)  # (B, feat_dim + 64)


class StatisticalBranch(nn.Module):
    """Branch C: Global statistical features.

    Extracts algorithm-agnostic statistics from noise residuals:
      - Global covariance pooling (captures 2nd-order feature interactions)
      - Channel-wise moments (mean, variance, skewness, kurtosis)

    Reference: US-CovNet (JCST 2022) — covariance pooling for universal steganalysis.
    """

    def __init__(self, in_channels: int = 63, feat_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels

        # Reduce channels before covariance (full 63x63 cov matrix is too large)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Covariance matrix: 32x32 = 1024 upper-triangle elements
        # + 4 moments per reduced channel = 32*4 = 128
        cov_dim = 32 * (32 + 1) // 2  # upper triangle = 528
        moment_dim = 32 * 4  # mean, var, skew, kurt per channel

        self.fc = nn.Sequential(
            nn.Linear(cov_dim + moment_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.out_dim = feat_dim

    def _compute_moments(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-channel statistical moments.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, C*4) — mean, variance, skewness, kurtosis per channel
        """
        B, C = x.shape[:2]
        flat = x.reshape(B, C, -1)  # (B, C, H*W)

        mean = flat.mean(dim=2)
        var = flat.var(dim=2)
        std = var.sqrt().clamp(min=1e-6)
        centered = flat - mean.unsqueeze(2)
        skew = (centered ** 3).mean(dim=2) / (std ** 3)
        kurt = (centered ** 4).mean(dim=2) / (std ** 4) - 3.0  # excess kurtosis

        return torch.cat([mean, var, skew, kurt], dim=1)  # (B, C*4)

    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute upper-triangle of global covariance matrix.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, C*(C+1)//2) — upper triangle of covariance matrix
        """
        B, C = x.shape[:2]
        flat = x.reshape(B, C, -1)  # (B, C, N)
        mean = flat.mean(dim=2, keepdim=True)
        centered = flat - mean
        cov = torch.bmm(centered, centered.transpose(1, 2)) / (flat.shape[2] - 1)
        # Extract upper triangle
        idx = torch.triu_indices(C, C, device=x.device)
        return cov[:, idx[0], idx[1]]  # (B, C*(C+1)//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, out_dim) statistical feature vector."""
        x_reduced = self.reduce(x)
        moments = self._compute_moments(x_reduced)
        covariance = self._compute_covariance(x_reduced)
        combined = torch.cat([covariance, moments], dim=1)
        return self.fc(combined)


class ForensicStream(nn.Module):
    """Complete forensic analysis stream (Stream 1).

    Combines three branches:
      - Spatial CNN (SRNet-style) -> 512d
      - Frequency DWT + cross-subband correlation -> 256d + 64d = 320d
      - Statistical moments + covariance -> 256d
    Total before fusion: 1088d -> 512d

    Input: preprocessed features from ForensicPreprocessor (B, 92, H, W)
    Output: forensic feature vector (B, 512)
    """

    def __init__(
        self,
        in_channels: int = 92,
        spatial_base: int = 64,
        freq_dim: int = 256,
        stat_dim: int = 256,
    ):
        super().__init__()
        self.spatial = SpatialBranch(in_channels, spatial_base)
        self.frequency = FrequencyBranch(in_channels, freq_dim)
        self.statistical = StatisticalBranch(in_channels, stat_dim)

        total_dim = self.spatial.out_dim + self.frequency.out_dim + self.statistical.out_dim
        self.fuse = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: preprocessed features, shape (B, 92, H, W)
        Returns:
            Forensic feature vector, shape (B, 512)
        """
        f_spatial = self.spatial(x)
        f_freq = self.frequency(x)
        f_stat = self.statistical(x)
        combined = torch.cat([f_spatial, f_freq, f_stat], dim=1)
        return self.fuse(combined)
