"""
UniSteg Gated Fusion Module

Fuses forensic stream features with context stream features using
a learned gating mechanism. Context features modulate forensic features
via element-wise gating — the context stream controls HOW MUCH each
forensic feature dimension contributes to the final representation.

Previous version used cross-attention with single-token Q/K/V which
degenerates to identity (softmax over 1x1 = always 1.0). This gated
fusion actually performs content-adaptive modulation.

Reference: CNN-Transformer Gated Fusion (Scientific Reports, 2025)
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Gated fusion of forensic and context features.

    Context features produce per-dimension gates that modulate forensic
    features, enabling content-adaptive forensic analysis.

    Architecture:
      1. Project both streams to common dimension
      2. Context → gate values via sigmoid
      3. Gated forensic = forensic * gate
      4. Concatenate gated + raw + context → final projection
    """

    def __init__(
        self,
        forensic_dim: int = 512,
        context_dim: int = 256,
        fused_dim: int = 512,
    ):
        super().__init__()

        # Project context to forensic dimension for gating
        self.gate_proj = nn.Sequential(
            nn.Linear(context_dim, forensic_dim),
            nn.Sigmoid(),
        )

        # Project context to forensic dimension for additive modulation
        self.shift_proj = nn.Linear(context_dim, forensic_dim)

        # Combine: gated forensic (forensic_dim) + raw context (context_dim)
        self.fuse = nn.Sequential(
            nn.Linear(forensic_dim + context_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
        )
        self.out_dim = fused_dim

    def forward(
        self,
        forensic_feat: torch.Tensor,
        context_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            forensic_feat: (B, forensic_dim)
            context_feat: (B, context_dim)

        Returns:
            Fused feature: (B, fused_dim)
        """
        # Context-adaptive gating: gate ∈ [0, 1] per forensic dimension
        gate = self.gate_proj(context_feat)       # (B, forensic_dim)
        shift = self.shift_proj(context_feat)     # (B, forensic_dim)

        # Modulate forensic features: scale + shift (FiLM-style)
        modulated = forensic_feat * gate + shift  # (B, forensic_dim)

        # Fuse modulated forensic with raw context
        combined = torch.cat([modulated, context_feat], dim=1)
        return self.fuse(combined)
