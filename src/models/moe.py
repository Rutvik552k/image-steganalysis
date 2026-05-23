"""
UniSteg Soft Mixture of Experts

Fully differentiable MoE layer — every expert contributes with learned
soft weights. No hard routing, no dropped tokens, no routing instability.

Experts:
  0: Shared (universal stego features)
  1: Direct embedding specialist (LSB, F5 patterns)
  2: Adaptive STC specialist (S-UNIWARD, HILL patterns)
  3: JPEG-domain specialist (DCT artifacts)
  4: Neural/generative specialist (distribution anomalies)

Reference: Puigcerver et al., "From Sparse to Soft Mixtures of Experts,"
ICLR 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert: two-layer MLP with GELU activation."""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoftMoE(nn.Module):
    """Soft Mixture of Experts layer.

    Every input contributes to every expert with learned soft weights.
    No discrete routing — fully differentiable.

    Architecture:
      1. Router computes soft assignment weights over all experts
      2. Each expert processes a weighted combination of inputs
      3. Output is weighted combination of all expert outputs
    """

    def __init__(
        self,
        dim: int = 512,
        num_experts: int = 5,
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim

        # Router: linear projection → softmax over experts
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(dim, hidden_mult) for _ in range(num_experts)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, dim) input features

        Returns:
            (B, dim) expert-processed features
        """
        # Compute soft routing weights: (B, num_experts)
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Each expert processes the full input, weighted by routing
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, num_experts, dim)

        # Weighted combination: (B, dim)
        out = torch.einsum("be,bed->bd", routing_weights, expert_outputs)

        # Residual + norm
        return self.norm(out + x)

    def get_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return routing weights for analysis/visualization.

        Useful for understanding which expert handles which algorithm type.
        """
        with torch.no_grad():
            return F.softmax(self.router(x), dim=-1)
