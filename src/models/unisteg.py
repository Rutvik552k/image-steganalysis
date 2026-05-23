"""
UniSteg: Universal Steganalysis Network

First single-model steganalyzer covering ALL algorithm classes:
  - Spatial (LSB, HUGO, WOW, S-UNIWARD, HILL, MiPOD)
  - JPEG (F5, nsF5, J-UNIWARD, JMiPOD, UERD)
  - Neural encoder-decoder (SteganoGAN)
  - Diffusion model (DiffStega)
  - Real-world tools (Steghide, OpenStego)

Architecture:
  1. ForensicPreprocessor: dual SRM + BayarConv + Gabor bank + entropy maps -> 92ch
  2. ForensicStream: spatial CNN + DWT frequency + statistical branches → 512d
  3. ContextStream: frozen DINOv2 + LoRA → 256d semantic guidance
  4. CrossAttentionFusion: context guides forensic → 512d fused features
  5. SoftMoE: 5 specialized experts with soft routing → 512d
  6. Multi-task heads: binary detection + algo class + algo ID + payload rate

Total: ~33M params, ~11M trainable. Single-GPU friendly.

Usage:
    model = UniSteg()
    outputs = model(images)  # images: (B, 1, H, W), raw [0,255]
    # outputs['binary']: (B, 2) cover/stego logits
    # outputs['algo_class']: (B, num_classes) algorithm class logits
    # outputs['algorithm']: (B, num_algos) specific algorithm logits
    # outputs['payload_rate']: (B,) estimated rate in [0,1]; multiply by 0.5 for bpp
"""

import torch
import torch.nn as nn

from .context_stream import ContextStream
from .forensic_stream import ForensicStream
from .fusion import GatedFusion
from .moe import SoftMoE
from .preprocessing import ForensicPreprocessor


class MultiTaskHead(nn.Module):
    """Multi-task output heads with GradNorm-compatible structure."""

    def __init__(
        self,
        in_dim: int = 512,
        num_algo_classes: int = 7,
        num_algorithms: int = 21,
    ):
        super().__init__()

        # Head 1: Binary detection (cover vs stego)
        self.binary_head = nn.Linear(in_dim, 2)

        # Head 2: Algorithm class (6 stego classes + 1 cover)
        self.algo_class_head = nn.Linear(in_dim, num_algo_classes)

        # Head 3: Specific algorithm (20 stego algos + 1 cover)
        self.algorithm_head = nn.Linear(in_dim, num_algorithms)

        # Head 4: Payload rate regression
        self.payload_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # output in [0, 1], scale to [0, 0.5] at loss time
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "binary": self.binary_head(x),
            "algo_class": self.algo_class_head(x),
            "algorithm": self.algorithm_head(x),
            "payload_rate": self.payload_head(x).squeeze(-1),
        }


class UniSteg(nn.Module):
    """Universal Steganalysis Network.

    Args:
        use_context_stream: If True, include DINOv2 context stream.
            Set False for lightweight/ablation experiments.
        dino_model: DINOv2 model name ('dinov2_vits14', 'dinov2_vitb14')
        lora_rank: LoRA rank for DINOv2 adaptation
        num_experts: Number of Soft MoE experts
        num_algo_classes: Number of algorithm class labels
        num_algorithms: Number of specific algorithm labels
        tlu_threshold: TLU truncation threshold for SRM residuals
    """

    def __init__(
        self,
        use_context_stream: bool = True,
        dino_model: str = "dinov2_vits14",
        lora_rank: int = 8,
        num_experts: int = 5,
        num_algo_classes: int = 7,
        num_algorithms: int = 21,
        tlu_threshold: float = 3.0,
    ):
        super().__init__()
        self.use_context_stream = use_context_stream

        # 1. Preprocessing: raw pixels -> 92-channel forensic features
        #    (30 fixed SRM + 30 learnable SRM + 3 BayarConv +
        #     24 Gabor + 3 learnable Gabor + 2 entropy maps)
        self.preprocessor = ForensicPreprocessor(tlu_threshold=tlu_threshold)

        # 2. Forensic stream: features -> 512d
        self.forensic_stream = ForensicStream(
            in_channels=self.preprocessor.out_channels,
            spatial_base=64,
            freq_dim=256,
            stat_dim=256,
        )

        # 3. Context stream (optional): raw pixels → 256d semantic guidance
        if use_context_stream:
            self.context_stream = ContextStream(
                model_name=dino_model,
                lora_rank=lora_rank,
                num_layers_to_use=4,
                context_dim=256,
            )

            # 4. Gated fusion: context modulates forensic features (FiLM-style)
            self.fusion = GatedFusion(
                forensic_dim=self.forensic_stream.out_dim,
                context_dim=self.context_stream.out_dim,
                fused_dim=512,
            )
            moe_in_dim = self.fusion.out_dim
        else:
            moe_in_dim = self.forensic_stream.out_dim

        # 5. Soft MoE: specialized expert routing
        self.moe = SoftMoE(
            dim=moe_in_dim,
            num_experts=num_experts,
        )

        # 6. Multi-task output heads
        self.heads = MultiTaskHead(
            in_dim=moe_in_dim,
            num_algo_classes=num_algo_classes,
            num_algorithms=num_algorithms,
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: raw grayscale image, shape (B, 1, H, W), values [0, 255]
               If input is (B, 31, H, W) from dataloader (1 raw + 30 SRM),
               we split and use raw channel for context, all for forensic.

        Returns:
            dict with keys: 'binary', 'algo_class', 'algorithm', 'payload_rate'
        """
        # Handle dataloader output: (B, 31, H, W) = 1 raw + 30 SRM
        if x.shape[1] > 1:
            x_raw = x[:, :1, :, :]  # first channel is raw [0, 255]
        else:
            x_raw = x

        # 1. Forensic preprocessing on raw pixels
        residuals = self.preprocessor(x_raw)  # (B, 63, H, W)

        # 2. Forensic stream
        forensic_feat = self.forensic_stream(residuals)  # (B, 512)

        # 3. Context stream + fusion (if enabled)
        if self.use_context_stream:
            context_feat = self.context_stream(x_raw)  # (B, 256)
            fused_feat = self.fusion(forensic_feat, context_feat)  # (B, 512)
        else:
            fused_feat = forensic_feat

        # 4. Soft MoE expert routing
        expert_feat = self.moe(fused_feat)  # (B, 512)

        # 5. Multi-task prediction
        return self.heads(expert_feat)

    def get_expert_routing(self, x: torch.Tensor) -> torch.Tensor:
        """Get MoE routing weights for analysis. Shape (B, num_experts)."""
        with torch.no_grad():
            if x.shape[1] > 1:
                x_raw = x[:, :1, :, :]
            else:
                x_raw = x
            residuals = self.preprocessor(x_raw)
            forensic_feat = self.forensic_stream(residuals)
            if self.use_context_stream:
                context_feat = self.context_stream(x_raw)
                fused_feat = self.fusion(forensic_feat, context_feat)
            else:
                fused_feat = forensic_feat
            return self.moe.get_routing_weights(fused_feat)


class UniStegLite(nn.Module):
    """Lightweight UniSteg without DINOv2 context stream.

    For quick experiments, ablation studies, or resource-constrained settings.
    Uses only the forensic stream + MoE + heads.
    ~11M params, all trainable.
    """

    def __init__(
        self,
        num_experts: int = 5,
        num_algo_classes: int = 7,
        num_algorithms: int = 21,
    ):
        super().__init__()
        self.model = UniSteg(
            use_context_stream=False,
            num_experts=num_experts,
            num_algo_classes=num_algo_classes,
            num_algorithms=num_algorithms,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)
