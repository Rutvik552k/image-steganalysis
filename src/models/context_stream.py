"""
UniSteg Context Stream (Stream 2)

Frozen DINOv2 backbone providing content-aware guidance:
  - Frozen DINOv2-S/14 (21M params, non-trainable)
  - LoRA adapters for forensic adaptation (~0.5M trainable)
  - Multi-scale patch tokens from last 4 layers

The context stream does NOT directly detect stego noise. It provides
semantic guidance — "where is the texture?" "what is the compression
quality?" — so the forensic stream knows WHERE to focus.

DINOv2 API (verified from github.com/facebookresearch/dinov2):
  model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
  features = model.forward_features(x)
  # Returns dict with 'x_norm_clstoken', 'x_norm_patchtokens'
  # get_intermediate_layers(x, n=4) returns features from last 4 layers

Input: raw grayscale image, shape (B, 1, H, W), values [0, 255]
Output: context feature vector, shape (B, context_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning.

    Adds low-rank trainable matrices to frozen linear layers.
    W_adapted = W_frozen + alpha * (B @ A)

    Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
    Models," ICLR 2022.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the LoRA delta to be added to frozen output."""
        return (x @ self.lora_A @ self.lora_B) * self.scale


class ContextStream(nn.Module):
    """DINOv2-based context stream with LoRA adaptation.

    Provides content-aware features that guide forensic analysis.
    DINOv2 backbone is FROZEN — only LoRA adapters are trained.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        lora_rank: int = 8,
        num_layers_to_use: int = 4,
        context_dim: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_layers_to_use = num_layers_to_use

        # Load frozen DINOv2
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get embedding dimension from model
        self.embed_dim = self.backbone.embed_dim  # 384 for vits14

        # Add LoRA adapters to each transformer block's QKV projection
        self.lora_layers = nn.ModuleList()
        for block in self.backbone.blocks:
            lora = LoRALayer(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                rank=lora_rank,
            )
            self.lora_layers.append(lora)

        # Input adaptation: grayscale 1ch → RGB 3ch for DINOv2
        # DINOv2 expects 3-channel input normalized with ImageNet stats
        # We replicate grayscale to 3 channels and normalize
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0,
        )

        # Project multi-scale features to context_dim
        # Last num_layers_to_use layers concatenated: embed_dim * num_layers
        proj_in = self.embed_dim * num_layers_to_use
        self.projector = nn.Sequential(
            nn.Linear(proj_in, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
        )
        self.out_dim = context_dim

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw grayscale [0,255] to DINOv2-expected RGB input.

        DINOv2 expects: 3-channel, ImageNet-normalized.
        Input size must be divisible by 14 (patch size).
        """
        B, C, H, W = x.shape

        # Replicate grayscale to 3 channels
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # Pad to next multiple of 14 (DINOv2 patch size requirement).
        # Pad instead of crop to avoid spatial misalignment with forensic stream.
        # Reflect padding preserves edge statistics.
        h_new = ((H + 13) // 14) * 14
        w_new = ((W + 13) // 14) * 14
        if h_new != H or w_new != W:
            pad_h = h_new - H
            pad_w = w_new - W
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Normalize with ImageNet stats (DINOv2 requirement)
        # This is OK here because context stream captures semantics, not noise
        x = (x - self.pixel_mean) / self.pixel_std

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw grayscale image, shape (B, 1, H, W), values [0, 255]
        Returns:
            Context feature vector, shape (B, context_dim)
        """
        x_rgb = self._prepare_input(x)

        # Extract features from last N layers.
        # NO torch.no_grad() here — frozen params (requires_grad=False) won't
        # accumulate gradients, but activations must carry gradient so LoRA
        # adapters receive proper input-dependent gradients during backprop.
        intermediate = self.backbone.get_intermediate_layers(
            x_rgb,
            n=self.num_layers_to_use,
            return_class_token=True,
        )

        # Apply LoRA to class tokens from each layer
        # intermediate is tuple of (patch_tokens, cls_token) per layer
        num_blocks = len(self.backbone.blocks)
        cls_features = []
        for i, (patch_tok, cls_tok) in enumerate(intermediate):
            layer_idx = num_blocks - self.num_layers_to_use + i
            # Add LoRA adaptation to cls token
            lora_delta = self.lora_layers[layer_idx](cls_tok)
            adapted_cls = cls_tok + lora_delta
            cls_features.append(adapted_cls)

        # Concatenate multi-layer features: (B, embed_dim * num_layers)
        multi_scale = torch.cat(cls_features, dim=1)

        # Project to context_dim
        return self.projector(multi_scale)
