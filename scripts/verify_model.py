"""
Verify UniSteg model architecture — shape checks, param counts, forward pass.

Usage:
    python scripts/verify_model.py                # full model (needs DINOv2 download)
    python scripts/verify_model.py --lite          # lite model (no DINOv2)
    python scripts/verify_model.py --lite --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.unisteg import UniSteg, UniStegLite
from src.models.preprocessing import ForensicPreprocessor
from src.models.forensic_stream import ForensicStream, SpatialBranch, FrequencyBranch, StatisticalBranch
from src.models.moe import SoftMoE
from src.models.fusion import GatedFusion
from src.training.losses import UniStegLoss


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def verify_preprocessing(device):
    print("=== ForensicPreprocessor ===")
    pp = ForensicPreprocessor().to(device)
    x = torch.randn(2, 1, 256, 256, device=device) * 127 + 128  # simulate [0,255]
    out = pp(x)
    print(f"  Input:  {x.shape} (raw pixels)")
    expected_ch = pp.out_channels
    print(f"  Output: {out.shape} (expected: [2, {expected_ch}, 256, 256])")
    assert out.shape == (2, expected_ch, 256, 256), f"Shape mismatch: {out.shape}"
    t, tr, fr = count_params(pp)
    print(f"  Params: {t:,} total, {tr:,} trainable, {fr:,} frozen")
    print(f"  PASS")


def verify_forensic_stream(device):
    print("\n=== ForensicStream ===")
    in_ch = ForensicPreprocessor().out_channels
    fs = ForensicStream(in_channels=in_ch).to(device)
    x = torch.randn(2, in_ch, 256, 256, device=device)
    out = fs(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} (expected: [2, 512])")
    assert out.shape == (2, 512), f"Shape mismatch: {out.shape}"

    # Check sub-branches
    sb = SpatialBranch(in_ch).to(device)
    fb = FrequencyBranch(in_ch).to(device)
    stb = StatisticalBranch(in_ch).to(device)
    print(f"  Spatial branch:     {sb(x).shape} -> {sb.out_dim}d")
    print(f"  Frequency branch:   {fb(x).shape} -> {fb.out_dim}d")
    print(f"  Statistical branch: {stb(x).shape} -> {stb.out_dim}d")

    t, tr, fr = count_params(fs)
    print(f"  Params: {t:,} total, {tr:,} trainable")
    print(f"  PASS")


def verify_fusion(device):
    print("\n=== GatedFusion ===")
    fuse = GatedFusion(forensic_dim=512, context_dim=256, fused_dim=512).to(device)
    f_feat = torch.randn(2, 512, device=device)
    c_feat = torch.randn(2, 256, device=device)
    out = fuse(f_feat, c_feat)
    print(f"  Forensic: {f_feat.shape}, Context: {c_feat.shape}")
    print(f"  Output:   {out.shape} (expected: [2, 512])")
    assert out.shape == (2, 512), f"Shape mismatch: {out.shape}"
    t, tr, _ = count_params(fuse)
    print(f"  Params: {t:,} trainable")
    print(f"  PASS")


def verify_moe(device):
    print("\n=== SoftMoE ===")
    moe = SoftMoE(dim=512, num_experts=5).to(device)
    x = torch.randn(2, 512, device=device)
    out = moe(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} (expected: [2, 512])")
    assert out.shape == (2, 512), f"Shape mismatch: {out.shape}"

    routing = moe.get_routing_weights(x)
    print(f"  Routing weights: {routing.shape} (expected: [2, 5])")
    print(f"  Routing example: {routing[0].cpu().tolist()}")
    assert routing.shape == (2, 5)
    assert torch.allclose(routing.sum(dim=1), torch.ones(2, device=device), atol=1e-5)

    t, tr, _ = count_params(moe)
    print(f"  Params: {t:,} trainable")
    print(f"  PASS")


def verify_loss(device):
    print("\n=== UniStegLoss ===")
    loss_fn = UniStegLoss().to(device)
    preds = {
        "binary": torch.randn(4, 2, device=device),
        "algo_class": torch.randn(4, 7, device=device),
        "algorithm": torch.randn(4, 21, device=device),
        "payload_rate": torch.sigmoid(torch.randn(4, device=device)),
    }
    labels = {
        "binary": torch.randint(0, 2, (4,), device=device),
        "algorithm_class": torch.randint(0, 7, (4,), device=device),
        "algorithm": torch.randint(0, 21, (4,), device=device),
        "payload_rate": torch.rand(4, device=device) * 0.5,
    }
    total, loss_dict = loss_fn(preds, labels)
    print(f"  Total loss: {total.item():.4f}")
    for k, v in loss_dict.items():
        if k not in ("weights", "log_var"):
            print(f"    {k}: {v.item():.4f}")
    print(f"  Task weights: {loss_dict['weights'].cpu().tolist()}")
    print(f"  Log variance: {loss_dict['log_var'].cpu().tolist()}")
    print(f"  PASS")


def verify_unisteg_lite(device):
    print("\n=== UniStegLite (no DINOv2) ===")
    model = UniStegLite(
        num_algo_classes=7,
        num_algorithms=21,
    ).to(device)

    x = torch.randn(2, 1, 256, 256, device=device) * 127 + 128
    outputs = model(x)

    print(f"  Input:  {x.shape}")
    for k, v in outputs.items():
        print(f"  Output '{k}': {v.shape}")

    assert outputs["binary"].shape == (2, 2)
    assert outputs["algo_class"].shape == (2, 7)
    assert outputs["algorithm"].shape == (2, 21)
    assert outputs["payload_rate"].shape == (2,)

    t, tr, fr = count_params(model)
    print(f"  Total params:     {t:,}")
    print(f"  Trainable params: {tr:,}")
    print(f"  Frozen params:    {fr:,}")
    print(f"  PASS")


def verify_unisteg_full(device):
    print("\n=== UniSteg Full (with DINOv2) ===")
    print("  Loading DINOv2 from torch.hub (may take a moment)...")
    model = UniSteg(
        use_context_stream=True,
        dino_model="dinov2_vits14",
        lora_rank=8,
        num_experts=5,
        num_algo_classes=7,
        num_algorithms=21,
    ).to(device)

    # Input must be multiple of 14 for DINOv2 patches
    x = torch.randn(2, 1, 252, 252, device=device) * 127 + 128  # 252 = 14*18

    outputs = model(x)
    print(f"  Input:  {x.shape}")
    for k, v in outputs.items():
        print(f"  Output '{k}': {v.shape}")

    assert outputs["binary"].shape == (2, 2)
    assert outputs["algo_class"].shape == (2, 7)
    assert outputs["algorithm"].shape == (2, 21)
    assert outputs["payload_rate"].shape == (2,)

    # Expert routing analysis
    routing = model.get_expert_routing(x)
    print(f"  Expert routing: {routing.shape}")
    print(f"  Expert weights: {routing[0].cpu().tolist()}")

    t, tr, fr = count_params(model)
    print(f"  Total params:     {t:,}")
    print(f"  Trainable params: {tr:,}")
    print(f"  Frozen params:    {fr:,}")
    print(f"  PASS")


def main():
    parser = argparse.ArgumentParser(description="Verify UniSteg architecture")
    parser.add_argument("--lite", action="store_true", help="Skip DINOv2 (lite mode)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Always verify components
    verify_preprocessing(device)
    verify_forensic_stream(device)
    verify_fusion(device)
    verify_moe(device)
    verify_loss(device)
    verify_unisteg_lite(device)

    # Full model only if not --lite
    if not args.lite:
        verify_unisteg_full(device)
    else:
        print("\n  [SKIP] Full UniSteg (--lite flag). Run without --lite to test DINOv2.")

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
