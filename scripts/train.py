"""
UniSteg Training Entry Point

Usage:
    python scripts/train.py                                    # defaults
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --resume checkpoints/last.pt
    python scripts/train.py --lite --epochs 50 --lr 5e-4       # CLI overrides
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.models.unisteg import UniSteg, UniStegLite
from src.training.train_loop import train


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str | None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train UniSteg")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lite", action="store_true", help="Use UniStegLite (no DINOv2)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    # CLI overrides
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.resume is not None:
        cfg["resume_path"] = args.resume
    if args.lite:
        model_cfg["variant"] = "lite"

    # Seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"Seed: {seed}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    variant = model_cfg.get("variant", "lite")
    if variant == "full":
        model = UniSteg(
            use_context_stream=True,
            dino_model=model_cfg.get("dino_model", "dinov2_vits14"),
            lora_rank=model_cfg.get("lora_rank", 8),
            num_experts=model_cfg.get("num_experts", 5),
            num_algo_classes=model_cfg.get("num_algo_classes", 7),
            num_algorithms=model_cfg.get("num_algorithms", 21),
            tlu_threshold=model_cfg.get("tlu_threshold", 3.0),
        )
        print(f"Model: UniSteg (full, with DINOv2)")
    else:
        model = UniStegLite(
            num_experts=model_cfg.get("num_experts", 5),
            num_algo_classes=model_cfg.get("num_algo_classes", 7),
            num_algorithms=model_cfg.get("num_algorithms", 21),
        )
        print(f"Model: UniStegLite (no DINOv2)")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable")

    # Data
    splits_dir = data_cfg.get("splits_dir", "data/splits")
    train_loader, val_loader, _ = create_dataloaders(
        splits_dir=splits_dir,
        batch_size=data_cfg.get("batch_size", 32),
        target_size=data_cfg.get("target_size", 256),
        apply_srm=data_cfg.get("apply_srm", False),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # TensorBoard
    writer = None
    if cfg.get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = cfg.get("log_dir", "runs")
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard: {log_dir}/")
        except ImportError:
            print("TensorBoard not installed, skipping")

    # Train
    best_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        output_dir=cfg.get("output_dir", "checkpoints"),
        resume_path=cfg.get("resume_path"),
        writer=writer,
    )

    if writer:
        writer.close()

    return best_acc


if __name__ == "__main__":
    main()
