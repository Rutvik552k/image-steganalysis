"""
UniSteg Checkpoint Save/Load Utilities

Features:
  - Atomic saves (write to tmp, rename) — no corruption on crash
  - Rich metadata: timestamp, config hash, git hash, epoch, step
  - Checkpoint rotation: keep last N + best
  - Safe loading with weights_only=True
"""

import glob
import os
import shutil
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from src.training.losses import UniStegLoss


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: UniStegLoss,
    epoch: int,
    metrics: dict,
    path: str,
    global_step: int = 0,
    best_val_acc: float = 0.0,
    config: dict | None = None,
):
    """Save training state to disk atomically.

    Writes to a temp file first, then renames. Prevents corruption
    if training is interrupted mid-save.
    """
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "metrics": metrics,
        "best_val_acc": best_val_acc,
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
    }
    if config:
        state["config"] = config

    # Atomic write: save to tmp, then rename
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, tmp_path)
    shutil.move(tmp_path, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    criterion: Optional[UniStegLoss] = None,
    map_location: Optional[str] = None,
) -> dict:
    """Load training state from disk.

    Returns full checkpoint dict with keys:
      epoch, global_step, metrics, best_val_acc, timestamp, git_hash
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if criterion and "criterion_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
    return checkpoint


def rotate_checkpoints(output_dir: str, keep_last: int = 3):
    """Keep only the last N epoch checkpoints + best.pt + last.pt.

    Removes older epoch_*.pt files to save disk space.
    """
    pattern = os.path.join(output_dir, "epoch_*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    # Also include step checkpoints
    step_pattern = os.path.join(output_dir, "step_*.pt")
    step_files = sorted(glob.glob(step_pattern), key=os.path.getmtime)
    files.extend(step_files)
    files.sort(key=os.path.getmtime)

    if len(files) > keep_last:
        for f in files[:-keep_last]:
            os.remove(f)
