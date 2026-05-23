"""
UniSteg Training Loop (hardened)

Features:
  - Mixed precision (AMP) with GradScaler
  - Gradient accumulation for effective large batches
  - BayarConv weight projection after each optimizer step
  - Cosine annealing with linear warmup + optional restarts
  - ReduceLROnPlateau fallback when cosine LR plateaus
  - Best-model checkpointing by binary accuracy
  - Periodic epoch checkpoints + intra-epoch step checkpoints
  - SIGINT handler: saves emergency checkpoint on Ctrl+C
  - Checkpoint rotation (keep last N)
  - Gradient norm monitoring (detect exploding/vanishing gradients)
  - Early stopping on validation loss plateau
  - TensorBoard logging

Usage:
    python scripts/train.py --config configs/training_config.yaml
"""

import math
import os
import signal
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import BayarConv2d
from src.training.checkpoint import save_checkpoint, load_checkpoint, rotate_checkpoints
from src.training.losses import UniStegLoss
from src.training.metrics import MetricsTracker


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with separate param groups: higher LR for heads, lower for backbone."""
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "heads" in name or "log_var" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * 3.0},
    ], weight_decay=weight_decay)


# ============================================================
# LR Scheduler: warmup + cosine + plateau fallback
# ============================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup + cosine annealing to min_lr."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-3)  # floor at 0.1% during warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale so LR decays to min_lr, not zero
        return max(cosine, min_lr / optimizer.defaults.get("lr", 1e-3))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PlateauReducer:
    """Reduce LR when validation metric plateaus. Supplements cosine schedule.

    If val loss doesn't improve for `patience` epochs, halve the LR
    multiplier applied on top of the cosine schedule.
    """

    def __init__(self, patience: int = 5, factor: float = 0.5, min_factor: float = 0.1):
        self.patience = patience
        self.factor = factor
        self.min_factor = min_factor
        self.best_loss = float("inf")
        self.wait = 0
        self.lr_multiplier = 1.0

    def step(self, val_loss: float) -> bool:
        """Returns True if LR was reduced."""
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.wait = 0
            return False

        self.wait += 1
        if self.wait >= self.patience:
            new_mult = max(self.lr_multiplier * self.factor, self.min_factor)
            if new_mult < self.lr_multiplier:
                self.lr_multiplier = new_mult
                self.wait = 0
                return True
        return False

    def apply(self, optimizer: torch.optim.Optimizer):
        """Apply current multiplier to all param groups."""
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("_base_lr", pg["lr"]) * self.lr_multiplier


# ============================================================
# BayarConv projection
# ============================================================

def project_bayar_weights(model: nn.Module):
    """Project BayarConv weights to feasible set after optimizer step."""
    for m in model.modules():
        if isinstance(m, BayarConv2d):
            m.project_weights()


# ============================================================
# Gradient health monitoring
# ============================================================

def compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ============================================================
# Training epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: UniStegLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    epoch: int = 0,
    global_step: int = 0,
    save_every_n_steps: int = 0,
    output_dir: str = "checkpoints",
    best_val_acc: float = 0.0,
    config: dict | None = None,
) -> tuple[dict, int, list[float]]:
    """Train for one epoch.

    Returns (metrics_dict, updated_global_step, grad_norms_list).
    """
    model.train()
    tracker = MetricsTracker()
    grad_norms = []

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

        with torch.autocast(device_type=device.type):
            predictions = model(images)
            loss, loss_dict = criterion(predictions, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            gn = compute_grad_norm(model)
            grad_norms.append(gn)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            project_bayar_weights(model)
            scheduler.step()
            global_step += 1

            # Intra-epoch step checkpoint
            if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, criterion, epoch,
                    {"step": global_step}, os.path.join(output_dir, f"step_{global_step}.pt"),
                    global_step=global_step, best_val_acc=best_val_acc, config=config,
                )

        tracker.update(predictions, labels, loss_dict)

    return tracker.compute(), global_step, grad_norms


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: UniStegLoss,
    device: torch.device,
) -> dict:
    """Validate. Returns metrics dict."""
    model.train(False)
    tracker = MetricsTracker()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

        with torch.autocast(device_type=device.type):
            predictions = model(images)
            _, loss_dict = criterion(predictions, labels)

        tracker.update(predictions, labels, loss_dict)

    return tracker.compute()


# ============================================================
# Formatting
# ============================================================

def format_metrics(metrics: dict, prefix: str = "") -> str:
    parts = []
    for k in ["binary_acc", "algo_class_acc", "algo_acc", "payload_rmse", "loss/total"]:
        if k in metrics:
            v = metrics[k]
            label = k.replace("loss/", "L_")
            parts.append(f"{label}={v:.4f}")
    return f"{prefix} " + " | ".join(parts)


# ============================================================
# Main training loop
# ============================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: str = "checkpoints",
    resume_path: str | None = None,
    writer=None,
):
    """Full training loop with hardened checkpointing.

    Checkpoint strategy:
      - best.pt:        saved whenever val binary_acc improves
      - last.pt:        saved every epoch (always resumable)
      - epoch_N.pt:     saved every `save_every` epochs
      - step_N.pt:      saved every `save_every_steps` steps (intra-epoch safety)
      - emergency.pt:   saved on Ctrl+C / SIGINT
      - Rotation:       keeps last 3 epoch/step checkpoints

    LR schedule:
      - Linear warmup (0 -> lr) over warmup_epochs
      - Cosine annealing (lr -> min_lr) over remaining epochs
      - Plateau reducer: halves LR multiplier if val loss stalls for 5 epochs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 5)
    accumulation_steps = config.get("accumulation_steps", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    patience = config.get("patience", 15)
    save_every = config.get("save_every", 10)
    save_every_steps = config.get("save_every_steps", 0)
    keep_last_checkpoints = config.get("keep_last_checkpoints", 3)
    min_lr = config.get("min_lr", 1e-6)
    plateau_patience = config.get("plateau_patience", 5)

    model = model.to(device)
    criterion = UniStegLoss().to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = build_scheduler(
        optimizer, warmup_epochs, epochs, len(train_loader), min_lr,
    )
    scaler = torch.amp.GradScaler(device.type)
    plateau = PlateauReducer(patience=plateau_patience)

    # Store base LR for plateau reducer
    for pg in optimizer.param_groups:
        pg["_base_lr"] = pg["lr"]

    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(
            resume_path, model, optimizer, scheduler, criterion,
            map_location=str(device),
        )
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, step {global_step}, best_acc={best_val_acc:.4f}")
        if "timestamp" in ckpt:
            print(f"  Checkpoint from: {ckpt['timestamp']}")

    # SIGINT handler — save emergency checkpoint on Ctrl+C
    interrupted = [False]

    def _sigint_handler(signum, frame):
        if interrupted[0]:
            print("\nForce quit.")
            sys.exit(1)
        interrupted[0] = True
        print(f"\n[INTERRUPTED] Saving emergency checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler, criterion, epoch + 1,
            {}, os.path.join(output_dir, "emergency.pt"),
            global_step=global_step, best_val_acc=best_val_acc, config=config,
        )
        print(f"  Saved to {output_dir}/emergency.pt")
        print(f"  Resume with: --resume {output_dir}/emergency.pt")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Print config
    print(f"\nTraining config:")
    print(f"  LR: {lr} (heads: {lr * 3}, min: {min_lr})")
    print(f"  Epochs: {epochs} (warmup: {warmup_epochs})")
    eff_batch = train_loader.batch_size * accumulation_steps
    print(f"  Batch: {train_loader.batch_size} x {accumulation_steps} accum = {eff_batch} effective")
    print(f"  Early stopping patience: {patience}")
    print(f"  Plateau LR reducer patience: {plateau_patience}")
    print(f"  Checkpoint: every {save_every} epochs, keep last {keep_last_checkpoints}")
    if save_every_steps > 0:
        print(f"  Step checkpoint: every {save_every_steps} steps")
    print(f"  Device: {device}")
    print(f"  AMP: enabled")
    print()

    epoch = start_epoch  # ensure epoch is defined for SIGINT handler

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_metrics, global_step, grad_norms = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, accumulation_steps, max_grad_norm,
            epoch=epoch, global_step=global_step,
            save_every_n_steps=save_every_steps,
            output_dir=output_dir, best_val_acc=best_val_acc, config=config,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]

        # Gradient health
        avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
        max_grad_norm_val = max(grad_norms) if grad_norms else 0.0

        # Logging
        print(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.0f}s) "
            f"lr={lr_current:.2e} grad={avg_grad_norm:.2f}/{max_grad_norm_val:.2f}"
        )
        print(f"  {format_metrics(train_metrics, 'Train')}")
        print(f"  {format_metrics(val_metrics, 'Val  ')}")

        # Gradient health warnings
        if avg_grad_norm < 1e-6:
            print(f"  [WARN] Vanishing gradients: avg_norm={avg_grad_norm:.2e}")
        if max_grad_norm_val > 100:
            print(f"  [WARN] Gradient spike: max_norm={max_grad_norm_val:.2e}")

        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", lr_current, epoch)
            writer.add_scalar("grad/avg_norm", avg_grad_norm, epoch)
            writer.add_scalar("grad/max_norm", max_grad_norm_val, epoch)
            writer.add_scalar("step", global_step, epoch)

        # Always save last.pt (crash recovery)
        save_checkpoint(
            model, optimizer, scheduler, criterion, epoch + 1,
            val_metrics, os.path.join(output_dir, "last.pt"),
            global_step=global_step, best_val_acc=best_val_acc, config=config,
        )

        # Best model checkpoint
        val_acc = val_metrics.get("binary_acc", 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, criterion, epoch + 1,
                val_metrics, os.path.join(output_dir, "best.pt"),
                global_step=global_step, best_val_acc=best_val_acc, config=config,
            )
            print(f"  ** New best: binary_acc={val_acc:.4f}")
        else:
            epochs_without_improvement += 1

        # Periodic epoch checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, criterion, epoch + 1,
                val_metrics, os.path.join(output_dir, f"epoch_{epoch + 1}.pt"),
                global_step=global_step, best_val_acc=best_val_acc, config=config,
            )

        # Rotate old checkpoints
        rotate_checkpoints(output_dir, keep_last=keep_last_checkpoints)

        # Plateau LR reducer
        val_loss = val_metrics.get("loss/total", float("inf"))
        if plateau.step(val_loss):
            print(f"  [LR PLATEAU] Reducing LR multiplier to {plateau.lr_multiplier:.3f}")
            plateau.apply(optimizer)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete. Best binary_acc: {best_val_acc:.4f}")
    print(f"Checkpoints in {output_dir}/: best.pt, last.pt")
    return best_val_acc
