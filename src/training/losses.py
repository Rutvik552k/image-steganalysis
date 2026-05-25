"""
UniSteg Multi-Task Loss with Uncertainty Weighting

Four tasks with automatically balanced weights via learned uncertainty:
  1. Binary detection (CE): cover vs stego
  2. Algorithm class (CE): 7-way classification
  3. Specific algorithm (CE): 21-way classification
  4. Payload rate (MSE): regression [0, 0.5]

Uses Kendall et al. (CVPR 2018) uncertainty weighting:
  L_total = sum_i (1/(2*sigma_i^2) * L_i + log(sigma_i))

The log(sigma) regularization term prevents weights from decaying to zero.
Without it, the optimizer would minimize total loss by shrinking all weights.

Reference: Kendall, Gal, Cipolla, "Multi-Task Learning Using Uncertainty
to Weigh Losses for Scene Geometry and Semantics," CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniStegLoss(nn.Module):
    """Multi-task loss with Kendall uncertainty weighting or fixed weights.

    Two modes:
      - "kendall": learns per-task log-variance (Kendall et al., CVPR 2018)
      - "fixed": uses static weights (better when auxiliary tasks are noisy)

    Default fixed weights prioritize binary detection (the primary task).
    """

    def __init__(
        self,
        num_tasks: int = 4,
        mode: str = "kendall",
        fixed_weights: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.mode = mode

        # Learnable log(sigma^2) per task — used in kendall mode
        self.log_var = nn.Parameter(torch.zeros(num_tasks))

        # Fixed weights: (binary, algo_class, algorithm, payload)
        if fixed_weights is None:
            fixed_weights = (1.0, 0.3, 0.1, 0.1)
        self.register_buffer(
            "fixed_w", torch.tensor(fixed_weights, dtype=torch.float32)
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            predictions: dict from UniSteg.forward()
            labels: dict from dataloader

        Returns:
            (total_loss, loss_dict) where loss_dict has per-task losses
        """
        # Task 1: Binary detection
        loss_binary = self.ce_loss(predictions["binary"], labels["binary"])

        # Task 2: Algorithm class
        loss_algo_class = self.ce_loss(
            predictions["algo_class"], labels["algorithm_class"]
        )

        # Task 3: Specific algorithm
        loss_algorithm = self.ce_loss(
            predictions["algorithm"], labels["algorithm"]
        )

        # Task 4: Payload rate regression (stego images only)
        # Model outputs [0, 1] via sigmoid; scale target from [0, 0.5] to [0, 1]
        # Only compute on stego images — cover images have no meaningful rate
        stego_mask = labels["binary"] == 1
        if stego_mask.any():
            pred_rate = predictions["payload_rate"][stego_mask]
            target_rate = labels["payload_rate"][stego_mask] / 0.5
            target_rate = target_rate.clamp(0.0, 1.0)
            loss_payload = self.mse_loss(pred_rate, target_rate)
        else:
            loss_payload = torch.tensor(0.0, device=loss_binary.device)

        losses = torch.stack([loss_binary, loss_algo_class, loss_algorithm, loss_payload])

        if self.mode == "kendall":
            # Kendall uncertainty weighting:
            # L_total = sum_i [ (1 / (2 * exp(log_var_i))) * L_i + 0.5 * log_var_i ]
            log_var_clamped = self.log_var.float().clamp(-6.0, 6.0)
            precision = torch.exp(-log_var_clamped)  # 1 / sigma^2
            total_loss = (precision * losses.float() + log_var_clamped).sum() * 0.5
            effective_weights = precision.detach()
        else:
            # Fixed weights — binary task dominates
            total_loss = (self.fixed_w * losses.float()).sum()
            effective_weights = self.fixed_w.detach()

        # NaN guard
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = loss_binary + loss_algo_class + loss_algorithm + loss_payload

        loss_dict = {
            "binary": loss_binary.detach(),
            "algo_class": loss_algo_class.detach(),
            "algorithm": loss_algorithm.detach(),
            "payload_rate": loss_payload.detach(),
            "total": total_loss.detach(),
            "weights": effective_weights,
            "log_var": self.log_var.detach(),
        }

        return total_loss, loss_dict
