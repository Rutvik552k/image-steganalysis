"""
UniSteg Evaluation Metrics Tracker
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class MetricsTracker:
    """Track and compute steganalysis evaluation metrics per epoch.

    Collects predictions across batches and computes:
      - Binary accuracy, AUC-ROC
      - Algorithm class accuracy
      - Specific algorithm accuracy
      - Payload RMSE (stego only)
      - Per-component losses
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.binary_correct = 0
        self.binary_total = 0
        self.algo_class_correct = 0
        self.algo_class_total = 0
        self.algo_correct = 0
        self.algo_total = 0
        self.payload_mse_sum = 0.0
        self.payload_count = 0
        self.loss_sums = {}
        self.loss_counts = 0

        # For AUC-ROC: collect all probs and labels
        self._binary_probs = []
        self._binary_labels = []

    def update(self, predictions: dict, labels: dict, loss_dict: dict):
        B = labels["binary"].shape[0]

        pred_binary = predictions["binary"].argmax(dim=1)
        self.binary_correct += (pred_binary == labels["binary"]).sum().item()
        self.binary_total += B

        # Collect softmax probs for AUC-ROC
        probs = F.softmax(predictions["binary"].detach(), dim=1)[:, 1]
        self._binary_probs.append(probs.cpu())
        self._binary_labels.append(labels["binary"].cpu())

        pred_ac = predictions["algo_class"].argmax(dim=1)
        self.algo_class_correct += (pred_ac == labels["algorithm_class"]).sum().item()
        self.algo_class_total += B

        pred_algo = predictions["algorithm"].argmax(dim=1)
        self.algo_correct += (pred_algo == labels["algorithm"]).sum().item()
        self.algo_total += B

        stego_mask = labels["binary"] == 1
        if stego_mask.any():
            pred_rate = predictions["payload_rate"][stego_mask] * 0.5
            true_rate = labels["payload_rate"][stego_mask]
            # Filter NaN/Inf predictions to prevent poison accumulation
            valid = ~(torch.isnan(pred_rate) | torch.isinf(pred_rate))
            if valid.any():
                self.payload_mse_sum += ((pred_rate[valid] - true_rate[valid]) ** 2).sum().item()
                self.payload_count += valid.sum().item()

        for k, v in loss_dict.items():
            if k not in ("weights", "log_var"):
                val = v.item()
                if not (math.isnan(val) or math.isinf(val)):
                    self.loss_sums[k] = self.loss_sums.get(k, 0.0) + val
        self.loss_counts += 1

    def compute(self) -> dict:
        metrics = {
            "binary_acc": self.binary_correct / max(self.binary_total, 1),
            "algo_class_acc": self.algo_class_correct / max(self.algo_class_total, 1),
            "algo_acc": self.algo_correct / max(self.algo_total, 1),
            "payload_rmse": (self.payload_mse_sum / max(self.payload_count, 1)) ** 0.5,
        }

        # AUC-ROC
        if self._binary_probs:
            all_probs = torch.cat(self._binary_probs).numpy()
            all_labels = torch.cat(self._binary_labels).numpy()
            # Filter NaN/Inf probs
            valid_mask = np.isfinite(all_probs)
            if valid_mask.sum() > 0:
                all_probs = all_probs[valid_mask]
                all_labels = all_labels[valid_mask]
            try:
                metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
            except ValueError:
                metrics["auc_roc"] = 0.0
        else:
            metrics["auc_roc"] = 0.0

        for k, v in self.loss_sums.items():
            metrics[f"loss/{k}"] = v / max(self.loss_counts, 1)
        return metrics
