"""
UniSteg Evaluation Metrics Tracker
"""

import torch


class MetricsTracker:
    """Track and compute steganalysis evaluation metrics per epoch."""

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

    def update(self, predictions: dict, labels: dict, loss_dict: dict):
        B = labels["binary"].shape[0]

        pred_binary = predictions["binary"].argmax(dim=1)
        self.binary_correct += (pred_binary == labels["binary"]).sum().item()
        self.binary_total += B

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
            self.payload_mse_sum += ((pred_rate - true_rate) ** 2).sum().item()
            self.payload_count += stego_mask.sum().item()

        for k, v in loss_dict.items():
            if k not in ("weights", "log_var"):
                self.loss_sums[k] = self.loss_sums.get(k, 0.0) + v.item()
        self.loss_counts += 1

    def compute(self) -> dict:
        metrics = {
            "binary_acc": self.binary_correct / max(self.binary_total, 1),
            "algo_class_acc": self.algo_class_correct / max(self.algo_class_total, 1),
            "algo_acc": self.algo_correct / max(self.algo_total, 1),
            "payload_rmse": (self.payload_mse_sum / max(self.payload_count, 1)) ** 0.5,
        }
        for k, v in self.loss_sums.items():
            metrics[f"loss/{k}"] = v / max(self.loss_counts, 1)
        return metrics
