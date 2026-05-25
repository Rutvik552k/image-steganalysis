"""
UniSteg Evaluation Metrics Tracker

Steganalysis-standard metrics:
  - P_E: detection error = (P_FA + P_MD) / 2 (primary metric in SRNet, YedroudjNet)
  - P_FA: false alarm rate (cover classified as stego)
  - P_MD: missed detection rate (stego classified as cover)
  - Balanced accuracy: average of per-class recall
  - AUC-ROC: area under ROC curve
  - Per-algorithm detection accuracy
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


class MetricsTracker:
    """Track and compute steganalysis evaluation metrics per epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Binary detection
        self.binary_tp = 0  # stego correctly detected
        self.binary_tn = 0  # cover correctly detected
        self.binary_fp = 0  # cover misclassified as stego (false alarm)
        self.binary_fn = 0  # stego misclassified as cover (missed detection)
        self.binary_total = 0

        # Multi-class
        self.algo_class_correct = 0
        self.algo_class_total = 0
        self.algo_correct = 0
        self.algo_total = 0

        # Per-algorithm tracking
        self._algo_preds = []
        self._algo_labels = []

        # Payload
        self.payload_mse_sum = 0.0
        self.payload_count = 0

        # Losses
        self.loss_sums = {}
        self.loss_counts = 0

        # For AUC-ROC
        self._binary_probs = []
        self._binary_labels = []

    def update(self, predictions: dict, labels: dict, loss_dict: dict):
        B = labels["binary"].shape[0]

        pred_binary = predictions["binary"].argmax(dim=1)
        true_binary = labels["binary"]
        self.binary_total += B

        # TP/TN/FP/FN for P_E calculation
        # label 0 = cover, label 1 = stego
        self.binary_tp += ((pred_binary == 1) & (true_binary == 1)).sum().item()
        self.binary_tn += ((pred_binary == 0) & (true_binary == 0)).sum().item()
        self.binary_fp += ((pred_binary == 1) & (true_binary == 0)).sum().item()
        self.binary_fn += ((pred_binary == 0) & (true_binary == 1)).sum().item()

        # Softmax probs for AUC-ROC
        probs = F.softmax(predictions["binary"].detach(), dim=1)[:, 1]
        self._binary_probs.append(probs.cpu())
        self._binary_labels.append(true_binary.cpu())

        # Algorithm class
        pred_ac = predictions["algo_class"].argmax(dim=1)
        self.algo_class_correct += (pred_ac == labels["algorithm_class"]).sum().item()
        self.algo_class_total += B

        # Specific algorithm
        pred_algo = predictions["algorithm"].argmax(dim=1)
        self.algo_correct += (pred_algo == labels["algorithm"]).sum().item()
        self.algo_total += B
        self._algo_preds.append(pred_algo.cpu())
        self._algo_labels.append(labels["algorithm"].cpu())

        # Payload RMSE (stego only)
        stego_mask = true_binary == 1
        if stego_mask.any():
            pred_rate = predictions["payload_rate"][stego_mask] * 0.5
            true_rate = labels["payload_rate"][stego_mask]
            valid = ~(torch.isnan(pred_rate) | torch.isinf(pred_rate))
            if valid.any():
                self.payload_mse_sum += ((pred_rate[valid] - true_rate[valid]) ** 2).sum().item()
                self.payload_count += valid.sum().item()

        # Losses
        for k, v in loss_dict.items():
            if k not in ("weights", "log_var"):
                val = v.item()
                if not (math.isnan(val) or math.isinf(val)):
                    self.loss_sums[k] = self.loss_sums.get(k, 0.0) + val
        self.loss_counts += 1

    def compute(self) -> dict:
        # P_FA, P_MD, P_E — standard steganalysis detection metrics
        total_cover = self.binary_tn + self.binary_fp
        total_stego = self.binary_tp + self.binary_fn
        p_fa = self.binary_fp / max(total_cover, 1)  # false alarm rate
        p_md = self.binary_fn / max(total_stego, 1)  # missed detection rate
        p_e = (p_fa + p_md) / 2  # detection error (lower is better)

        # Balanced accuracy — immune to class imbalance
        cover_recall = self.binary_tn / max(total_cover, 1)
        stego_recall = self.binary_tp / max(total_stego, 1)
        balanced_acc = (cover_recall + stego_recall) / 2

        # Precision, Recall, F1
        precision = self.binary_tp / max(self.binary_tp + self.binary_fp, 1)
        recall = self.binary_tp / max(self.binary_tp + self.binary_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        metrics = {
            "binary_acc": (self.binary_tp + self.binary_tn) / max(self.binary_total, 1),
            "balanced_acc": balanced_acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "p_e": p_e,
            "p_fa": p_fa,
            "p_md": p_md,
            "algo_class_acc": self.algo_class_correct / max(self.algo_class_total, 1),
            "algo_acc": self.algo_correct / max(self.algo_total, 1),
            "payload_rmse": (self.payload_mse_sum / max(self.payload_count, 1)) ** 0.5,
        }

        # AUC-ROC + threshold-swept min P_E
        if self._binary_probs:
            all_probs = torch.cat(self._binary_probs).numpy()
            all_labels = torch.cat(self._binary_labels).numpy()
            valid_mask = np.isfinite(all_probs)
            if valid_mask.sum() > 0:
                all_probs = all_probs[valid_mask]
                all_labels = all_labels[valid_mask]
            try:
                metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
                # Threshold-swept min P_E (canonical steganalysis metric)
                fpr, tpr, _ = roc_curve(all_labels, all_probs)
                p_md_curve = 1.0 - tpr  # missed detection = 1 - true positive rate
                p_e_curve = (fpr + p_md_curve) / 2.0
                metrics["min_p_e"] = float(np.min(p_e_curve))
            except ValueError:
                metrics["auc_roc"] = 0.0
                metrics["min_p_e"] = 0.5
        else:
            metrics["auc_roc"] = 0.0
            metrics["min_p_e"] = 0.5

        # Per-algorithm accuracy + confusion matrix
        if self._algo_preds:
            all_preds = torch.cat(self._algo_preds).numpy()
            all_labels_algo = torch.cat(self._algo_labels).numpy()
            unique_algos = np.unique(all_labels_algo)
            per_algo = {}
            for algo_id in unique_algos:
                mask = all_labels_algo == algo_id
                if mask.sum() > 0:
                    per_algo[int(algo_id)] = float((all_preds[mask] == algo_id).mean())
            metrics["per_algo_acc"] = per_algo

            # Confusion matrix (algo_id x algo_id)
            n_classes = int(max(all_labels_algo.max(), all_preds.max())) + 1
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for pred, true in zip(all_preds, all_labels_algo):
                cm[int(true), int(pred)] += 1
            metrics["confusion_matrix"] = cm.tolist()

            # Macro F1 for multi-class
            per_class_f1 = []
            for c in range(n_classes):
                tp_c = cm[c, c]
                fp_c = cm[:, c].sum() - tp_c
                fn_c = cm[c, :].sum() - tp_c
                prec_c = tp_c / max(tp_c + fp_c, 1)
                rec_c = tp_c / max(tp_c + fn_c, 1)
                f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-8)
                per_class_f1.append(f1_c)
            metrics["macro_f1"] = float(np.mean(per_class_f1))

        for k, v in self.loss_sums.items():
            metrics[f"loss/{k}"] = v / max(self.loss_counts, 1)
        return metrics
