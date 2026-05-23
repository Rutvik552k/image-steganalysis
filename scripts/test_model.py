"""
UniSteg Test Script

Full assessment of trained model on test set:
  1. Per-algorithm binary detection accuracy
  2. Per-rate detection accuracy
  3. Algorithm classification confusion matrix
  4. MoE expert routing analysis per algorithm class
  5. Payload rate estimation error

Usage:
    python scripts/test_model.py --checkpoint checkpoints/best.pt --splits data/splits
    python scripts/test_model.py --checkpoint checkpoints/best.pt --splits data/splits --save-dir results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.models.unisteg import UniSteg, UniStegLite
from src.training.checkpoint import load_checkpoint
from src.training.losses import UniStegLoss


# ============================================================
# Prediction collection
# ============================================================

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on entire dataset, collect all predictions and labels."""
    model.train(False)  # inference mode
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    all_routing = []
    all_paths = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"]

        with torch.autocast(device_type=device.type):
            predictions = model(images)

        all_preds["binary_logits"].append(predictions["binary"].cpu())
        all_preds["algo_class_logits"].append(predictions["algo_class"].cpu())
        all_preds["algorithm_logits"].append(predictions["algorithm"].cpu())
        all_preds["payload_rate"].append(predictions["payload_rate"].cpu() * 0.5)

        for k, v in labels.items():
            all_labels[k].append(v)

        # MoE routing weights
        if hasattr(model, "get_expert_routing"):
            routing = model.get_expert_routing(images)
            all_routing.append(routing.cpu())
        elif hasattr(model, "model") and hasattr(model.model, "get_expert_routing"):
            routing = model.model.get_expert_routing(images)
            all_routing.append(routing.cpu())

        all_paths.extend(batch["path"])

    result = {
        "binary_logits": torch.cat(all_preds["binary_logits"]),
        "algo_class_logits": torch.cat(all_preds["algo_class_logits"]),
        "algorithm_logits": torch.cat(all_preds["algorithm_logits"]),
        "payload_rate": torch.cat(all_preds["payload_rate"]),
        "binary": torch.cat(all_labels["binary"]),
        "algorithm_class": torch.cat(all_labels["algorithm_class"]),
        "algorithm": torch.cat(all_labels["algorithm"]),
        "payload_rate_true": torch.cat(all_labels["payload_rate"]),
        "paths": all_paths,
    }

    if all_routing:
        result["routing"] = torch.cat(all_routing)

    return result


# ============================================================
# Metric computation
# ============================================================

def compute_overall_metrics(data: dict) -> dict:
    """Compute aggregate metrics."""
    binary_pred = data["binary_logits"].argmax(dim=1)
    binary_true = data["binary"]
    binary_acc = (binary_pred == binary_true).float().mean().item()

    algo_class_pred = data["algo_class_logits"].argmax(dim=1)
    algo_class_acc = (algo_class_pred == data["algorithm_class"]).float().mean().item()

    algo_pred = data["algorithm_logits"].argmax(dim=1)
    algo_acc = (algo_pred == data["algorithm"]).float().mean().item()

    stego_mask = binary_true == 1
    if stego_mask.any():
        payload_rmse = ((data["payload_rate"][stego_mask] - data["payload_rate_true"][stego_mask]) ** 2).mean().sqrt().item()
    else:
        payload_rmse = 0.0

    tp = ((binary_pred == 1) & (binary_true == 1)).sum().item()
    fp = ((binary_pred == 1) & (binary_true == 0)).sum().item()
    fn = ((binary_pred == 0) & (binary_true == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "binary_acc": binary_acc,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "algo_class_acc": algo_class_acc,
        "algo_acc": algo_acc,
        "payload_rmse": payload_rmse,
        "total_samples": len(binary_true),
        "stego_samples": stego_mask.sum().item(),
        "cover_samples": (~stego_mask).sum().item(),
    }


def compute_per_algorithm_metrics(data: dict, label_maps: dict) -> dict:
    """Compute binary detection accuracy per algorithm."""
    binary_pred = data["binary_logits"].argmax(dim=1)
    binary_true = data["binary"]
    algo_id_to_name = {v: k for k, v in label_maps["algorithm"].items()}

    results = {}
    for algo_id in data["algorithm"].unique().tolist():
        mask = data["algorithm"] == algo_id
        if mask.sum() == 0:
            continue
        acc = (binary_pred[mask] == binary_true[mask]).float().mean().item()
        algo_name = algo_id_to_name.get(algo_id, f"algo_{algo_id}")
        results[algo_name] = {"binary_acc": acc, "count": mask.sum().item()}

    return results


def compute_per_rate_metrics(data: dict) -> dict:
    """Compute binary detection accuracy per payload rate."""
    binary_pred = data["binary_logits"].argmax(dim=1)
    binary_true = data["binary"]
    rates = data["payload_rate_true"]
    stego_mask = binary_true == 1

    results = {}
    unique_rates = sorted(set(rates[stego_mask].numpy().round(2).tolist()))

    for rate in unique_rates:
        mask = stego_mask & (torch.abs(rates - rate) < 0.01)
        if mask.sum() == 0:
            continue
        acc = (binary_pred[mask] == binary_true[mask]).float().mean().item()
        results[f"rate_{rate:.2f}"] = {"binary_acc": acc, "count": mask.sum().item()}

    cover_mask = ~stego_mask
    if cover_mask.sum() > 0:
        cover_acc = (binary_pred[cover_mask] == binary_true[cover_mask]).float().mean().item()
        results["cover"] = {"binary_acc": cover_acc, "count": cover_mask.sum().item()}

    return results


def compute_confusion_matrix(data: dict, label_maps: dict, task: str = "algorithm_class") -> dict:
    """Compute confusion matrix for classification task."""
    if task == "algorithm_class":
        logits = data["algo_class_logits"]
        true = data["algorithm_class"]
        id_to_name = {v: k for k, v in label_maps["algorithm_class"].items()}
    else:
        logits = data["algorithm_logits"]
        true = data["algorithm"]
        id_to_name = {v: k for k, v in label_maps["algorithm"].items()}

    pred = logits.argmax(dim=1)
    n_classes = logits.shape[1]

    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(true, pred):
        cm[t, p] += 1

    class_names = [id_to_name.get(i, f"class_{i}") for i in range(n_classes)]
    cm_dict = {
        "matrix": cm.numpy().tolist(),
        "labels": class_names,
        "per_class_acc": {},
    }
    for i, name in enumerate(class_names):
        total = cm[i].sum().item()
        cm_dict["per_class_acc"][name] = cm[i, i].item() / max(total, 1)

    return cm_dict


def compute_routing_analysis(data: dict, label_maps: dict) -> dict:
    """Analyze MoE expert routing per algorithm class."""
    if "routing" not in data:
        return {}

    routing = data["routing"]
    algo_class = data["algorithm_class"]
    id_to_name = {v: k for k, v in label_maps["algorithm_class"].items()}

    results = {}
    for class_id in algo_class.unique().tolist():
        mask = algo_class == class_id
        class_name = id_to_name.get(class_id, f"class_{class_id}")
        mean_routing = routing[mask].mean(dim=0).numpy().tolist()
        std_routing = routing[mask].std(dim=0).numpy().tolist()
        dominant_expert = int(routing[mask].mean(dim=0).argmax())

        results[class_name] = {
            "mean_routing": [round(r, 4) for r in mean_routing],
            "std_routing": [round(r, 4) for r in std_routing],
            "dominant_expert": dominant_expert,
            "count": mask.sum().item(),
        }

    return results


# ============================================================
# Printing
# ============================================================

def print_results(overall, per_algo, per_rate, confusion, routing):
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"\n--- Overall ---")
    print(f"  Binary Acc:       {overall['binary_acc']:.4f}")
    print(f"  Binary F1:        {overall['binary_f1']:.4f}")
    print(f"  Precision/Recall: {overall['binary_precision']:.4f} / {overall['binary_recall']:.4f}")
    print(f"  Algo Class Acc:   {overall['algo_class_acc']:.4f}")
    print(f"  Algorithm Acc:    {overall['algo_acc']:.4f}")
    print(f"  Payload RMSE:     {overall['payload_rmse']:.4f}")
    print(f"  Samples:          {overall['total_samples']} ({overall['cover_samples']} cover, {overall['stego_samples']} stego)")

    print(f"\n--- Per Algorithm (binary detection) ---")
    for algo, m in sorted(per_algo.items()):
        print(f"  {algo:20s}  acc={m['binary_acc']:.4f}  n={m['count']}")

    print(f"\n--- Per Rate (binary detection) ---")
    for rate, m in sorted(per_rate.items()):
        print(f"  {rate:12s}  acc={m['binary_acc']:.4f}  n={m['count']}")

    if confusion:
        print(f"\n--- Algorithm Class Confusion (per-class accuracy) ---")
        for cls, acc in confusion["per_class_acc"].items():
            print(f"  {cls:25s}  acc={acc:.4f}")

    if routing:
        print(f"\n--- MoE Expert Routing ---")
        for cls, r in sorted(routing.items()):
            weights = " ".join(f"{w:.3f}" for w in r["mean_routing"])
            print(f"  {cls:25s}  [{weights}]  dominant=E{r['dominant_expert']}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test UniSteg model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, default="data/splits")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    label_maps_path = os.path.join(args.splits, "label_maps.json")
    with open(label_maps_path) as f:
        label_maps = json.load(f)

    num_algo_classes = len(label_maps["algorithm_class"])
    num_algorithms = len(label_maps["algorithm"])

    if args.lite:
        model = UniStegLite(num_algo_classes=num_algo_classes, num_algorithms=num_algorithms)
    else:
        model = UniSteg(use_context_stream=True, num_algo_classes=num_algo_classes, num_algorithms=num_algorithms)

    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    _, val_loader, test_loader = create_dataloaders(
        splits_dir=args.splits, batch_size=args.batch_size, target_size=256,
        apply_srm=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    loader = test_loader if args.split == "test" else val_loader
    print(f"Running on {args.split} split...")

    data = collect_predictions(model, loader, device)
    overall = compute_overall_metrics(data)
    per_algo = compute_per_algorithm_metrics(data, label_maps)
    per_rate = compute_per_rate_metrics(data)
    confusion = compute_confusion_matrix(data, label_maps, "algorithm_class")
    routing = compute_routing_analysis(data, label_maps)

    print_results(overall, per_algo, per_rate, confusion, routing)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        results = {
            "overall": overall,
            "per_algorithm": per_algo,
            "per_rate": per_rate,
            "confusion_matrix": confusion,
            "routing_analysis": routing,
        }
        out_path = os.path.join(args.save_dir, f"results_{args.split}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
