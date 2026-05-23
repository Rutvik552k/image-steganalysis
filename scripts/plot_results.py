"""
Generate paper figures from saved evaluation results.

Produces:
  1. Expert routing heatmap (Fig. routing)
  2. Per-rate accuracy curves (Fig. rate)
  3. Algorithm class confusion matrix (Fig. confusion)
  4. Training loss/accuracy curves from TensorBoard logs

Usage:
    python scripts/plot_results.py --results results/results_test.json --output paper/figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def plot_routing_heatmap(routing: dict, output_path: str):
    """Expert routing weights heatmap per algorithm class."""
    import matplotlib.pyplot as plt

    classes = sorted(routing.keys())
    n_experts = len(routing[classes[0]]["mean_routing"])

    data = np.array([routing[c]["mean_routing"] for c in classes])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.replace("class_", "").replace("_", " ") for c in classes],
                       fontsize=9)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Algorithm Class")

    # Annotate cells
    for i in range(len(classes)):
        for j in range(n_experts):
            val = data[i, j]
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Routing Weight")
    plt.title("Soft MoE Expert Routing Weights")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_rate_accuracy(per_rate: dict, output_path: str):
    """Detection accuracy vs payload rate."""
    import matplotlib.pyplot as plt

    rates = []
    accs = []
    for key in sorted(per_rate.keys()):
        if key.startswith("rate_"):
            rate = float(key.replace("rate_", ""))
            rates.append(rate)
            accs.append(per_rate[key]["binary_acc"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rates, accs, "o-", color="#2196F3", linewidth=2, markersize=6)

    if "cover" in per_rate:
        ax.axhline(y=per_rate["cover"]["binary_acc"], color="gray",
                   linestyle="--", label=f"Cover acc: {per_rate['cover']['binary_acc']:.3f}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Payload Rate (bpp)")
    ax.set_ylabel("Binary Detection Accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("Detection Accuracy vs Payload Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix(confusion: dict, output_path: str):
    """Algorithm class confusion matrix."""
    import matplotlib.pyplot as plt

    cm = np.array(confusion["matrix"])
    labels = confusion["labels"]
    short_labels = [l.replace("class_", "").replace("_", "\n") for l in labels]

    # Normalize rows
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Normalized Count")
    plt.title("Algorithm Class Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_algorithm_bar(per_algo: dict, output_path: str):
    """Per-algorithm binary detection accuracy bar chart."""
    import matplotlib.pyplot as plt

    algos = sorted(per_algo.keys())
    accs = [per_algo[a]["binary_acc"] for a in algos]
    counts = [per_algo[a]["count"] for a in algos]
    short_names = [a.replace("_", "\n") for a in algos]

    colors = []
    for a in algos:
        if a in ("none", "cover"):
            colors.append("#9E9E9E")
        elif "lsb" in a:
            colors.append("#4CAF50")
        elif a in ("s_uniward_sim", "hill_sim", "hugo", "wow", "mipod", "s_uniward", "hill"):
            colors.append("#2196F3")
        elif a in ("j_uniward", "jmipod", "uerd"):
            colors.append("#FF9800")
        elif a in ("steganogan", "hidden"):
            colors.append("#9C27B0")
        elif a in ("diffstega", "diffusion_stego"):
            colors.append("#F44336")
        else:
            colors.append("#607D8B")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(algos)), accs, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Binary Detection Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Per-Algorithm Binary Detection Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print("Generating figures...")

    if "routing_analysis" in results and results["routing_analysis"]:
        plot_routing_heatmap(
            results["routing_analysis"],
            os.path.join(args.output, "routing_heatmap.pdf"),
        )

    if "per_rate" in results:
        plot_per_rate_accuracy(
            results["per_rate"],
            os.path.join(args.output, "per_rate_accuracy.pdf"),
        )

    if "confusion_matrix" in results:
        plot_confusion_matrix(
            results["confusion_matrix"],
            os.path.join(args.output, "confusion_matrix.pdf"),
        )

    if "per_algorithm" in results:
        plot_per_algorithm_bar(
            results["per_algorithm"],
            os.path.join(args.output, "per_algorithm_accuracy.pdf"),
        )

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
