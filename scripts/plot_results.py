"""
Generate paper-quality figures from saved evaluation results.

Produces (all vector PDF, 300 DPI fallback PNG):
  1. ROC curves — overall + per-algorithm (Fig. roc)
  2. Expert routing heatmap (Fig. routing)
  3. Per-rate accuracy curves (Fig. rate)
  4. Algorithm class confusion matrix (Fig. confusion)
  5. Per-algorithm accuracy bar chart (Fig. per_algo)
  6. Payload rate scatter: predicted vs true (Fig. payload)
  7. Training curves: loss + accuracy + LR over epochs (Fig. training)
  8. Per-algorithm AUC-ROC bar chart (Fig. auc_bar)

Usage:
    python scripts/plot_results.py --results results/results_test.json --output paper/figures/
    python scripts/plot_results.py --results results/results_test.json --training-log runs/training_log.json --output paper/figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Paper style defaults
FONT_SIZE = 10
TITLE_SIZE = 11
TICK_SIZE = 8
LEGEND_SIZE = 8
LINE_WIDTH = 1.5
FIG_DPI = 300

# Color palette — colorblind-friendly (Tol's bright)
COLORS = {
    "spatial": "#4477AA",    # blue
    "jpeg": "#EE6677",       # red
    "neural": "#228833",     # green
    "diffusion": "#CCBB44",  # yellow
    "realworld": "#AA3377",  # purple
    "direct": "#66CCEE",     # cyan
    "cover": "#BBBBBB",      # gray
    "main": "#4477AA",       # default blue
}

ALGO_CLASS_COLORS = {
    "class_a_direct": COLORS["direct"],
    "class_b_stc_spatial": COLORS["spatial"],
    "class_d_stc_jpeg": COLORS["jpeg"],
    "class_e_neural": COLORS["neural"],
    "class_f_diffusion": COLORS["diffusion"],
    "class_h_realworld": COLORS["realworld"],
    "cover": COLORS["cover"],
}


def _algo_color(algo_name: str) -> str:
    """Map algorithm name to color."""
    if algo_name in ("none", "cover"):
        return COLORS["cover"]
    if "lsb" in algo_name or algo_name in ("f5", "nsf5", "outguess"):
        return COLORS["direct"]
    if algo_name in ("s_uniward", "hill", "hugo", "wow", "mipod"):
        return COLORS["spatial"]
    if algo_name in ("j_uniward", "jmipod", "uerd"):
        return COLORS["jpeg"]
    if algo_name in ("steganogan", "hidden"):
        return COLORS["neural"]
    if algo_name in ("diffstega", "diffusion_stego"):
        return COLORS["diffusion"]
    if algo_name in ("steghide", "openstego"):
        return COLORS["realworld"]
    return "#607D8B"


def _setup_style():
    """Set matplotlib style for paper figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig, output_path: str):
    """Save as PDF (vector) and PNG (raster fallback)."""
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=FIG_DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved: {output_path} + .png")


# ============================================================
# 1. ROC CURVES
# ============================================================

def plot_roc_curves(overall: dict, per_algo: dict, output_path: str):
    """Overall + per-algorithm ROC curves."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: overall ROC
    roc = overall.get("roc_curve", {})
    if roc.get("fpr"):
        ax1.plot(roc["fpr"], roc["tpr"], color=COLORS["main"], linewidth=2,
                 label=f'UniSteg (AUC = {overall["binary_auc_roc"]:.3f})')
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Overall ROC Curve")
    ax1.legend(loc="lower right")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Right: per-algorithm ROC
    for algo_name in sorted(per_algo.keys()):
        m = per_algo[algo_name]
        if algo_name in ("none", "cover"):
            continue
        roc_data = m.get("roc_curve", {})
        if not roc_data.get("fpr"):
            continue
        auc_val = m.get("auc_roc", 0)
        if np.isnan(auc_val):
            continue
        label = f'{algo_name.replace("_", "-")} ({auc_val:.2f})'
        ax2.plot(roc_data["fpr"], roc_data["tpr"], linewidth=LINE_WIDTH,
                 color=_algo_color(algo_name), label=label)

    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Per-Algorithm ROC Curves")
    ax2.legend(loc="lower right", fontsize=6, ncol=2)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 2. EXPERT ROUTING HEATMAP
# ============================================================

def plot_routing_heatmap(routing: dict, output_path: str):
    """Expert routing weights heatmap per algorithm class."""
    import matplotlib.pyplot as plt
    _setup_style()

    classes = sorted(routing.keys())
    n_experts = len(routing[classes[0]]["mean_routing"])
    data = np.array([routing[c]["mean_routing"] for c in classes])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.replace("class_", "").replace("_", " ").title()
                        for c in classes], fontsize=TICK_SIZE)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Algorithm Class")

    for i in range(len(classes)):
        for j in range(n_experts):
            val = data[i, j]
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Routing Weight", shrink=0.8)
    ax.set_title("Soft MoE Expert Routing Weights")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 3. PER-RATE ACCURACY CURVES
# ============================================================

def plot_per_rate_accuracy(per_rate: dict, output_path: str):
    """Detection accuracy vs payload rate."""
    import matplotlib.pyplot as plt
    _setup_style()

    rates = []
    accs = []
    for key in sorted(per_rate.keys()):
        if key.startswith("rate_"):
            rate = float(key.replace("rate_", ""))
            rates.append(rate)
            accs.append(per_rate[key]["binary_acc"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rates, accs, "o-", color=COLORS["main"], linewidth=2,
            markersize=6, label="UniSteg")

    if "cover" in per_rate:
        ax.axhline(y=per_rate["cover"]["binary_acc"], color=COLORS["cover"],
                   linestyle="--", linewidth=1,
                   label=f'Cover acc: {per_rate["cover"]["binary_acc"]:.3f}')

    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.3, label="Random")
    ax.set_xlabel("Payload Rate (bpp)")
    ax.set_ylabel("Binary Detection Accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("Detection Accuracy vs Payload Rate")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(confusion: dict, output_path: str):
    """Algorithm class confusion matrix (normalized)."""
    import matplotlib.pyplot as plt
    _setup_style()

    cm = np.array(confusion["matrix"])
    labels = confusion["labels"]
    short_labels = [l.replace("class_", "").replace("_", " ").title()
                    for l in labels]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Proportion", shrink=0.8)
    ax.set_title("Algorithm Class Confusion Matrix")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 5. PER-ALGORITHM ACCURACY BAR CHART
# ============================================================

def plot_per_algorithm_bar(per_algo: dict, output_path: str):
    """Per-algorithm binary detection accuracy bar chart."""
    import matplotlib.pyplot as plt
    _setup_style()

    algos = sorted([a for a in per_algo.keys() if a not in ("none", "cover")])
    accs = [per_algo[a]["binary_acc"] for a in algos]
    colors = [_algo_color(a) for a in algos]
    short_names = [a.replace("_", "-") for a in algos]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(range(len(algos)), accs, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.7)

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Binary Detection Accuracy")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.3, label="Random")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.set_title("Per-Algorithm Binary Detection Accuracy")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 6. PER-ALGORITHM AUC-ROC BAR CHART
# ============================================================

def plot_auc_bar(per_algo: dict, output_path: str):
    """Per-algorithm AUC-ROC bar chart."""
    import matplotlib.pyplot as plt
    _setup_style()

    algos = []
    aucs = []
    for a in sorted(per_algo.keys()):
        if a in ("none", "cover"):
            continue
        auc = per_algo[a].get("auc_roc", float("nan"))
        if np.isnan(auc):
            continue
        algos.append(a)
        aucs.append(auc)

    if not algos:
        print("  [SKIP] No AUC-ROC data for bar chart")
        return

    colors = [_algo_color(a) for a in algos]
    short_names = [a.replace("_", "-") for a in algos]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(range(len(algos)), aucs, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.7)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.3, label="Random")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.set_title("Per-Algorithm AUC-ROC")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 7. PAYLOAD RATE SCATTER
# ============================================================

def plot_payload_scatter(results_path: str, output_path: str):
    """Predicted vs true payload rate scatter plot.

    Requires raw predictions saved separately (payload_predictions.json).
    """
    import matplotlib.pyplot as plt
    _setup_style()

    pred_path = results_path.replace("results_test", "payload_predictions")
    if not os.path.exists(pred_path):
        print(f"  [SKIP] No payload predictions file: {pred_path}")
        return

    with open(pred_path) as f:
        data = json.load(f)

    true_rates = np.array(data["true"])
    pred_rates = np.array(data["predicted"])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(true_rates, pred_rates, s=4, alpha=0.3, color=COLORS["main"],
               edgecolors="none")

    # Perfect prediction line
    lims = [0, max(true_rates.max(), pred_rates.max()) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect")

    # Linear fit
    if len(true_rates) > 10:
        z = np.polyfit(true_rates, pred_rates, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, p(x_fit), color=COLORS["jpeg"], linewidth=1,
                linestyle="-", alpha=0.7,
                label=f"Fit: y={z[0]:.2f}x+{z[1]:.3f}")

    rmse = np.sqrt(np.mean((true_rates - pred_rates) ** 2))
    ax.text(0.05, 0.92, f"RMSE = {rmse:.4f}", transform=ax.transAxes,
            fontsize=FONT_SIZE, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("True Payload Rate (bpp)")
    ax.set_ylabel("Predicted Payload Rate (bpp)")
    ax.set_title("Payload Rate Estimation")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# 8. TRAINING CURVES
# ============================================================

def plot_training_curves(log_path: str, output_path: str):
    """Training loss, accuracy, and LR curves from training log JSON."""
    import matplotlib.pyplot as plt
    _setup_style()

    with open(log_path) as f:
        log = json.load(f)

    epochs = log.get("epochs", list(range(len(log.get("train_loss", [])))))
    train_loss = log.get("train_loss", [])
    val_loss = log.get("val_loss", [])
    train_acc = log.get("train_binary_acc", [])
    val_acc = log.get("val_binary_acc", [])
    lr_history = log.get("lr", [])
    val_auc = log.get("val_auc_roc", [])

    n_plots = 2 + (1 if lr_history else 0) + (1 if val_auc else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Loss curves
    if train_loss:
        axes[idx].plot(epochs[:len(train_loss)], train_loss,
                       color=COLORS["main"], label="Train", linewidth=LINE_WIDTH)
    if val_loss:
        axes[idx].plot(epochs[:len(val_loss)], val_loss,
                       color=COLORS["jpeg"], label="Val", linewidth=LINE_WIDTH)
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Loss")
    axes[idx].set_title("Training & Validation Loss")
    axes[idx].legend()
    idx += 1

    # Accuracy curves
    if train_acc:
        axes[idx].plot(epochs[:len(train_acc)], train_acc,
                       color=COLORS["main"], label="Train", linewidth=LINE_WIDTH)
    if val_acc:
        axes[idx].plot(epochs[:len(val_acc)], val_acc,
                       color=COLORS["jpeg"], label="Val", linewidth=LINE_WIDTH)
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Binary Accuracy")
    axes[idx].set_title("Training & Validation Accuracy")
    axes[idx].set_ylim(0.4, 1.02)
    axes[idx].legend()
    idx += 1

    # AUC-ROC curve over epochs
    if val_auc:
        axes[idx].plot(epochs[:len(val_auc)], val_auc,
                       color=COLORS["neural"], linewidth=LINE_WIDTH)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("AUC-ROC")
        axes[idx].set_title("Validation AUC-ROC")
        axes[idx].set_ylim(0.4, 1.02)
        idx += 1

    # LR schedule
    if lr_history:
        axes[idx].plot(epochs[:len(lr_history)], lr_history,
                       color=COLORS["diffusion"], linewidth=LINE_WIDTH)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Learning Rate")
        axes[idx].set_title("Learning Rate Schedule")
        axes[idx].set_yscale("log")
        idx += 1

    fig.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results_test.json from test_model.py")
    parser.add_argument("--training-log", type=str, default=None,
                        help="Path to training_log.json for training curves")
    parser.add_argument("--output", type=str, default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print("Generating paper figures...")

    # 1. ROC curves (overall + per-algorithm)
    if "overall" in results:
        plot_roc_curves(
            results["overall"],
            results.get("per_algorithm", {}),
            os.path.join(args.output, "roc_curves.pdf"),
        )

    # 2. Expert routing heatmap
    if results.get("routing_analysis"):
        plot_routing_heatmap(
            results["routing_analysis"],
            os.path.join(args.output, "routing_heatmap.pdf"),
        )

    # 3. Per-rate accuracy
    if "per_rate" in results:
        plot_per_rate_accuracy(
            results["per_rate"],
            os.path.join(args.output, "per_rate_accuracy.pdf"),
        )

    # 4. Confusion matrix
    if "confusion_matrix" in results:
        plot_confusion_matrix(
            results["confusion_matrix"],
            os.path.join(args.output, "confusion_matrix.pdf"),
        )

    # 5. Per-algorithm accuracy bars
    if "per_algorithm" in results:
        plot_per_algorithm_bar(
            results["per_algorithm"],
            os.path.join(args.output, "per_algorithm_accuracy.pdf"),
        )

    # 6. Per-algorithm AUC-ROC bars
    if "per_algorithm" in results:
        plot_auc_bar(
            results["per_algorithm"],
            os.path.join(args.output, "per_algorithm_auc.pdf"),
        )

    # 7. Payload scatter
    plot_payload_scatter(
        args.results,
        os.path.join(args.output, "payload_scatter.pdf"),
    )

    # 8. Training curves
    if args.training_log and os.path.exists(args.training_log):
        plot_training_curves(
            args.training_log,
            os.path.join(args.output, "training_curves.pdf"),
        )

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
