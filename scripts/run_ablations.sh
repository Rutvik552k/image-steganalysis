#!/bin/bash
# Run all ablation experiments sequentially
# Usage: bash scripts/run_ablations.sh [--device cuda]

set -euo pipefail

DEVICE="${1:---device cpu}"

echo "=== UniSteg Ablation Experiments ==="
echo "Device: $DEVICE"
echo ""

# Main lite model (baseline for ablations)
echo "--- [1/5] Lite baseline (5 experts) ---"
python scripts/train.py --config configs/training_config.yaml $DEVICE

# No MoE (1 expert)
echo "--- [2/5] No MoE (1 expert) ---"
python scripts/train.py --config configs/ablation_no_moe.yaml $DEVICE

# 3 experts
echo "--- [3/5] 3 experts ---"
python scripts/train.py --config configs/ablation_experts_3.yaml $DEVICE

# 8 experts
echo "--- [4/5] 8 experts ---"
python scripts/train.py --config configs/ablation_experts_8.yaml $DEVICE

# Full model with DINOv2 (needs GPU)
echo "--- [5/5] Full model with DINOv2 ---"
python scripts/train.py --config configs/ablation_full_model.yaml $DEVICE

echo ""
echo "=== All ablations complete ==="
echo "Run: python scripts/test_model.py --checkpoint <path> --splits data/splits --lite"
