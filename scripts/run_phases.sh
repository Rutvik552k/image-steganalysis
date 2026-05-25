#!/bin/bash
# Phased training launcher for Thunder Compute
# Usage: nohup bash scripts/run_phases.sh > phased_training.log 2>&1 &

set -e

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

cd ~/unisteg

echo "=== Phased Training Start: $(date) ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run all phases sequentially, 20 epochs each
python3 -u scripts/phased_train.py \
    --splits-dir data/splits \
    --output-dir checkpoints \
    --epochs-per-phase 20 \
    --batch-size 32 \
    --lr 1e-3 \
    --seed 42 \
    --num-workers 4

echo "=== Phased Training Complete: $(date) ==="
