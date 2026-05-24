#!/bin/bash
# ============================================================
# UniSteg Cloud Pipeline — A6000 (6 vCPU, 48GB RAM, 250GB disk)
# Full parallel: download + generate + train in <5 hours
# ============================================================
set -euo pipefail

WORKDIR="$HOME/unisteg"
DATA_DIR="$WORKDIR/data/raw"
PROCESSED_DIR="$WORKDIR/data/processed"
SPLITS_DIR="$WORKDIR/data/splits"
CHECKPOINT_DIR="$WORKDIR/checkpoints"
LOG_DIR="$WORKDIR/runs"
STEGO_WORKERS=5

START_TIME=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "============================================"
log " UniSteg Cloud Pipeline"
log "============================================"

# ── System info ──
log "=== SYSTEM ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || log "GPU not ready"
log "CPUs: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}') | Disk free: $(df -h $HOME | awk 'NR==2{print $4}')"

# ── Setup ──
log "=== SETUP ==="
mkdir -p "$WORKDIR" "$DATA_DIR"/{bossbase,alaska2,text} "$PROCESSED_DIR" "$SPLITS_DIR" "$CHECKPOINT_DIR"

pip install -q conseal jpeglib pillow numpy pandas scikit-learn \
    torch torchvision tensorboard pyyaml tqdm matplotlib 2>&1 | tail -3
log "Deps installed"

# ── Download BOSSbase (must complete before stego gen) ──
log "=== DOWNLOAD: BOSSbase ==="
if [ "$(ls "$DATA_DIR"/bossbase/*.pgm 2>/dev/null | wc -l)" -gt 5000 ]; then
    log "[SKIP] BOSSbase present"
else
    cd "$DATA_DIR/bossbase"
    wget -q --show-progress "http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip" -O bossbase.zip
    unzip -q -o bossbase.zip && rm -f bossbase.zip
    log "BOSSbase: $(ls *.pgm 2>/dev/null | wc -l) images"
fi

# ── Download ALASKA2 in background ──
log "=== DOWNLOAD: ALASKA2 (background) ==="
(
    if [ "$(ls "$DATA_DIR"/alaska2/Cover/*.jpg 2>/dev/null | wc -l)" -gt 5000 ]; then
        log "[SKIP] ALASKA2 present"
    else
        cd "$DATA_DIR/alaska2"
        if command -v kaggle &>/dev/null; then
            kaggle competitions download alaska2-image-steganalysis -p .
            unzip -q -o "*.zip" && rm -f *.zip
        else
            log "[WARN] No kaggle CLI. Install: pip install kaggle && set KAGGLE_USERNAME/KEY"
            log "  Manual: kaggle competitions download alaska2-image-steganalysis"
        fi
    fi
    log "ALASKA2 download done"
) &
PID_ALASKA=$!

# ── Download text corpus in background ──
(
    if [ -f "$DATA_DIR/text/english_corpus.txt" ]; then
        log "[SKIP] Text corpus present"
    else
        cd "$DATA_DIR/text"
        for id in 1342 11 1661 84 98 74 2701 1232 345 135; do
            wget -q "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -O "book_$id.txt" 2>/dev/null || true
        done
        cat book_*.txt > english_corpus.txt 2>/dev/null
        log "Text corpus: $(wc -c < english_corpus.txt) bytes"
    fi
) &
PID_TEXT=$!

# ── Generate spatial stego (parallel, 5 workers) ──
log "=== STEGO GENERATION ($STEGO_WORKERS workers) ==="
cd "$WORKDIR"

python scripts/generate_stego_fast.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$PROCESSED_DIR" \
    --workers $STEGO_WORKERS \
    --algorithms s_uniward hill \
    --rates 0.2 0.4 \
    --skip-alaska2 \
    2>&1 | tee "$WORKDIR/stego_gen.log"

log "Spatial stego generation complete"

# ── Wait for ALASKA2 + register ──
log "Waiting for ALASKA2 download..."
wait $PID_ALASKA || true
wait $PID_TEXT || true

# Register ALASKA2 if downloaded
if [ -d "$DATA_DIR/alaska2/Cover" ]; then
    log "Registering ALASKA2..."
    python -c "
import sys; sys.path.insert(0, '.')
from scripts.generate_stego_fast import register_alaska2
import pandas as pd, os

records = register_alaska2('$DATA_DIR/alaska2')
# Append to existing metadata
meta_path = '$PROCESSED_DIR/metadata.csv'
if os.path.exists(meta_path):
    existing = pd.read_csv(meta_path)
    combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
else:
    combined = pd.DataFrame(records)
combined.to_csv(meta_path, index=False)
print(f'Total records: {len(combined)}')
"
    log "ALASKA2 registered"
else
    log "[WARN] ALASKA2 not available — training with spatial only"
fi

# ── Build splits ──
log "=== BUILD SPLITS ==="
python scripts/build_splits.py \
    --metadata "$PROCESSED_DIR/metadata.csv" \
    --output-dir "$SPLITS_DIR" \
    2>&1 | tee "$WORKDIR/splits.log"

log "Splits built"
wc -l "$SPLITS_DIR"/*.csv

# ── Train ──
log "=== TRAINING (UniStegLite, A6000) ==="
log "Batch 128, AMP, torch.compile, channels_last"

python scripts/train.py \
    --config configs/training_config.yaml \
    --lite \
    --batch-size 128 \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CHECKPOINT_DIR" \
    --epochs 100 \
    --lr 1e-3 \
    2>&1 | tee "$WORKDIR/training.log"

# ── Done ──
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

log "============================================"
log " PIPELINE COMPLETE in ${ELAPSED} minutes"
log "============================================"
log "Checkpoints: $CHECKPOINT_DIR/"
log "Training log: $WORKDIR/training.log"
log "TensorBoard: tensorboard --logdir $LOG_DIR --bind_all"

ls -lh "$CHECKPOINT_DIR"/{best,last}.pt 2>/dev/null
