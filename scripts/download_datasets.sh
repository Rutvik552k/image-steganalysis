#!/bin/bash
# =============================================================
# Global Steganalysis - Dataset Download Script
# Downloads all required cover image datasets
# =============================================================
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

echo "=== Global Steganalysis Dataset Download ==="
echo "Target directory: $DATA_DIR"
echo ""

# ----- 1. BOSSbase 1.01 (Spatial covers) -----
download_bossbase() {
    local dir="$DATA_DIR/bossbase"
    if [ -d "$dir" ] && [ "$(ls -1 "$dir"/*.pgm 2>/dev/null | wc -l)" -ge 10000 ]; then
        echo "[SKIP] BOSSbase already downloaded ($(ls -1 "$dir"/*.pgm | wc -l) images)"
        return
    fi
    echo "[DOWNLOAD] BOSSbase 1.01 (10,000 grayscale PGM, ~1.5 GB)..."
    mkdir -p "$dir"
    wget -q --show-progress -O "$dir/BOSSbase_1.01.zip" \
        "http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
    echo "[EXTRACT] Unpacking BOSSbase..."
    unzip -q -o "$dir/BOSSbase_1.01.zip" -d "$dir"
    rm -f "$dir/BOSSbase_1.01.zip"
    echo "[OK] BOSSbase: $(ls -1 "$dir"/*.pgm 2>/dev/null | wc -l) images"
}

# ----- 2. BOWS2 (Spatial holdout) -----
download_bows2() {
    local dir="$DATA_DIR/bows2"
    if [ -d "$dir" ] && [ "$(ls -1 "$dir"/*.pgm 2>/dev/null | wc -l)" -ge 10000 ]; then
        echo "[SKIP] BOWS2 already downloaded"
        return
    fi
    echo "[DOWNLOAD] BOWS2 (10,000 grayscale PGM)..."
    mkdir -p "$dir"
    wget -q --show-progress -O "$dir/BOWS2OrigEp3.tgz" \
        "http://bows2.ec-lille.fr/BOWS2OrigEp3.tgz"
    echo "[EXTRACT] Unpacking BOWS2..."
    tar -xzf "$dir/BOWS2OrigEp3.tgz" -C "$dir"
    rm -f "$dir/BOWS2OrigEp3.tgz"
    echo "[OK] BOWS2: $(ls -1 "$dir"/*.pgm 2>/dev/null | wc -l) images"
}

# ----- 3. ALASKA2 (JPEG primary - via Kaggle) -----
download_alaska2() {
    local dir="$DATA_DIR/alaska2"
    if [ -d "$dir" ] && [ -d "$dir/Cover" ]; then
        echo "[SKIP] ALASKA2 already downloaded"
        return
    fi
    echo "[DOWNLOAD] ALASKA2 (300K JPEG images, ~50 GB)..."
    echo "  Requires: pip install kaggle && kaggle config"
    mkdir -p "$dir"

    if ! command -v kaggle &>/dev/null; then
        echo "[ERROR] kaggle CLI not found. Install with: pip install kaggle"
        echo "  Then set up API key: https://www.kaggle.com/docs/api"
        return 1
    fi

    kaggle competitions download -c alaska2-image-steganalysis -p "$dir"
    echo "[EXTRACT] Unpacking ALASKA2 (this takes a while)..."
    unzip -q -o "$dir/alaska2-image-steganalysis.zip" -d "$dir"
    rm -f "$dir/alaska2-image-steganalysis.zip"

    # ALASKA2 structure: Cover/, JMiPOD/, JUNIWARD/, UERD/
    echo "[OK] ALASKA2 structure:"
    for subdir in Cover JMiPOD JUNIWARD UERD; do
        if [ -d "$dir/$subdir" ]; then
            echo "  $subdir: $(ls -1 "$dir/$subdir"/*.jpg 2>/dev/null | wc -l) images"
        fi
    done
}

# ----- 4. IStego100K (Cross-dataset test) -----
download_istego100k() {
    local dir="$DATA_DIR/istego100k"
    if [ -d "$dir" ] && [ "$(find "$dir" -name '*.jpg' 2>/dev/null | wc -l)" -ge 1000 ]; then
        echo "[SKIP] IStego100K already downloaded"
        return
    fi
    echo "[DOWNLOAD] IStego100K..."
    echo "  Note: Large dataset. Download from https://github.com/YangzlTHU/IStego100K"
    echo "  Follow instructions on GitHub (Baidu Cloud or Google Drive links)"
    mkdir -p "$dir"
    echo "  Manual download required — see README for links"
}

# ----- 5. FFHQ (Diffusion stego covers) -----
download_ffhq() {
    local dir="$DATA_DIR/ffhq"
    if [ -d "$dir" ] && [ "$(find "$dir" -name '*.png' 2>/dev/null | wc -l)" -ge 1000 ]; then
        echo "[SKIP] FFHQ already downloaded"
        return
    fi
    echo "[DOWNLOAD] FFHQ 256x256 (70K face images)..."
    mkdir -p "$dir"

    # Use the thumbnails (256x256) version for efficiency
    echo "  Download from: https://github.com/NVlabs/ffhq-dataset"
    echo "  Recommended: Use thumbnails128x128 or thumbnails256x256 via Google Drive"
    echo "  Or: pip install gdown && gdown <drive_id>"
}

# ----- 6. DIV2K (Neural stego covers) -----
download_div2k() {
    local dir="$DATA_DIR/div2k"
    if [ -d "$dir" ] && [ "$(find "$dir" -name '*.png' 2>/dev/null | wc -l)" -ge 800 ]; then
        echo "[SKIP] DIV2K already downloaded"
        return
    fi
    echo "[DOWNLOAD] DIV2K (1000 2K-resolution images)..."
    mkdir -p "$dir"
    wget -q --show-progress -O "$dir/DIV2K_train_HR.zip" \
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    unzip -q -o "$dir/DIV2K_train_HR.zip" -d "$dir"
    rm -f "$dir/DIV2K_train_HR.zip"
    echo "[OK] DIV2K: $(find "$dir" -name '*.png' | wc -l) images"
}

# ----- 7. Text Corpora -----
download_text_corpora() {
    local dir="$DATA_DIR/text"
    mkdir -p "$dir"

    if [ -f "$dir/english_corpus.txt" ] && [ -s "$dir/english_corpus.txt" ]; then
        echo "[SKIP] English corpus already exists"
        return
    fi

    echo "[DOWNLOAD] Text corpora for payload generation..."

    # Download several public domain books from Project Gutenberg
    echo "  Downloading English text (Project Gutenberg)..."
    local books=(
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"   # Pride and Prejudice
        "https://www.gutenberg.org/cache/epub/84/pg84.txt"       # Frankenstein
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"   # Sherlock Holmes
        "https://www.gutenberg.org/cache/epub/11/pg11.txt"       # Alice in Wonderland
        "https://www.gutenberg.org/cache/epub/98/pg98.txt"       # Tale of Two Cities
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"   # Moby Dick
        "https://www.gutenberg.org/cache/epub/1260/pg1260.txt"   # Jane Eyre
        "https://www.gutenberg.org/cache/epub/16/pg16.txt"       # Peter Pan
        "https://www.gutenberg.org/cache/epub/74/pg74.txt"       # Tom Sawyer
        "https://www.gutenberg.org/cache/epub/1952/pg1952.txt"   # Yellow Wallpaper
    )

    > "$dir/english_corpus.txt"
    for url in "${books[@]}"; do
        wget -q -O - "$url" >> "$dir/english_corpus.txt" 2>/dev/null || true
        echo "" >> "$dir/english_corpus.txt"
    done

    local size=$(wc -c < "$dir/english_corpus.txt")
    echo "[OK] English corpus: $(( size / 1024 / 1024 )) MB"
}

# =============================================================
# MAIN
# =============================================================
echo "Phase 1: Core datasets (BOSSbase + ALASKA2 + text)"
echo "---------------------------------------------------"
download_bossbase
download_bows2
download_alaska2
download_text_corpora

echo ""
echo "Phase 2: Additional covers (FFHQ + DIV2K)"
echo "-------------------------------------------"
download_ffhq
download_div2k

echo ""
echo "Phase 3: Cross-dataset test sets"
echo "---------------------------------"
download_istego100k

echo ""
echo "=== Download Summary ==="
echo "Core:     BOSSbase (spatial), ALASKA2 (JPEG), text corpora"
echo "Covers:   FFHQ (diffusion), DIV2K (neural)"
echo "Test:     BOWS2, IStego100K, StegoAppDB (manual)"
echo ""
echo "Next: Run python scripts/generate_stego.py to create stego images"
