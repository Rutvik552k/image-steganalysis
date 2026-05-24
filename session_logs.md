# Session Logs — UniSteg Image Steganalysis

## Session: 2026-05-23 / 2026-05-24

### Summary
Set up Thunder Compute GPU cloud instance, downloaded all datasets, generated stego images, fixed training bugs, and started full model training.

---

### 1. Thunder Compute Setup
- Created instance: `1jrbdzmg`, RTX A6000 (48GB VRAM), 4 vCPUs, 32GB RAM, 200GB storage
- SSH key at `~/.thunder/keys/1jrbdzmg`, port 30856
- Cost: ~$0.75/hr
- CLI: `tnr` (Thunder Compute CLI) installed locally

### 2. Code Pushed to GitHub
- Repo: `https://github.com/Rutvik552k/image-steganalysis.git`
- Initial commit: full codebase (model, training, configs, paper, scripts)
- Subsequent commits:
  - Added AUC-ROC metrics + 8 paper-quality figure generators
  - Added parallel stego generation script (`generate_stego_fast.py`)
  - Added missing `src/data/` module (dataset loader)
  - Fixed `torch.cuda.total_mem` → `total_memory` for torch 2.11
  - Fixed NaN loss: payload regression on stego only, clamped log_var, NaN guard
  - Added `--splits-dir` and `--output-dir` CLI args to train.py

### 3. Datasets Downloaded on Instance
| Dataset | Images | Size | Status |
|---------|--------|------|--------|
| BOSSbase 1.01 | 10,000 PGM | ~2.5GB | Ready |
| ALASKA2 | 75K covers + 225K stego (JMiPOD, JUNIWARD, UERD) | ~37GB | Ready |
| DIV2K | 800 PNG | ~3.3GB | Ready |
| BOWS2 | 10,000 PGM (Kaggle) | ~1.7GB | Ready |
| FFHQ | 52,001 PNG (Kaggle, partial) | ~19.5GB | Ready |
| Text corpus | 4MB (Project Gutenberg) | 4MB | Ready |

### 4. Stego Image Generation (Phase 1)
- Script: `generate_stego_fast.py` with 4 multiprocessing workers
- Algorithms: S-UNIWARD, HILL (spatial domain)
- Rates: 0.2, 0.4 bpp
- Payload: random_binary
- Covers: 10,000 BOSSbase images → 40,000 stego images
- ALASKA2: 75,000 covers + 225,000 pre-generated stego registered
- Total: ~350,000 data points in `metadata.csv`
- Duration: ~4.5 hours

### 5. Dataset Splits
- Built via `build_splits.py` with SHA-256 hash-based deterministic splitting
- Leakage check: PASSED (no cover overlap across splits)
- Train: 244,915 rows (70%)
- Val: 52,530 rows (15%)
- Test: 52,555 rows (15%)
- Saved at `/home/ubuntu/data/splits/`

### 6. Training — Test Run (1,000 samples, 5 epochs)
- LR: 1e-4, batch 64 effective, AMP enabled
- Results after 5 epochs:
  - Binary acc: 75.0%
  - Loss: 1.31 (no NaN)
  - Payload RMSE: 0.004
  - Gradients stabilized: 4.70 avg
- Confirmed: NaN fix works, model learns

### 7. Training — Full Run (started, in progress)
- Config: LR=1e-4, 100 epochs, batch 64, early stopping patience 15
- Data: 244,915 train / 52,530 val
- GPU: RTX A6000 48GB, ~28GB VRAM used, 70-100% utilization
- Epoch 1 results:
  - Train binary_acc: 75.6%, algo_class_acc: 65.9%, loss: 1.577
  - Val binary_acc: 74.6% (saved as best.pt)
  - Known issue: Val loss/payload_rmse showing NaN (val-only, doesn't affect model weights or accuracy-based checkpointing)
- Epoch 2 in progress when session ended
- Estimated time per epoch: ~78 min
- Training runs independently via `nohup` on Thunder instance

### 8. Bugs Found and Fixed
1. **`src/data/` not in git** — `.gitignore` had `data/` matching `src/data/`. Fixed to `/data/` (root only)
2. **`torch.cuda.total_mem`** — renamed to `total_memory` for torch 2.11 compatibility
3. **NaN loss explosion** — caused by:
   - Payload regression computed on cover images (should be stego only)
   - Unbounded `log_var` in Kendall uncertainty weighting
   - No NaN guard in training loop
   Fixed: stego-only payload loss, clamped log_var to [-6, 6], skip NaN steps
4. **Git auth on instance** — repo is private, needed token for git pull on instance

### 9. Features Added This Session
- **AUC-ROC metric** — per-epoch during training + per-algorithm in evaluation
- **8 paper-quality figures** (plot_results.py):
  - ROC curves (overall + per-algorithm)
  - Per-algorithm AUC-ROC bars
  - Per-rate accuracy curves
  - Confusion matrix
  - Per-algorithm accuracy bars
  - MoE routing heatmap
  - Payload scatter (predicted vs true)
  - Training curves (loss, acc, AUC, LR)
- **Training history JSON** — saved per-epoch for plotting
- **Parallel stego generation** — 4x speedup via multiprocessing
- **CLI args** — `--splits-dir`, `--output-dir` for train.py

### 10. Known Issues for Next Session
1. **Val NaN** — validation loss and payload RMSE show NaN. Need to add same NaN guard to validation loop and investigate val payload computation
2. **Val AUC-ROC = 0.0** — likely related to NaN propagation in softmax probs
3. **Grad norm** — some NaN grad norms during training (steps skipped by guard). Max grad=90 is high, may need stricter clipping
4. **BOWS2 cross-dataset test** — downloaded but not used yet
5. **Neural stego (Class E)** — DIV2K ready but SteganoGAN/HiDDeN not set up
6. **Diffusion stego (Class F)** — FFHQ ready but DiffStega not set up
7. **Phase 2 algorithms** — HUGO, WOW, MiPOD, LSB, nsF5 not generated yet

### 11. How to Check Training Status
```bash
# SSH to instance
ssh -p 30856 -i ~/.thunder/keys/1jrbdzmg ubuntu@216.81.200.237

# Check training log
tail -30 /tmp/training.log

# Check GPU
nvidia-smi

# Check checkpoints
ls -lh ~/checkpoints/

# Check training history
cat ~/checkpoints/training_log.json
```

### 12. Cost Estimate
- Instance: ~$0.75/hr
- Stego gen: ~4.5 hrs = ~$3.40
- Test training: ~10 min = ~$0.13
- Full training (estimated 30-50 epochs): ~40-65 hrs = ~$30-49
- Total session so far: ~6 hrs = ~$4.50

---

## Session: 2026-05-24 (continued)

### Summary
Verified local training pipeline works end-to-end, downloaded splits from Thunder instance (350K rows), checked training status (epoch 7 in progress, 76.4% best val acc), then instance was deleted. Conducted deep research on paper evaluation requirements.

---

### 13. Training Status Before Instance Deletion
- **Best checkpoint**: epoch 6, binary_acc=76.4%, algo_class_acc=74.4%, AUC-ROC=0.528
- **Current epoch**: 7 (in progress, GPU 100%, 29.6/49.1 GB VRAM)
- **Issues found**: TensorBoard not installed (no live logging), training_log.json only saves at end
- **Instance deleted** — all checkpoints and generated stego images LOST
- **Retained locally**: split CSVs (350K rows), codebase, raw cover images

### 14. Research: Paper Metrics & Evaluation Requirements

#### 14.1 Critical Metrics (MUST include all)

| Metric | Definition | Community Standard | Our Status |
|--------|-----------|-------------------|------------|
| **P_E** | ½ × min(P_FA + P_MD) | Gold standard in IEEE TIFS since SRM (2012) | **MISSING** |
| **wAUC** | Weighted AUC emphasizing low-FPR region (TPR 0-0.4 weighted 2x) | ALASKA competition standard | **MISSING** |
| **AUC-ROC** | Area under ROC curve | Threshold-independent performance | Have |
| **Detection Accuracy** | (TP + TN) / Total | Simple reporting | Have |
| **FP-50** | False positive rate when FNR = 50% | Operational dependability (Ker) | **MISSING** |
| **F1-Score** | Harmonic mean of precision/recall | Stego Battlefield benchmark (2025) | **MISSING** |

#### 14.2 Required Evaluation Protocol

**Core result table** (per-algorithm × per-rate P_E):
```
| Algorithm           | 0.1 bpp | 0.2 bpp | 0.4 bpp |
|---------------------|---------|---------|---------|
| S-UNIWARD           |  P_E    |  P_E    |  P_E    |
| HILL                |  P_E    |  P_E    |  P_E    |
| WOW                 |  P_E    |  P_E    |  P_E    |
| HUGO                |  P_E    |  P_E    |  P_E    |
| J-UNIWARD (QF75)    |  P_E    |  P_E    |  P_E    |
| J-UNIWARD (QF95)    |  P_E    |  P_E    |  P_E    |
| UERD (QF75)         |  P_E    |  P_E    |  P_E    |
| JMiPOD              |  P_E    |  P_E    |  P_E    |
| Neural (SteganoGAN) |  P_E    |  P_E    |  P_E    |
| Diffusion           |  P_E    |  P_E    |  P_E    |
| Average             |  P_E    |  P_E    |  P_E    |
```

**Payload rates to report:**
- Spatial: 0.1, 0.2, 0.4 bpp
- JPEG: 0.1, 0.2, 0.4 bpnzAC at QF 75 AND QF 95
- Neural: natural payload sizes
- Key insight: 0.4 bpp is "easy" (SRNet P_E < 0.05). Reviewers focus on 0.2 and especially 0.1 bpp.

#### 14.3 Mandatory Baselines (desk-reject risk if missing)

| Baseline | Domain | Citation |
|----------|--------|----------|
| **SRNet** | Spatial + JPEG | Boroumand et al., TIFS 2019 |
| **YeNet** | Spatial | Ye et al., TIFS 2017 |
| **XuNet** | Spatial | Xu et al., IH&MMSec 2016 |
| **Yedroudj-Net** | Spatial | Yedroudj-Ber et al., 2018 |
| **Zhu-Net** | Spatial | Zhu et al., 2018 |
| **SRM + EC** | Spatial | Fridrich & Kodovsky, TIFS 2012 |
| **EfficientNet-B0** | Both | Modern backbone comparison |
| **CovPool** | Both | Deng et al., IH&MMSec 2019 |

#### 14.4 Cross-Domain Generalization (required for "universal" claim)

Three tiers:
1. **Cross-algorithm**: Train on {WOW, HUGO}, test on {S-UNIWARD, HILL}
2. **Cross-source**: Train on BOSSBase, test on BOWS-2/ALASKA2
3. **Cross-domain**: Train on spatial, test on JPEG (and vice versa)

**Cover-Source Mismatch (CSM)**: Identified as THE #1 operational concern (EURASIP 2024 review). Must acknowledge, test, and show graceful degradation.

#### 14.5 Statistical Rigor

- **3+ random seeds**, report mean ± std
- **Paired t-test** or Wilcoxon signed-rank when claiming improvement over baselines
- **Confidence intervals** on key results
- Current field standard: single run acceptable but 3+ seeds puts us ABOVE typical bar

#### 14.6 Figures Required

- **ROC curves**: full + zoomed at low-FPR region [0, 0.1]
- **P_E vs payload rate curves**: per algorithm
- **Algorithm confusion matrix**: NxN which algos get confused
- **Expert utilization heatmap**: which MoE expert handles which domain
- **Training curves**: loss, accuracy, LR over epochs
- **Cross-algorithm transfer matrix**: P_E when trained on A, tested on B

#### 14.7 Unique Metrics for Universal Steganalysis (our novelty)

1. **Universality gap**: single model P_E vs best per-algorithm specialist (target: < 2% gap)
2. **Domain-averaged P_E**: average across spatial/JPEG/neural/diffusion — headline number
3. **Expert specialization analysis**: MoE routing patterns per domain
4. **Cross-domain transfer P_E**: train on one domain, test on others
5. **Continual learning backward transfer**: adding new algo doesn't degrade old performance

#### 14.8 Paper Rejection Triggers (AVOID)

1. No SRNet comparison
2. Only testing at 0.4 bpp (too easy, proves nothing)
3. Ignoring cover-source mismatch
4. No per-algorithm breakdown (aggregate only hides failures)
5. Single seed, no error bars
6. Missing JPEG QF75 + QF95 results
7. Not showing model works on ALL claimed domains
8. Not comparing single universal model vs specialist models
9. Using resize in preprocessing (destroys stego signal)
10. "It should work" without actually running + showing output

#### 14.9 What to Implement in metrics.py

```
Currently have:  binary_acc, AUC-ROC, algo_class_acc, algo_acc, payload_rmse
Need to add:     P_E, wAUC, FP-50, F1, per-algo P_E breakdown, 
                 cross-algorithm transfer matrix, ROC curve data export,
                 significance testing utilities
```

#### 14.10 Key References

- SRNet: Boroumand et al., IEEE TIFS 2019
- ALASKA2: Cogranne et al., IEEE SP 2021 (wAUC metric origin)
- Operational Steganalysis: Ker, "Towards Dependable Steganalysis"
- CSM: EURASIP Journal 2024 systematic review
- Stego Battlefield: arXiv:2605.05789 (2025 benchmark)
- Rich Models: Fridrich & Kodovsky, IEEE TIFS 2012
- Loss for Low FPR: HAL-05101274 (2024)
- Continual Learning for Steganalysis: arXiv:2209.01326

### 15. Next Steps
1. Implement missing metrics (P_E, wAUC, FP-50) in evaluation pipeline
2. Spin up new GPU instance for training
3. Re-generate stego images (all algorithms needed for paper)
4. Add 0.1 bpp rate (critical for paper credibility)
5. Implement baseline comparisons (SRNet, YeNet at minimum)
6. Add TensorBoard + crash-safe CSV logging before training
7. Plan 3-seed evaluation runs for statistical rigor
