# Plan: Experiment Framework & Research-Driven Improvement Roadmap

## Context

The current multimodal relapse prediction system uses a transformer fusion architecture (per-modality encoders + fusion transformer + MLP classifier, ~316K params) trained with LOSO CV on 9 patients. **The model severely overfits**: val AUROC 0.61–0.86 but test AUROC 0.15–0.77 (excluding the anomalous fold 8 with perfect test scores). F1=0 in most folds because the model never predicts "relapse" at threshold 0.5. Best epoch is typically 1-5, meaning the model starts overfitting almost immediately.

Root causes: (1) too many parameters for N=9 patients, (2) no patient-invariance regularization, (3) suboptimal fusion strategy, (4) no threshold calibration, (5) limited augmentation.

**Goal**: Restructure the project for modular experiment management, then implement a series of transformer-focused improvements ranked by expected impact.

---

## Step 1: Merge baseline & restructure for modular experiments

### 1a. Merge `model-implementation` → `main`

The training code (model.py, dataset.py, train.py, configs/train.json) currently lives only on `model-implementation` (3 commits ahead). Merge it to establish the baseline on main.

### 1b. Restructure `src/` for modularity

Reorganize so that model variants, fusion strategies, and loss functions are swappable via config:

```
src/
├── models/
│   ├── __init__.py          # Registry: MODEL_REGISTRY["transformer_v1"] → class
│   ├── base.py              # Abstract base class with shared interface
│   ├── transformer_v1.py    # Current model (moved from model.py, unchanged)
│   ├── transformer_v2.py    # Experiment 1: bottleneck fusion (new)
│   └── transformer_v3.py    # Experiment 2: gated fusion + DANN (new)
├── losses/
│   ├── __init__.py          # LOSS_REGISTRY["focal"] → class
│   ├── weighted_bce.py      # Current (extracted from train.py)
│   └── focal.py             # Current (extracted from train.py)
├── dataset.py               # Unchanged
├── train.py                 # Modified: reads `model_name` from config, uses registry
├── feature_extractor.py     # Unchanged (enhanced in later steps)
├── preprocess_loso.py       # Unchanged
├── data_loader.py           # Unchanged
├── executor.py              # Unchanged
└── utils.py                 # Unchanged
```

### 1c. Config-driven experiment selection

Extend `configs/train.json` pattern. Each experiment is a new config file that selects its model variant:

```
configs/
├── train.json               # Baseline: transformer_v1 (preserved, never modified)
├── exp_bottleneck.json      # Experiment: transformer_v2 with bottleneck fusion
├── exp_gated_dann.json      # Experiment: transformer_v3 with gated fusion + DANN
├── exp_focal.json           # Experiment: transformer_v1 with focal loss
├── exp_augmented.json       # Experiment: transformer_v1 with data augmentation
└── preprocessing.json       # Unchanged
```

The config gains a `model_name` field (default: `"transformer_v1"` for backward compat). `train.py` uses `MODEL_REGISTRY[model_name]` to instantiate.

**Key principle**: `configs/train.json` is NEVER modified. It always reproduces the baseline.

### Critical files to modify:
- `src/model.py` → move to `src/models/transformer_v1.py` (rename only)
- `src/train.py` → add `model_name` CLI option, use registry lookup
- New: `src/models/__init__.py`, `src/models/base.py`
- New: `src/losses/__init__.py`

---

## Step 2: Threshold calibration (Immediate F1 fix)

**Why**: F1=0 across almost all folds because threshold=0.5 is too high for a 1:3 class imbalance. The model's ranking (AUROC) is often decent but hard predictions are useless.

**What**: Add post-training threshold sweep in `train.py`:
- After restoring best model, sweep thresholds [0.05, 0.10, ..., 0.95] on validation set
- Select threshold maximizing F1 on validation
- Report test metrics at both default (0.5) and calibrated threshold
- Save calibrated threshold in output JSON

**Config addition**: `"calibrate_threshold": true` (default false for backward compat)

**Files**: `src/train.py` only

---

## Step 3: Attention Bottleneck Fusion (Experiment 1)

**Why**: The current fusion transformer allows full pairwise attention across 5 modality tokens. With 5 modalities this is manageable parameter-wise, but the bottleneck forces the model to compress cross-modal information through a small number of latent tokens, acting as a structural regularizer.

**What**: Create `src/models/transformer_v2.py` — identical to v1 except:
- Replace `FusionTransformer` with `BottleneckFusion`
- Add `num_bottleneck_tokens` learnable latent tokens (default: 4)
- Modality tokens attend to bottleneck tokens via cross-attention
- Bottleneck tokens attend to each other via self-attention
- Final representation = mean-pool of bottleneck tokens

**Config**: `configs/exp_bottleneck.json` with `"model_name": "transformer_v2"`, `"num_bottleneck_tokens": 4`

**New files**: `src/models/transformer_v2.py`
**Reuses**: `ModalityEncoder` from `transformer_v1.py` (import directly)

---

## Step 4: Gated Fusion + Domain Adversarial Training (Experiment 2)

**Why**: Research shows domain adversarial training (DANN) is highly effective for LOSO generalization. Gated fusion adaptively weights modalities, which is especially valuable when sensors have missing data.

**What**: Create `src/models/transformer_v3.py`:
- Replace `FusionTransformer` with `GatedFusion`: learned sigmoid gates per modality, weighted sum
- Add `PatientDiscriminator`: small MLP head that predicts patient ID (9-class) from fused features
- Add `GradientReversalLayer` between encoder output and patient discriminator
- Training loss: `L = L_bce + lambda * L_adversarial` (lambda annealed 0→0.3 over first 20 epochs)

**Config**: `configs/exp_gated_dann.json` with `"model_name": "transformer_v3"`, `"dann_lambda": 0.3`, `"dann_warmup_epochs": 20`

**Requires**: Patient ID to be available in each batch → modify `dataset.py` to include patient index, modify `collate_fn`

**New files**: `src/models/transformer_v3.py`
**Modified files**: `src/dataset.py` (add patient_id to samples), `src/train.py` (DANN loss branch)

---

## Step 5: Data augmentation (can combine with any experiment)

**Why**: With ~2900 training windows from 8 patients, augmentation effectively multiplies the training set. Research shows jittering, magnitude warping, and mixup are most effective for wearable time-series.

**What**: Add augmentation to `RelapseDataset`:
- **Jittering**: Add Gaussian noise (σ=0.05) to features
- **Magnitude warping**: Multiply by smooth random curve (cubic spline with σ=0.1)
- **Cross-patient mixup**: Interpolate features and labels between two random samples (alpha from Beta(0.2, 0.2))

**Config additions**: `"augmentation": "none"` (default) / `"jitter"` / `"warp"` / `"mixup"` / `"all"`

**Files**: `src/dataset.py` (add augmentation logic, controlled by flag passed from train.py)

---

## Step 6: Focal loss + increased dropout (low-hanging fruit)

**Why**: Focal loss down-weights easy negatives, forcing the model to focus on the hard relapse boundary. Higher dropout (0.3 vs 0.1) provides additional regularization.

**What**: Already implemented as `FocalLoss` in train.py. Just needs configs:
- `configs/exp_focal.json`: `"loss_fn": "focal"`, `"focal_alpha": 0.75`, `"focal_gamma": 2.0`, `"dropout": 0.3`
- Also try smaller model: `"d_model": 32`, `"nhead": 2`

**Files**: New config only (no code changes needed — focal loss already exists in train.py)

---

## Step 7: Feature engineering improvements

**Why**: The current 7-day rolling deviation pattern (only implemented for sleep onset/wake times) is the most patient-invariant feature type. Extending it to all modalities should improve cross-patient generalization. Research shows circadian features predict mood episodes with AUC≥0.94.

**What**: Extend `FeatureExtractor` to add:
- **Temporal differences**: day-over-day change (Δ) for all 121 features
- **Rolling deviation**: 7-day rolling mean deviation for key features (HR mean, step count, sleep duration, accel energy)
- **Cross-modality ratios**: HR_mean/total_steps, sleep_duration×HR_rmssd

This changes feature dimensions, so it requires re-running preprocessing. The new features would be additive (existing features unchanged, new features appended).

**Config**: `configs/preprocessing_v2.json` with `"extended_features": true`
**Files**: `src/feature_extractor.py` (add methods), `src/preprocess_loso.py` (pass flag)

---

## Experiment Priority Order

| Priority | Experiment | Expected Impact | Effort | Config |
|----------|-----------|----------------|--------|--------|
| 1 | Threshold calibration | High (F1: 0→meaningful) | Low | train.json + flag |
| 2 | Focal loss + smaller model + higher dropout | Medium-High | Very Low | exp_focal.json |
| 3 | Attention bottleneck fusion | High | Medium | exp_bottleneck.json |
| 4 | Data augmentation (mixup) | Medium-High | Medium | exp_augmented.json |
| 5 | Gated fusion + DANN | High | Medium-High | exp_gated_dann.json |
| 6 | Extended features | Medium | Medium | preprocessing_v2.json |

Each experiment is independent and can be run via:
```bash
bash scripts/submit_slurm.sh -n exp_bottleneck   # runs all 9 LOSO folds
bash scripts/submit_slurm.sh -n exp_focal         # etc.
```

---

## Verification

1. **Baseline preservation**: Run `bash scripts/run.sh -n train` on fold 0 and confirm output matches existing `outputs/train__fold=0.json` exactly
2. **New experiments**: Each new config should produce outputs named `outputs/{exp_name}__fold=N.json` — no collision with baseline
3. **Regression test**: After restructuring, all existing imports (`from src.model import ...`) must still work (handled via re-exports in `__init__.py`)
4. **Metric comparison**: Compare val/test AUROC, AUPRC, and calibrated-F1 across experiments per fold

---

## Summary of All Files to Create/Modify

**New files:**
- `src/models/__init__.py` — model registry
- `src/models/base.py` — abstract base class
- `src/models/transformer_v1.py` — current model (moved from model.py)
- `src/models/transformer_v2.py` — bottleneck fusion variant
- `src/models/transformer_v3.py` — gated fusion + DANN variant
- `src/losses/__init__.py` — loss registry
- `src/losses/weighted_bce.py` — extracted from train.py
- `src/losses/focal.py` — extracted from train.py
- `configs/exp_bottleneck.json`
- `configs/exp_focal.json`
- `configs/exp_gated_dann.json`
- `configs/exp_augmented.json`

**Modified files:**
- `src/train.py` — model registry lookup, threshold calibration, DANN loss, augmentation flag
- `src/dataset.py` — patient_id in samples, augmentation support
- `src/feature_extractor.py` — extended features (Step 7)

**Preserved unchanged:**
- `configs/train.json` — baseline config, never modified
- `src/data_loader.py`, `src/preprocess_loso.py`, `src/executor.py`, `src/utils.py`
- All existing outputs in `outputs/`
