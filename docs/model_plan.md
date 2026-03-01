# Plan: Multimodal Transformer Fusion Model

## Context

The preprocessing pipeline (data loader → feature extractor → LOSO windowing) is complete, producing 7-day sliding windows with 121 features/day across 5 modalities. The experiment dispatch infrastructure (`executor.py` + `run.sh` + `submit_slurm.sh`) is ready but currently wired to a placeholder voting simulation. We need to implement the actual model layer: PyTorch Dataset, transformer architecture, training loop, and experiment config.

## Files to Create

### 1. `src/dataset.py` — PyTorch Dataset for LOSO folds

- `RelapseDataset(Dataset)` — loads a fold pickle (`fold_N/{train,val,test}.pkl`), returns per-sample dicts with per-modality feature tensors, masks, and last-day label
- `collate_fn()` — custom collate that stacks the dict-based samples into batched tensors
- `MODALITY_ORDER` list — canonical ordering: `["accel", "gyr", "hr", "step", "sleep"]`
- Label handling: clamp unknown labels (-1) to 0, track validity via `label_valid` bool
- Only extracts the **last day's label** (the window_end_day prediction target)

### 2. `src/model.py` — Multimodal Transformer Fusion Architecture

Three-stage architecture (~300K parameters with default hyperparameters):

**Stage 1 — `ModalityEncoder`** (one per modality, independent weights):
- Projects raw features to `d_model` via Linear + LayerNorm + GELU
- Prepends a learnable CLS token + adds positional embeddings
- Runs a small `TransformerEncoder` (pre-norm, GELU, `batch_first=True`)
- Masks attention using combined `padding_mask & modality_mask` (inverted for PyTorch convention)
- Returns the CLS token output as a `(B, d_model)` modality summary

**Stage 2 — `FusionTransformer`**:
- Receives 5 modality tokens `(B, 5, d_model)`, adds learnable modality-type embeddings
- Masks unavailable modalities (modality has no data for ANY day in the window)
- Runs a `TransformerEncoder` over the 5 tokens
- Mean-pools over available modalities → `(B, d_model)`

**Stage 3 — `MultimodalRelapseTransformer`** (top-level module):
- `nn.ModuleDict` of 5 `ModalityEncoder` instances (keyed by modality name)
- One `FusionTransformer`
- Classification head: `Linear → GELU → Dropout → Linear(1)` → scalar logit

### 3. `src/train.py` — Training Script (Click CLI)

Follows the existing `src/experiment.py` pattern for executor integration:
- **Executor-injected params**: `--exp_name`, `--output_dir`, `--output_filename`
- **Experiment params**: `--fold`, `--d_model`, `--nhead`, `--num_encoder_layers`, `--num_fusion_layers`, `--dropout`, `--lr`, `--weight_decay`, `--batch_size`, `--epochs`, `--patience`, `--loss_fn`, `--focal_alpha`, `--focal_gamma`, `--seed`, `--threshold`

Training logic:
- **Loss**: `weighted_bce` (default, auto-computes `pos_weight = n_neg/n_pos`) or `focal` (with tunable alpha/gamma). `FocalLoss` class defined inline.
- **Optimizer**: AdamW with weight decay + CosineAnnealingLR scheduler
- **Early stopping** on val AUROC with configurable patience
- **Gradient clipping** at `max_norm=1.0`
- **Label masking**: only samples with `label_valid=True` contribute to loss
- **Evaluation**: AUROC, AUPRC, F1, precision, recall, accuracy (via scikit-learn)
- **Output**: JSON with all hyperparameters, val/test metrics, and full training history

### 4. `configs/train.json` — Experiment Config

```json
{
    "fold": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "d_model": 64, "nhead": 4,
    "num_encoder_layers": 1, "num_fusion_layers": 1,
    "dropout": 0.1, "lr": 1e-4, "weight_decay": 0.01,
    "batch_size": 32, "epochs": 100, "patience": 15,
    "loss_fn": "weighted_bce", "seed": 42, "threshold": 0.5,
    "executor": { "exec_name": "train", ... },
    "slurm": { "gres": "gpu:1", "memory": "16G", "time": "0-02:00", ... }
}
```

Any scalar param can be made into a list for hyperparameter sweeps (executor generates Cartesian product).

### 5. `requirements.txt` — Add `scikit-learn`

Needed for `roc_auc_score`, `average_precision_score`, `f1_score`, etc.

## Implementation Order

1. Add `scikit-learn` to `requirements.txt`
2. Create `src/dataset.py`
3. Create `src/model.py`
4. Create `src/train.py`
5. Create `configs/train.json`

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Per-modality encoders | Independent transformer per modality | Handles heterogeneous feature dims (38 vs 9); naturally masks missing sensors |
| Fusion method | Transformer + mean pooling | Cross-attention learns inter-modality relationships; mean pool robust to missing modalities |
| Model size | ~300K params (d_model=64) | Small dataset (9 patients) → small model to avoid overfitting |
| Pre-norm transformers | `norm_first=True` | More stable training with limited data |
| Early stopping metric | Val AUROC | Threshold-agnostic; appropriate for imbalanced binary classification |
| Loss function | Weighted BCE (default) | Auto-computes class weights from training data; simpler than focal loss |

## Critical Existing Files

- `src/feature_extractor.py:31-37` — `MODALITY_DIMS` dict (reused by dataset and model)
- `src/preprocess_loso.py:1-22` — Window dict schema (dataset must match this format)
- `src/experiment.py` — Click CLI pattern to follow for train.py
- `src/executor.py:56-66` — How CLI args are constructed (train.py params must be compatible)
- `configs/experiment.json` — Reference config structure for train.json

## Verification

1. **Unit test dataset**: Load a fold pickle, iterate `RelapseDataset`, verify tensor shapes match `MODALITY_DIMS`
2. **Unit test model**: Forward pass with random tensors, verify output shape `(B,)` for batch of B
3. **Local single-fold run**: `bash scripts/run.sh -n train` (after setting `fold` to `[0]` in config)
4. **Full LOSO run**: `bash scripts/submit_slurm.sh -n train` (all 9 folds on SLURM)
5. **Check outputs**: `outputs/train__fold=*.json` should contain val/test metrics for each fold
