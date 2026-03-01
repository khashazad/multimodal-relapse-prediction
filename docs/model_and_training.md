# Model Architecture and Training Pipeline

Comprehensive reference documentation for the Multimodal Transformer Fusion model used in Track 1 non-psychotic relapse detection.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset (`src/dataset.py`)](#2-dataset)
3. [Model Architecture (`src/model.py`)](#3-model-architecture)
4. [Training Pipeline (`src/train.py`)](#4-training-pipeline)
5. [Configuration (`configs/train.json`)](#5-configuration)
6. [Parameter Count Breakdown](#6-parameter-count-breakdown)
7. [Design Decisions](#7-design-decisions)
8. [Known Issues and Edge Cases](#8-known-issues-and-edge-cases)

---

## 1. Overview

### Prediction Task

The model performs **binary per-day relapse classification** (relapse=1 vs. stable=0) for 9 psychiatric patients (P1--P9) using 5 wearable sensor modalities. Each input sample is a sliding window of W consecutive days (default W=7). The model predicts the **last day** in each window.

### Modalities

| Modality | Key       | Features/Day | Description |
|----------|-----------|-------------|-------------|
| Accelerometer | `accel` | 38 | Statistical, activity-level, and frequency-domain features from linear acceleration |
| Gyroscope     | `gyr`   | 38 | Identical feature pipeline applied to gyroscope data |
| Heart Rate    | `hr`    | 26 | HR/RR statistics, HRV time/frequency domain, Poincare, coverage fraction |
| Steps         | `step`  | 10 | Daily aggregates (steps, distance, calories) and temporal pattern features |
| Sleep         | `sleep` |  9 | Duration, episode count, circular-encoded timing, rolling deviation |

Total features per day: 121 (38 + 38 + 26 + 10 + 9).

### LOSO Evaluation Setup

Leave-One-Subject-Out cross-validation with 9 folds. For fold N (test patient = P_k):

- **Train**: ALL sequences (train\_\*, val\_\*, test\_\*) from the remaining 8 patients. train\_\* sequences are treated as stable (label=0); val\_\*/test\_\* sequences use their actual relapse labels.
- **Val**: val\_\* sequences from P_k only.
- **Test**: test\_\* sequences from P_k only.

Each fold trains an independent model from scratch. Final performance is reported as a per-fold metric table plus cross-fold aggregates.

---

## 2. Dataset

**File**: `src/dataset.py`

### `MODALITY_ORDER`

```python
MODALITY_ORDER: List[str] = ["accel", "gyr", "hr", "step", "sleep"]
```

This canonical ordering is used consistently throughout the model and dataset. The fusion transformer receives modality tokens in this exact order.

### `RelapseDataset`

```python
class RelapseDataset(Dataset):
    def __init__(self, pkl_path: str | Path) -> None
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
```

**Initialization**: Loads a single pickle file (e.g., `data/processed/track1/fold_0/train.pkl`) containing a list of window dicts produced by `LOSOPreprocessor`.

**Sample structure**: Each call to `__getitem__` returns a dict with the following keys:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `accel_features` | `(W, 38)` | `float32` | Accelerometer features for each day in the window |
| `accel_mask` | `(W,)` | `bool` | True where the accelerometer had data that day |
| `gyr_features` | `(W, 38)` | `float32` | Gyroscope features |
| `gyr_mask` | `(W,)` | `bool` | Gyroscope data availability mask |
| `hr_features` | `(W, 26)` | `float32` | Heart rate features |
| `hr_mask` | `(W,)` | `bool` | Heart rate data availability mask |
| `step_features` | `(W, 10)` | `float32` | Step features |
| `step_mask` | `(W,)` | `bool` | Step data availability mask |
| `sleep_features` | `(W, 9)` | `float32` | Sleep features |
| `sleep_mask` | `(W,)` | `bool` | Sleep data availability mask |
| `padding_mask` | `(W,)` | `bool` | True = real day, False = left-padding position |
| `label` | scalar | `float32` | Last-day relapse label (0.0 or 1.0) |
| `label_valid` | scalar | `bool` | Whether the label is valid (known) |

**Label extraction logic**: The dataset extracts the label from the **last position** of the window's `labels` array:

```python
last_label = int(w["labels"][-1])
sample["label"] = torch.tensor(max(last_label, 0), dtype=torch.float32)
sample["label_valid"] = torch.tensor(bool(w["label_mask"][-1]), dtype=torch.bool)
```

Unknown labels (value -1 in the preprocessed data, from test\_\* sequences with no `relapse` column) are clamped to 0, but `label_valid` is set to False so these samples are excluded from loss computation and evaluation metrics.

### Window Dict Schema (from `preprocess_loso.py`)

The dataset reads pickles containing lists of dicts with this schema:

```python
{
    'patient_id':     str,               # e.g. 'P1'
    'sequence_name':  str,               # e.g. 'train_0', 'val_1', 'test_0'
    'window_end_day': int,               # the day being predicted (last in window)
    'window_days':    List[int],         # day indices; -1 = left-padded position
    'features':       Dict[str, np.ndarray],  # {modality: (W, F_mod) float32}
    'modality_masks': Dict[str, np.ndarray],  # {modality: (W,) bool}
    'padding_mask':   np.ndarray,        # (W,) bool  — True = real day
    'labels':         np.ndarray,        # (W,) int32 — 0=stable, 1=relapse, -1=unknown
    'label_mask':     np.ndarray,        # (W,) bool  — True = label valid
}
```

### `collate_fn`

```python
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]
```

Simple dict-based collation: for each key present in the sample dicts, stacks the corresponding tensors along a new batch dimension using `torch.stack`. This produces batched tensors of shape `(B, W, F)` for features, `(B, W)` for masks, `(B,)` for labels, etc.

---

## 3. Model Architecture

**File**: `src/model.py`

The model is a three-stage pipeline:

```
Input (5 modalities) --> [ModalityEncoder x5] --> [FusionTransformer] --> [Classification Head] --> logit
```

### 3.1 Stage 1: ModalityEncoder

```python
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=1, dropout=0.1, max_seq_len=8)
    def forward(self, x, padding_mask, modality_mask) -> torch.Tensor
```

**Purpose**: Encodes a single modality's temporal feature sequence into a fixed-size summary vector. There are 5 independent ModalityEncoder instances (one per modality), each with its own learned weights. They are stored in an `nn.ModuleDict` keyed by modality name.

**Why independent encoders**: Each modality has a different feature dimensionality (38 vs. 26 vs. 10 vs. 9), different statistical properties, and different missing-data patterns. Independent encoders allow each modality to learn its own temporal representation without interference. The shared `d_model` output dimension provides a uniform interface for fusion.

#### Input Projection

```python
self.proj = nn.Sequential(
    nn.Linear(input_dim, d_model),    # (B, W, input_dim) -> (B, W, d_model)
    nn.LayerNorm(d_model),
    nn.GELU(),
)
```

Maps modality-specific feature dimensions to a common `d_model`-dimensional space. LayerNorm stabilizes the projected features. GELU provides nonlinearity.

Tensor shape: `(B, W, input_dim)` --> `(B, W, d_model)`

#### CLS Token Mechanism

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
```

A learnable token prepended to the sequence. After the transformer processes the sequence, the CLS token's output serves as a fixed-size summary of the entire modality time series.

```python
cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
x = torch.cat([cls, x], dim=1)          # (B, W+1, d_model)
```

Initialization scale of 0.02 follows the convention from BERT/ViT to keep initial token representations small.

#### Positional Embeddings

```python
self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
```

Learnable absolute positional embeddings. `max_seq_len = window_size + 1` (e.g., 8 for a 7-day window) to accommodate the CLS token at position 0 followed by W day positions.

```python
x = x + self.pos_embed[:, :W+1, :]
```

Only the first `W+1` positions are used (supports variable-length windows up to `max_seq_len`).

#### Transformer Encoder

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,           # Hidden dimension: 64
    nhead=nhead,               # Attention heads: 4 (head_dim = 16)
    dim_feedforward=d_model*4, # FFN width: 256
    dropout=dropout,           # 0.1
    activation="gelu",         # GELU instead of ReLU
    batch_first=True,          # Input shape: (B, seq_len, d_model)
    norm_first=True,           # Pre-norm: LayerNorm before attention/FFN
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

Configuration details:
- **Pre-norm** (`norm_first=True`): Applies LayerNorm before attention and FFN sublayers rather than after. This provides more stable gradients during training with limited data.
- **GELU activation**: Used in both the FFN and the input projection. Smoother than ReLU, standard in modern transformers.
- **`batch_first=True`**: Tensors are `(B, seq_len, d_model)` throughout, avoiding transpose operations.
- **`dim_feedforward = 4 * d_model`**: Standard 4x expansion ratio in the position-wise FFN.

#### Attention Masking

The masking logic combines two sources of invalidity:

```python
# CLS token is always valid
cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
combined_mask = padding_mask & modality_mask  # (B, W)
combined_mask = torch.cat([cls_valid, combined_mask], dim=1)  # (B, W+1)

# PyTorch convention: True = IGNORE in src_key_padding_mask
attn_mask = ~combined_mask  # (B, W+1)
```

A position is **valid** (attends and is attended to) only if:
1. `padding_mask[t] == True` (position is a real day, not left-padding), AND
2. `modality_mask[t] == True` (this modality's sensor had data on that day)

The CLS token at position 0 is always valid.

**PyTorch inverted convention**: `src_key_padding_mask` uses `True` to mean "ignore this position". The code inverts the combined mask via `~` to comply with this convention.

#### CLS Extraction

```python
return x[:, 0, :]  # (B, d_model)
```

Returns the CLS token output (position 0) as the modality summary vector. Through self-attention, the CLS token has aggregated information from all valid day positions.

#### Full Forward Pass Shape Trace

For a batch of size B with window size W=7:

```
Input:    x             (B, 7, F_mod)     -- F_mod varies by modality
          padding_mask  (B, 7)
          modality_mask (B, 7)

proj:     x             (B, 7, 64)        -- after Linear+LN+GELU

CLS:      cls           (B, 1, 64)
concat:   x             (B, 8, 64)        -- CLS prepended

pos_embed:x             (B, 8, 64)        -- positional embeddings added

mask:     attn_mask     (B, 8)            -- combined & inverted

encoder:  x             (B, 8, 64)        -- transformer output

output:   cls_out       (B, 64)           -- CLS token extracted
```

### 3.2 Stage 2: FusionTransformer

```python
class FusionTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=1, num_modalities=5, dropout=0.1)
    def forward(self, tokens, modality_available) -> torch.Tensor
```

**Purpose**: Fuses the 5 modality summary tokens via cross-modal self-attention, producing a single patient-level representation.

**Why this fusion approach**: Rather than simple concatenation or averaging, a transformer over modality tokens enables the model to learn inter-modality relationships (e.g., how sleep disruption correlates with changes in heart rate variability or activity patterns). Mean-pooling over available modalities makes the fusion robust to missing modalities -- if a sensor was not worn, its token is masked out rather than contributing noise.

#### Modality-Type Embeddings

```python
self.modality_embed = nn.Parameter(torch.randn(1, num_modalities, d_model) * 0.02)
```

Learnable embeddings that encode modality identity. Shape: `(1, 5, 64)`. Each of the 5 positions gets a different learned vector added to its CLS summary token. This tells the fusion transformer which modality each token represents (analogous to segment embeddings in BERT or modality embeddings in multimodal transformers).

```python
tokens = tokens + self.modality_embed  # (B, 5, 64)
```

#### Modality Availability Masking

```python
attn_mask = ~modality_available  # (B, M) -- PyTorch True=ignore convention
tokens = self.encoder(tokens, src_key_padding_mask=attn_mask)
```

`modality_available` is a `(B, M)` boolean tensor where `True` means the modality had data for at least one day in the window. This is computed in the top-level module:

```python
avail = mod_mask.any(dim=1)  # (B,) -- True if ANY day in window has data
```

A modality is fully masked out of fusion attention if its sensor recorded no data for any day in the entire window.

#### Transformer Encoder

Same architecture as the modality encoder's transformer:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

Operates over a sequence of length 5 (one token per modality).

#### Mean-Pooling

```python
avail = modality_available.unsqueeze(-1).float()  # (B, M, 1)
n_avail = avail.sum(dim=1).clamp(min=1.0)         # (B, 1)
fused = (tokens * avail).sum(dim=1) / n_avail      # (B, d_model)
```

Averages only over available modality tokens. The `clamp(min=1.0)` prevents division by zero when all modalities are missing (defensive; should not happen in practice).

#### Full Forward Pass Shape Trace

```
Input:    tokens             (B, 5, 64)   -- stacked CLS outputs
          modality_available (B, 5)       -- bool availability

embed:    tokens             (B, 5, 64)   -- + modality_embed
mask:     attn_mask          (B, 5)       -- inverted availability
encoder:  tokens             (B, 5, 64)   -- transformer output
pool:     fused              (B, 64)      -- mean over available modalities
```

### 3.3 Stage 3: Classification Head

```python
self.classifier = nn.Sequential(
    nn.Linear(d_model, d_model),   # (B, 64) -> (B, 64)
    nn.GELU(),
    nn.Dropout(dropout),           # p=0.1
    nn.Linear(d_model, 1),         # (B, 64) -> (B, 1)
)
```

A two-layer MLP that maps the fused representation to a single raw logit:

1. `Linear(64, 64)` -- hidden layer maintains dimension
2. `GELU` -- nonlinear activation
3. `Dropout(0.1)` -- regularization to prevent overfitting
4. `Linear(64, 1)` -- single output logit

The output is squeezed to shape `(B,)`:

```python
logits = self.classifier(fused).squeeze(-1)  # (B,)
```

**Design rationale**: The head outputs a raw logit (not a probability). Sigmoid is applied externally: by `BCEWithLogitsLoss` during training (for numerical stability) and explicitly via `torch.sigmoid()` during evaluation. This two-layer design adds a small amount of nonlinear capacity beyond what the fusion transformer provides.

### 3.4 Top-Level Module: MultimodalRelapseTransformer

```python
class MultimodalRelapseTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=1,
                 num_fusion_layers=1, dropout=0.1, window_size=7)
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor
```

Wires together all components.

#### Constructor

```python
# 5 independent modality encoders
self.encoders = nn.ModuleDict({
    mod: ModalityEncoder(
        input_dim=MODALITY_DIMS[mod],       # 38, 38, 26, 10, or 9
        d_model=d_model,
        nhead=nhead,
        num_layers=num_encoder_layers,
        dropout=dropout,
        max_seq_len=window_size + 1,        # +1 for CLS token
    )
    for mod in MODALITY_ORDER
})

# Fusion transformer
self.fusion = FusionTransformer(
    d_model=d_model, nhead=nhead,
    num_layers=num_fusion_layers,
    num_modalities=len(MODALITY_ORDER),     # 5
    dropout=dropout,
)

# Classification head
self.classifier = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
)
```

#### Forward Pass Flow

```python
def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
```

Step-by-step:

1. **Extract padding mask** from batch: `padding_mask = batch["padding_mask"]` shape `(B, W)`

2. **Encode each modality independently**:
   ```python
   for mod in MODALITY_ORDER:  # ["accel", "gyr", "hr", "step", "sleep"]
       features = batch[f"{mod}_features"]    # (B, W, F_mod)
       mod_mask = batch[f"{mod}_mask"]         # (B, W)
       cls_out = self.encoders[mod](features, padding_mask, mod_mask)  # (B, d_model)
       modality_tokens.append(cls_out)
       avail = mod_mask.any(dim=1)             # (B,)
       modality_available.append(avail)
   ```

3. **Stack modality tokens**: `tokens = torch.stack(modality_tokens, dim=1)` shape `(B, 5, d_model)`

4. **Stack availability**: `avail_mask = torch.stack(modality_available, dim=1)` shape `(B, 5)`

5. **Fuse modalities**: `fused = self.fusion(tokens, avail_mask)` shape `(B, d_model)`

6. **Classify**: `logits = self.classifier(fused).squeeze(-1)` shape `(B,)`

#### Complete Shape Trace (B=32, W=7, d_model=64)

```
batch["accel_features"]   (32, 7, 38)   -->  ModalityEncoder["accel"]  -->  (32, 64)
batch["gyr_features"]     (32, 7, 38)   -->  ModalityEncoder["gyr"]    -->  (32, 64)
batch["hr_features"]      (32, 7, 26)   -->  ModalityEncoder["hr"]     -->  (32, 64)
batch["step_features"]    (32, 7, 10)   -->  ModalityEncoder["step"]   -->  (32, 64)
batch["sleep_features"]   (32, 7,  9)   -->  ModalityEncoder["sleep"]  -->  (32, 64)

torch.stack(dim=1)                       -->  tokens         (32, 5, 64)
torch.stack(dim=1)                       -->  avail_mask     (32, 5)

FusionTransformer(tokens, avail_mask)    -->  fused          (32, 64)

classifier(fused).squeeze(-1)            -->  logits         (32,)
```

---

## 4. Training Pipeline

**File**: `src/train.py`

### 4.1 CLI Parameters and Executor Integration

The training script uses [Click](https://click.palletsprojects.com/) to define a CLI that integrates with the experiment dispatch infrastructure (`src/executor.py`).

#### How Parameters Flow

```
configs/train.json
       |
       v
scripts/run.sh (or submit_slurm.sh) -n train
       |
       v
src/executor.py  -- reads JSON, separates fixed vs. list params,
                    generates Cartesian product of list params,
                    injects exp_name/output_dir/output_filename,
                    constructs CLI command string
       |
       v
python src/train.py --fold 0 --d_model 64 --nhead 4 ... --exp_name train --output_dir outputs --output_filename "train__fold=0.json"
```

The executor (`src/executor.py`) handles:
1. **Fixed vs. variable param separation**: Scalar values in the JSON config become fixed params; list values become variable (sweep) params.
2. **Cartesian product**: All combinations of variable params are enumerated. The `--index` argument selects one combination.
3. **Injected params**: `--exp_name` (from `-n` CLI arg), `--output_dir` (from `executor.output_dir` in config), and `--output_filename` (auto-generated from variable param values) are injected.
4. **Output filename format**: `{exp_name}__{param1}={val1}__{param2}={val2}.json`

#### Executor-Injected Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `--exp_name` | `-n` argument to `run.sh`/`submit_slurm.sh` | Name of the experiment |
| `--output_dir` | `executor.output_dir` in config JSON | Directory for output JSONs |
| `--output_filename` | Auto-generated by executor | Unique filename per parameter combination |

#### Experiment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--fold` | int | required | LOSO fold index (0--8, maps to test patient P1--P9) |
| `--d_model` | int | 64 | Transformer hidden dimension |
| `--nhead` | int | 4 | Number of attention heads (head_dim = d_model / nhead) |
| `--num_encoder_layers` | int | 1 | TransformerEncoder layers per modality encoder |
| `--num_fusion_layers` | int | 1 | TransformerEncoder layers in fusion |
| `--dropout` | float | 0.1 | Dropout rate throughout the model |
| `--lr` | float | 1e-4 | Initial learning rate for AdamW |
| `--weight_decay` | float | 0.01 | L2 weight decay for AdamW |
| `--batch_size` | int | 32 | Training batch size |
| `--epochs` | int | 100 | Maximum number of training epochs |
| `--patience` | int | 15 | Early stopping patience (epochs without val AUROC improvement) |
| `--loss_fn` | choice | `weighted_bce` | Loss function: `weighted_bce` or `focal` |
| `--focal_alpha` | float | 0.25 | Focal loss alpha (only used if `--loss_fn=focal`) |
| `--focal_gamma` | float | 2.0 | Focal loss gamma (only used if `--loss_fn=focal`) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--threshold` | float | 0.5 | Classification threshold for computing F1/precision/recall |
| `--data_dir` | str | `data/processed/track1` | Path to preprocessed LOSO fold data |

### 4.2 Loss Functions

#### Weighted Binary Cross-Entropy (default)

```python
pw = compute_pos_weight(train_ds)  # n_neg / n_pos
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
```

`pos_weight` is computed from the training dataset by counting valid-label samples:

```python
def compute_pos_weight(dataset: RelapseDataset) -> float:
    n_pos = 0
    n_neg = 0
    for w in dataset.windows:
        if w["label_mask"][-1]:       # only valid last-day labels
            if w["labels"][-1] == 1:
                n_pos += 1
            else:
                n_neg += 1
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
```

The `pos_weight` scales the loss on positive (relapse) samples upward to counteract class imbalance. For example, if there are 3x more stable days than relapse days, `pos_weight = 3.0` makes each relapse sample contribute 3x more to the loss.

**Rationale**: Simple and effective for imbalanced binary classification. Auto-adapts to the actual class ratio per fold. Works directly with logits via `BCEWithLogitsLoss` for numerical stability (log-sum-exp trick).

#### Focal Loss (alternative)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
```

Implementation:

```python
probs = torch.sigmoid(logits)
ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
p_t = probs * targets + (1 - probs) * (1 - targets)
alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
loss = alpha_t * ((1 - p_t) ** self.gamma) * ce
```

- `alpha` (default 0.25): Balancing factor. alpha is applied to positive samples, (1-alpha) to negatives.
- `gamma` (default 2.0): Focusing parameter. Higher values down-weight easy-to-classify samples more aggressively, focusing training on hard examples.
- Returns per-sample losses (shape matches input); `.mean()` is applied in the training loop.

**Rationale**: Focal loss addresses class imbalance differently from weighted BCE -- it focuses on hard examples regardless of class. Useful when the model quickly learns to classify most stable days correctly but struggles with borderline relapse cases.

**Note**: When `loss_fn == "focal"`, the per-sample losses are explicitly averaged in the training loop:

```python
per_sample = criterion(logits[valid], labels[valid])
loss = per_sample.mean()
```

### 4.3 Optimizer and Scheduler

#### AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

- **Learning rate**: 1e-4 (default). Moderate for a small transformer.
- **Weight decay**: 0.01 (default). L2 regularization applied correctly via AdamW (decoupled from gradient updates, unlike vanilla Adam).

**Why AdamW**: The standard optimizer for transformer models. Decoupled weight decay provides better regularization than L2 penalty in Adam, which is important for a model that must generalize across patients.

#### CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

Smoothly anneals the learning rate from the initial value to 0 following a cosine curve over `T_max` epochs. `scheduler.step()` is called once per epoch after the training loop.

**Why cosine annealing**: Provides a smooth learning rate decay without requiring tuning of step/milestone schedules. The gradual reduction in later epochs allows fine-grained weight adjustments as the model converges.

### 4.4 Training Loop

```python
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    n_valid = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)       # (B,)
        labels = batch["label"]     # (B,)
        valid = batch["label_valid"]  # (B,)

        if not valid.any():
            continue

        # Loss only on valid-label samples
        loss = criterion(logits[valid], labels[valid])  # or .mean() for focal

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * valid.sum().item()
        n_valid += valid.sum().item()

    scheduler.step()
```

Key details:

- **Label masking**: Only samples where `label_valid == True` contribute to the loss. Batches with no valid labels are skipped entirely.
- **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` prevents gradient explosion, particularly important with small datasets and transformer attention.
- **Loss accumulation**: Weighted by number of valid samples per batch for accurate epoch-level average.
- **NaN divergence detection**: After each epoch, checks if `avg_loss` is NaN and terminates early if so:
  ```python
  if np.isnan(avg_loss):
      print(f"Training diverged at epoch {epoch} (NaN loss). Stopping.")
      break
  ```

### 4.5 Early Stopping

```python
best_val_auroc = -1.0
best_epoch = 0
best_state = None

# After each epoch:
if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
    best_val_auroc = val_auroc
    best_epoch = epoch
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if epoch - best_epoch >= patience:
    print(f"Early stopping at epoch {epoch} (patience={patience})")
    break
```

- **Metric**: Validation AUROC (threshold-agnostic, robust for imbalanced data).
- **Patience**: 15 epochs (default). If val AUROC does not improve for 15 consecutive epochs, training stops.
- **Model checkpoint**: Best model weights are stored in CPU memory (deep-copied to avoid GPU memory accumulation) and restored before final evaluation.
- **NaN handling**: NaN val AUROC (from single-class validation folds) does not count as an improvement, so training continues until patience runs out or max epochs.

### 4.6 Evaluation Metrics

```python
def evaluate(model, loader, device, threshold=0.5) -> dict
```

Computes metrics only on samples with `label_valid == True`:

| Metric | Computation | Notes |
|--------|-------------|-------|
| `auroc` | `roc_auc_score(labels, probs)` | Requires both classes present; NaN otherwise |
| `auprc` | `average_precision_score(labels, probs)` | Requires both classes present; NaN otherwise |
| `f1` | `f1_score(labels, preds, zero_division=0)` | Binary, threshold-based |
| `precision` | `precision_score(labels, preds, zero_division=0)` | Binary, threshold-based |
| `recall` | `recall_score(labels, preds, zero_division=0)` | Binary, threshold-based |
| `accuracy` | `accuracy_score(labels, preds)` | Overall accuracy |
| `n_samples` | `len(labels)` | Number of valid-label samples |
| `n_pos` | `int(labels.sum())` | Number of relapse samples |
| `n_neg` | `int(len(labels) - n_pos)` | Number of stable samples |

**NaN probability guard**: If the model produces NaN predictions (training diverged), they are replaced with 0.5:

```python
nan_mask = np.isnan(probs)
if nan_mask.any():
    probs = np.where(nan_mask, 0.5, probs)
```

**Empty/single-class handling**: If no valid-label samples exist, all metrics are NaN and `n_samples=0`. If only one class is present, AUROC and AUPRC are NaN (undefined) but F1/precision/recall/accuracy are still computed.

### 4.7 Output Format

The training script saves a JSON file per fold with this structure:

```json
{
  "exp_name": "train",
  "fold": 0,
  "hyperparameters": {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 1,
    "num_fusion_layers": 1,
    "dropout": 0.1,
    "lr": 0.0001,
    "weight_decay": 0.01,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,
    "loss_fn": "weighted_bce",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "seed": 42,
    "threshold": 0.5,
    "window_size": 7
  },
  "model_params": 316033,
  "device": "cuda",
  "best_epoch": 42,
  "val_metrics": {
    "auroc": 0.85,
    "auprc": 0.65,
    "f1": 0.72,
    "precision": 0.80,
    "recall": 0.66,
    "accuracy": 0.88,
    "n_samples": 150,
    "n_pos": 30,
    "n_neg": 120
  },
  "test_metrics": { "..." },
  "history": [
    {"epoch": 1, "train_loss": 0.693, "val_auroc": 0.50, "val_auprc": 0.30, "val_f1": 0.0, "lr": 0.0001},
    {"epoch": 2, "...": "..."}
  ],
  "train_size": 1200,
  "val_size": 80,
  "test_size": 60,
  "elapsed_seconds": 45.3
}
```

The file is saved to `{output_dir}/{output_filename}` (e.g., `outputs/train__fold=0.json`).

---

## 5. Configuration

**File**: `configs/train.json`

```json
{
    "fold": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 1,
    "num_fusion_layers": 1,
    "dropout": 0.1,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,
    "loss_fn": "weighted_bce",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "seed": 42,
    "threshold": 0.5,
    "data_dir": "data/processed/track1",
    "executor": {
        "exec_name": "train",
        "output_dir": "outputs",
        "log_dir": "outputs/logs"
    },
    "slurm": {
        "cores": 4,
        "nodes": 1,
        "time": "0-02:00",
        "memory": "16G",
        "partition": "gpunodes",
        "gres": "gpu:rtx_4090:1",
        "email": "...",
        "email_type": "END,FAIL",
        "log_dir": "outputs/slurm_logs",
        "job_name": "train"
    }
}
```

### Parameter Sections

#### Experiment Parameters (top-level)

| Key | Value | Sweep-able | Description |
|-----|-------|-----------|-------------|
| `fold` | `[0,1,2,3,4,5,6,7,8]` | **YES** (list) | Which LOSO fold to train. Currently a list, creating 9 jobs (one per fold). |
| `d_model` | `64` | Scalar (fixable to list for sweep) | Transformer hidden dimension |
| `nhead` | `4` | Scalar | Number of attention heads |
| `num_encoder_layers` | `1` | Scalar | Layers per modality encoder |
| `num_fusion_layers` | `1` | Scalar | Layers in fusion transformer |
| `dropout` | `0.1` | Scalar | Dropout probability |
| `lr` | `1e-4` | Scalar | Learning rate |
| `weight_decay` | `0.01` | Scalar | AdamW weight decay |
| `batch_size` | `32` | Scalar | Training batch size |
| `epochs` | `100` | Scalar | Maximum training epochs |
| `patience` | `15` | Scalar | Early stopping patience |
| `loss_fn` | `"weighted_bce"` | Scalar | Loss function choice |
| `focal_alpha` | `0.25` | Scalar | Focal loss alpha |
| `focal_gamma` | `2.0` | Scalar | Focal loss gamma |
| `seed` | `42` | Scalar | Random seed |
| `threshold` | `0.5` | Scalar | Classification threshold |
| `data_dir` | `"data/processed/track1"` | Scalar | Preprocessed data directory |

**Sweep mechanism**: Any scalar parameter can be converted to a list to trigger a hyperparameter sweep. The executor generates the Cartesian product of all list-valued parameters. For example, setting `"lr": [1e-4, 1e-3]` and `"fold": [0,1,2,3,4,5,6,7,8]` would produce 18 jobs (2 LRs x 9 folds).

#### Executor Section

| Key | Value | Description |
|-----|-------|-------------|
| `exec_name` | `"train"` | Script to run: `src/train.py` |
| `output_dir` | `"outputs"` | Directory for JSON result files |
| `log_dir` | `"outputs/logs"` | Directory for stdout/stderr logs |

#### SLURM Section

| Key | Value | Description |
|-----|-------|-------------|
| `cores` | `4` | CPUs per task |
| `nodes` | `1` | Number of nodes |
| `time` | `"0-02:00"` | Max runtime (2 hours) |
| `memory` | `"16G"` | Memory per job |
| `partition` | `"gpunodes"` | SLURM partition |
| `gres` | `"gpu:rtx_4090:1"` | GPU resource request (1 RTX 4090) |
| `job_name` | `"train"` | SLURM job name |
| `email_type` | `"END,FAIL"` | Email notification triggers |

---

## 6. Parameter Count Breakdown

With default hyperparameters (`d_model=64`, `nhead=4`, `num_encoder_layers=1`, `num_fusion_layers=1`, `window_size=7`):

### Per ModalityEncoder

Each ModalityEncoder has a fixed overhead plus an input-dimension-dependent projection layer.

| Component | Formula | Parameters |
|-----------|---------|-----------|
| **Input projection** | | |
| Linear(F, 64) | F * 64 + 64 | F-dependent |
| LayerNorm(64) | 64 + 64 | 128 |
| **CLS token** | 1 * 1 * 64 | 64 |
| **Positional embed** | 1 * 8 * 64 | 512 |
| **TransformerEncoderLayer** (1 layer) | | |
| Self-attention in_proj (Q,K,V) | 3 * 64 * 64 + 3 * 64 | 12,480 |
| Self-attention out_proj | 64 * 64 + 64 | 4,160 |
| FFN linear1(64, 256) | 64 * 256 + 256 | 16,640 |
| FFN linear2(256, 64) | 256 * 64 + 64 | 16,448 |
| LayerNorm x2 (norm1, norm2) | 2 * (64 + 64) | 256 |
| **Per-encoder subtotal** | 64*F + 64 + 128 + 64 + 512 + 49,984 | **64F + 50,752** |

### All 5 Modality Encoders

| Modality | F | Parameters |
|----------|---|-----------|
| accel | 38 | 64 * 38 + 50,752 = **53,184** |
| gyr | 38 | 64 * 38 + 50,752 = **53,184** |
| hr | 26 | 64 * 26 + 50,752 = **52,416** |
| step | 10 | 64 * 10 + 50,752 = **51,392** |
| sleep | 9 | 64 * 9 + 50,752 = **51,328** |
| **Total** | | **261,504** |

### FusionTransformer

| Component | Formula | Parameters |
|-----------|---------|-----------|
| Modality embeddings | 1 * 5 * 64 | 320 |
| TransformerEncoderLayer (1 layer) | same as above | 49,984 |
| **Total** | | **50,304** |

### Classification Head

| Component | Formula | Parameters |
|-----------|---------|-----------|
| Linear(64, 64) | 64 * 64 + 64 | 4,160 |
| Linear(64, 1) | 64 * 1 + 1 | 65 |
| **Total** | | **4,225** |

### Grand Total

| Component | Parameters | Percentage |
|-----------|-----------|-----------|
| Modality Encoders (x5) | 261,504 | 82.7% |
| Fusion Transformer | 50,304 | 15.9% |
| Classification Head | 4,225 | 1.3% |
| **Grand Total** | **316,033** | **100%** |

The model is deliberately small (~316K parameters) because the training dataset contains only 9 patients with limited sequences each. A larger model would overfit rapidly.

---

## 7. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Per-modality encoders | Independent transformer per modality with separate weights | Handles heterogeneous feature dimensions (38 vs. 9); each modality can learn its own temporal patterns; naturally handles missing sensors by masking individual encoders |
| Fusion method | Transformer self-attention + mean pooling | Cross-modal attention learns inter-modality relationships (e.g., sleep disruption and HR changes); mean pooling is robust to missing modalities (unavailable modalities are masked out) |
| Model size | ~316K params with d_model=64 | Very small dataset (9 patients, ~1200 training windows per fold) demands a small model to avoid overfitting; 64-dim is sufficient to capture the 121 input features |
| Pre-norm transformers | `norm_first=True` in all TransformerEncoderLayers | More stable training gradients, especially important with limited data and few layers; avoids need for careful learning rate warm-up |
| CLS token for summarization | Learnable CLS prepended to each modality sequence | Standard approach (BERT, ViT) for producing a fixed-size representation from a variable-length sequence; allows the model to learn what information to aggregate |
| Positional embeddings | Learnable absolute embeddings | With W=7 days, there are only 8 positions (including CLS). Learnable embeddings can capture the temporal structure. Sinusoidal would also work but offers no advantage at this scale |
| Early stopping metric | Validation AUROC | Threshold-agnostic (unlike F1/accuracy); robust for imbalanced binary classification; does not require choosing a threshold during training |
| Primary loss function | Weighted BCE with auto-computed pos_weight | Directly addresses class imbalance by up-weighting the minority class; simpler than focal loss; pos_weight auto-adapts to per-fold class ratios |
| Optimizer | AdamW with cosine annealing | AdamW is the standard for transformers; decoupled weight decay provides proper L2 regularization; cosine schedule smoothly reduces LR without milestone tuning |
| Gradient clipping | max_norm=1.0 | Prevents gradient explosions that can occur with transformer attention and small datasets; 1.0 is a conservative standard value |
| Classification threshold | 0.5 (configurable) | Default threshold for converting probabilities to binary predictions for F1/precision/recall; can be tuned post-hoc using the saved probabilities |
| Last-day prediction | Only the final day in each window is predicted | The window provides temporal context (6 days of history + current day); predicting only the last day avoids label leakage from future days within the window |
| Label masking | Samples with unknown labels excluded from loss and metrics | test_\* sequences from non-held-out patients contribute to training data but may have unknown labels (label=-1); masking ensures clean gradient signals |

---

## 8. Known Issues and Edge Cases

### NaN Training Divergence

The training loop includes NaN detection:

```python
if np.isnan(avg_loss):
    print(f"Training diverged at epoch {epoch} (NaN loss). Stopping.")
    break
```

NaN loss can occur when:
- The `pos_weight` is extremely large (very few positive samples in training)
- Learning rate is too high for the data scale
- A batch contains only invalid labels, causing division by zero in loss accumulation

The evaluation function also guards against NaN predictions:

```python
nan_mask = np.isnan(probs)
if nan_mask.any():
    probs = np.where(nan_mask, 0.5, probs)
```

### Single-Class Validation Folds

Some folds may have validation sets with only stable days (no relapse days) or vice versa. In this case:

```python
if n_pos == 0 or n_neg == 0:
    auroc = float("nan")
    auprc = float("nan")
```

AUROC and AUPRC are undefined with a single class. The early stopping logic handles this gracefully:

```python
if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
    # NaN AUROC is never an improvement, so best_state stays at previous best
```

If val AUROC is NaN for every epoch, `best_state` remains `None`, and the final model is whatever weights were last set during training. This is a degenerate case that should be flagged in results analysis.

### Fold 3 Behavior

Fold 3 (test patient = P4) may exhibit unusual behavior depending on P4's data characteristics. If P4's validation set has no relapse days, the early stopping mechanism cannot meaningfully select a best model. The output JSON will show `"auroc": NaN` in `val_metrics` for such folds. When aggregating cross-fold results, NaN folds should either be excluded or handled explicitly.

### Fold 8 Perfect Score

Fold 8 (test patient = P9) may produce near-perfect or perfect AUROC/AUPRC on validation/test. This can happen when the held-out patient has a very clear relapse pattern that is easy to detect, or when the validation set is very small with clear separation. A perfect score (AUROC=1.0) should be noted but may be legitimate given the patient-specific nature of relapse patterns.

### Missing Modality Handling

If a sensor modality recorded no data for any day in a window:
- `modality_mask` is all-False for that modality
- The ModalityEncoder still processes the (zero-filled) features, but all day positions are masked in attention; only the CLS token attends to itself
- `modality_available` is False for that modality in fusion, so it is masked out of cross-modal attention and excluded from mean-pooling
- This is handled correctly but means the CLS output for a fully-missing modality is essentially the initial CLS token embedding plus positional embedding at position 0, transformed by the encoder -- it contributes no useful information and is properly excluded from fusion

### Class Imbalance

Relapse days are a significant minority (~20 relapse vs. ~60 stable in a typical validation sequence). The weighted BCE loss addresses this at training time. However, the evaluation threshold (default 0.5) may not be optimal. Consider:
- Analyzing precision-recall curves per fold
- Tuning the threshold on validation data post-hoc
- Using AUPRC as a complementary metric to AUROC (AUPRC is more sensitive to performance on the minority class)

### Data Loader Memory

The entire fold's pickle is loaded into memory at initialization:

```python
def __init__(self, pkl_path: str | Path) -> None:
    with open(pkl_path, "rb") as f:
        self.windows: List[Dict] = pickle.load(f)
```

For training folds with ~1200 windows, each containing 5 modality arrays of shape `(7, F)`, memory usage is moderate. However, if window size or number of modalities increases significantly, consider lazy loading or memory-mapped arrays.

### Reproducibility

Seeds are set for:
- `torch.manual_seed(seed)` -- PyTorch CPU operations
- `np.random.seed(seed)` -- NumPy operations
- `torch.cuda.manual_seed_all(seed)` -- CUDA operations (if available)

However, full reproducibility also requires `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, which are not currently set. Results may vary slightly across runs on GPU due to non-deterministic CUDA operations.
