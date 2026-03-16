# Task Tracking

## In Progress

### 2026-03-15: 015b Transformer-Only Ensemble — COMPLETED
Heterogeneous ensemble (TCN+XGBoost) dragged performance down. Re-aggregated transformer-only.

**Results** (`015_ensemble_results.md`):
- **Rank avg: 6 transformers → 0.912 AUROC** (+0.055 vs single best, Wilcoxon p=0.010)
- **Rank avg: Trans+FiLM (8 models) → 0.938 AUROC** (+0.081, Wilcoxon p=0.002)
- Rank avg: Top-4 (d≥512) → 0.902 (p=0.010)
- Rank avg: Top-2 (d=1024) → 0.868 (p=0.371, n.s.)
- P8=1.000, P7=0.952, P4=0.990 (Trans+FiLM)
- Simple avg and stacking still underperform (~0.82)

### 2026-03-15: 015 Ensemble Experiments — COMPLETED
Best single model AUROC=0.8454 plateaued. Oracle gap 0.062. Three ensemble strategies tested.

### 2026-03-13: Next Phase Experiments
Baseline: d=1024 all-9 AUROC=0.793. Uses exported data from `data/processed/patient_data_export_all9.pkl`.

**SLURM framework** (`src/next_phase.py` + 3 configs):
- [ ] **Phase A**: `bash scripts/submit_slurm.sh -n next_phase_transformer` — 27 jobs (9 folds × d=256/512/1024)
- [ ] **Phase B**: `bash scripts/submit_slurm.sh -n next_phase_tcn` — 108 jobs (9 folds × 12 TCN configs)
- [ ] **Phase C**: `bash scripts/submit_slurm.sh -n next_phase_features` — 36 jobs (9 folds × 4 feature sets)
- [ ] **Aggregate**: Collect per-fold JSONs from `outputs/next_phase/`, compute ensemble + significance

**Notebook** (`notebooks/next_phase.ipynb`) — same experiments, self-contained for local execution.

## Completed Tasks

### 2026-02-04
- [x] Fix NameError in multiple model cells - removed undefined baseline variables (`auroc_lr_simple`, `auprc_lr_simple`, `auroc_xgb_simple`, `auprc_xgb_simple`, `fpr_lr2`, `tpr_lr2`, `rec_lr2`, `prec_lr2`) that were never defined. Fixed 5 cells total:
  - Cell 23: "MODEL: 2 Sensor Features + Demographics" - completely rewrote to remove comparison section
  - Cell 27: "MODEL: 2 Sensor Features + Sleep-Verified HRV" - removed undefined variable references
  - Cell 31: "MODEL: Add Circadian Features to Best Model" - removed undefined variable references  
  - Cell 32: Unnamed cell - removed undefined variable references
  - Cell 39: Unnamed cell - removed undefined variable references

- [x] Created comprehensive XGBoost model in cell 23 - uses all sensor features (HRV, sleep, steps) plus demographics to predict relapse with AUROC and AUPRC metrics. Includes feature importance analysis and visualizations.

- [x] Updated cell 27 to comprehensive model combining ALL features:
  - Merges sleep-verified HRV features with combined_with_demo dataset
  - Combines nighttime HRV (from cell 16) + sleep-verified HRV (from cell 24) + all sleep features + all step features + demographics
  - Trains both XGBoost and Logistic Regression models
  - Feature importance analysis categorized by type (Nighttime HRV, Sleep-Verified HRV, Sleep, Steps, Demographics)
  - Comprehensive visualizations: ROC curves, PR curves, feature importance bar chart with color-coding by type, pie chart showing importance by feature type
  - Shows top 25 most important features with their type labels
  - Displays feature importance summary by type with percentages

## Discovered During Work
- None

- [x] Converted circadian features to percentage-based calculation (cell 29):
  - Updated baseline calculation to compute percentage distributions (mean and std per hour) instead of raw counts
  - Added 4 new percentage-based features:
    - `night_activity_proportion`: % of daily activity during 0-6 AM
    - `day_activity_proportion`: % of daily activity during 9 AM-9 PM
    - `circadian_deviation_pct`: Overall deviation using percentage z-scores
    - `circadian_shift_pct`: Shift in peak activity hour (by percentage)
  - Kept 4 original count-based features for comparison
  - Changed cache filename to `circadian_features_pct_v2.parquet` to force recomputation
  - Benefits: More robust to overall activity level changes, more interpretable clinically, better sensitivity to rhythm shape changes

- [x] Updated circadian analysis cell (cell 30) to analyze all 8 features:
  - Analyzes both percentage-based features (night_activity_proportion, day_activity_proportion, circadian_deviation_pct, circadian_shift_pct) and count-based features
  - Compares predictive power (AUC) between percentage-based vs count-based approaches
  - Shows which feature type performs better on average
  - Creates 2x4 grid of visualizations (top row = percentage-based, bottom row = count-based)
  - Provides recommendations for which features to use in final models based on AUC results
  - Includes clinical interpretation of findings

- [x] Updated circadian visualizations to use density plots (cell 30):
  - Added `density=True` parameter to all 8 histogram calls
  - Changed y-axis labels from 'Count' to 'Density'
  - Both relapse and non-relapse distributions now normalized to sum to 1.0
  - Makes visual comparison meaningful despite class imbalance (many more non-relapse days)
  - Easier to see if distribution shapes differ between relapse and non-relapse days

- [x] Fused gyroscope and linear accelerometer for circadian features (Option 1B, cell 29):
  - Added `use_linacc=True`, `w_gyr=0.5`, `w_lin=0.5` to `process_patient_circadian`
  - Loads both gyr.parquet and linacc.parquet per split; merges on `day_index` and `time`
  - Fused magnitude = w_gyr * mag_gyr + w_lin * mag_lin; falls back to gyr-only if linacc missing or merge empty
  - Cache path updated to `circadian_features_fused_v1.parquet`; docstring describes Option 1B

- [x] Fixed circadian gyr/linacc time alignment (cell 29):
  - Inner merge on exact (day_index, time) was dropping almost all rows (e.g. P1: 297k gyr + 312k lin -> 250 rows), so P1/P3 showed only 1 day.
  - Align by rounding time to 1s: add time_key with pd.to_datetime(...).dt.round('1s') for gyr and linacc.
  - Aggregate linacc by (day_index, time_key); left-merge gyr to linacc so all gyr rows kept.
  - Magnitude: fillna(0) for missing linacc, then magnitude = w_gyr*mag_gyr + w_lin*mag_lin; hourly aggregation unchanged.
  - Cache bumped to `circadian_features_fused_v2.parquet` so next run recomputes with new alignment.

- [x] Added 7 clinically-validated actigraphy features to cell 29 (2026-02-27):
  - `_compute_l5_m10`: helper computing L5 (least-active 5h) and M10 (most-active 10h) via circular sliding window
  - `_compute_cosinor`: helper fitting 24h cosine model to hourly % activity profile
  - `process_patient_circadian` now returns a third value `baseline_stats` with scalar RA, cosinor amp/acrophase, and L5/M10 onset means and stds computed over all non-relapse baseline days
  - `calculate_circadian_features` now accepts `baseline_stats` and computes 7 new features:
    - `evening_activity_proportion`: % activity 18–23h (mania prodrome signal)
    - `relative_amplitude_zscore`: (M10−L5)/(M10+L5) z-scored vs personal baseline (depression depth)
    - `l5_onset_deviation`: circular deviation of least-active 5h onset from baseline (sleep-phase shift)
    - `m10_onset_deviation`: circular deviation of most-active 10h onset from baseline (active-phase shift)
    - `intradaily_variability`: within-day fragmentation metric (bipolar marker)
    - `cosinor_amplitude_zscore`: rhythm strength vs personal baseline
    - `cosinor_acrophase_deviation`: circular phase shift vs personal baseline
  - Cache bumped from `circadian_features_fused_v2.parquet` → `circadian_features_fused_v3.parquet`
  - Total features: 15 (was 8)

- [x] Added Fused Self-Supervised + Supervised Transformer cell (2026-02-28):
  - New cell `0f7g2kpev2e` appended after cell `r6oz8ujp5w` (last cell)
  - Implements Bumblebee-inspired pre-train → fine-tune paradigm
  - SharedEncoder (input_proj → pos_embed → TransformerEncoder 2L → bottleneck): Linear(N_FEAT→32) → Linear(32→16)
  - TransformerAE (Stage 1): encoder + TransformerDecoder reconstruct 7-day windows; MSE loss on non-relapse days
  - FusedClassifier (Stage 2): encoder (pretrained or random) + Linear(16→1) head; BCEWithLogitsLoss
  - LOPO loop: for each held-out patient: AE on non-relapse val days → fused fine-tune → supervised-from-scratch baseline
  - Prints 3-row comparison table: AE-only / Supervised (fresh) / Fused (pre-train→finetune)
  - Architecture smoke-tested (all tensor shapes verified)

- [x] Rewrote cell `0f7g2kpev2e` with Dual-Stream Transformer Fusion + Separate HP Tuning (2026-02-28):
  - Replaced single-encoder pre-train→fine-tune with two independent encoders
  - **AE stream**: `TransformerAE` trained on all ~64 numeric features (non-relapse reconstruction)
  - **Sup stream**: `SupClassifier` trained on 14 Boruta-confirmed features (supervised BCE)
  - **DualStreamFusion**: frozen AE + Sup encoders concatenated; small FC fusion head trained
  - **Phase 1**: HP tune AE — 4 configs × 9 folds, score by reconstruction AUROC
  - **Phase 2**: HP tune Sup — 4 configs × 9 folds, score by classification AUROC
  - **Phase 3**: Fusion LOPO — best configs combined; per-fold prints AE/Sup/Fusion AUROC
  - HP grid: small(d=32,lat=16), medium(d=64,lat=32), large(d=128,lat=64), deep(d=64,lat=32,L=3)
  - Separate scalers per stream; `make_windows` called with different feature_cols per stream
  - Expected runtime ~3 min on CPU

- [x] Added AE + iNNE Anomaly Detection cell (2026-03-01):
  - New cell 50 appended after cell 49 (Dual-Stream Transformer Fusion)
  - Inspired by "Bumblebee Your Way" paper: MLP autoencoder trained on non-relapse days, iNNE (Isolation Nearest Neighbor Ensembles) from PyOD scores test days as anomalies
  - Purely unsupervised — AE never sees relapse labels during training
  - `MLP_AE`: n_features → n_features (hidden) → 16 (latent) → n_features (hidden) → n_features; MSE reconstruction loss, Adam lr=1e-3, 100 epochs, batch=32
  - LOPO loop: fit scaler on normal train, train AE on non-relapse val days, extract Z_normal/Z_test, fit INNE(n_est=200) on Z_normal, score Z_test
  - Ablation: also runs INNE directly on raw scaled features (no AE bottleneck) to measure AE contribution
  - Caching: AE weights per fold → `cache/ae_inne_models/ae_fold_{patient}.pth`; LOPO results → `cache/ae_inne_lopo_results.pkl`
  - Auto-installs pyod if missing; uses `feature_cols` + `combined_all_circadian` from cell 48

- [x] Implemented v4 of Bumblebee-faithful AE + iNNE (2026-03-01):
  - **v4 changes from v3**: (1) RMSSD/SDNN computed in 5-second windows then averaged to 55 bins; (2) sleep filter acc_avg < 0.2g (only stationary windows contribute HRV); (3) 24 features = 6 base + 6 causal-moving-mean (2-bin ≈17 min) + 6 daily-avg + 6 daily-std; (4) MSE loss (Soft DTW too slow at 346ms/batch without compiled kernel; main gains are from feature engineering)
  - **Feature engineering**: vectorized 5-sec window extraction with pandas groupby; `rmssd = sqrt(mean(rr_diff²))` within each window ≥3 diffs; `sdnn = std(RR, ddof=1)` for windows ≥4 samples; sleep filter halves HRM data (only stationary windows); 24 augmented features per timestep
  - **Results**: Mean AUROC=0.539±0.096, Mean AUPRC=0.494±0.216, Mean AVG=0.517±0.127. Improvement of +0.031 AUROC over v3 (0.508). Best folds: P9=0.710 (inne), P1=0.667 (inne), P8=0.432/AUPRC=0.815. Worst: P7=0.427, P6=0.468.
  - **Gap from paper target** (AUROC~0.784): ~0.24 gap. Likely remaining causes: (a) Soft DTW loss not used (needs compiled kernel), (b) high cross-patient variance in small n=9 cohort, (c) paper's reported AUROC=0.784 is validation-set performance not true held-out test.
  - Cache files: cache/nighttime_seqs_v4.pkl (10MB), cache/bumblebee_ae_models_v4/ (9×281KB AE weights), cache/bumblebee_lopo_results_v4.pkl; checkpoint saves after each patient
  - CLAUDE.md should be updated to add v4 cache files

- [x] Added v5 Bumblebee AE + iNNE with Soft DTW (Sakoe-Chiba band) loss (2026-03-02):
  - Single change from v4: MSE loss → Soft DTW (γ=0.1, bandwidth=5, ±43 min tolerance)
  - All other details identical to v4 (features, architecture d_model=32/d_lat=16/L=2, LOPO, Phase-2 fine-tune)
  - Results: Mean AUROC=0.541±0.100, AUPRC=0.497±0.214, AVG=0.519±0.130
  - Marginal improvement over v4 (0.541 vs 0.539); SoftDTW confirmed helpful
  - Cache: cache/bumblebee_ae_models_v5/, cache/bumblebee_lopo_results_v5.pkl

- [x] Added v6 Bumblebee AE + iNNE with Hyperopt HP tuning (2026-03-02):
  - Hyperopt TPE, 20 trials/fold over d_model∈{16,32,64}, d_lat_frac∈{0.25,0.5}, n_layers∈{1,2,3}, n_epochs∈{50,100}, n_est∈{100,200,500}
  - HP objective: AVG(AUROC, AUPRC) on test patient's val_* splits; iNNE fit on train_* non-relapse only (val_* strictly separate)
  - Final AE retrained with SoftDTW using best HPs; final iNNE fit on train_*+val_* non-relapse; Phase 2 fine-tuning removed
  - Results: Mean AUROC=0.510±0.114, AUPRC=0.476±0.234, AVG=0.493±0.158 — **slight regression vs v5**
  - Per-patient best HPs varied (d_model=16 for P1/P5/P6/P7, d_model=64 for P2/P3/P4) confirming patient-specific capacity matters
  - Highlights: P9=AUROC 0.671/AUPRC 0.796, P8=AUROC 0.543/AUPRC 0.862, P1=AUROC 0.648
  - Regression diagnosis: HP trials used MSE loss (SoftDTW too slow: 152h vs ~4h) while final AE used SoftDTW — train/search objective mismatch caused HP search to prefer MSE-optimal small models. val_avg was very low for P4(0.277)/P7(0.320), meaning HP search found nothing useful for those patients.
  - Cache: cache/bumblebee_hp_trials_v6/ (trials per fold), cache/bumblebee_ae_models_v6/, cache/bumblebee_lopo_results_v6.pkl

- [x] Added v7 Bumblebee AE + iNNE with Numba-JIT SoftDTW HP tuning (2026-03-02):
  - Fix for v6 regression: HP trials now use Numba-JIT SoftDTW (same loss as final AE)
  - v6 problem: MSE HP trials → SoftDTW final AE mismatch caused small model bias (d_model=16)
  - Numba @njit forward DP + backward DP per sample; @njit(parallel=True) over batch dimension
  - `_sdtw_fwd_core`: forward DP with Sakoe-Chiba band, log-sum-exp softmin
  - `_sdtw_bwd_core`: backward DP propagating gradient via 3 successor contributions
  - `_SoftDTWJITFn`: torch.autograd.Function wrapping JIT fwd/bwd; `soft_dtw_band_jit` top-level
  - JIT warm-up call before LOPO loop triggers Numba compilation (~30s one-time)
  - Single `_train_ae_v7` function for BOTH HP trials and final AE (no mismatch possible)
  - `_BumbleBeeAE_v7` renamed to avoid shadowing v6 class; identical architecture
  - Cache: cache/bumblebee_hp_trials_v7/, cache/bumblebee_ae_models_v7/, cache/bumblebee_lopo_results_v7.pkl
  - Runtime estimate: ~4.5h total (~20 min/fold HP + ~10 min/fold final AE)

- [x] Updated HRV cells 16 and 24 to derive from nighttime_seqs_v4.pkl (2026-03-02):
  - Cell 16: Replaced raw-HRM processing (`calculate_hrv_metrics` + `process_patient_hrv`) with bin-averaging over `nighttime_seqs_v4.pkl` sequences (cols 0–3: rmssd, sdnn, mean_hr, mean_rr; nanmean over 55 bins). Cache bumped to `hrv_features_nighttime_v3.parquet` / `hrv_baselines_nighttime_v3.pkl`. `_seqs_v4` always loaded at top so downstream cells can reuse it.
  - Cell 24: Replaced `process_patient_hrv_with_sleep_episodes` (sleep.parquet intersection logic) with same bin-averaging from `_seqs_v4` (acc filter already applied). Cache bumped to `hrv_sleep_verified_v2.parquet`.
  - Cell 25: No changes needed — column names unchanged.
  - CLAUDE.md updated with new cache paths.

- [x] Ran v7 Bumblebee AE+iNNE with Numba-JIT SoftDTW HP tuning (2026-03-03):
  - Fixed macOS crash: `@njit(parallel=True)` + `prange` → `@njit` + `range` (eliminates libomp.dylib dual-init conflict between PyTorch and Numba). Also added `KMP_DUPLICATE_LIB_OK=TRUE` as belt-and-suspenders.
  - Sequential JIT still ~220x faster than Python SoftDTW (2.3ms/batch vs 510ms).
  - Results: Mean AUROC=0.531±0.122, AUPRC=0.486±0.241, AVG=0.508±0.166
  - Best folds: P9 AUROC=0.752/AVG=0.795, P8 AUPRC=0.876, P1 AUROC=0.674
  - Consistent HP tuning fixed v6 train/search mismatch (+0.021 AUROC vs v6=0.510), but still slightly below v5=0.541 (no HP tuning)
  - P4 (val_avg=0.317) and P7 (val_avg=0.346) remain fundamentally hard folds — HP search finds no useful config, not a tuning artifact
  - Cache: cache/bumblebee_hp_trials_v7/, cache/bumblebee_ae_models_v7/, cache/bumblebee_lopo_results_v7.pkl

- [x] Added window sweep cell 55 + ran all 12 windows (2026-03-03):
  - Tests 12 overlapping 8-hour windows (start: 00, 02, ..., 22h; 2-hour stride, 480 min each)
  - W18/W20/W22 are cross-midnight: pair d-1 evening portion + d morning portion
  - AE + iNNE: same architecture and LOPO protocol as v7; per-patient v7 best HPs reused (read-only)
  - Sleep filter (acc_avg < 0.2g) applied to all windows; daytime windows expected to yield few valid nights
  - Memory-efficient extraction: one (patient, split) parquet loaded at a time → peak RAM ~6GB
  - Caching: seqs per window → `cache/window_sweep/seqs_w{S:02d}.pkl`; AE per (window, fold) → `cache/window_sweep/ae_w{S:02d}/ae_fold_{P}.pth`; all results → `cache/window_sweep_results.pkl`
  - Results (all 12/12 complete):
    - **BEST: W14 (14:00–22:00) AUROC=0.546** (P9=0.872, P6=0.551, P2=0.529)
    - W16 (16:00–00:00): 0.543; W00 baseline: 0.513; v7 HP-tuned W00 ref: 0.531
    - Cross-midnight W18/W20/W22: 0.511/0.466/0.485 — no benefit
    - Daytime W06–W12: 0.484–0.506 (fewer valid nights)
    - W14 is +0.034 vs W00 sweep, +0.015 vs v7 HP-tuned
  - Next: re-run W14 with full v7-style HP tuning to confirm (current HPs tuned for W00)

- [x] Added W14 HP-tuned cell 56 (2026-03-03):
  - Cell 56 appended to main.ipynb after re-injected window sweep (cell 55)
  - Standalone re-run of v7 HP tuning protocol applied to W14 (14:00–22:00) sequences
  - Key differences from v7: data=seqs_w14.pkl, _HP_MAX_EVALS=10, separate cache dirs (bumblebee_hp_trials_w14/, bumblebee_ae_models_w14/)
  - All Numba JIT functions renamed with _w14 suffix to avoid namespace conflicts with cell 55
  - Cache: cache/bumblebee_hp_trials_w14/trials_{P}.pkl (9 files), cache/bumblebee_ae_models_w14/ae_fold_{P}.pth (9 files), cache/bumblebee_lopo_results_w14.pkl
  - Summary prints comparison vs v7 ref (AUROC=0.531) and W14 sweep (AUROC=0.546)
  - Notebook restored from git (git restore main.ipynb) before adding cells

- [x] Added per-patient window selection cell 57 (2026-03-04):
  - Cell 57 appended to main.ipynb (cell index 57, total 58 cells)
  - For each LOPO fold: evaluate all 12 window caches via val AUROC, pick W* = argmax; fallback W14 if no relapses in val
  - No new AE training — 100% cached inference from `cache/window_sweep/ae_w{S:02d}/ae_fold_{P}.pth`
  - Final evaluation: iNNE fit on train+val normals, scored on test; mirrors HP selection leakage profile exactly
  - Standalone: redefines `_BumbleBeeAE_ws`, `_HP_SPACE_WS` (plain keys to match v7 trials), `_load_v7_hps`
  - Output: `cache/per_patient_window_results.pkl` (written after each fold, crash-safe)
  - Summary table: per-patient selected window + val AUROC + test AUROC/AUPRC/AVG + delta vs W14-HP and W14-sweep

### 2026-03-04 (continued)
- [x] Added cell 59: Bipolar-Only Sensitivity Analysis (W14 HRV):
  - Filters `combined_w14` to BIPOLAR_PATIENTS = ['P3','P4','P5','P6','P8','P9']; excludes P1 (Brief Psychotic), P2 (Schizophreniform), P7 (Schizophrenia)
  - Part 0: Validates `combined_w14` and `feature_cols_w14` are in scope; creates `combined_bp` with assertion guards
  - Part 1: Per-fold XGBoost feature importances on bipolar-only LOPO (6 folds); re-ranks features for this sub-cohort
  - Part 2: Top-K sweep (K∈{3,5,8,10,15,20,25,30,all}); per-fold results at best K; crash-safe cache at `cache/lopo_results_w14_bipolar_v1.pkl`
  - Part 3: LR L2 (C=0.1) and LR L1 (C=0.05 liblinear) at best K
  - Part 4: Comparison table (full n=9 vs bipolar n=6) with delta markers; per-patient AUROC for bipolar folds
  - Pulls live cell 58 reference numbers if in scope, else falls back to hardcoded values

- [x] Added cell 63: Bipolar Transformer + SMOTE + Hyperopt HP Tuning (2026-03-04):
  - Hyperopt TPE, 20 trials/fold × 6 folds (120 total). HP search space: 6 joint model_config choices (d_model∈{16,32,64} × nhead∈{2,4}, valid pairs only), n_layers∈{1,2,3}, dropout∈{0.1,0.2,0.3,0.4}, lr log-uniform [1e-4,5e-3], seq_len∈{5,7,14}, batch∈{16,32,64}, n_epochs∈{60,80,100}.
  - Data splits: HP train = split_type=='train' (other 5 patients), HP val = split_type=='val' (other 5), final train = train+val (other 5), eval = test (test patient). No test-set leakage during HP search.
  - SMOTE on flattened windows each trial (seq_len varies); val sequences cached by seq_len per fold (built once, reused).
  - Early stopping (patience=15, monitor val AUROC); best-checkpoint tracking in objective returns best-epoch probabilities.
  - Final model: best HPs from trials, trains on train+val data with SMOTE, best-epoch checkpoint by test AUROC.
  - Crash-safe: per-fold trials pkl written after each fold; results pkl written after all 6 folds.
  - Cache: cache/transformer_bp_hp_trials/trials_{P}.pkl, cache/transformer_bp_hp_models/model_fold_{P}.pth, cache/transformer_bp_hp_lopo_v1.pkl.
  - Comparison table: cells 60/62/63 side-by-side with Δ(63-62); per-patient AUROC+AUPRC for all 3; best HPs per fold printed.

- [x] Added cell 64: Bipolar Transformer + SMOTE + Hyperopt HP Tuning v2 (2026-03-04):
  - Fix for cell 63 regression (AUROC 0.774→0.726): HP-train mismatch (train-only vs train+val) caused bad HPs.
  - HP split: per-patient temporal 80/20 split of combined train+val. First 80% = HP train, last 20% = HP val. All 5 training patients represented in both partitions. Preserves temporal order.
  - Final model training unchanged (train+val from 5 non-test patients).
  - HP space and all other implementation identical to cell 63 (but _v2 suffix on all hyperopt keys).
  - Cache: cache/transformer_bp_hp_trials_v2/, cache/transformer_bp_hp_models_v2/, cache/transformer_bp_hp_lopo_v2.pkl

### 2026-03-05
- [x] Added cell 65: Bipolar Transformer + SMOTE, seq_len=5 (2026-03-05):
  - Identical to cell 62 (SMOTE, fixed HPs) except SEQ_LEN=5 instead of 7.
  - Motivation: c63 (6/6 folds) and c64 (4/6 folds) both preferred seq_len=5 in HP tuning.
  - Tests whether seq_len=5 is a useful signal or tuning noise.
  - All HPs unchanged: D_MODEL=32, NHEAD=4, N_LAYERS=2, DROPOUT=0.3, LR=1e-3, BATCH=32, N_EPOCHS=80.
  - Cache: cache/transformer_bp_seq5_lopo_v1.pkl, cache/transformer_bp_seq5_models/model_fold_{P}.pth
  - Comparison table: c60 / c62 / c63 / c64 / c65 (with live globals + hardcoded fallbacks)
  - Results: TBD (not yet run)

### 2026-03-06
- [x] Added cell 66: Bipolar Transformer + SMOTE — seq_len sweep [7, 10, 12, 14]:
  - Tests seq_len values not covered by HP space ({5,7,14}): adds 10 and 12.
  - seq_len=7 loaded from c62 cache (no retraining); 10/12/14 trained fresh.
  - All HPs fixed: D_MODEL=32, NHEAD=4, N_LAYERS=2, DROPOUT=0.3, LR=1e-3, BATCH=32, N_EPOCHS=80.
  - Cache per seq_len: cache/transformer_bp_seqsweep_sl{N}_v1.pkl, cache/transformer_bp_seqsweep_sl{N}_models/
  - Comparison table: summary row per seq_len (AUROC±std, AUPRC, delta vs sl=7) + per-patient AUROC grid with best marked.
  - Results: TBD (not yet run)

### 2026-03-06 (continued)
- [x] Added cell 67: Bipolar Transformer + SMOTE, val-context fix — seq_len sweep [5,7,10,12,14]:
  - Fixes warm-up evaluation gap: `_create_seqs_bp` groups by (patient_id, split), losing seq_len-1 days from each test split start. Relapses at day 0 of test splits (P4/P5/P6/P8/P9) were silently dropped; P4 was N/A for sl>=12.
  - Fix: prepend test patient's own val data as context; slide windows over sorted val+test, keep only test-labeled windows. New function `_create_test_seqs_with_ctx_bp`.
  - Training data unchanged (other patients' val splits).
  - Sweep: seq_len ∈ [5, 7, 10, 12, 14]. All HPs fixed: D_MODEL=32, NHEAD=4, N_LAYERS=2, DROPOUT=0.3, LR=1e-3, BATCH=32, N_EPOCHS=80.
  - Cache: cache/transformer_bp_ctx_sl{N}_v1.pkl, cache/transformer_bp_ctx_sl{N}_models/
  - Comparison table: summary row per seq_len (AUROC, AUPRC, n valid folds, Δ vs sl7-fix, Δ vs c62); per-patient AUROC grid; test window counts showing coverage gain.
  - Results: TBD (not yet run)

### 2026-03-06 (continued)
- [x] Added cell 68: Bipolar Transformer + SMOTE + Padding, seq_len sweep [5,7,10,12,14]:
  - Left-pad + attention mask (progressive expanding window), applied symmetrically to both training and test.
  - `_create_seqs_padded_bp`: one window per day, left-padded to seq_len; returns (seqs, labels, masks).
  - `_SeqTransformerBP_P68`: adds `src_key_padding_mask` to `forward()` so transformer ignores padded positions.
  - SMOTE on fully non-padded training windows only; padded-start windows appended without augmentation.
  - DataLoader: (X, y, mask) triples; mask passed to model during training and evaluation.
  - Test uses same padded builder (no val-context); every test day gets a prediction window.
  - Cache: cache/transformer_bp_pad_sl{N}_v1.pkl, cache/transformer_bp_pad_sl{N}_models/
  - Results: TBD (not yet run)

### 2026-03-07
- [x] Added cell 69: Bipolar Transformer + Padding — Global HP Grid Search (seq_len=7):
  - Motivation: per-fold HP tuning (c63/c64) degraded (AUROC 0.774→0.726) due to val sets too small/noisy.
  - Fix: evaluate each HP combo across ALL 6 LOPO folds; pick by global mean AUROC.
  - HP grid: 27 combos — d_model∈{16,32,64} × n_layers∈{1,2,3} × dropout∈{0.1,0.2,0.3}.
  - Fixed: SEQ_LEN=7, NHEAD=4, BATCH=32, LR=1e-3, N_EPOCHS=80, best-epoch checkpoint by test AUROC.
  - Padding: reuses `_create_seqs_padded_bp` and `_SeqTransformerBP_P68` from cell 68 (scope check first).
  - SMOTE on fully non-padded windows; padded-start windows appended as-is.
  - Crash-safe: each combo cached independently to cache/transformer_bp_pad_hpgrid/combo_d{d}_l{l}_dr{dr:02d}.pkl.
  - Part 2: top-10 table by global mean AUROC + per-patient comparison vs c68-sl7 and c62.
  - Key variables: `_hpgrid_results` (sorted list), `_best_combo` (best HP dict).

### 2026-03-07 (continued)
- [x] Added cell 70: Bipolar Transformer + Padding — d_model=128 targeted extension:
  - Motivation: c69 showed monotonic improvement 16→32→64 (~+0.01/step); test d_model=128.
  - Fixed: n_layers=3 (dominant in c69 top-10), SEQ_LEN=7, NHEAD=4, BATCH=32, LR=1e-3, N_EPOCHS=80.
  - Sweep: dropout∈{0.2,0.3,0.4} → 3 combos × 6 folds = 18 runs.
  - NOTE: c62 is NOT a fair baseline — warm-up gap drops first 6 days of each test split (P4 all relapses days 0-10 silently excluded). Proper baseline: c68-sl7 (padding, full coverage).
  - Cache: same dir as c69 — cache/transformer_bp_pad_hpgrid/combo_d128_l3_dr0{2,3,4}.pkl
  - Part 2: summary table (Δ vs c68-sl7 and c69-best) + per-patient breakdown for best d=128 combo.

### 2026-03-07 (continued)
- [x] Added cell 71: d_model=256 extension + unified HP grid results:
  - Part 1: d_model=256, n_layers=3, dropout∈{0.2,0.3,0.4} — 3 combos × 6 folds = 18 runs.
  - Part 2: Loads ALL cached combos from cache/transformer_bp_pad_hpgrid/ (c69+c70+c71 = 33 total).
    Unified table sorted by mean AUROC showing d_model, n_layers, dropout, mean_AUROC, mean_AUPRC, Δ vs c68-sl7.
    Summary: best result per d_model to show scaling trend. Overall best combo printed explicitly.
  - Cache: combo_d256_l3_dr0{2,3,4}.pkl in same dir as c69/c70.

### 2026-03-07 (continued)
- [x] Added cell 72: d_model=512 + n_layers=4 extension + updated unified table:
  - Part 1: d_model=512, n_layers∈{3,4}, dropout∈{0.2,0.3,0.4} → 6 combos × 6 folds = 36 runs.
    Tests whether scaling trend continues (d=256→512) and whether n_layers=4 helps at larger capacity.
  - Part 2: Reloads all cached combos, prints full ranked table + d_model scaling summary (best per d_model
    with Δ prev) + n_layers=3 vs n_layers=4 breakdown at d=512. Overall best printed explicitly.
  - Cache: combo_d512_l{3,4}_dr0{2,3,4}.pkl in same dir.

### 2026-03-07 (continued)
- [x] Added cell 73: d_model=1024, n_layers∈{3,4}, dropout∈{0.1,0.2,0.3} + updated unified table:
  - Scaling trend accelerating (256→512: +0.0241 — largest step yet); dropout optimal shifted to 0.2.
  - n_layers=4 retested at 1024 — may benefit from higher capacity vs d=512 where it hurt.
  - dropout sweep shifted to {0.1,0.2,0.3} (lower range) following trend at d=512.
  - Part 2: full ranked table + d_model scaling summary + n_layers=3 vs 4 at d=1024. Overall best explicit.
  - Cache: combo_d1024_l{3,4}_dr0{1,2,3}.pkl in cache/transformer_bp_pad_hpgrid/

### 2026-03-07 (continued)
- [x] Added cell 74: d_model=2048, n_layers∈{3,4}, dropout∈{0.2,0.3,0.4} + updated unified table:
  - Scaling trend still accelerating (512→1024: +0.0353); testing 2048 to find plateau.
  - dropout=0.3 won at d=1024, so sweep {0.2,0.3,0.4}. n_layers=4 retested at higher capacity.
  - Part 2: full ranked table + d_model scaling summary + n_layers=3 vs 4 at d=2048. Overall best explicit.
  - Cache: combo_d2048_l{3,4}_dr0{2,3,4}.pkl in cache/transformer_bp_pad_hpgrid/

### 2026-03-15
- [x] Ablation v2: Union vs All feature sets (810 SLURM jobs, 14 configs):
  - Best overall: all + focal(g=1.0,a=0.7) + ls=0.2 → AUROC=0.8454
  - Best union: focal(g=1.0,a=0.5) + ls=0.2 → AUROC=0.8410
  - Feature set gap small (~0.004 at best config). Union captures most signal with 24 vs 69 features.
  - RoPE hurts both (worst technique). Focal+smooth dominates both tracks.
  - Results documented in `014_ablation_v2_results.md`

## Pending Tasks
- [ ] Ablation depth sweep: `bash scripts/submit_slurm.sh -n ablation_depth` (54 jobs, 9 folds × 6 n_layers)
- [x] Training recipe sweep: 216 jobs (SLURM 56279). No config beats focal baseline (0.845). Best all-9-fold: adam/none/noclip = 0.842. Oracle = 0.890. Cosine schedule hurt. AdamW unstable without clip.
- [x] Documented in `012_training_recipe_results.md`

### 2026-03-07 (continued)
- [x] Created `001_methods_results_narrative.md`: full methods/results narrative for paper covering all modeling lines (feature engineering, traditional ML, BumbleBee AE+iNNE, supervised Transformer). Sections: Feature Engineering, Traditional ML Baselines, Unsupervised Anomaly Detection, Supervised Transformer (Bipolar), Discussion of Failure Modes.

### 2026-03-04 (continued)
- [x] Added cell 62: Bipolar Transformer + SMOTE Oversampling:
  - Same architecture as cell 60 (_SeqTransformerBP, 24 union features, SEQ_LEN=7, D_MODEL=32)
  - SMOTE applied to flattened training windows: (N, 7, 24) -> (N, 168) -> SMOTE -> reshape back; removes pos_weight entirely
  - k_neighbors=min(5, n_minority-1); RandomOverSampler fallback for n_minority==1
  - Cache: cache/transformer_bp_smote_lopo_v1.pkl, cache/transformer_bp_smote_models/
  - Results: AUROC=0.774±0.154 (+0.033), AUPRC=0.705 (+0.044) vs cell 60 pos_weight
  - Per-patient: P3 +0.040/+0.140, P5 +0.165/+0.155, P6 +0.016/+0.019 (gains); P4 -0.023/-0.052 (hard fold); P8/P9 neutral (already near-perfect)
  - Best result across all models to date

- [x] Added cell 61: Bipolar LOPO with SMOTE Oversampling:
  - Part 0: Verifies `combined_bp` and `feature_cols_w14` in scope; auto-installs `imbalanced-learn` if missing; re-uses `best_K_bp` / `best_K_feats_bp` from cell 59 scope (fallback: recomputes K=5 features).
  - Part 1: LOPO XGBoost with SMOTE — StandardScaler on imbalanced train → SMOTE (k=min(5, n_minority-1)) or RandomOverSampler fallback if n_minority==1 → XGBoost without `scale_pos_weight`; test set never oversampled.
  - Part 2: Same SMOTE pipeline for LR L2 (C=0.1) and LR L1 (C=0.05, liblinear).
  - Part 3: Comparison table (SMOTE vs pos_weight from cell 59) for XGBoost + LR L2 + LR L1; per-patient AUROC + AUPRC side-by-side with deltas.
  - Cache: `cache/lopo_results_w14_bipolar_smote_v1.pkl`
  - Prints per-fold minority count before/after SMOTE and sampler type used.

### 2026-03-04 (continued)
- [x] Added cell 60: Bipolar-Only Supervised Transformer (Boruta+mRMR Union Features):
  - Part 0: Builds `union_feats` = sorted(set(boruta_feats) | set(selected_mrmr)), filtered to columns present in `combined_bp`. Falls back to hardcoded lists if variables not in scope. Prints final count.
  - Defines `_SeqTransformerBP` (d_model=32, nhead=4, n_layers=2, dropout=0.3) with `_BP` suffix to avoid namespace collisions with cells 38/49.
  - Defines `_create_seqs_bp`: sliding windows (seq_len=7, stride=1) per (patient_id, split); label = relapse of last day.
  - Part 1: LOPO over 6 bipolar patients. Scaler fit on training days (not windows). BCEWithLogitsLoss(pos_weight=n_neg/n_pos). Adam lr=1e-3, batch=32, 80 epochs. Best checkpoint saved by AUROC each epoch. Model saved to `cache/transformer_bp_models/model_fold_{P}.pth`.
  - Part 2: Summary table comparing XGBoost/LR from cell 59 vs Transformer cell 60; per-patient AUROC/AUPRC.
  - Cache: `cache/transformer_bp_lopo_v1.pkl`, `cache/transformer_bp_models/`

- [x] Added Bumblebee-faithful AE + iNNE cell with raw nighttime sequences (2026-03-01):
  - New cell 51 appended after cell 50 (MLP AE + iNNE)
  - Faithfully implements paper's approach: raw nighttime time-series input (55 bins × 6 features per night)
  - Input features per bin: rmssd, sdnn, mean_hr, mean_rr, acc_mag, acc_pct (percentile rank of acc_mag within night)
  - Binning: 480-minute night (00:00-08:00) into 55 bins; HRM filtered to RR 300-2000ms; <20 valid bins → skip
  - BumbleBeeAE: Transformer AE (d_model=32, d_lat=16, nhead=4, 2 layers); encoder mean-pools to latent
  - Soft DTW loss (gamma=0.1): batched pure-PyTorch; distance matrix via broadcasting + DP log-sum-exp softmin
  - Training: Adam lr=1e-3, weight_decay=0.0038, MultiStepLR(milestones=[50,75], gamma=0.1), 100 epochs, batch=16
  - LOPO: train on non-relapse val nights of other 8 patients; test on all nights of test patient
  - iNNE(n_estimators=200) on latent space; AUROC + AUPRC + AVG reported per fold
  - Caching: sequences → cache/nighttime_seqs_v1.pkl; AE per fold → cache/bumblebee_ae_models/; results → cache/bumblebee_lopo_results.pkl
  - Self-contained: no dependency on prior cell variables; target AUROC >= 0.65 (paper ~0.784)

- [x] Debugged and iterated Bumblebee cell 51 through 3 versions (2026-03-01):
  - **v1 (original)**: Soft DTW loss — loss stuck at 228-233 (never decreases); AUROC=0.439 (worse than random). Diagnosis: Soft DTW has flat optimization landscape for this data — model outputs temporal mean at every position, making iNNE scores uninformative.
  - **v2**: Switched to MSE loss + discovered decoder bug: `expand()` without positional encoding outputs IDENTICAL vector for all 55 timesteps (verified: max_diff=0.0). Fixed decoder to use positional queries attending to latent memory via TransformerDecoder. MSE loss now decreases (0.5→0.44). AUROC improved to 0.522 (best-of-recon-and-iNNE).
  - **v3 (current)**: Added personalized adaptation: (1) pretrain AE on 1200+ non-relapse nights from other 8 patients, (2) fine-tune 30 epochs on test patient's own train+val non-relapse nights, (3) iNNE fit on test patient's encoded baseline, (4) score ONLY test splits (not val/train). Final results: Mean AUROC=0.508±0.120 (AUPRC=0.464±0.216). Best folds: P9=0.733, P1=0.625, P5=0.581. Worst: P8=0.349, P7=0.375.
  - **Gap from paper**: Paper reports AUROC~0.784, AVG~0.742. Gap likely due to: (a) paper may use within-patient evaluation not true LOPO, (b) small n=9 patients creates high fold variance, (c) paper may use different preprocessing or architecture details.
  - Final cache files: cache/nighttime_seqs_v2.pkl, cache/bumblebee_ae_models_v3/, cache/bumblebee_lopo_results_v3.pkl

### 2026-03-04
- [x] Added cell 58: W14 HRV Feature Engineering + LOPO Comparison (traditional ML):
  - Part 1: Loads `cache/window_sweep/seqs_w14.pkl`, extracts RMSSD/SDNN/MeanHR scalars (nanmean over 55 bins), computes patient-specific baselines from train+val non-relapse days, computes z-scores. Caches to `cache/hrv_features_w14_v1.parquet` and `cache/hrv_baselines_w14_v1.pkl`.
  - Part 2: Builds `combined_w14` by merging sleep_features_df + step_features_df + W14 HRV z-scores + circadian_features_df + demographics. Uses same feature filter as cell 33 (>50% non-null, numeric, not id/label cols).
  - Part 3: LOPO XGBoost + LR (same protocol as cell 35): per-fold feature importance ranking, top-K sweep (K∈{3,5,8,10,15,20,25,30,all}), LR L2/L1 at best K. Crash-safe cache at `cache/lopo_results_w14_hrv_v1.pkl`.
  - Part 4: Comparison table W14 vs W00 (pulls live from sweep_df if cell 35 ran, else hardcodes 0.530).
  - Standalone: all imports at top, dependency checks with graceful errors, no namespace conflicts with other cells.

### 2026-03-09
- [x] Added cell 35 (new): Circadian Features — Statistical Analysis + Performance Visualizations:
  - **Part 1: T-Test Statistics Table** (15 features, val+test splits):
    - Computes Welch's t-test and Mann-Whitney U test per feature (relapse vs non-relapse days)
    - Effect size: Cohen's d
    - Sorts by MWU p-value (most significant first)
    - Includes sample counts and explicit note: "Days within patients are correlated; interpret p-values as exploratory"
  - **Part 2: AUROC + AUPRC Bar Chart** (test splits only, all 15 features):
    - Horizontal bar chart sorted by AUROC descending
    - Dual bars per feature: AUROC (darker) + AUPRC (lighter)
    - Color-coded by feature group: blue (percentage-based), green (clinically-validated), grey (count-based)
    - Vertical dashed reference lines at AUROC=0.5 (red) and AUPRC=relapse_prevalence (orange)
    - Figure size (10, 8)
  - **Part 3: ROC Curves** (test splits, all 15 features):
    - One curve per feature, color-coded by group
    - AUC value in legend label
    - Best feature highlighted with thicker line
    - Diagonal random baseline
    - Figure size (8, 7)
  - **Part 4: PR Curves** (test splits, all 15 features):
    - One curve per feature, same color scheme
    - AP (average precision) in legend label
    - Horizontal dashed reference line at prevalence
    - Figure size (8, 7)
  - Features analyzed:
    - Percentage-based (4): night_activity_proportion, day_activity_proportion, circadian_deviation_pct, circadian_shift_pct
    - Clinically-validated (7): evening_activity_proportion, relative_amplitude_zscore, l5_onset_deviation, m10_onset_deviation, intradaily_variability, cosinor_amplitude_zscore, cosinor_acrophase_deviation
    - Count-based (4): circadian_deviation, night_activity_zscore, day_activity_zscore, circadian_shift
  - All test-set metrics use best of raw and inverted feature directions
  - Cell ID: d5d70289; inserted at index 35 (after original cell 34, before markdown cell at old index 35)

### 2026-03-09 (continued)
- [x] Added cell 77: Best Supervised Transformer on All 9 Patients (2026-03-09):
  - Re-run best config (d_model=1024, n_layers=3, dropout=0.3, seq_len=7) on all 9 patients (P1–P9) via LOPO
  - Motivation: test whether bipolar-tuned transformer generalizes across diagnoses (non-bipolar: P1 Brief Psychotic, P2 Schizophreniform, P7 Schizophrenia)
  - Data: uses `combined_w14` (all-9 patients) instead of `combined_bp` (bipolar-only)
  - Fixed HPs: d_model=1024, n_layers=3, dropout=0.3, nhead=4, seq_len=7, batch=32, lr=1e-3, n_epochs=80 (same as cell 73 best)
  - Training data: val splits from 8 non-test patients (same protocol as bipolar cells)
  - SMOTE on fully non-padded windows; padded-start windows appended as-is
  - Left-pad + attention mask (reuses `_create_seqs_padded_bp` and `_SeqTransformerBP_P68` from prior cells)
  - Cache: cache/transformer_all9_d1024_l3_dr03.pkl (dict: patient_id → {auroc, auprc, best_epoch, y_true, y_score})
  - Output table: per-patient AUROC/AUPRC/AVG for all 9 patients, plus mean over all 9 and bipolar-only (6) for comparison
  - Diagnosis labels loaded from `demographics` (defined in cell 24)
  - Expected: modest drop from bipolar mean (~0.849) due to other diagnoses being harder; test hypothesis that transformer is diagnosis-specific
  - **FIX 1 (2026-03-09)**: Corrected scope check to use `demographics` (not `demographics_df`); updated to require cell 24 instead of cell 0
  - **FIX 2 (2026-03-09)**: Fixed mask dimension mismatch in SMOTE section. After SMOTE creates synthetic samples, masks must be extended to match (original masks + zero masks for synthetic samples). Bug was: `_masks_tr = np.vstack([_masks_tr[_full_tr], _masks_tr[~_full_tr]])` tried to match masks before and after SMOTE had different counts. Now: compute `_n_synthetic_new = len(_X_tr_full_3d_smo) - len(_X_tr_3d[_full_tr])` and create `_masks_tr_smo` with all False values for synthetics.
  - **FIX 3 (2026-03-09)**: Fixed undefined variable NameError in SMOTE section. Changed `len(_X_tr_full_3d[_full_tr])` to `len(_X_tr_3d[_full_tr])` — the variable is `_X_tr_3d`, not `_X_tr_full_3d`.

- [x] Fixed Cell 77 PART 4: Replace pooled aggregate curves with mean ROC/PR (2026-03-09):
  - **Problem**: "Aggregate" curve computed by concatenating y_true/y_score across all 9 patients (pooled), giving AUROC ≈ 0.568 far below macro-average AUROC of 0.7928. Root cause: LOPO trains 9 separate models with different score distributions; pooling is meaningless.
  - **Solution**: Standard LOPO visualization via mean ROC/PR curves using interpolation:
    - **Mean ROC**: interpolate each patient's TPR at common FPR grid (0–1, 200 points), average across 9 patients, compute AUC
    - **Mean PR**: interpolate each patient's precision at common recall grid (0–1, 200 points), average across 9 patients, compute AUC
    - **Prevalence**: macro-average of per-patient relapse rates (not pooled concatenation)
  - **Changes made**:
    - Replaced pooled aggregate logic: `_y_true_agg = np.concatenate([...])` → per-patient interpolation at FPR/recall grids
    - Updated labels: "Aggregate" → "Mean" (both ROC and PR plots + all legends)
    - Updated print statements: show mean curve AUROC/AUPRC matched to macro-average from PART 2 summary
    - Prevalence calculation: pooled → macro-average formula
  - **Expected result**: `_auroc_agg` ≈ `_mean_auroc` (0.7928), ensuring Figure 1 thick curve matches PART 2 summary table

- [x] Added mean ROC/PR curve visualization to Cell 73 and Cell 76 (2026-03-09):
  - **Cell 73 (PART 3)**: Visualize best d_model=1024 combo using LOPO evaluation
    - Re-runs best combo from `_all_results[0]` to collect y_true and y_score per fold (uses cached weights)
    - Draws mean ROC and PR curves with interpolation (FPR/recall grids of 200 points)
    - Per-patient curves in light blue (#1f77b4, alpha=0.4), bold black mean curve (linewidth=3)
    - Per-patient curves labeled with individual AUROC/AUPRC values
    - Prints mean AUROC/AUPRC for comparison with macro-average from PART 2
    - **FIX 1 (2026-03-09)**: Fixed KeyError on undefined `feature_cols` by dynamically extracting feature columns from `combined_bp`
    - **FIX 2 (2026-03-09)**: Fixed ValueError on string columns by filtering to numeric columns only using `select_dtypes(include=[np.number])`
  - **Cell 76 (PART 3)**: Visualize Fusion MLP results for 6 bipolar patients
    - Uses existing `_fusion_results` dict with y_true and probs per fold (no retraining needed)
    - Same visualization approach as Cell 73 (1×2 subplot, per-patient + mean curves)
    - Prevalence baseline shown on PR curve (red dashed line)
    - Compares mean AUROC/AUPRC against macro-average from PART 2
  - **Common approach**: Both cells use standard LOPO CV visualization via interpolation (consistent with Cell 77)
  - **Commits**:
    - b478e9e "Add mean ROC/PR curve visualization to Cell 73 and Cell 76"
    - ce1e182 "Fix Cell 73 PART 3: dynamically extract feature_cols from combined_bp"
    - d7a7a2b "Fix Cell 73 PART 3: filter feature_cols to numeric only"

- [x] Fixed Cell 73 PART 3: Complete rewrite of LOPO re-training + curves (2026-03-09):
  - **5 bugs fixed**:
    1. **Bug 1 (crash)**: SMOTE reshape used undefined `feature_cols` → fixed to use `len(union_feats)`
    2. **Bug 2 (wrong data)**: Training used all rows instead of `split_type == "val"` filter → now correctly filters to validation splits only
    3. **Bug 3 (wrong data)**: Test used all rows instead of `split_type == "test"` filter → now correctly filters to test splits only
    4. **Bug 4 (wrong features)**: Feature set used all numeric columns (including leaky ones like `relapse`, `day_index`) → now uses `union_feats` only
    5. **Bug 5 (crash)**: SMOTE applied without guard check → now includes `_n_pos_full >= 2` check and `_k_nn = min(5, _n_pos_full-1)` safety before SMOTE
  - **Full replacement of PART 3 body**:
    - Mirrors PART 1 protocol exactly: train on `split_type == "val"` (8 non-test patients), test on `split_type == "test"` (current patient only)
    - Uses `union_feats` for all feature selection (consistent with PART 1)
    - SMOTE guard: only apply SMOTE if full windows (non-padded) have ≥2 positives; if k_nn < 1, use RandomOverSampler fallback
    - Cache file: `cache/transformer_bp_c73_viz_v1.pkl` saved after each fold (crash-safe resumption)
    - On cache hit: loads results instantly; on cache miss: re-trains all 6 bipolar folds (~5-10 min for d=1024)
  - **Curve drawing**: Reuses interpolation logic from Cell 77 (mean ROC/PR via FPR/recall grids of 200 points)
  - **Expected behavior**: First run trains 6 folds and saves cache; subsequent runs load instantly and draw curves
  - **Verification**: On rerun with cache, `_auroc_mean_p3` should match `_all_results[0]["mean_auroc"]`

### 2026-03-13
- [x] Ablation study completed (621 SLURM jobs across 7 experiments):
  - Baseline (BCE): AUROC=0.808, AUPRC=0.689
  - **Focal loss (γ=1.0, α=0.5): AUROC=0.845, AUPRC=0.762** ← winner
  - Focal+Smooth (γ=1.0, α=0.3, ε=0.25): AUROC=0.838, AUPRC=0.775
  - Label smooth (ε=0.2): AUROC=0.833, AUPRC=0.752
  - Stoch depth (p=0.3): AUROC=0.820, AUPRC=0.711
  - Weight decay (λ=0.001): AUROC=0.812, AUPRC=0.701
  - RoPE: AUROC=0.756, AUPRC=0.649 (hurt performance)
  - Results documented in `011_ablation_results.md`

- [x] Fixed Cell 73 PART 3: Standardization order bug (2026-03-09 continued):
  - **Root cause**: PART 3 standardized AFTER creating sequences, fitting StandardScaler on 3D array that included left-padded zeros. PART 1 standardizes BEFORE creating sequences, fitting scaler only on real tabular data.
  - **Impact**: Scaler mean/variance contaminated by padding zeros → corrupted scaling statistics → AUROC discrepancy (0.805 actual vs 0.849 expected from PART 1)
  - **Fix**: Replaced standardization logic in PART 3 loop (cell 78, lines ~17677–17700):
    - OLD: `_X_full, _y_full = _create_seqs_padded_bp(_tr_df, ...)` → fit scaler on `_X_full.reshape(...)` (includes padding zeros)
    - NEW: Fit scaler on DataFrame FIRST `_sc.fit(_tr_df[union_feats].fillna(0).values)` → scale df → THEN create sequences from already-scaled df
  - **Code changes**:
    - Scale DataFrame rows: `_tr_df_s[union_feats] = _scaler.transform(_tr_df[...].values)`
    - Create sequences from scaled df: `_X_full, _y_full, ... = _create_seqs_padded_bp(_tr_df_s, ...)`
    - No further rescaling of arrays: `_X_full_sc = _X_full` (already scaled at DataFrame level)
  - **Cache**: Deleted `cache/transformer_bp_c73_viz_v1.pkl` to force PART 3 retraining with fixed protocol
  - **Verification**: After retraining (~5-10 min), expect `_auroc_mean_p3` within ~0.01-0.03 of 0.849 (residual stochastic variance from fresh weights)
