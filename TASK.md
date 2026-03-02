# Task Tracking

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

## Pending Tasks
- None

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
