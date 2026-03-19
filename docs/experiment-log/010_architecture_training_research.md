# 010 - Architecture & Training Research for Improving AUROC/AUPRC

## Summary

Research into techniques for improving binary per-day relapse prediction (current best: AUROC 0.834, AUPRC 0.718) from wearable sensor data. 9 patients, LOPO CV, ~2900 training windows, ~20% relapse prevalence, 7-day windows of 69 features. Below: 8 research areas with concrete findings, ranked by expected impact and feasibility.

---

## 1. Attention Mechanisms for Short Sequences

### Findings

**Relative Positional Encoding / RoPE**: Standard sinusoidal positional encoding is suboptimal for very short sequences (T=7). Rotary Position Embedding (RoPE) encodes relative distances directly in the dot-product attention, decaying inter-token dependency with distance. A survey on positional encoding for time series transformers (arXiv:2502.12370, 2025) confirms that data characteristics like sequence length significantly influence PE method effectiveness. For T=7, relative PE matters less than for long sequences, but RoPE is a near-zero-cost swap that marginally helps.

**Feature-Temporal Dual Attention**: Yao et al. (2025, J. Forecasting) show that *separating* feature attention (FA) and temporal attention (TA) into distinct mechanisms, then combining them (FATA), adapts better to different data regimes. For short-sequence clinical tabular data, Feature Attention is often more important than Temporal Attention.

**Cross-Attention Between Modality Groups**: Rather than concatenating all 69 features, group them into modality blocks (sleep, steps, HRV, circadian, demographics) and use cross-attention across groups. The ArmFormer architecture (Huang et al., 2025, Expert Systems) demonstrates multi-scale patch encoding + lead-wise attention achieves 97.36% AUC on cardiac wearable data. Analogously, modality-wise attention lets the model learn which modality interactions matter per-patient.

**Recommendation**: Try (a) RoPE as a drop-in PE replacement, (b) modality-group cross-attention where each modality group has its own encoder branch, then cross-attend. Expected impact: +1-3% AUROC from cross-attention.

---

## 2. Multi-Task / Auxiliary Losses

### Findings

**Days-until-relapse regression**: The MIMIC-III benchmark (Harutyunyan et al., 2019, Scientific Data) established that combining in-hospital mortality (binary), length-of-stay (ordinal/regression), decompensation (binary), and phenotyping (multi-label) as multi-task objectives improves all individual tasks. The key insight: auxiliary regression tasks act as regularizers.

For relapse prediction, the natural auxiliary task is **days-to-relapse onset** (continuous regression, 0 for relapse day, increasing for stable days). This creates a smoother gradient landscape near decision boundaries. The loss:

```
L_total = w_cls * L_focal + w_reg * L_huber(days_to_relapse)
```

Zhang et al. (2026, Geophysical Research Letters) confirm that multi-task learning "mitigates the smoothing effect of regression loss on infrequent events" while enhancing classification reliability.

**Ordinal auxiliary**: Convert days-to-relapse into ordinal bins (0-1 days, 2-3 days, 4-7 days, 7+ days) and use ordinal cross-entropy as a third loss head. This is analogous to SurvTRACE's approach for survival analysis.

**Recommendation**: Add a regression head for days-to-relapse with Huber loss. Weight ratio ~0.7 classification / 0.3 regression. Expected impact: +2-4% AUROC, +3-5% AUPRC (better ranking near decision boundary).

---

## 3. Patient-Adaptive Methods

### Findings

**MAML for LOPO**: Wang et al. (2025, Med Research) directly address small-data mental health forecasting with MAML. The key: treat each patient as a "task" in the meta-learning sense. During meta-training, inner-loop adapts to each training patient's data, outer-loop optimizes for cross-patient generalization. At test time, use the held-out patient's val set for a few gradient steps. With only 9 patients (8 train, 1 test), MAML's bi-level optimization is feasible but the task distribution is extremely narrow.

**Reptile** (first-order MAML approximation) is more practical here: lower memory cost, simpler implementation, and Wang et al. note it "reduces computational overhead while still enabling fast adaptation." For 9 patients, Reptile is preferred over full MAML.

**Learnable Patient Embeddings**: Add a small (d=16-32) learnable embedding per patient ID, concatenated to the feature vector at each timestep. At test time, initialize the new patient's embedding to the mean of training embeddings, then optionally fine-tune during the few val-set gradient steps. This is cheap and compatible with the existing transformer.

**Prototypical Networks**: Embed each patient's trajectory in latent space; classify test windows by proximity to per-class prototypes computed from training patients. Especially relevant when patient-level clustering exists (some patients have more similar relapse patterns than others).

**Test-Time Adaptation (TTA)**: A 2024 comprehensive survey on TTA (IJCV, 2024) covers methods like TENT (entropy minimization on test batch statistics) and TTT (self-supervised auxiliary task at test time). For LOPO, the held-out patient's unlabeled test windows could be used for batch-norm adaptation or entropy minimization. Low-risk, potentially +1-2% AUROC.

**Recommendation**: Try patient embeddings first (simplest, no arch change beyond concat). Then Reptile meta-learning. Expected impact: patient embeddings +1-2%, Reptile +2-4% AUROC.

---

## 4. Temporal Modeling

### Findings

**Bidirectional Attention**: BiTimelyGPT (arXiv:2402.09558, 2024) alternates forward and backward attention across layers for healthcare time series. It outperforms unidirectional transformers on Sleep-EDF and ECG tasks. For 7-day windows where the prediction target is the last day, bidirectional attention lets the model use "future" context from within the window (all 7 days are observed; we predict the label of the final day). This is valid because the window is fully observed at prediction time.

**Causal Masking**: Not appropriate here. Causal masking would prevent day 7 from attending to... itself. Since all 7 days are observed inputs and the target is a classification of the window's endpoint, full bidirectional attention is correct and preferred.

**Multi-Scale Windows**: The MSTformer (Wu et al., 2024) demonstrates learnable multi-scale attention that divides sequences into varying temporal scales. For your problem: instead of just 7-day windows, create parallel branches for 3-day, 7-day, and 14-day windows, each processed by a shared-weight encoder, then fuse representations. The 3-day branch captures acute changes; the 14-day branch captures gradual trends. A multiscale model for multivariate time series (Scientific Reports, 2024) confirms that multi-resolution inputs improve classification by capturing patterns invisible at any single scale.

**Implementation**: Simplest approach is to feed all three window sizes through the same transformer (with positional encoding adapted per scale), then concatenate the [CLS] tokens and classify.

**Recommendation**: (a) Switch to bidirectional attention (free improvement). (b) Add 3-day and 14-day window branches. Expected impact: bidirectional +1-2%, multi-scale +2-4% AUROC.

---

## 5. Regularization for Tiny Datasets

### Findings

**Mixup on Time Series**: Zhu et al. (2026, IET Image Processing) demonstrate that combining Mixup + Label Smoothing cuts error by >50% on small imbalanced datasets. For time series specifically, a 2024 study (PMC12099431) shows Mixup on wearable sensor temporal representations outperforms standard augmentation. Implementation for your case:

```python
# Window-level mixup
lam = np.random.beta(alpha, alpha)
x_mixed = lam * x_i + (1 - lam) * x_j
y_mixed = lam * y_i + (1 - lam) * y_j
```

**Manifold Mixup**: Verma et al. (ICML 2019) perform interpolation in hidden layers rather than input space. This creates smoother decision boundaries in representation space. UMAP Mixup (arXiv:2312.13141) ensures synthetic samples stay on the data manifold.

**Label Smoothing**: Replace hard labels (0/1) with soft labels (0.05/0.95). Harshey & Pharwaha (2024, IJIST) show this combined with stochastic depth improves generalization on imbalanced medical data. Label smoothing is especially important when using focal loss, as it prevents the model from becoming overconfident on easy examples.

**Stochastic Depth**: Randomly drop entire transformer layers during training (survival probability ~0.8-0.9). Acts as implicit ensemble and reduces overfitting. Well-established for small datasets.

**MixMatch / FixMatch Semi-Supervised**: Li et al. (2025, CPE) show that combining contrastive pretraining with FixMatch achieves satisfactory recognition with only 1% labeled data. Your train_* sequences (confirmed stable, label=0) could be used as unlabeled data in a semi-supervised framework.

**Recommendation**: Layer these: Label Smoothing (eps=0.1) + Manifold Mixup (alpha=0.2) + Stochastic Depth (p=0.1). Each is near-zero cost. Expected impact: +2-4% AUROC collectively, primarily through reduced overfitting.

---

## 6. Ensemble Strategies

### Findings

**Stacking > Simple Averaging**: Multiple clinical prediction studies (PMC10785929, 2024) confirm stacking outperforms simple averaging and weighted averaging. A meta-learner (logistic regression or small MLP) trained on held-out validation predictions from base models learns optimal combination weights. For LOPO: use inner CV within training folds to generate stacking features.

**Heterogeneous Ensembles**: Combining architecturally different models (your d=256/512/1024 transformers, plus XGBoost, plus possibly a TCN) yields more diverse predictions than ensembling variants of the same architecture. Diversity is the key driver of ensemble improvement.

**Recommended Strategy**:
1. Train d=256, d=512, d=1024 transformers + XGBoost on each LOPO fold
2. Use out-of-fold predictions (from inner 8-fold CV within training patients) as features for a logistic regression meta-learner
3. The meta-learner outputs calibrated probabilities

**Rank-based averaging**: A simpler but effective alternative -- average the rank-transformed predictions from each model. This is robust to score miscalibration across models.

**Expected impact**: +2-5% AUROC, +3-6% AUPRC from stacking heterogeneous models vs single best model.

---

## 7. Calibration & AUPRC

### Findings

**Critical insight**: Post-hoc calibration (Platt scaling, isotonic regression, temperature scaling) primarily improves probability quality (Brier score, ECE) but does NOT reliably improve ranking metrics (AUROC, AUPRC). Sakai et al. (2026, J. Neuroimaging) directly measured this: isotonic calibration *decreased* PR-AUC from 0.86 to 0.75 (delta -0.11). Platt scaling had zero effect on PR-AUC.

**What DOES improve AUPRC**:
- **Focal loss** (already used): inherently improves calibration by increasing prediction entropy, which indirectly helps AUPRC through better-ranked minority-class predictions.
- **Temperature scaling during training**: Rather than post-hoc, use a learnable temperature parameter that's optimized on validation AUPRC directly.
- **Threshold-moving**: Not calibration per se, but optimizing the classification threshold on val-set AUPRC can significantly improve F1/precision/recall without changing the model.
- **Cost-sensitive focal loss**: Focal loss with asymmetric class weights alpha_1 > alpha_0 further emphasizes minority class.

**Recommendation**: Do NOT invest in post-hoc calibration for AUPRC improvement. Instead: (a) tune focal loss gamma and alpha on val AUPRC, (b) use ensemble predictions (which are naturally better calibrated). Expected AUPRC improvement: +1-3% from focal loss tuning.

---

## 8. Contrastive / Self-Supervised Pretraining

### Findings

**TS-TCC** (Eldele et al., IJCAI 2021): Temporal and Contextual Contrasting for time series. Cross-view prediction task + contextual similarity maximization. Works with fully unlabeled data, then fine-tunes with few labels.

**TS2Vec** (Yue et al., AAAI 2022): Hierarchical contrastive learning preserving both local and global temporal consistency. Produces multi-scale representations.

**TF-C** (Zhang et al., NeurIPS 2022): Time-Frequency Consistency. Self-supervised pretraining that aligns time-domain and frequency-domain views. Outperforms baselines by 15.4% F1 in cross-domain transfer settings. Evaluated on EEG, EMG, HAR datasets.

**MACL** (Dixon et al., 2024, CPE): Modality-Aware Contrastive Learning. Uses different sensing modalities as different views of the same input for contrastive learning. Directly relevant -- your 5 modality groups (sleep, steps, HRV, circadian, demographics) are natural view candidates.

**Applicability to your problem**: You have ~2900 training windows labeled + the train_* sequences (all stable, label=0) which provide additional unlabeled-ish data. The pretraining strategy:
1. Pretrain encoder on ALL available windows (including stable-only train_* data) using TS-TCC or MACL
2. Fine-tune with labels on the labeled windows

**Caveat**: With only ~2900 windows total, the benefit of self-supervised pretraining is uncertain. SSL shines when unlabeled data vastly exceeds labeled data. Here the ratio isn't extreme. The raw sensor data (before feature extraction) might be a better pretraining substrate, but that requires a different architecture.

**Recommendation**: Try MACL-style pretraining using modality groups as views. Low risk. Expected impact: +1-3% AUROC if implemented well, but uncertain given dataset size.

---

## Priority Ranking (by expected impact * feasibility)

| Rank | Technique | Expected AUROC Gain | Expected AUPRC Gain | Effort |
|------|-----------|-------------------|---------------------|--------|
| 1 | Multi-task (days-to-relapse regression) | +2-4% | +3-5% | Medium |
| 2 | Regularization stack (label smoothing + mixup + stochastic depth) | +2-4% | +2-3% | Low |
| 3 | Stacking ensemble (heterogeneous models) | +2-5% | +3-6% | Medium |
| 4 | Multi-scale windows (3+7+14 day) | +2-4% | +2-3% | Medium |
| 5 | Bidirectional attention | +1-2% | +1-2% | Low |
| 6 | Patient embeddings | +1-2% | +1-2% | Low |
| 7 | Modality cross-attention | +1-3% | +1-3% | Medium-High |
| 8 | Reptile meta-learning | +2-4% | +2-3% | High |
| 9 | Self-supervised pretraining (MACL) | +1-3% | +1-2% | High |
| 10 | Focal loss gamma/alpha tuning | +0-1% | +1-3% | Low |
| 11 | Test-time batch-norm adaptation | +1-2% | +0-1% | Low |
| 12 | RoPE positional encoding | +0-1% | +0-1% | Low |

**Quick wins (low effort, try first)**: #2 (regularization stack), #5 (bidirectional attention), #6 (patient embeddings), #10 (focal loss tuning), #12 (RoPE)

**Medium effort, high payoff**: #1 (multi-task), #3 (stacking), #4 (multi-scale)

**High effort, uncertain payoff**: #8 (Reptile), #9 (SSL pretraining)

**Do NOT pursue**: Post-hoc calibration (Platt/isotonic) -- evidence shows it hurts AUPRC.

---

## Key References

- Wang et al. (2025) "Harnessing Small-Data ML for Mental Health Forecasting" - Med Research 1(2):226-238
- Harutyunyan et al. (2019) "Multitask learning and benchmarking with clinical time series data" - Scientific Data
- Zhang et al. (2022) "Self-Supervised Contrastive Pre-Training via Time-Frequency Consistency" - NeurIPS 2022
- Dixon et al. (2024) "Modality Aware Contrastive Learning for Multimodal HAR" - CPE 36(16)
- Eldele et al. (2021) "Time-Series Representation Learning via Temporal and Contextual Contrasting" - IJCAI
- Wu et al. (2024) "MSTformer: Multiscale spatial-temporal transformer" - CPE 36(27)
- Lilli et al. (2021) "A Calibrated Multiexit Neural Network" - CMMM 2021 (focal loss + temperature scaling)
- Sakai et al. (2026) "Explainable ML for Carotid Plaque" - J. Neuroimaging (calibration hurts PR-AUC)
- BiTimelyGPT (arXiv:2402.09558, 2024) - bidirectional generative pretraining for healthcare time series
- Zhu et al. (2026) "RASMN" - IET Image Processing (mixup + label smoothing on small datasets)
- ArmFormer (Huang et al., 2025) - multi-scale temporal transformer for wearable cardiac monitoring
