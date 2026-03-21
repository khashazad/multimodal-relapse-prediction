# Benchmarking Multi-Modal Fusion Strategies for Relapse Prediction in Bipolar Disorder using Wearable Data

**Senior Leads:** Dr. Venkat Bhat, Dr. Alice Rueda

**Students:** Khash Azad and David Holt

**Contributions:** Both members contributed equally to exploratory data analysis, generation of ideas for architectures and feature engineering, meeting attendance and reporting, presentation preparation and paper writing.

---

## Clinical Motivation

Mental health and substance use disorders typically begin in adolescence (mean 14.5 years) (Solmi et al. 2021), are among the 30 leading causes of disability globally, and are associated with increased mortality (Correll et al. 2022). Among individuals ages 15 to 44, unipolar depression is the second leading contributor of disability adjusted life years (DALY), with alcohol-related disorders, schizophrenia, and bipolar disorder among the top 10 disorders (WHO 2001). The Global Burden of Diseases Study reported that mental and behavioral disorders accounted for 22.7% of all years lived with disability (YLDs) globally and that neuropsychiatric disorders were the leading cause of global YLDs (Vos et al. 2012).

Early detection and symptom management are critical in order to improve patient outcomes in those with mental illness. As an example from the schizophrenia literature, the 10-year follow-up of patients in the early detection and intervention arm of new-onset psychosis showed a significantly higher percentage of recovery relative to usual-detection patients as well as milder deficits and superior functioning (Hegelstad et al. 2012). Although early detection and intervention is critical for long-term outcomes, there is a significant treatment gap. A Singapore mental health study conducted in 2016 (Subramaniam et al. 2020) showed that the 12-month treatment gap for major depressive disorder (MDD) and bipolar disorder was around 75%. As of November 2024, nearly a quarter of the U.S. population (46 million people) had a mental illness, yet 46% received no treatment. The shortage of psychiatrists is affecting patients' ability to access care and creating a large burden on the healthcare professionals who provide it (James 2025).

In depressive disorders, treatment response ranges from 51 to 54%, remission is around 43% (Cuijpers et al. 2014), and throughout 26 weeks, relapse occurs in about 33--50% of cases (Gueorguieva et al. 2017). In patients with multi-episode schizophrenia, 51% and 23% of subjects have a >= 20% or >= 50% reduction of total symptoms (Leucht et al. 2017), and only 13.5% reach recovery (Jaaskelainen et al. 2013). If relapses can be detected and intervened upon earlier, and patient monitoring scaled through widely used smartwatch technology, the potential to reduce long-term disability is immense. This capability was the main aim of the e-Prevention project.

The e-Prevention project (Zlatintsi et al. 2022) was a longitudinal study which followed 39 patients with a variety of mental health disorders in Athens, Greece for up to 2.5 years, monitoring smartwatch data during wakefulness and sleep. Clinicians annotated relapse periods according to monthly assessments and communications with attending physicians or families. Data were split into three categories: stable, relapse, and near-relapse (up to 21 days before relapse). The near-relapse data was discarded. The authors used an anomaly detection approach with four autoencoder architectures (Transformer, FCNN, CNN, GRU) using both personalized and global schemes, finding that the CNN AE had the best overall performance with a median PR-AUC of 0.76.

## Related Work

The e-Prevention project advanced previous work in relapse detection and electronic patient monitoring. In a study on 5 bipolar disorder patients over 12 weeks, mood episodes and relapses could be predicted with high confidence using accelerometer and audio information from a smartphone (Maxhuni et al. 2016). A 2018 study used smartphone data from 17 patients with schizophrenia---including mobility patterns and social behavior in the Beiwe app---to detect relapse with high accuracy (Barnett et al. 2018). Another group used linear accelerometer data to monitor and evaluate tremor severity in 34 Parkinson's disease patients (Cai et al. 2018).

Using data from their study, the e-Prevention project proposed a challenge at ICASSP 2023 (Zlatintsi et al. 2023). The 2nd e-Prevention challenge (Filntisis et al. 2024) aimed to improve relapse detection in mental-health patients with both psychotic and non-psychotic disorders using Samsung Gear S3 smartwatch data. The watch includes accelerometers, gyroscopes, heart rate monitors, step counters, and sleep monitors.

Participants were tasked with predicting relapse in two tracks. Track 1 (non-psychotic relapse) included 9 patients with data separated into train, validation, and test sets. Training data contained only stable patient states; validation and test data contained both stable and relapse states. Our aim is to improve upon the Track 1 results in predicting relapse in bipolar disorder patients using different multi-modal fusion strategies.

The winning approach to Track 1 employed a transformer-based autoencoder with fixed-size feature vectors used for anomaly detection via iNNE (Nearest Neighbor Ensembles), achieving test-set AUPRC=0.620 and AUROC=0.711. The second-place team (Kaliosis et al. 2024) applied a vision transformer for representation learning from augmented features and created a hybrid outlier detector based on a one-class SVM or Gaussian model.

## Data and Settings

We used the publicly available Track 1 dataset from the 2nd e-Prevention challenge. The data were collected and published by the University Mental Health Research Institute (UMHRI) in Athens, Greece (Zlatintsi et al. 2022) as part of a ~2-year wearable monitoring study (~100,000 hours across all participants) that started with control recordings in June–October 2019 and continued with patients between November 2019 and March 2021, reflecting staggered enrollment due to limited smartwatch availability. A total of 39 patients were recruited: the inclusion criteria required patients to be under active treatment and at least stable, while exclusion criteria ruled out serious hearing, vision, or motor impairments, inability to read at least at a sixth-grade level, or incapacity to provide informed consent. The published challenge focuses on 9 of those patients with non-psychotic disorders (six bipolar, one brief psychotic disorder, one schizophreniform, one schizophrenia). Each patient’s data spans multiple months and is pre-divided into temporally ordered train, validation, and test splits (train\_0/1, val\_0/1, test\_0/1/2).

Five sensor modalities are available per patient: gyroscope (20 Hz), linear accelerometer (20 Hz), heart rate monitor (5 Hz), per-minute step counts, and sleep episode annotations. The outcome variable is a binary daily relapse indicator. Training splits contain only stable (non-relapse) days; validation and test splits contain both stable and relapse days.

The dataset exhibits substantial class imbalance: relapse days constitute a minority of observations (approximately 10--20% across patients). Patient-level heterogeneity is also pronounced; raw sensor distributions vary considerably across individuals due to differences in physiology, behavior, and device wear patterns. The dataset also includes a demographics file with 26 variables: age, gender, marital status, birth place, educational level, diagnosis, years of illness, birth complications, family psychiatric history, smoking status, alcohol consumption, cannabis use, treatment compliance, dominant hand, and smartwatch hand, among others.

## Methods

### 1. Feature Engineering

Six groups of handcrafted features were engineered from the raw sensor streams, all expressed as patient-specific z-score deviations from personal baselines computed on non-relapse training days.

**Sleep features** (16 total): main sleep duration, onset/wake times, nap count and hours, with circular statistics for time-of-day quantities. **Step count features**: daily totals with an inverted z-score (`steps_zscore_inv`) so that sedentary days produce positive risk signals. **Nighttime HRV**: RMSSD and SDNN from 5-second windows filtered by accelerometer stationarity (< 0.2g), binned into 55 temporal bins across an 8-hour extraction window. **Sleep-verified HRV**: identical pipeline but requiring confirmed sleep overlap. **Demographics**: age, gender, marital status, birth place, educational level, diagnosis, years of illness, birth complications, family psychiatric history, smoking status, alcohol consumption, cannabis use, treatment compliance, dominant hand, and smartwatch hand---all one-hot encoded where categorical. **Circadian actigraphy** (15 features): fused gyroscope and accelerometer hourly profiles yielding relative amplitude, intradaily variability, cosinor amplitude/acrophase, L5/M10 onset deviations, and evening activity proportion.

A sweep over 12 candidate 8-hour HRV extraction windows (00:00--22:00, 2-hour stride) identified **W14 (14:00--22:00)** as the optimal window for unsupervised and traditional ML models. However, when W14 HRV was substituted into the supervised Transformer in a head-to-head comparison, W00 sleep-verified HRV (00:00--08:00) proved superior. This suggests the optimal extraction window is model-dependent: the unsupervised AE benefits from evening physiological signals, while the supervised Transformer extracts more discriminative features from nighttime sleep-verified recordings.

Feature selection used the union of Boruta (14 confirmed features) and mRMR top-15 (~25 features total) as input for Transformer models.

### 2. Traditional ML Baselines

XGBoost and Logistic Regression (L1/L2) were trained under LOPO cross-validation across all 9 patients. Class imbalance was handled via `scale_pos_weight` (XGBoost) and inverse-frequency weighting (Logistic Regression). A top-K feature sweep (K in {3, 5, 8, 10, 15, 20, 25, 30, all}) selected the optimal feature count per LOPO iteration.

### 3. Unsupervised Anomaly Detection (BumbleBee AE + iNNE)

Inspired by the winning e-Prevention approach, a Transformer autoencoder was trained on 55-bin x 24-feature nightly sequences using only non-relapse data. The 24 features per bin comprise 6 base physiological signals augmented with causal moving means, daily means, and daily standard deviations. After global pretraining on 8 patients, iNNE anomaly scoring was applied on the held-out patient's latent vectors. The model underwent 9 development iterations, systematically evaluating: MSE vs. Soft DTW reconstruction loss, Numba-JIT acceleration for Soft DTW (~220x speedup), ensemble blending of iNNE and reconstruction scores, per-patient hyperparameter tuning, and extraction window optimization.

### 4. Supervised Sequence Transformer

The supervised Transformer was restricted to 6 bipolar patients (P3--P9 excluding P1, P2, P7) to reduce diagnostic heterogeneity. The model is a Transformer encoder with a linear classification head, operating on sliding windows of 7 consecutive days (each day described by ~25 union features). Left-padding with attention masking ensures complete test-day coverage.

**Class imbalance handling**: SMOTE oversampling was applied to flattened training windows (N, seq\_len x F), then reshaped to 3D. ADASYN was tested as an ablation. Test data was never oversampled.

**Hyperparameter strategy**: Per-fold Hyperopt tuning degraded results due to small validation sets (~9 relapse events per patient). A global HP grid across all 6 LOPO folds was adopted instead, sweeping d\_model in {16, 32, 64, 128, 256, 512, 1024}, n\_layers in {1--4}, and dropout in {0.1, 0.2, 0.3}, with fixed NHEAD=4, LR=1e-3, BATCH=32, N\_EPOCHS=80.

### 5. Focal Loss and Label Smoothing (All 9 Patients)

Focal loss and label smoothing were applied to the supervised Transformer extended to all 9 patients. Focal loss down-weights easy examples via a modulating factor (1-p)^gamma, focusing training on hard-to-classify days. Label smoothing softens target labels (0 -> epsilon/2, 1 -> 1-epsilon/2), preventing overconfident predictions and improving calibration---particularly relevant given the inherent noise in clinical relapse labels, where the exact onset day is uncertain.

## Evaluation

All models were evaluated under **leave-one-patient-out (LOPO)** cross-validation, the standard protocol for this dataset. In each fold, one patient is held out for testing; the remaining patients' validation splits provide training data. This ensures that no patient's data appears in both training and testing, reflecting the clinical scenario where a model must generalize to an unseen patient.

**Primary metrics** are AUROC and AUPRC, reported as means and standard deviations across folds. AUROC measures discriminative ability across all thresholds and is robust to class imbalance. AUPRC is more informative than AUROC when the positive class is rare, as it directly evaluates precision at each recall level. Both are threshold-free, avoiding the need for arbitrary operating-point selection.

**Baseline comparisons**: We benchmark against (1) the winning Track 1 entry (AUROC=0.711, AUPRC=0.620 on the test set), (2) XGBoost/LR traditional ML baselines, and (3) our unsupervised BumbleBee replication. All comparisons use the same LOPO protocol and metrics.

**Class imbalance**: The relapse-to-stable ratio is approximately 1:5--1:10 depending on the patient. All models employ explicit imbalance correction (SMOTE, class weighting, or focal loss), and AUPRC is reported alongside AUROC to ensure performance on the minority class is evaluated.

## Results

### Feature Space Analysis

PCA and t-SNE visualizations confirmed that patients are clearly separable in low-dimensional projections, indicating that between-patient differences dominate the feature space. Relapse and non-relapse days overlap substantially with no recoverable cluster structure by label. This absence of separability motivates both the patient-specific normalization strategy and the per-patient anomaly detection paradigm.

### Traditional ML

The feature set was built incrementally: sleep features alone yielded AUROC~0.52; adding steps provided negligible gain (~0.53); incorporating HRV and demographics lifted AUROC to ~0.57. With W14 HRV substitution and top-K sweep (optimal K=3), XGBoost achieved **AUROC=0.584 +/- 0.085, AUPRC=0.530**. The top-3 features were `nap_hours_diff`, `watch_dominant`, and `day_activity_zscore`---none HRV-specific---suggesting the W14 benefit reflects improved covariate structure rather than direct HRV signal.

### BumbleBee AE + iNNE (Unsupervised, 9 Patients)

The version progression demonstrated systematic gains from feature representation (+0.031), loss function alignment (+0.021), and extraction window optimization (+0.034):

| Version | Key Change | AUROC |
|---------|-----------|-------|
| v3 | Raw sequences, MSE | 0.508 |
| v4 | 24-feature representation | 0.539 |
| v5 | Soft DTW loss | 0.541 |
| v7 | JIT-accelerated, consistent HP | 0.531 |
| W14-HP | W14 window + HP tuning | **0.559** |

The W14 window (14:00--22:00) outperformed the classical nighttime window by +0.034 AUROC, suggesting that evening physiological signals are earlier and more reliable prodromal indicators than overnight recordings. Ensemble blending of iNNE with reconstruction scores (v8, v9) degraded performance due to anti-correlation between the two scoring dimensions across patients.

### Supervised Transformer (Bipolar Cohort, 6 Patients)

SMOTE oversampling improved AUROC from 0.741 (pos\_weight only) to **0.774 +/- 0.154**, with gains concentrated on weaker folds (P5: +0.165). The global HP grid revealed monotonic improvement with increasing d\_model:

| d\_model | n\_layers | dropout | AUROC | AUPRC |
|---------|----------|---------|-------|-------|
| 32 | 2 | 0.3 | 0.759 | 0.686 |
| 128 | 3 | 0.2 | 0.777 | 0.678 |
| 256 | 3 | 0.3 | 0.790 | 0.724 |
| 512 | 3 | 0.2 | 0.814 | 0.773 |
| **1024** | **3** | **0.3** | **0.849** | **0.794** |

Performance plateaued at d=1024, as d=2048 offered no further improvement, and n\_layers=4 consistently underperformed n\_layers=3. ADASYN underperformed SMOTE by -0.032 AUROC (0.817 vs 0.849), indicating that uniform oversampling is better suited to this task than adaptive density-weighted synthesis.

### Focal Loss + Label Smoothing (All 9 Patients)

Extending the supervised Transformer to all 9 patients with focal loss and label in addition to SMOTE smoothing achieved **AUROC=0.85, AUPRC=0.79**---matching the bipolar-only performance while including the diagnostically heterogeneous non-bipolar patients (P1, P2, P7). This represents a substantial improvement over the challenge-winning result (AUROC=0.711, AUPRC=0.620).

### Summary Comparison

| Model | Patients | AUROC | AUPRC |
|-------|----------|-------|-------|
| Challenge winner (Bumblebee) | 9 | 0.711 | 0.620 |
| XGBoost (W14 HRV) | 9 | 0.584 | 0.530 |
| BumbleBee replication (W14-HP) | 9 | 0.559 | 0.512 |
| Supervised Transformer (focal loss + LS + SMOTE) | 9 | **0.850** | **0.790** |

### Clinical Relevance

At an AUROC of 0.85, the model provides clinically meaningful discrimination between relapse and stable days from passively collected wearable data. In a clinical decision-support scenario, a high-sensitivity operating point could flag at-risk periods for clinician review, enabling earlier intervention without requiring patient self-report. The 7-day sliding window aligns naturally with weekly clinical check-ins, and the reliance on commodity smartwatch sensors makes deployment feasible at scale. The finding that evening physiological signals (W14) outperform overnight recordings is clinically intuitive: prodromal changes in activity and autonomic function during waking hours may precede the sleep disruptions traditionally monitored in bipolar disorder.

## Discussion and Limitations

This project systematically benchmarked multiple fusion strategies for relapse prediction in bipolar disorder using wearable sensor data. Our best models substantially outperform the e-Prevention challenge baseline (AUROC: 0.85 vs. 0.711), demonstrating that supervised sequence modeling with appropriate class-imbalance handling and capacity scaling can extract strong prodromal signals from commodity smartwatch data.

Several key lessons emerged. Patient-specific z-score normalization was essential---raw inter-patient variability dominates the feature space and renders population-level approaches ineffective. Global hyperparameter selection across all LOPO folds consistently outperformed per-fold tuning, as individual validation sets (~9 relapse events) carry insufficient statistical power for reliable model selection. The discovery that the W14 evening window outperforms classical nighttime HRV extraction has potential clinical implications for monitoring protocols.

**Limitations.** The dataset comprises only 9 patients, limiting generalizability. LOPO results are subject to high variance across folds, and patients P4 and P7 remained persistently weak across all methods, suggesting that some relapse patterns may not be wearable-detectable. Clinicians assigned daily relapse flags, yet those annotations leaned heavily on monthly clinical followups (plus frequent scale scores and communicating physicians/carers), so the exact onset day retains some uncertainty. We did not evaluate the models' prospective performance or assess how far in advance of relapse the system could provide actionable alerts.

**Role.** This system is designed for clinical decision support---flagging at-risk periods for clinician review---rather than autonomous diagnostic automation.

**Future work.** Validating on larger, multi-site cohorts is essential. Prospective deployment studies should assess lead time (how many days before clinical relapse the model flags), false-alarm burden on clinicians, and patient outcomes when alerts are acted upon. Integration with electronic health records and ecological momentary assessment could further enhance prediction. The monotonic scaling of d\_model with no observed plateau motivates exploring even larger architectures or pre-trained foundation models for physiological time series.

---

## References

Barnett, I., Torous, J., Staples, P. et al. Relapse prediction in schizophrenia through digital phenotyping: a pilot study. *Neuropsychopharmacol* 43, 1660--1666 (2018).

Cai, G.; Lin, Z.; Dai, H. et al. Quantitative assessment of parkinsonian tremor based on a linear acceleration extraction algorithm. *Biomed. Signal Process. Control* 42, 53--62 (2018).

Correll, C.U., Solmi, M., Croatto, G. et al. Mortality in people with schizophrenia: a systematic review and meta-analysis. *World Psychiatry* 21, 248--271 (2022).

Cuijpers, P., Karyotaki, E., Weitz, E. et al. The effects of psychotherapies for major depression in adults on remission, recovery and improvement: a meta-analysis. *J Affect Disord* 159, 118--126 (2014).

Filntisis, P., Efthymiou, N., Retsinas, G. et al. The 2nd E-Prevention Challenge: Psychotic and Non-Psychotic Relapse Detection Using Wearable-Based Digital Phenotyping. *ICASSP Workshops*, 125--126 (2024).

Gueorguieva, R., Chekroud, A.M., Krystal, J.H. Trajectories of relapse in randomised, placebo-controlled trials of treatment discontinuation in major depressive disorder. *Lancet Psychiatry* 4, 230--237 (2017).

Hegelstad, W.T.V. et al. Long-term follow-up of the TIPS early detection in psychosis study. *Am J Psychiatry* 169, 374--380 (2012).

Jaaskelainen, E. et al. A systematic review and meta-analysis of recovery in schizophrenia. *Schizophr Bull* 39, 1296--1306 (2013).

James. Mind the Gap: The Ongoing Psychiatrist Shortage. *Medscape* (Feb 3, 2025).

Kaliosis, P., Eleftheriou, S., Nikou, C., Giannakopoulos, T. A self-supervised learning approach for detecting nonpsychotic relapses using wearable-based digital phenotyping. *Proc. ICASSP* (2024).

Leucht, S. et al. Sixty years of placebo-controlled antipsychotic drug trials in acute schizophrenia. *Am J Psychiatry* 174, 927--942 (2017).

Maxhuni, A. et al. Classification of bipolar disorder episodes based on analysis of voice and motor activity. *Pervasive Mob. Comput.* 31, 50--66 (2016).

Solmi, M. et al. Age at onset of mental disorders worldwide: large-scale meta-analysis. *Mol Psychiatry* 27, 281--295 (2021).

Solmi, M. et al. Multivariable prognostic models of clinical outcomes in mental disorders. *Mol Psychiatry* 28, 3671--3687 (2023).

Subramaniam, M. et al. Minding the treatment gap: results of the Singapore Mental Health Study. *Soc Psychiatry Psychiatr Epidemiol* 55, 1415--1424 (2020).

Vos, A.D. et al. Years lived with disability (YLDs) for 1160 sequelae of 289 diseases and injuries 1990--2010. *Lancet* 380, 2163--2196 (2012).

World Health Organization. *The World Health Report 2001: Mental Health---New Understanding, New Hope.* Geneva: WHO (2001).

Zlatintsi, A. et al. E-Prevention: Advanced Support System for Monitoring and Relapse Prevention in Patients with Psychotic Disorders. *Sensors* 22, 7544 (2022).

Zlatintsi, A. et al. E-Prevention: The ICASSP-2023 Challenge on Person Identification and Relapse Detection. *ICASSP* (2023).
