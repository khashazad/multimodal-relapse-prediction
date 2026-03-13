# 009: Next Phase — Ensemble + TCN + Feature Expansion

## Summary
Experiments to improve on baseline all-9-patient AUROC of 0.793 (d=1024 transformer). Two execution paths: SLURM cluster (per-fold parallelism) and local notebook (sequential).

## Files Created
| File | Purpose |
|------|---------|
| `src/next_phase.py` | Click CLI training script (executor-compatible) |
| `configs/next_phase_transformer.json` | Phase A: 27 jobs (9 folds × d=256/512/1024) |
| `configs/next_phase_tcn.json` | Phase B: 108 jobs (9 folds × 12 TCN configs) |
| `configs/next_phase_features.json` | Phase C: 36 jobs (9 folds × 4 feature sets) |
| `notebooks/next_phase.ipynb` | Self-contained notebook (same experiments, local execution) |

## SLURM Execution
```bash
# Phase A: multi-scale transformer ensemble
bash scripts/submit_slurm.sh -n next_phase_transformer

# Phase B: TCN baseline sweep
bash scripts/submit_slurm.sh -n next_phase_tcn

# Phase C: feature expansion
bash scripts/submit_slurm.sh -n next_phase_features

# Or run locally (sequential):
bash scripts/run.sh -n next_phase_transformer
bash scripts/run.sh -n next_phase_tcn
bash scripts/run.sh -n next_phase_features
```

Results saved to `outputs/next_phase/` as per-fold JSONs. Aggregation (ensemble averaging, significance testing) done post-hoc.

## Output JSON Schema
Each job produces `outputs/next_phase/{exp_name}__fold={N}__...json` with:
- `test_metrics.auroc`, `test_metrics.auprc` — per-fold metrics
- `y_true`, `y_score` — raw predictions for ensemble aggregation
- `history` — per-epoch training curves
- `hyperparameters`, `model_params`, `elapsed_seconds`

## Notebook Structure
| Cell | Purpose |
|------|---------|
| 0 | Title markdown |
| 1 | Setup: imports + load `patient_data_export_all9.pkl` |
| 2 | Infrastructure: SeqTransformer, TCN, SMOTE, LOPO loop |
| 3 | **Phase A**: Train d=256/512/1024, simple + weighted ensemble |
| 4 | **Phase B**: TCN (12 HP configs), ranking, param comparison |
| 5 | **Phase C**: Interaction features, univariate screening, 4 feature sets |
| 6 | **Phase D**: TF+TCN meta-ensemble, comparison table, Wilcoxon test, viz |

## Status
Script + configs created and smoke-tested (all 3 methods verified). Not yet submitted to cluster.
