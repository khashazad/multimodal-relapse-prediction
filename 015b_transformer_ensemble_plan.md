# 015b — Transformer-Only Ensemble Re-Aggregation

## Summary
Heterogeneous ensemble (015) failed — TCN (0.657) and XGBoost dragged down performance. Filtered to transformer-only base models and re-ran aggregation. Rank-averaged 6-transformer ensemble achieves **0.912 AUROC** (+0.055 over single best 0.857, Wilcoxon p=0.010). Adding FiLM yields **0.938 AUROC** (p=0.002).

## Output
- Modified `scripts/aggregate_ensemble.py`: added `--model_filter` CLI arg (choices: all, transformer, transformer+film)
- Subset analysis: automatically computes rank/simple avg for all-6, top-4, top-2, and transformer+FiLM subsets
- Results saved to `015_ensemble_results.md`

## Key Results
| Subset | Rank Avg AUROC | Wilcoxon p |
|--------|---------------|-----------|
| All 6 transformers | 0.912 | 0.010 |
| Top-4 (d>=512) | 0.902 | 0.010 |
| Top-2 (d=1024) | 0.868 | 0.371 |
| Trans+FiLM (8) | **0.938** | **0.002** |

## Status: COMPLETED
