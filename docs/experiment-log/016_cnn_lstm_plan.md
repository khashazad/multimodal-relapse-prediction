# 016: CNN+LSTM Per-Modality Architecture

## Summary
Implemented CNN+LSTM architecture adapting a cardiac arrest prediction paper to bipolar relapse prediction. Key insight: raw LSTM on noisy physiological signals fails; CNN-derived latent features from binned signals stabilize training.

## Architecture
- **5-minute bins** (288 timesteps/day): mean+std per channel per bin
  - IMU (linacc/gyr): 8 channels (X/Y/Z/mag × mean/std)
  - HRM: 4 channels (hr/rr × mean/std)
- **Day-level CNN**: Conv1d→Tanh→MaxPool→Dropout (×2) → GlobalAvgPool → 16-dim latent
- **Across-day LSTM**: 7 daily latents → final hidden → classification
- **Episodic branches** (step/sleep): FC→ReLU→LSTM for hand-crafted features
- **Three fusion modes**: learned (softmax weights), mean, concat

## Model Size
- Full ensemble (learned): 22,330 params
- Single branch (linacc): 6,081 params
- Much lighter than transformer baseline (~316K params)

## Files Created
12 files total — see TASK.md for full list.

## Baselines to Beat
| Method | Test AUROC |
|--------|-----------|
| Single best transformer | 0.857 |
| Rank-avg 6 transformers | 0.912 |
| Trans+FiLM rank-avg | 0.938 |

## Run Instructions
```bash
# 1. Preprocess (slow: ~30-60min locally, or use SLURM --patient mode)
python scripts/preprocess_cnn_data.py

# 2. Joint training (all branches end-to-end)
bash scripts/submit_slurm.sh -n cnn_lstm_joint

# 3. Independent per-modality training
bash scripts/submit_slurm.sh -n cnn_lstm_independent

# 4. Post-hoc ensemble (after step 3)
python -m src.ensemble_cnn

# 5. Hyperparameter sweep
bash scripts/submit_slurm.sh -n cnn_lstm_sweep
```
