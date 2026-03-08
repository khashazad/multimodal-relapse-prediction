#!/bin/bash
# Run new experiments sequentially, with folds parallelized within each experiment.
# Each experiment's parameter combinations (mostly folds) are launched as background
# processes, then we wait for all to finish before starting the next experiment.

set -e

cd "$(dirname "$0")/.."
source env/bin/activate

EXPERIMENTS=(
    exp_ablation_dropout_only
    exp_ablation_loss_only
    exp_ablation_small_model
    exp_asymmetric_focal
    exp_dann_focal
    exp_focal_alpha_sweep
    exp_focal_augmented
    exp_focal_earlystop_auprc
    exp_focal_gamma_sweep
    exp_focal_label_smoothing
    exp_focal_lr_warmup
    exp_gated_focal
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for EXP in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    CONFIG="configs/${EXP}.json"

    # Count parameter combinations
    PARAM_COMBINATIONS=$(python3 -c "
import json, functools, operator
d = json.load(open('$CONFIG'))
lists = [v for k, v in d.items() if isinstance(v, list) and k not in ('executor', 'slurm') and not k.startswith('_')]
print(functools.reduce(operator.mul, [len(l) for l in lists], 1))
")

    echo ""
    echo "=============================================="
    echo "[$CURRENT/$TOTAL] Running: $EXP ($PARAM_COMBINATIONS jobs in parallel)"
    echo "=============================================="

    # Launch all parameter combinations in parallel
    PIDS=()
    for INDEX in $(seq 0 $((PARAM_COMBINATIONS - 1))); do
        python src/executor.py --config="$CONFIG" --name="$EXP" --index="$INDEX" --lang="python" --source="train.py" &
        PIDS+=($!)
    done

    # Wait for all parallel jobs and track failures
    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "WARNING: $FAILED/$PARAM_COMBINATIONS jobs failed for $EXP"
    else
        echo "DONE: $EXP ($PARAM_COMBINATIONS/$PARAM_COMBINATIONS succeeded)"
    fi
done

echo ""
echo "All experiments complete."
deactivate
