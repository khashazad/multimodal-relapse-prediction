#!/bin/bash
# Master submission script: preprocess → ablations → exploration experiments.
#
# Ablation configs run immediately (data exists in patient_data_export_all9.pkl).
# Exploration configs depend on preprocessing (data/processed/track1/fold_*/).
#
# Usage:
#   bash scripts/submit_all.sh [--dry-run]

set -e

DRY_RUN=""
for arg in "$@"; do
    case $arg in
        --dry-run|-d) DRY_RUN="--dry-run" ;;
        --help|-h)
            echo "Usage: submit_all.sh [--dry-run]"
            echo ""
            echo "Submit all experiments to SLURM with dependency chaining."
            echo "Ablations run immediately; exploration waits for preprocessing."
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS_DIR="configs"

# Skip non-experiment configs
SKIP="preprocessing"

# Track submitted jobs
ABLATION_JOBS=()
EXPLORE_JOBS=()
TOTAL_TASKS=0

echo "=== Phase 1: Preprocessing ==="
echo ""

if [ -n "$DRY_RUN" ]; then
    echo "(dry-run) bash scripts/submit_preprocess.sh --dry-run"
    bash scripts/submit_preprocess.sh --dry-run 2>&1
    MERGE_JOB_ID="DRY_RUN"
else
    # Capture the merge job ID (last line of submit_preprocess.sh output contains it)
    PREPROCESS_OUTPUT=$(bash scripts/submit_preprocess.sh 2>&1)
    echo "$PREPROCESS_OUTPUT"
    MERGE_JOB_ID=$(echo "$PREPROCESS_OUTPUT" | grep -o 'Merge job submitted: *[0-9]*' | grep -o '[0-9]*')
    if [ -z "$MERGE_JOB_ID" ]; then
        echo "ERROR: failed to capture merge job ID from preprocessing" >&2
        exit 1
    fi
    echo ""
    echo "Merge job ID: $MERGE_JOB_ID"
fi

echo ""
echo "=== Phase 2: Ablation experiments (immediate) ==="
echo ""

for config in "$CONFIGS_DIR"/ablation_*.json "$CONFIGS_DIR"/ablv2_*.json; do
    [ -f "$config" ] || continue
    name=$(basename "$config" .json)

    # Count jobs
    n_jobs=$(jq -r '
        to_entries
        | map(select(.value | type == "array"))
        | map(.value | length)
        | reduce .[] as $item (1; . * $item)
    ' "$config")
    TOTAL_TASKS=$((TOTAL_TASKS + n_jobs))

    echo "[$name] $n_jobs jobs"
    if [ -n "$DRY_RUN" ]; then
        bash scripts/submit_slurm.sh -n "$name" --dry-run 2>&1 | tail -2
    else
        OUTPUT=$(bash scripts/submit_slurm.sh -n "$name" 2>&1)
        echo "$OUTPUT" | tail -2
        JOB_ID=$(echo "$OUTPUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        [ -n "$JOB_ID" ] && ABLATION_JOBS+=("$JOB_ID")
    fi
    echo ""
done

echo "=== Phase 3: Exploration experiments (after preprocessing) ==="
echo ""

for config in "$CONFIGS_DIR"/exp_*.json "$CONFIGS_DIR"/train.json; do
    [ -f "$config" ] || continue
    name=$(basename "$config" .json)

    # Count jobs
    n_jobs=$(jq -r '
        to_entries
        | map(select(.value | type == "array"))
        | map(.value | length)
        | reduce .[] as $item (1; . * $item)
    ' "$config")
    TOTAL_TASKS=$((TOTAL_TASKS + n_jobs))

    echo "[$name] $n_jobs jobs (depends on preprocessing)"
    if [ -n "$DRY_RUN" ]; then
        bash scripts/submit_slurm.sh -n "$name" --dry-run 2>&1 | tail -2
    else
        # Submit with dependency on preprocessing merge job
        # We need to add --dependency to the sbatch command
        # submit_slurm.sh doesn't support --dependency directly, so we
        # temporarily set SBATCH_DEPENDENCY env var
        export SLURM_DEPENDENCY="afterok:${MERGE_JOB_ID}"
        OUTPUT=$(bash scripts/submit_slurm.sh -n "$name" 2>&1)
        unset SLURM_DEPENDENCY
        echo "$OUTPUT" | tail -2
        JOB_ID=$(echo "$OUTPUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        [ -n "$JOB_ID" ] && EXPLORE_JOBS+=("$JOB_ID")
    fi
    echo ""
done

echo "=== Summary ==="
echo "Total tasks: $TOTAL_TASKS"
if [ -z "$DRY_RUN" ]; then
    echo "Preprocessing merge job: $MERGE_JOB_ID"
    echo "Ablation jobs: ${ABLATION_JOBS[*]}"
    echo "Exploration jobs: ${EXPLORE_JOBS[*]}"
    ALL_JOBS="${MERGE_JOB_ID}"
    [ ${#ABLATION_JOBS[@]} -gt 0 ] && ALL_JOBS+=",$(IFS=,; echo "${ABLATION_JOBS[*]}")"
    [ ${#EXPLORE_JOBS[@]} -gt 0 ] && ALL_JOBS+=",$(IFS=,; echo "${EXPLORE_JOBS[*]}")"
    echo ""
    echo "Monitor all:"
    echo "  squeue --jobs=$ALL_JOBS"
else
    echo "(dry-run — no jobs submitted)"
fi
