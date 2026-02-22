#!/bin/bash
# Submit the preprocessing pipeline as a two-phase SLURM job.
#
# Phase 1 — array job (one task per patient):
#   Each task runs:  python -u scripts/preprocess_data.py --patient <PATIENT_ID>
#   Output:          data/processed/track<N>/patients/<PATIENT_ID>.pkl
#
# Phase 2 — merge job (runs after ALL array tasks succeed):
#   Runs:  python -u scripts/preprocess_data.py --merge
#   Output: data/processed/track<N>/fold_*/  + metadata.json + patient_scalers.pkl
#
# SLURM settings are read from the 'slurm' section of configs/preprocessing.json.
#
# Usage:
#   bash scripts/submit_preprocess.sh [--dry-run]

set -e

CONFIG_FILE="configs/preprocessing.json"
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run|-d) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: submit_preprocess.sh [--dry-run]"
            echo ""
            echo "Submit the preprocessing pipeline to SLURM."
            echo "SLURM settings are read from configs/preprocessing.json."
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Read config
# ---------------------------------------------------------------------------

DATA_ROOT=$(jq -r '.data_root'           "$CONFIG_FILE")
TRACK=$(jq -r '.track'                   "$CONFIG_FILE")
TRACK_DIR="${DATA_ROOT}/track${TRACK}"

SLURM_CORES=$(jq -r     '.slurm.cores       // 4'              "$CONFIG_FILE")
SLURM_TIME=$(jq -r      '.slurm.time        // "0-04:00"'      "$CONFIG_FILE")
SLURM_MEMORY=$(jq -r    '.slurm.memory      // "32G"'          "$CONFIG_FILE")
SLURM_MERGE_TIME=$(jq -r '.slurm.merge_time // "0-00:30"'      "$CONFIG_FILE")
SLURM_MERGE_MEM=$(jq -r  '.slurm.merge_memory // "8G"'         "$CONFIG_FILE")
SLURM_LOG_DIR=$(jq -r   '.slurm.log_dir    // "outputs/slurm_logs"' "$CONFIG_FILE")
SLURM_JOB_NAME=$(jq -r  '.slurm.job_name   // "preprocess"'   "$CONFIG_FILE")
SLURM_PARTITION=$(jq -r '.slurm.partition  // empty'           "$CONFIG_FILE")
SLURM_ACCOUNT=$(jq -r   '.slurm.account    // empty'           "$CONFIG_FILE")
SLURM_EMAIL=$(jq -r     '.slurm.email      // empty'           "$CONFIG_FILE")
SLURM_EMAIL_TYPE=$(jq -r '.slurm.email_type // "FAIL"'         "$CONFIG_FILE")

# ---------------------------------------------------------------------------
# Discover patients
# ---------------------------------------------------------------------------

if [ ! -d "$TRACK_DIR" ]; then
    echo "Error: track directory not found: $TRACK_DIR" >&2
    exit 1
fi

readarray -t PATIENTS 2>/dev/null < <(ls -d "${TRACK_DIR}"/P* 2>/dev/null | xargs -I{} basename {} | sort) \
    || PATIENTS=($(ls -d "${TRACK_DIR}"/P* 2>/dev/null | xargs -I{} basename {} | sort))
N_PATIENTS=${#PATIENTS[@]}

if [ "$N_PATIENTS" -eq 0 ]; then
    echo "Error: no patient directories found in $TRACK_DIR" >&2
    exit 1
fi

echo "Found ${N_PATIENTS} patients: ${PATIENTS[*]}"

# ---------------------------------------------------------------------------
# Write a small patients index file (worker reads it by line number)
# ---------------------------------------------------------------------------

mkdir -p "$SLURM_LOG_DIR"
PATIENTS_FILE="${SLURM_LOG_DIR}/patients.txt"
printf "%s\n" "${PATIENTS[@]}" > "$PATIENTS_FILE"
echo "Patient list written to $PATIENTS_FILE"

# ---------------------------------------------------------------------------
# Build common sbatch option string
# ---------------------------------------------------------------------------

COMMON_OPTS=""
[ -n "$SLURM_PARTITION" ]  && COMMON_OPTS+=" --partition=$SLURM_PARTITION"
[ -n "$SLURM_ACCOUNT" ]    && COMMON_OPTS+=" --account=$SLURM_ACCOUNT"
[ -n "$SLURM_EMAIL" ]      && COMMON_OPTS+=" --mail-user=$SLURM_EMAIL"
[ -n "$SLURM_EMAIL_TYPE" ] && COMMON_OPTS+=" --mail-type=$SLURM_EMAIL_TYPE"

# ---------------------------------------------------------------------------
# Phase 1: per-patient array job
# ---------------------------------------------------------------------------

ARRAY_CMD=(
    sbatch --parsable
    --array="0-$((N_PATIENTS - 1))"
    --cpus-per-task="$SLURM_CORES"
    --time="$SLURM_TIME"
    --mem="$SLURM_MEMORY"
    --job-name="${SLURM_JOB_NAME}_extract"
    --output="${SLURM_LOG_DIR}/%A_%a.out"
    --error="${SLURM_LOG_DIR}/%A_%a.err"
    $COMMON_OPTS
    --export="ALL,PREPROCESS_PATIENTS_FILE=${PATIENTS_FILE}"
    --wrap="source env/bin/activate && \
        PATIENT_ID=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" \"\$PREPROCESS_PATIENTS_FILE\") && \
        echo \"Task \$SLURM_ARRAY_TASK_ID → patient \$PATIENT_ID\" && \
        python -u scripts/preprocess_data.py --patient \"\$PATIENT_ID\""
)

echo ""
echo "Phase 1 — array job command:"
echo "  ${ARRAY_CMD[*]}"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "(dry-run — not submitting)"
    exit 0
fi

ARRAY_JOB_ID=$("${ARRAY_CMD[@]}")
echo "Array job submitted: ${ARRAY_JOB_ID}"

# ---------------------------------------------------------------------------
# Phase 2: merge job (depends on all array tasks succeeding)
# ---------------------------------------------------------------------------

MERGE_CMD=(
    sbatch --parsable
    --dependency="afterok:${ARRAY_JOB_ID}"
    --cpus-per-task=1
    --time="$SLURM_MERGE_TIME"
    --mem="$SLURM_MERGE_MEM"
    --job-name="${SLURM_JOB_NAME}_merge"
    --output="${SLURM_LOG_DIR}/%A_merge.out"
    --error="${SLURM_LOG_DIR}/%A_merge.err"
    $COMMON_OPTS
    --wrap="source env/bin/activate && python -u scripts/preprocess_data.py --merge"
)

MERGE_JOB_ID=$("${MERGE_CMD[@]}")
echo "Merge job submitted:  ${MERGE_JOB_ID} (runs after ${ARRAY_JOB_ID} completes)"

echo ""
echo "Monitor with:"
echo "  squeue --job=${ARRAY_JOB_ID},${MERGE_JOB_ID}"
echo "  tail -f ${SLURM_LOG_DIR}/${ARRAY_JOB_ID}_0.out   # patient 0 log"
echo "  tail -f ${SLURM_LOG_DIR}/${MERGE_JOB_ID}_merge.out"
