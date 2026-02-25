#!/bin/bash
source /w/20252/khashazad/env/bin/activate
PATIENT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PREPROCESS_PATIENTS_FILE")
echo "Task $SLURM_ARRAY_TASK_ID → patient $PATIENT_ID"
python -u scripts/preprocess_data.py --patient "$PATIENT_ID"
