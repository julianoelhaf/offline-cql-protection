#!/bin/bash
#SBATCH --job-name=pg-gate4
#SBATCH --partition=a100mig
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100med:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

set -euo pipefail

: "${PROJECT:?set PROJECT to the committed cluster checkout}"
: "${GATE4_ENV:?set GATE4_ENV to the pinned cluster environment}"
: "${GATE4_OUTPUT:?set GATE4_OUTPUT to the isolated output root}"
: "${POWER_GRID_DATA_DIR:?set POWER_GRID_DATA_DIR to the reference CSV directory}"
: "${PHASOR_SHA256_LEDGER:?set PHASOR_SHA256_LEDGER to the corrected feature ledger}"

PYTHON="$GATE4_ENV/bin/python"
MATRIX="$PROJECT/gate4/matrix.json"
TAG="$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["tag"])' \
    "$MATRIX" "$SLURM_ARRAY_TASK_ID")"

cd "$PROJECT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PROJECT"

"$PYTHON" gate4/run_one.py "$TAG" \
    --matrix "$MATRIX" \
    --output-root "$GATE4_OUTPUT/runs"
