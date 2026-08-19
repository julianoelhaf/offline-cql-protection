#!/bin/bash
#SBATCH --job-name=pg-g4-smoke
#SBATCH --partition=a100mig
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100med:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

set -euo pipefail

: "${PROJECT:?set PROJECT to the committed cluster checkout}"
: "${GATE4_ENV:?set GATE4_ENV to the pinned cluster environment}"
: "${POWER_GRID_DATA_DIR:?set POWER_GRID_DATA_DIR to the reference CSV directory}"

cd "$PROJECT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PROJECT"

"$GATE4_ENV/bin/python" gate4/smoke_repro.py
