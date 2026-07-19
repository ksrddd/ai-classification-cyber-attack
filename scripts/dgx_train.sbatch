#!/usr/bin/env bash
# Submit from the repository root after creating logs/dgx:
#   mkdir -p logs/dgx && sbatch scripts/dgx_train.sbatch
# Override paths or the run label at submit time, for example:
#   PROJECT_ROOT=/workspace/cyberml RUN_LABEL=thesis_v1 sbatch scripts/dgx_train.sbatch
#SBATCH --job-name=cyberml-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dgx/%x-%j.out
#SBATCH --error=logs/dgx/%x-%j.err

# This script intentionally uses one node/GPU. The current trainer is a
# single-process program; requesting more GPUs would not make it distributed.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-dgx}"
RUN_LABEL="${RUN_LABEL:-full}"
# Override RUN_NAME to resume an interrupted run directory. New submissions
# receive a collision-resistant name by default.
RUN_NAME="${RUN_NAME:-dgx_${RUN_LABEL}_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)}"
MANIFEST="${SPLIT_MANIFEST:-${PROJECT_ROOT}/configs/splits/source_holdout_v3_full_70_30.json}"
LOG_DIR="${PROJECT_ROOT}/logs/dgx"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

exec > >(tee -a "${LOG_DIR}/${RUN_NAME}.log") 2>&1

echo "[$(date -u --iso-8601=seconds)] project=${PROJECT_ROOT} run=${RUN_NAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi is unavailable; this job must run on a DGX GPU node." >&2
  exit 2
}
nvidia-smi -L
nvidia-smi

# Slurm normally sets CUDA_VISIBLE_DEVICES. Preserve it; use the first GPU
# only for interactive/local invocation where Slurm did not set it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

test -f "${MANIFEST}" || {
  echo "ERROR: split manifest not found: ${MANIFEST}" >&2
  exit 2
}
"${VENV_DIR}/bin/python" main.py --stage audit --split-manifest "${MANIFEST}"
"${VENV_DIR}/bin/python" scripts/dgx_preflight.py \
  --mode gpu \
  --manifest "${MANIFEST}" \
  --gpu-devices 0 \
  --output "${PROJECT_ROOT}/results/${RUN_NAME}_dgx_preflight.json"

# Explicit full-data run. Slurm allocates/isolates the GPU and the runtime
# logs visible GPUs above; GPU-capable libraries use their installed backend.
# A unique --run-name makes artefacts resumable without overwriting another
# DGX experiment. Do not add --force when resuming this same run directory.
"${VENV_DIR}/bin/python" main.py --stage train \
  --run-name "${RUN_NAME}" \
  --preset full \
  --profile overnight \
  --split-manifest "${MANIFEST}" \
  --model all \
  --accelerator gpu \
  --gpu-devices 0 \
  --imbalance-strategy class_weight \
  --target-max-fpr 0.02

echo "[$(date -u --iso-8601=seconds)] complete: results/${RUN_NAME}"
