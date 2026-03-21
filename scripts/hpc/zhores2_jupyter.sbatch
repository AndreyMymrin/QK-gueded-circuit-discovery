#!/usr/bin/env bash
#SBATCH -J logic-jupyter
#SBATCH -p ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o %x-%j.out

set -euo pipefail

IMAGE_SIF="${IMAGE_SIF:-/gpfs/gpfs0/$USER/containers/prop-logic-transformer.sif}"
PROJECT_DIR="${PROJECT_DIR:-/gpfs/gpfs0/$USER/prop-logic-transformer-circuit}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/gpfs/gpfs0/$USER/hf-cache}"
RESULTS_DIR="${RESULTS_DIR:-/gpfs/gpfs0/$USER/logic-results}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-change-me}"

mkdir -p "$PROJECT_DIR" "$HF_CACHE_DIR" "$RESULTS_DIR"

if ! command -v apptainer >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load apptainer || true
  fi
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer command not found. Try: module load apptainer" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_SIF" ]]; then
  echo "ERROR: IMAGE_SIF not found: $IMAGE_SIF" >&2
  echo "Hint: apptainer pull \"$IMAGE_SIF\" docker://<registry>/<user>/prop-logic-transformer:<tag>" >&2
  exit 1
fi

export APPTAINERENV_HF_HOME=/workspace/.cache/huggingface
export APPTAINERENV_TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"
export APPTAINERENV_JUPYTER_PORT="$JUPYTER_PORT"
export APPTAINERENV_JUPYTER_TOKEN="$JUPYTER_TOKEN"

echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "Compute node: $(hostname)"
echo "Image: $IMAGE_SIF"
echo "Notebook: /workspace/project/analysis_walkthrough/LLM Analysis Part 1 Circuit search, interpretation, and verification.ipynb"
echo
echo "From your laptop, create SSH tunnel:"
echo "  ssh -N -L ${JUPYTER_PORT}:$(hostname):${JUPYTER_PORT} ${USER}@cdise.login"
echo "Then open:"
echo "  http://127.0.0.1:${JUPYTER_PORT}/lab?token=${JUPYTER_TOKEN}"
echo

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/project" \
  --bind "${HF_CACHE_DIR}:/workspace/.cache/huggingface" \
  --bind "${RESULTS_DIR}:/workspace/results" \
  --pwd /workspace/project \
  "$IMAGE_SIF" \
  bash -lc "jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --ServerApp.token='${JUPYTER_TOKEN}'"
