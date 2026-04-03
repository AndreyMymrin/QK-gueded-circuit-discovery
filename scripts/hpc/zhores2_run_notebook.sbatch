#!/usr/bin/env bash
#SBATCH -J logic-notebook
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
NOTEBOOK_REL="${NOTEBOOK_REL:-analysis_walkthrough/LLM Analysis Part 1 Circuit search, interpretation, and verification.ipynb}"
OUTPUT_NOTEBOOK="${OUTPUT_NOTEBOOK:-LLM_Analysis_Part1_executed_${SLURM_JOB_ID:-local}.ipynb}"

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
export APPTAINERENV_PYTHONPATH=/workspace/project
export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"

echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "Compute node: $(hostname)"
echo "Image: $IMAGE_SIF"
echo "Notebook: $NOTEBOOK_REL"
echo "Output: /workspace/results/${OUTPUT_NOTEBOOK}"
echo

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/project" \
  --bind "${HF_CACHE_DIR}:/workspace/.cache/huggingface" \
  --bind "${RESULTS_DIR}:/workspace/results" \
  --pwd /workspace/project \
  "$IMAGE_SIF" \
  bash -lc "jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output='${OUTPUT_NOTEBOOK}' --output-dir='/workspace/results' '/workspace/project/${NOTEBOOK_REL}'"
