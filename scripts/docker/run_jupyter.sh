#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <username> <gpu_id> [tag]"
  echo "Example: $0 alice 1 20260316"
  exit 1
fi

username="$1"
gpu_id="$2"
tag="${3:-$(date +%Y%m%d)}"

image_name="${username}/prop-logic-transformer:${tag}"
container_name="${username}-logic-gpu${gpu_id}"

project_dir="${PROJECT_DIR:-/mnt/data/${username}/prop-logic-transformer-circuit}"
hf_cache_dir="${HF_CACHE_DIR:-/mnt/data/${username}/hf-cache}"
results_dir="${RESULTS_DIR:-/mnt/data/${username}/logic-results}"
host_port="${HOST_JUPYTER_PORT:-8888}"

mkdir -p "${project_dir}" "${hf_cache_dir}" "${results_dir}"

echo "Starting container: ${container_name}"
echo "Image: ${image_name}"
echo "GPU: ${gpu_id}"
echo "Project mount: ${project_dir} -> /workspace/project"
echo "HF cache mount: ${hf_cache_dir} -> /workspace/.cache/huggingface"
echo "Results mount: ${results_dir} -> /workspace/results"
echo "Jupyter URL: http://127.0.0.1:${host_port}"

docker run --rm -it \
  --name "${container_name}" \
  --gpus "device=${gpu_id}" \
  --shm-size=16g \
  -p "127.0.0.1:${host_port}:8888" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  -e PYTHONPATH=/workspace/project \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e JUPYTER_PORT=8888 \
  -e JUPYTER_TOKEN="${JUPYTER_TOKEN:-}" \
  -v "${project_dir}:/workspace/project" \
  -v "${hf_cache_dir}:/workspace/.cache/huggingface" \
  -v "${results_dir}:/workspace/results" \
  "${image_name}"
