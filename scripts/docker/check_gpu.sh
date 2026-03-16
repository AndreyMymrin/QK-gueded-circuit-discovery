#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 3 ]]; then
  echo "Usage: $0 <username> <gpu_id> <tag>"
  echo "Example: $0 alice 1 20260316"
  exit 1
fi

username="$1"
gpu_id="$2"
tag="$3"

image_name="${username}/prop-logic-transformer:${tag}"

docker run --rm \
  --gpus "device=${gpu_id}" \
  "${image_name}" \
  python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
