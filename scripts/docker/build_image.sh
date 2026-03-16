#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <username> [tag]"
  echo "Example: $0 alice 20260316"
  exit 1
fi

username="$1"
tag="${2:-$(date +%Y%m%d)}"
image_name="${username}/prop-logic-transformer:${tag}"

docker build -t "${image_name}" .

echo "Built image: ${image_name}"
