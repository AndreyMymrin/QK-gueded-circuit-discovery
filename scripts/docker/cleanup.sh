#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 1 ]]; then
  echo "Usage: $0 <username>"
  echo "Example: $0 alice"
  exit 1
fi

username="$1"

exited_containers="$(docker ps -a --filter "name=${username}-logic-gpu" --filter "status=exited" -q)"
if [[ -n "${exited_containers}" ]]; then
  echo "${exited_containers}" | xargs docker rm
else
  echo "No exited containers found for ${username}-logic-gpu*"
fi

# Remove dangling layers and build cache.
docker image prune -f
docker builder prune -f

echo "Cleanup completed."
