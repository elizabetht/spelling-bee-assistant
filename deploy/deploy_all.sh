#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/deploy_model.sh"
MANIFEST=deploy/vllm-nemotron-nano-30b.yaml DEPLOYMENT=vllm-nemotron-nano-30b "${SCRIPT_DIR}/deploy_model.sh"
"${SCRIPT_DIR}/deploy_redis.sh"
"${SCRIPT_DIR}/deploy_backend.sh"

echo "All deployments completed."
