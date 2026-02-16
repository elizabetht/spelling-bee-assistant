#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST=deploy/vllm-nemotron-nano-30b.yaml \
DEPLOYMENT=vllm-nemotron-nano-30b \
"${SCRIPT_DIR}/deploy_model.sh"
