#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
IMAGE_PATH="${2:-}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 <base-url> <image-path>"
  echo "Example: $0 http://127.0.0.1:30088 ./sample_words.png"
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Image file not found: ${IMAGE_PATH}"
  exit 1
fi

echo "[1/3] Health check: ${BASE_URL}/healthz"
HEALTH_JSON="$(curl -fsS "${BASE_URL}/healthz")"
echo "Health: ${HEALTH_JSON}"

echo "[2/3] Upload image: ${BASE_URL}/upload-image"
UPLOAD_JSON="$(curl -fsS -F "file=@${IMAGE_PATH}" "${BASE_URL}/upload-image")"
echo "Upload response: ${UPLOAD_JSON}"

SESSION_ID="$(printf '%s' "${UPLOAD_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("session_id",""))')"
if [[ -z "${SESSION_ID}" ]]; then
  echo "FAILED: session_id missing in upload response"
  exit 1
fi

WS_URL="$(python3 - <<'PY' "${BASE_URL}" "${SESSION_ID}"
import sys
base, sid = sys.argv[1], sys.argv[2]
if base.startswith('https://'):
    ws = 'wss://' + base[len('https://'):]
elif base.startswith('http://'):
    ws = 'ws://' + base[len('http://'):]
else:
    ws = 'ws://' + base
print(f"{ws}/pipecat/ws?session_id={sid}")
PY
)"

echo "[3/3] Basic endpoint verification passed"
echo "Session ID: ${SESSION_ID}"
echo "Pipecat WS URL: ${WS_URL}"
