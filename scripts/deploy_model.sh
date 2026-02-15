#!/usr/bin/env bash
set -euo pipefail

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
NAMESPACE="${NAMESPACE:-spellingbee}"
MANIFEST="${MANIFEST:-vllm-nemotron-nano-vl-8b.yaml}"
DEPLOYMENT="${DEPLOYMENT:-vllm-nemotron-nano-vl-8b}"
CHECK_HF_SECRET="${CHECK_HF_SECRET:-true}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}"
  exit 1
fi

if [[ "${CHECK_HF_SECRET}" == "true" ]]; then
  echo "[1/4] Checking hf-token secret"
  ${KUBECTL_BIN} -n "${NAMESPACE}" get secret hf-token >/dev/null
else
  echo "[1/4] Skipping hf-token secret check"
fi

echo "[2/4] Applying model manifest"
${KUBECTL_BIN} apply -f "${MANIFEST}"

echo "[3/4] Restarting deployment ${DEPLOYMENT}"
${KUBECTL_BIN} -n "${NAMESPACE}" rollout restart deploy/"${DEPLOYMENT}"

echo "[4/4] Waiting for rollout"
${KUBECTL_BIN} -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=600s

${KUBECTL_BIN} -n "${NAMESPACE}" get pods -o wide
${KUBECTL_BIN} -n "${NAMESPACE}" get svc "${DEPLOYMENT}"

echo "Model deploy complete."
