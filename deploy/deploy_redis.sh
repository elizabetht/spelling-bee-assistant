#!/usr/bin/env bash
set -euo pipefail

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
NAMESPACE="${NAMESPACE:-spellingbee}"
MANIFEST="${MANIFEST_REDIS:-deploy/redis.k8s.yaml}"
DEPLOYMENT="redis"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}"
  exit 1
fi

echo "[1/3] Applying Redis manifest"
${KUBECTL_BIN} apply -f "${MANIFEST}"

echo "[2/3] Waiting for rollout"
${KUBECTL_BIN} -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=120s

echo "[3/3] Verifying Redis service"
${KUBECTL_BIN} -n "${NAMESPACE}" get pods -l app=redis -o wide
${KUBECTL_BIN} -n "${NAMESPACE}" get svc redis

echo "Redis deploy complete."
