#!/usr/bin/env bash
set -euo pipefail

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
NAMESPACE="${NAMESPACE:-spellingbee}"
IMAGE_TAG="${IMAGE_TAG:-0.1}"
IMAGE_REPO="${IMAGE_REPO:-localhost:32000/spelling-bee-agent-backend}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
MANIFEST="${MANIFEST:-deploy/spelling-bee-agent-backend.k8s.yaml}"
DEPLOYMENT="${DEPLOYMENT:-spelling-bee-agent-backend}"
CREATE_SECRET="${CREATE_SECRET:-false}"
NVIDIA_API_KEY_VALUE="${NVIDIA_API_KEY_VALUE:-}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}"
  exit 1
fi

echo "[1/6] Building backend image: ${IMAGE}"
docker build --no-cache -t "${IMAGE}" .

echo "[2/6] Pushing backend image"
docker push "${IMAGE}"

if [[ "${CREATE_SECRET}" == "true" ]]; then
  if [[ -z "${NVIDIA_API_KEY_VALUE}" ]]; then
    echo "CREATE_SECRET=true requires NVIDIA_API_KEY_VALUE"
    exit 1
  fi
  echo "[3/6] Creating/updating nvidia-api-key secret"
  ${KUBECTL_BIN} -n "${NAMESPACE}" create secret generic nvidia-api-key \
    --from-literal=api-key="${NVIDIA_API_KEY_VALUE}" \
    --dry-run=client -o yaml | ${KUBECTL_BIN} apply -f -
else
  echo "[3/6] Skipping secret creation (CREATE_SECRET=false)"
fi

echo "[4/6] Applying backend manifest"
${KUBECTL_BIN} apply -f "${MANIFEST}"

echo "[5/6] Restarting deployment ${DEPLOYMENT}"
${KUBECTL_BIN} -n "${NAMESPACE}" rollout restart deploy/"${DEPLOYMENT}"

echo "[6/6] Waiting for rollout"
${KUBECTL_BIN} -n "${NAMESPACE}" rollout status deploy/"${DEPLOYMENT}" --timeout=300s

${KUBECTL_BIN} -n "${NAMESPACE}" get pods -o wide
${KUBECTL_BIN} -n "${NAMESPACE}" get svc "${DEPLOYMENT}"

echo "Backend deploy complete."
