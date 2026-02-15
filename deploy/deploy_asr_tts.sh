#!/usr/bin/env bash
# Build and deploy the Nemotron Speech ASR + Magpie TTS container to microk8s.
#
# This builds PyTorch + NeMo from source for CUDA 13.1 (DGX Spark / GB10).
# Expected build time: ~2-3 hours on a Spark node.
#
# Usage:
#   ./deploy/deploy_asr_tts.sh [--build-only] [--deploy-only]

set -euo pipefail

REGISTRY="localhost:32000"
IMAGE_NAME="nemotron-speech-asr-tts"
IMAGE_TAG="0.1"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
MANIFEST="deploy/nemotron-speech-asr-tts.k8s.yaml"
DOCKERFILE="deploy/Dockerfile.asr-tts"
NAMESPACE="spellingbee"

BUILD=true
DEPLOY=true

for arg in "$@"; do
  case $arg in
    --build-only)  DEPLOY=false ;;
    --deploy-only) BUILD=false ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

if $BUILD; then
  echo "=== Building ASR+TTS container image ==="
  echo "Image: ${FULL_IMAGE}"
  echo "Dockerfile: ${DOCKERFILE}"
  echo ""
  echo "WARNING: This build compiles PyTorch from source for sm_121."
  echo "Expected build time: 2-3 hours on DGX Spark."
  echo ""

  docker build -f "${DOCKERFILE}" -t "${FULL_IMAGE}" .

  echo ""
  echo "=== Pushing to local registry ==="
  docker push "${FULL_IMAGE}"
  echo "Pushed: ${FULL_IMAGE}"
fi

if $DEPLOY; then
  echo ""
  echo "=== Deploying ASR+TTS to microk8s ==="

  # Ensure namespace exists
  microk8s kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || \
    microk8s kubectl create namespace "${NAMESPACE}"

  # Ensure hf-token secret exists
  if ! microk8s kubectl get secret hf-token -n "${NAMESPACE}" >/dev/null 2>&1; then
    echo "ERROR: Secret 'hf-token' not found in namespace '${NAMESPACE}'."
    echo "Create it with:"
    echo "  microk8s kubectl create secret generic hf-token --from-literal=token=<YOUR_HF_TOKEN> -n ${NAMESPACE}"
    exit 1
  fi

  microk8s kubectl apply -f "${MANIFEST}"

  # Rollout restart if deployment already exists
  microk8s kubectl rollout restart deployment/nemotron-speech-asr-tts -n "${NAMESPACE}" 2>/dev/null || true

  echo ""
  echo "=== Waiting for rollout ==="
  microk8s kubectl rollout status deployment/nemotron-speech-asr-tts -n "${NAMESPACE}" --timeout=600s

  echo ""
  echo "=== ASR+TTS deployment complete ==="
  echo "ASR endpoint (cluster): ws://nemotron-speech-asr-tts:8080"
  echo "TTS endpoint (cluster): http://nemotron-speech-asr-tts:8001"
  echo ""
  echo "Pod status:"
  microk8s kubectl get pods -n "${NAMESPACE}" -l app=nemotron-speech-asr-tts
fi
