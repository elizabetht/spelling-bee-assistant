#!/usr/bin/env bash
# Opens a kubectl port-forward to the spelling-bee-agent-backend.
# Usage: bash deploy/tunnel.sh [port]
#
# Tunnels the backend service to localhost so the browser can access
# getUserMedia (requires HTTPS or localhost).
#
# If run on the controller node, forwards directly.
# If run from a remote machine (e.g. Mac), tunnels via SSH.
set -uo pipefail

PORT="${1:-8080}"
HOST="nvidia@192.168.1.75"
CONTROLLER_HOSTNAME="controller"
SVC="svc/spelling-bee-agent-backend"
NS="spellingbee"

# Kill any stale port-forward processes for this service
cleanup_port_forward() {
  local pids
  pids=$(pgrep -f "port-forward ${SVC}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "Killing stale port-forward (PIDs: ${pids})..."
    echo "${pids}" | xargs kill -9 2>/dev/null || true
    sleep 2
  fi
}

# Kill anything bound to the local port
cleanup_local_port() {
  lsof -ti:"${PORT}" 2>/dev/null | xargs kill -9 2>/dev/null || true
}

if [[ "$(hostname)" == "${CONTROLLER_HOSTNAME}" ]]; then
  # Running directly on the controller
  cleanup_port_forward
  cleanup_local_port
  sleep 1
  echo "Forwarding spelling-bee-agent-backend to http://localhost:${PORT}"
  echo "Press Ctrl+C to stop"
  microk8s kubectl -n "${NS}" port-forward "${SVC}" "${PORT}:8080"
else
  # Running from a remote machine — tunnel via SSH
  cleanup_local_port
  ssh "${HOST}" "kill \$(pgrep -f 'port-forward ${SVC}') 2>/dev/null" 2>/dev/null || true
  sleep 2
  echo "Tunneling spelling-bee-agent-backend to http://localhost:${PORT}"
  echo "Press Ctrl+C to stop"
  ssh -L "${PORT}:127.0.0.1:8080" "${HOST}" \
    "microk8s kubectl -n ${NS} port-forward ${SVC} 8080:8080"
fi
