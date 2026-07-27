#!/usr/bin/env bash
# Simple wrapper to exercise the Python socket client via CLI flags.
# Examples:
#   ./scripts/test_socket_client.sh --method status
#   ./scripts/test_socket_client.sh --method validate_setup
#   ./scripts/test_socket_client.sh --method transform_data --params '{"profile": {"url": "http://example.org/StructureMap/test"}}' --data '{"resourceType":"Patient","id":"123"}'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

METHOD=""
PARAMS=""
DATA=""
SOCKET_TYPE="UNIX"
SOCKET_PATH="/tmp/fsh_nifi_bridge.sock"
SOCKET_HOST="127.0.0.1"
SOCKET_PORT=9999

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --params) PARAMS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --socket-type) SOCKET_TYPE="$2"; shift 2 ;;
    --socket-path) SOCKET_PATH="$2"; shift 2 ;;
    --socket-host) SOCKET_HOST="$2"; shift 2 ;;
    --socket-port) SOCKET_PORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --method <method> [--params '<json>'] [--data '<json>'] [--socket-type UNIX|TCP] [--socket-path PATH] [--socket-host HOST] [--socket-port PORT]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${METHOD}" ]]; then
  echo "--method is required" >&2
  exit 1
fi

${PYTHON_BIN} "${REPO_ROOT}/scripts/test_socket_client.py" \
  --method "${METHOD}" \
  ${PARAMS:+--params "${PARAMS}"} \
  ${DATA:+--data "${DATA}"} \
  --socket-type "${SOCKET_TYPE}" \
  --socket-path "${SOCKET_PATH}" \
  --socket-host "${SOCKET_HOST}" \
  --socket-port "${SOCKET_PORT}" \
  || exit $?
