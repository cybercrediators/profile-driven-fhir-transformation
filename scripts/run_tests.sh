#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

usage() {
  cat <<'EOF'
Usage: run_tests.sh [all|unit|integration|e2e|fast]
  all          Run full test suite (default)
  unit         Run unit tests only (-m unit)
  integration  Run integration tests only (-m integration)
  e2e          Run end-to-end tests only (currently skipped placeholders)
  fast         Run unit tests without coverage (quick check)

Environment:
  PYTHON_BIN   Python executable to use (default: python)
EOF
}

suite=${1:-all}
cd "${REPO_ROOT}"

case "${suite}" in
  unit)
    exec ${PYTHON_BIN} -m pytest -m unit
    ;;
  integration)
    exec ${PYTHON_BIN} -m pytest -m integration
    ;;
  e2e)
    exec ${PYTHON_BIN} -m pytest tests/e2e
    ;;
  fast)
    exec ${PYTHON_BIN} -m pytest -q -m unit
    ;;
  all)
    exec ${PYTHON_BIN} -m pytest
    ;;
  -h|--help)
    usage; exit 0 ;;
  *)
    echo "Unknown suite: ${suite}" >&2
    usage
    exit 1
    ;;
esac
