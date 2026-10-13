#!/usr/bin/env bash
# Usage:
#   ./run.sh                  # runner.protocol from config/config.yaml
#   ./run.sh qwen3
#   ./run.sh uitars
# Extra hydra overrides still work, e.g.:
#   ./run.sh qwen3 coordinator.max_concurrent_envs=8
#
# An explicit first argument selects protocol + isolated Ray cluster.
# If omitted, runner.protocol in config/config.yaml is used.
# storage.root comes from config/config.yaml unless you override it on the command line.
# Two jobs can run at once: one qwen3, one uitars. Do not `ray stop` the other job.

source "${HOME}/miniconda3/bin/activate"
conda activate verl
set -euo pipefail

export PYTHONUNBUFFERED=1

CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/config.yaml"

read_protocol_from_config() {
  awk '
    /^runner:/ { in_runner=1; next }
    in_runner && /^[^[:space:]#]/ { in_runner=0 }
    in_runner && /^[[:space:]]+protocol:/ {
      line=$0
      sub(/^[[:space:]]+protocol:[[:space:]]*/, "", line)
      sub(/[[:space:]]*#.*/, "", line)
      gsub(/["'\'']/, "", line)
      gsub(/[[:space:]]/, "", line)
      print line
      exit
    }
  ' "${CONFIG_FILE}"
}

if [[ $# -ge 1 && ( "${1}" == "qwen3" || "${1}" == "uitars" ) ]]; then
  PROTOCOL="${1}"
  shift
else
  if [[ $# -ge 1 ]]; then
    echo "[ERROR] First argument must be qwen3 or uitars, got: ${1}" >&2
    echo "Usage: ./run.sh [qwen3|uitars] [hydra overrides...]" >&2
    echo "Omit the protocol to use runner.protocol from config/config.yaml." >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG_FILE}" >&2
    exit 1
  fi
  PROTOCOL="$(read_protocol_from_config)"
  if [[ -z "${PROTOCOL}" ]]; then
    echo "[ERROR] runner.protocol is missing in ${CONFIG_FILE}" >&2
    exit 1
  fi
  echo "[INFO] No protocol argument; using runner.protocol=${PROTOCOL} from ${CONFIG_FILE}"
fi

case "${PROTOCOL}" in
  qwen3|uitars) ;;
  *)
    echo "[ERROR] protocol must be qwen3 or uitars, got: ${PROTOCOL}" >&2
    exit 1
    ;;
esac

if [[ "${PROTOCOL}" == "qwen3" ]]; then
  RAY_PORT=6380
  RAY_DASHBOARD_PORT=8266
  RAY_TEMP_DIR=/tmp/ray_qwen3
  RAY_NAMESPACE=qwen3
else
  RAY_PORT=6379
  RAY_DASHBOARD_PORT=8265
  RAY_TEMP_DIR=/tmp/ray_uitars
  RAY_NAMESPACE=uitars
fi

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
export RAY_NAMESPACE

port_open() {
  bash -c "echo >/dev/tcp/127.0.0.1/${1}" 2>/dev/null
}

if port_open "${RAY_PORT}"; then
  echo "[INFO] Ray head already on ${RAY_ADDRESS} namespace=${RAY_NAMESPACE}"
else
  mkdir -p "${RAY_TEMP_DIR}"
  echo "[INFO] Starting Ray head ${RAY_ADDRESS} dashboard=${RAY_DASHBOARD_PORT} temp=${RAY_TEMP_DIR}"
  ray start --head --port="${RAY_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}" --temp-dir="${RAY_TEMP_DIR}"
fi

LOG_FILE="run_${PROTOCOL}_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] protocol=${PROTOCOL} (storage.root from config.yaml unless overridden)"
echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS} RAY_NAMESPACE=${RAY_NAMESPACE}"
echo "[INFO] Logging to: ${LOG_FILE}"
echo "[INFO] Starting: python -m src.run runner.protocol=${PROTOCOL} $*"

python -u -m src.run \
  "runner.protocol=${PROTOCOL}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
