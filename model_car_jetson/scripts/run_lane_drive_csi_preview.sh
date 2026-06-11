#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[deprecated] Use ./lane_following/run.sh from model_car_jetson."
extra=()
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  extra=(--sensor-id "$1")
  shift
fi

exec "${ROOT_DIR}/lane_following/run.sh" "${extra[@]}" "$@"
