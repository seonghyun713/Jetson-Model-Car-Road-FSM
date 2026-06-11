#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA="${1:-0}"
shift || true

python3 "${SCRIPT_DIR}/jetson_combined_runner.py" \
  --camera "${CAMERA}" \
  --web \
  --detect-every 3 \
  --stream-fps 4 \
  --stream-width 1280 \
  --device auto \
  "$@"
