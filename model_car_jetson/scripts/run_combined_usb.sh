#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA="${1:-0}"
shift || true

python3 "${SCRIPT_DIR}/jetson_combined_runner.py" \
  --camera "${CAMERA}" \
  --display \
  --save-video \
  --device auto \
  "$@"
