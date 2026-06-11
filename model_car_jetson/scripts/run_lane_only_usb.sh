#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/jetson_lane_only_runner.py" \
  --camera "${1:-0}" \
  --display \
  --save-video \
  --device auto
