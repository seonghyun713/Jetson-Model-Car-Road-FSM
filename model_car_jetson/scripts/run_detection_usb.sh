#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/jetson_yolo_detection_smoke.py" \
  --camera "${1:-0}" \
  --display \
  --device auto
