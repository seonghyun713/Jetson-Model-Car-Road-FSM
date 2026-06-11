#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSOR_ID="${1:-0}"
shift || true

PIPELINE="nvarguscamerasrc sensor-id=${SENSOR_ID} ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"

python3 "${SCRIPT_DIR}/jetson_yolo_detection_smoke.py" \
  --camera "${PIPELINE}" \
  --display \
  --device auto \
  "$@"
