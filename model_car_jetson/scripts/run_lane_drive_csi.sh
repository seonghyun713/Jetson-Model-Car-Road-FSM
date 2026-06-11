#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSOR_ID="${1:-0}"
shift || true

if [[ "${MODEL_CAR_NO_X11:-0}" == "1" ]]; then
  unset DISPLAY
  unset XAUTHORITY
else
  if [[ -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=":1"
  elif [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=":0"
  fi

  if [[ -z "${XAUTHORITY:-}" ]]; then
    for candidate in \
      "/run/user/$(id -u)/gdm/Xauthority" \
      "/run/user/$(id -u)/.mutter-Xwaylandauth.*" \
      "${HOME}/.Xauthority"; do
      for auth_file in ${candidate}; do
        if [[ -r "${auth_file}" ]]; then
          export XAUTHORITY="${auth_file}"
          break 2
        fi
      done
    done
  fi
fi

PIPELINE="nvarguscamerasrc sensor-id=${SENSOR_ID} ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"

python3 "${SCRIPT_DIR}/jetson_lane_drive_runner.py" \
  --camera "${PIPELINE}" \
  --display \
  --half \
  --device auto \
  --threaded-capture \
  "$@"
