#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_dir> [extra run_lane_following_offline.py args...]"
  exit 2
fi

INPUT_DIR="$1"
shift

python3 \
  "${ROOT_DIR}/scripts/run_lane_following_offline.py" \
  --input-dir "${INPUT_DIR}" \
  --model-dir "${ROOT_DIR}/weights/segmentation/segformer_b0_yellow_line_best_model" \
  --output-dir "${ROOT_DIR}/logs/offline_lane_following" \
  --device auto \
  --save-intermediate \
  "$@"
