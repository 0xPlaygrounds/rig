#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
RIG_CANDLE_POSE_MODEL_DIR="$MODEL_DIR" \
    "$SCRIPT_DIR/../../crates/rig-candle/tests/download_yolov8_pose.sh"
echo "Run: cargo run --release -p candle_pose"
