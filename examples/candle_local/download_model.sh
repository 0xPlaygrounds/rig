#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
MODEL_DIR="$MODEL_DIR" "$SCRIPT_DIR/../candle_wasm_chat/download_model.sh"
echo "Run: cargo run -p candle_local -- 'Say hello in one short sentence.'"
