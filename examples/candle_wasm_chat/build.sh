#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/download_model.sh"

wasm-pack build \
    "$SCRIPT_DIR" \
    --target web \
    --release \
    --out-dir "$SCRIPT_DIR/www/pkg" \
    --out-name candle_wasm_chat

WASM="$SCRIPT_DIR/www/pkg/candle_wasm_chat_bg.wasm"
echo "WASM size: $(wc -c < "$WASM" | tr -d ' ') bytes"

echo
echo "WASM chat app built in $SCRIPT_DIR/www"
echo "Run: $SCRIPT_DIR/serve.sh"
echo "Smoke test: node $SCRIPT_DIR/smoke.mjs"
