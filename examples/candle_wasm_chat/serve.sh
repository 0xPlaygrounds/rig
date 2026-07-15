#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8080}"

if [[ ! -f "$SCRIPT_DIR/www/pkg/candle_wasm_chat.js" ]]; then
    echo "Build output is missing. Run $SCRIPT_DIR/build.sh first." >&2
    exit 1
fi

echo "Serving http://127.0.0.1:$PORT"
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$SCRIPT_DIR/www"
