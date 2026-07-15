#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
MODEL_REPO="${MODEL_REPO:-yujiepan/llama-3-tiny-random}"
REVISION="${REVISION:-main}"
BASE_URL="https://huggingface.co/$MODEL_REPO/resolve/$REVISION"

mkdir -p "$MODEL_DIR"

download() {
    local filename="$1"
    local destination="$MODEL_DIR/$filename"
    local temporary="$destination.part"

    if [[ -s "$destination" ]]; then
        echo "Already present: $destination"
        return
    fi

    echo "Downloading $MODEL_REPO/$filename"
    curl \
        --fail \
        --location \
        --retry 3 \
        --progress-bar \
        --output "$temporary" \
        "$BASE_URL/$filename"
    mv "$temporary" "$destination"
}

download config.json
download tokenizer.json
download model.safetensors

echo
echo "Model files are ready in $MODEL_DIR"
echo "Run: cd '$SCRIPT_DIR' && cargo run -p candle_local -- 'Say hello in one short sentence.'"
