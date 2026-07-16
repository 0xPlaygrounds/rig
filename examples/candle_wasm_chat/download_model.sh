#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
MODEL_REVISION="a10cc1512eabd3dde888204e902eca88bddb4951"
GGUF_REVISION="7be6f65f1db715fe5dc5a4634c0d459b4eed42ec"

mkdir -p "$MODEL_DIR"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    local filename="$1"
    local expected="$2"
    local url="$3"
    local destination="$MODEL_DIR/$filename"
    local temporary="$destination.part"

    if [[ -f "$destination" ]] && [[ "$(sha256 "$destination")" == "$expected" ]]; then
        echo "Verified $filename ($(wc -c < "$destination" | tr -d ' ') bytes)"
        return
    fi
    rm -f "$temporary"
    echo "Downloading pinned $filename"
    curl --fail --location --retry 5 --retry-all-errors --progress-bar \
        --output "$temporary" "$url"
    local actual
    actual="$(sha256 "$temporary")"
    if [[ "$actual" != "$expected" ]]; then
        rm -f "$temporary"
        echo "SHA-256 mismatch for $filename: expected $expected, got $actual" >&2
        exit 1
    fi
    mv -f "$temporary" "$destination"
    echo "Verified $filename ($(wc -c < "$destination" | tr -d ' ') bytes)"
}

MODEL_BASE="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/$MODEL_REVISION"
GGUF_BASE="https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/$GGUF_REVISION"
download config.json 224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee "$MODEL_BASE/config.json"
download tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c "$MODEL_BASE/tokenizer.json"
download model.gguf 2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2 "$GGUF_BASE/SmolLM2-360M-Instruct-Q4_K_M.gguf"

echo "Model artifacts are ready in $MODEL_DIR"
