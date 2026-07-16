#!/usr/bin/env bash
set -euo pipefail

# Official Qwen repositories pinned to immutable commits.
MODEL_REVISION="bc640142c66e1fdd12af0bd68f40445458f3869b"
TOKENIZER_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_DIR="${RIG_CANDLE_TEST_MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/test-models/qwen3-4b-q4-k-m}"

mkdir -p "$MODEL_DIR"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    local name="$1"
    local url="$2"
    local expected_sha="$3"
    local expected_size="$4"
    local destination="$MODEL_DIR/$name"

    if [[ -f "$destination" ]] \
        && [[ "$(wc -c < "$destination" | tr -d ' ')" == "$expected_size" ]] \
        && [[ "$(sha256 "$destination")" == "$expected_sha" ]]; then
        echo "verified $name"
        return
    fi

    local temporary="$destination.part.$$"
    trap 'rm -f "$temporary"' RETURN
    rm -f "$temporary"
    echo "downloading $name"
    curl --fail --location --retry 5 --retry-all-errors --continue-at - \
        --output "$temporary" "$url"

    local actual_size
    actual_size="$(wc -c < "$temporary" | tr -d ' ')"
    [[ "$actual_size" == "$expected_size" ]] || {
        echo "$name size mismatch: expected $expected_size, got $actual_size" >&2
        exit 1
    }
    local actual_sha
    actual_sha="$(sha256 "$temporary")"
    [[ "$actual_sha" == "$expected_sha" ]] || {
        echo "$name checksum mismatch: expected $expected_sha, got $actual_sha" >&2
        exit 1
    }
    mv -f "$temporary" "$destination"
    trap - RETURN
    echo "installed $destination"
}

download \
    config.json \
    "https://huggingface.co/Qwen/Qwen3-4B/resolve/$TOKENIZER_REVISION/config.json" \
    8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a \
    726
download \
    tokenizer.json \
    "https://huggingface.co/Qwen/Qwen3-4B/resolve/$TOKENIZER_REVISION/tokenizer.json" \
    aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    11422654
download \
    model.gguf \
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/$MODEL_REVISION/Qwen3-4B-Q4_K_M.gguf" \
    7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 \
    2497280256

echo "Qwen3 artifacts are ready in $MODEL_DIR"
