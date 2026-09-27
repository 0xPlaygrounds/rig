#!/usr/bin/env bash
set -euo pipefail

# Candle's safetensors copy of the Ultralytics YOLOv8 pose weights
# (AGPL-3.0), and the photo Candle's YOLOv8 example runs on, pinned to
# immutable commits.
WEIGHTS_REVISION="be388c6fab95ae3035a039070e1b883b9c5a1325"
PHOTO_REVISION="7a62aad24a5d8b1d8cb351e5ef77a7b3457a74b8"
MODEL_DIR="${RIG_CANDLE_POSE_MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/test-models/yolov8n-pose}"

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
    yolov8n-pose.safetensors \
    "https://huggingface.co/lmz/candle-yolo-v8/resolve/$WEIGHTS_REVISION/yolov8n-pose.safetensors" \
    08c76047b41744c027c8150e81d1dbbd94f996656cf5be266491367ae7fad072 \
    6650412
download \
    bike.jpg \
    "https://raw.githubusercontent.com/huggingface/candle/$PHOTO_REVISION/candle-examples/examples/yolo-v8/assets/bike.jpg" \
    317e4a9d2d2be7859ba0ab8726a526f5ece9d77daf92857f3e93fb7b367824c1 \
    182991

echo "YOLOv8 pose artifacts are ready in $MODEL_DIR"
