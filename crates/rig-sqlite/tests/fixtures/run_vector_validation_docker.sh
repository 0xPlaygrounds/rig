#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMAGE="${RIG_SQLITE_VECTOR_VALIDATION_IMAGE:-rig-sqlite-vector-validation}"
DATA_DIR="${RIG_SQLITE_VECTOR_DATA_DIR:-$ROOT_DIR/target/sqlite-vector-validation}"

mkdir -p "$DATA_DIR"

docker build \
  -f "$ROOT_DIR/crates/rig-sqlite/tests/fixtures/Dockerfile.vector-validation" \
  -t "$IMAGE" \
  "$ROOT_DIR/crates/rig-sqlite/tests/fixtures"

docker run --rm \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}" \
  -e CARGO_HOME=/data/cargo-home \
  -e CARGO_INCREMENTAL=0 \
  -e CARGO_PROFILE_DEV_DEBUG=0 \
  -e CARGO_PROFILE_TEST_DEBUG=0 \
  -e CARGO_TARGET_DIR=/data/cargo-target \
  -e RUSTUP_HOME=/data/rustup \
  -v "$ROOT_DIR:/workspace" \
  -v "$DATA_DIR:/data" \
  "$IMAGE" \
  bash crates/rig-sqlite/tests/fixtures/run_vector_validation.sh "$@"
