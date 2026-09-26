#!/usr/bin/env bash
# Build the Linux rig-coder binary the Harbor adapter uploads into task
# containers, in Docker so it links against the containers' glibc. Cargo's
# registry and target directory persist in named volumes between builds.
#
#   test-support/rig-coder/bench/build-linux.sh          # aarch64 on Apple Silicon
#   ARCH=x86_64 test-support/rig-coder/bench/build-linux.sh
set -euo pipefail
BENCH="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$BENCH/../../.." && pwd)"
ARCH="${ARCH:-$( [ "$(uname -m)" = arm64 ] && echo aarch64 || uname -m )}"
case "$ARCH" in
  aarch64) PLATFORM=linux/arm64 ;;
  x86_64) PLATFORM=linux/amd64 ;;
  *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
RUST="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$ROOT/rust-toolchain.toml")"
OUT="$BENCH/bin/rig-coder-linux-$ARCH"
mkdir -p "$BENCH/bin"
rm -f "$OUT" "$OUT.build.json"
# Debian bullseye's glibc (2.31) keeps the binary runnable on older task images.
docker run --rm --platform "$PLATFORM" \
  -v "$ROOT":/src:ro -w /src \
  -v "rig-coder-cargo-$ARCH":/usr/local/cargo/registry \
  -v "rig-coder-target-$ARCH":/target \
  -v "$BENCH/bin":/out \
  -e CARGO_TARGET_DIR=/target \
  "rust:$RUST-bullseye" \
  bash -c "cargo build --locked --release -p rig-coder && cp /target/release/rig-coder /out/rig-coder-linux-$ARCH"
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
DIRTY="$( [ -z "$(git -C "$ROOT" status --porcelain -- test-support/rig-coder crates Cargo.toml Cargo.lock)" ] && echo false || echo true )"
SHA="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
printf '{"source_head": "%s", "dirty": %s, "arch": "%s", "sha256": "%s", "built_at": "%s"}\n' \
  "$HEAD" "$DIRTY" "$ARCH" "$SHA" "$(date -u +%FT%TZ)" > "$OUT.build.json"
echo "built $OUT ($HEAD, dirty=$DIRTY)"
