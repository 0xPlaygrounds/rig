#!/usr/bin/env bash
# `rust-toolchain.toml` is what cargo itself obeys, but every workflow also
# carries a `RUST_VERSION` env used as the `setup-rust-toolchain` input. That
# is four copies of one number, and a partial bump is silent: the gate goes
# green on the new toolchain while a workflow that was missed — nightly.yaml
# is also cd.yaml's pre-release gate — keeps compiling on the old one, so a
# failure there gets misread as a runtime regression rather than a stale pin.
#
# Assert the copies agree. This is a text check on purpose: it must fail on a
# workflow nobody remembered to update, which a parser keyed off the workflows
# that *do* declare the variable would not do.
#
# Why the copies exist at all: omitting the `toolchain` input would make
# `setup-rust-toolchain` read rust-toolchain.toml directly — no copies, no
# desync, no script. But rust-toolchain.toml also pins components
# (rust-analyzer, rust-src) and the wasm target for local development, and
# honoring it in CI would download those in every job. Until that trade is
# taken, this guard keeps the existing copies honest. A workflow with no
# RUST_VERSION needs no checking (it takes its toolchain from
# rust-toolchain.toml), so a repo that deletes every copy passes vacuously —
# that is the desired end state, not an error to steer people away from.
set -euo pipefail

cd "$(dirname "$0")/../.."

channel=$(grep -E '^\s*channel\s*=' rust-toolchain.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "$channel" ]; then
  echo "::error::could not read [toolchain] channel from rust-toolchain.toml"
  exit 1
fi

status=0
for workflow in .github/workflows/*.yaml .github/workflows/*.yml; do
  [ -e "$workflow" ] || continue
  # Extract the scalar value only: strip a CR from a CRLF checkout, any
  # trailing comment, surrounding quotes, and whitespace — a cosmetic edit
  # like `RUST_VERSION: 1.94.0 # keep in sync` must not fail the gate by
  # comparing a value with the comment text embedded in it.
  while IFS= read -r pinned; do
    if [ "$pinned" != "$channel" ]; then
      echo "::error file=$workflow::RUST_VERSION is $pinned but rust-toolchain.toml pins $channel"
      status=1
    fi
  done < <(grep -E '^\s*RUST_VERSION:' "$workflow" \
    | sed -E 's/\r$//; s/^[[:space:]]*RUST_VERSION:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//; s/^"([^"]*)"$/\1/; s/^'"'"'([^'"'"']*)'"'"'$/\1/')
done

if [ "$status" -eq 0 ]; then
  echo "ok: every declared RUST_VERSION matches rust-toolchain.toml ($channel)"
fi
exit "$status"
