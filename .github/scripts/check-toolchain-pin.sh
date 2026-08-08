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
set -euo pipefail

cd "$(dirname "$0")/../.."

channel=$(grep -E '^\s*channel\s*=' rust-toolchain.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "$channel" ]; then
  echo "::error::could not read [toolchain] channel from rust-toolchain.toml"
  exit 1
fi

status=0
found=0
for workflow in .github/workflows/*.yaml .github/workflows/*.yml; do
  [ -e "$workflow" ] || continue
  # Only workflows that declare the variable are checked; a workflow with no
  # `RUST_VERSION` takes its toolchain from rust-toolchain.toml already.
  while IFS= read -r pinned; do
    found=1
    if [ "$pinned" != "$channel" ]; then
      echo "::error file=$workflow::RUST_VERSION is $pinned but rust-toolchain.toml pins $channel"
      status=1
    fi
  done < <(grep -E '^\s*RUST_VERSION:' "$workflow" | sed -E 's/.*RUST_VERSION:[[:space:]]*//' | tr -d '"'"'"'')
done

if [ "$found" -eq 0 ]; then
  echo "::error::no workflow declares RUST_VERSION — this guard is checking nothing, so either restore the pins or delete the guard"
  exit 1
fi

if [ "$status" -eq 0 ]; then
  echo "ok: every workflow RUST_VERSION matches rust-toolchain.toml ($channel)"
fi
exit "$status"
