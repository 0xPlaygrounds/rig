#!/usr/bin/env bash
# `rust-toolchain.toml` is the single toolchain source: the rust-setup
# composite action (.github/actions/rust-setup) resolves its channel at run
# time, and no workflow carries a RUST_VERSION copy anymore. This guard
# exists to keep it that way — it fails on the two bypasses that would
# silently build one job on a stale toolchain:
#
#   * a reintroduced `RUST_VERSION:` env copy that drifts from the channel
#     (the pre-single-source failure mode: the gate goes green on the new
#     toolchain while a missed workflow — nightly.yaml is also cd.yaml's
#     pre-release gate — keeps compiling on the old one, misread as a
#     runtime regression);
#   * an inline literal `toolchain:` input in a `with:` block, which
#     sidesteps the composite action entirely.
#
# This is a text check on purpose: it must fail on a workflow nobody
# remembered to convert, which a parser keyed off the workflows that use the
# composite action would not do. With no matches anywhere it passes
# vacuously — that IS the desired state.
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

  # Also catch the bypass this env-var check cannot see: a job that pins a
  # literal version directly in `setup-rust-toolchain`'s `with:` block
  # (`toolchain: 1.93.0`) declares no RUST_VERSION and would pass vacuously
  # while silently building on a stale toolchain. `${{ env.RUST_VERSION }}`
  # references are skipped — they resolve to a value already checked above.
  while IFS= read -r inline; do
    if [ "$inline" != "$channel" ]; then
      echo "::error file=$workflow::inline toolchain pin is $inline but rust-toolchain.toml pins $channel"
      status=1
    fi
  done < <(grep -E '^\s*toolchain:' "$workflow" | grep -vF '${{' \
    | sed -E 's/\r$//; s/^[[:space:]]*toolchain:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//; s/^"([^"]*)"$/\1/; s/^'"'"'([^'"'"']*)'"'"'$/\1/')
done

if [ "$status" -eq 0 ]; then
  echo "ok: every declared RUST_VERSION matches rust-toolchain.toml ($channel)"
fi
exit "$status"
