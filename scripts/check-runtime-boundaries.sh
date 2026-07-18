#!/usr/bin/env bash
set -euo pipefail

assert_absent() {
  local package=$1
  local forbidden=$2
  shift 2
  if cargo tree -p "$package" -e normal,build "$@" --prefix none | grep -Eq "^(${forbidden})( |$)"; then
    echo "forbidden runtime dependency in $package: $forbidden" >&2
    exit 1
  fi
}

assert_present() {
  local package=$1
  local required=$2
  shift 2
  if ! cargo tree -p "$package" -e normal,build "$@" --prefix none | grep -Eq "^${required}( |$)"; then
    echo "expected runtime dependency missing from $package: $required" >&2
    exit 1
  fi
}

assert_absent rig-core 'rig-agent|rig-bevy'
assert_absent rig-agent 'rig-bevy'
assert_absent rig-bevy 'rig-agent'

assert_absent rig 'rig-agent|rig-bevy' --no-default-features
assert_present rig rig-agent --no-default-features --features agent
assert_absent rig rig-bevy --no-default-features --features agent
assert_present rig rig-bevy --no-default-features --features bevy
assert_absent rig rig-agent --no-default-features --features bevy
assert_present rig rig-agent --no-default-features --features agent,bevy
assert_present rig rig-bevy --no-default-features --features agent,bevy

if rg -n \
  'crate::agent|rig_core::agent|completion::(Prompt|Chat|TypedPrompt|PromptError|StructuredOutputError)|\bAgentBuilder\b|\bAgentRunner\b|\bToolContext\b|\bToolSet\b|\bDynamicTool\b' \
  crates/rig-core/src --glob '*.rs'; then
  echo 'classic-runtime implementation leaked into rig-core' >&2
  exit 1
fi

cargo check --manifest-path \
  crates/rig-derive/tests/fixtures/renamed-dependencies/Cargo.toml --locked
cargo check --manifest-path \
  crates/rig-derive/tests/fixtures/renamed-root/Cargo.toml --locked
