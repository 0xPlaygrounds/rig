#!/usr/bin/env bash

set -euo pipefail

check_declared_graph() {
  local metadata
  metadata="$(mktemp)"
  cargo metadata --format-version 1 "$@" >"$metadata"
  jq -e '
  . as $metadata |
  def package($name): $metadata.packages[] | select(.name == $name);
  def has_package($name): any($metadata.packages[]; .name == $name);
  def dependency_names($name): [package($name).dependencies[].name];
  def production_dependency_names($name):
    [package($name).dependencies[] | select(.kind != "dev") | .name];
  def excludes($dependencies; $forbidden):
    all($forbidden[]; . as $name | ($dependencies | index($name) | not));

  (dependency_names("rig-core")) as $core |
  (excludes($core; ["rig-agent", "rig-bevy", "bevy_ecs"])) and

  (if has_package("rig-agent") then
     (dependency_names("rig-agent")) as $agent |
     ($agent | index("rig-core") != null) and
     excludes($agent; ["rig-bevy"])
   else true end) and

  (if has_package("rig-bevy") then
     (dependency_names("rig-bevy")) as $bevy |
     ($bevy | index("rig-core") != null) and
     excludes($bevy; ["rig-agent"])
   else true end) and

  all([
    "rig-bedrock", "rig-candle", "rig-fastembed", "rig-gemini-grpc",
    "rig-helixdb", "rig-lancedb", "rig-memory", "rig-milvus",
    "rig-mongodb", "rig-neo4j", "rig-postgres", "rig-qdrant",
    "rig-s3vectors", "rig-scylladb", "rig-sqlite", "rig-surrealdb",
    "rig-vectorize", "rig-vertexai"
  ][]; . as $name |
    production_dependency_names($name) as $dependencies |
    excludes($dependencies; ["rig-agent", "rig-bevy"])
  )
' "$metadata" >/dev/null
  rm -f "$metadata"
}

check_rig_agent_root_surface() {
  local lib="crates/rig-agent/src/lib.rs"
  local allowlist="ci/rig-agent-root-reexport-allowlist.txt"

  # A blanket glob re-export at the crate root turns rig-agent back into an
  # implicit second facade. It is never permitted, regardless of the allowlist.
  # Root items sit at column 0; the `core` module's own glob is indented.
  if grep -Eq '^pub use rig_core::\*' "$lib"; then
    echo "forbidden: blanket 'pub use rig_core::*' at the rig-agent crate root" >&2
    echo "  portable contracts must be reached through rig_agent::core" >&2
    return 1
  fi

  # Every enumerated root re-export from the portable crates must be declared in
  # the allowlist, so widening rig-agent's public surface is always a reviewed,
  # intentional change rather than an accidental one.
  local actual expected
  actual="$(grep -E '^pub use (rig_core|rig_derive)' "$lib" | sed 's/[[:space:]]*$//' | sort -u)"
  expected="$(grep -vE '^[[:space:]]*(#|$)' "$allowlist" | sed 's/[[:space:]]*$//' | sort -u)"

  if [[ "$actual" != "$expected" ]]; then
    echo "rig-agent root re-export surface drifted from the allowlist:" >&2
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
    echo "  update ${allowlist} only if the change is intentional" >&2
    return 1
  fi
}

check_resolved_tree_excludes() {
  local package="$1"
  shift
  local tree
  tree="$(cargo tree -p "$package" --all-features --target all -e all)"
  for forbidden in "$@"; do
    if grep -Eq "(^|[^[:alnum:]_-])${forbidden} v" <<<"$tree"; then
      echo "forbidden resolved dependency: ${package} -> ${forbidden}" >&2
      return 1
    fi
  done
}

check_declared_graph
check_declared_graph --all-features
check_declared_graph --no-default-features

check_resolved_tree_excludes rig-core rig-agent rig-bevy bevy_ecs
check_resolved_tree_excludes rig-agent rig-bevy bevy_ecs
check_resolved_tree_excludes rig-bevy rig-agent

check_rig_agent_root_surface

echo "runtime dependency graph guard passed"
