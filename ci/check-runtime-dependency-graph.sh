#!/usr/bin/env bash

set -euo pipefail

metadata="$(mktemp)"
trap 'rm -f "$metadata"' EXIT

cargo metadata --format-version 1 >"$metadata"

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

echo "runtime dependency graph guard passed"
