#!/usr/bin/env bash

set -euo pipefail

ROOT="rig-integrations"

find "$ROOT" \
  -type f \
  -name "Cargo.toml" \
  -not -path "*/target/*" \
  -print0 |
while IFS= read -r -d '' file; do
  perl -0777 -i -pe '
    s{
      (rig-core\s*=\s*["\'])(\d+)\.(\d+)\.(\d+)(["\'])
    }{
      my ($p, $maj, $min, $patch, $s) = ($1, $2, $3, $4, $5);
      $p . $maj . "." . ($min + 1) . ".0" . $s
    }egx
  ' "$file"
done
