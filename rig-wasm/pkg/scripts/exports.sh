#!/usr/bin/env bash
# Automates the build pipeline for `rig-wasm`.
set -euo pipefail

OUT_DIR="out/esm"
CORE_MODULE="src"
VECTOR_STORES_MODULE="src/vector_stores"
PROVIDERS_MODULE="src/providers"
EXPORTS_FILE="exports.json"

echo '{' > "$EXPORTS_FILE"

# Add the root export manually
echo '  ".": {' >> "$EXPORTS_FILE"
echo '    "import": "./out/esm/index.js",' >> "$EXPORTS_FILE"
echo '    "require": "./out/cjs/index.cjs",' >> "$EXPORTS_FILE"
echo '    "types": "./out/esm/index.d.ts"' >> "$EXPORTS_FILE"
echo '  },' >> "$EXPORTS_FILE"

# Gather all .js files (excluding index.js)
mapfile -t core_files < <(find "$CORE_MODULE" -maxdepth 1 -type f -name "*.ts" | sort)
echo "Found ${#core_files[@]} core modules. Building exports..."

for i in "${!core_files[@]}"; do
  file="${core_files[$i]}"
  base=$(basename "$file" .js)

  echo "  \"./${base%.ts}\": {" >> "$EXPORTS_FILE"
  echo "    \"import\": \"./$OUT_DIR/${base%.ts}.js\"," >> "$EXPORTS_FILE"
  echo "    \"types\": \"./$OUT_DIR/${base%.ts}.d.ts\"" >> "$EXPORTS_FILE"
  echo "  }," >> "$EXPORTS_FILE"
done

# Gather all .js files (excluding index.js)
mapfile -t vector_stores_files < <(find "$VECTOR_STORES_MODULE" -maxdepth 1 -type f -name "*.ts" | sort)
echo "Found ${#vector_stores_files[@]} vector store modules. Building exports..."

for i in "${!vector_stores_files[@]}"; do
  file="${vector_stores_files[$i]}"
  base=$(basename "$file" .js)

  echo "  \"./${base%.ts}\": {" >> "$EXPORTS_FILE"
  echo "    \"import\": \"./$OUT_DIR/${base%.ts}.js\"," >> "$EXPORTS_FILE"
  echo "    \"types\": \"./$OUT_DIR/vector_stores/${base%.ts}.d.ts\"" >> "$EXPORTS_FILE"
  echo "  }," >> "$EXPORTS_FILE"
done

# Gather all .js files (excluding index.js)
mapfile -t providers_files < <(find "$PROVIDERS_MODULE" -maxdepth 1 -type f -name "*.ts" | sort)
echo "Found ${#providers_files[@]} provider modules. Building exports..."

for i in "${!providers_files[@]}"; do
  file="${providers_files[$i]}"
  base=$(basename "$file" .js)

  echo "  \"./${base%.ts}\": {" >> "$EXPORTS_FILE"
  echo "    \"import\": \"./$OUT_DIR/${base%.ts}.js\"," >> "$EXPORTS_FILE"
  echo "    \"types\": \"./$OUT_DIR/providers/${base%.ts}.d.ts\"" >> "$EXPORTS_FILE"

  if [[ $i -lt $((${#providers_files[@]} - 1)) ]]; then
    echo "  }," >> "$EXPORTS_FILE"
  else
    echo "  }" >> "$EXPORTS_FILE"
  fi
done

echo "}" >> "$EXPORTS_FILE"

echo "✅ Wrote updated exports to $EXPORTS_FILE"

jq --slurpfile exports exports.json '.exports = $exports[0]' package.json > package.new.json && mv package.new.json package.json

echo "✅ Updated exports in package.json"

rm $EXPORTS_FILE
