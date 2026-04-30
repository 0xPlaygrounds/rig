#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_MANIFEST="${ROOT_DIR}/Cargo.toml"
ROOT_LIB="${ROOT_DIR}/src/lib.rs"

failures=0

check() {
  local description="$1"
  shift

  if ! "$@"; then
    printf 'error: %s\n' "${description}" >&2
    failures=$((failures + 1))
  fi
}

contains_fixed() {
  local file="$1"
  local needle="$2"

  grep -Fq "${needle}" "${file}"
}

root_optional_integration_packages() {
  awk '
    /^\[dependencies\]/ {
      in_dependencies = 1
      next
    }

    /^\[/ {
      in_dependencies = 0
    }

    in_dependencies && /^rig-[A-Za-z0-9_-]+[[:space:]]*=/ && /optional[[:space:]]*=[[:space:]]*true/ {
      package = $1
      if (package != "rig-core") {
        print package
      }
    }
  ' "${ROOT_MANIFEST}" | sort
}

echo "Validating root integration facade..."

check "root facade must re-export rig-core" \
  contains_fixed "${ROOT_LIB}" "pub use rig_core::*;"

check "root default features must forward only rig-core/default" \
  contains_fixed "${ROOT_MANIFEST}" 'default = ["rig-core/default"]'

mapfile -t packages < <(root_optional_integration_packages)

if (( ${#packages[@]} == 0 )); then
  printf 'error: no optional rig-* integration dependencies found in root Cargo.toml\n' >&2
  failures=$((failures + 1))
fi

for package in "${packages[@]}"; do
  feature="${package#rig-}"
  module="${feature//-/_}"
  crate_name="${package//-/_}"

  check "missing crate directory: crates/${package}" \
    test -d "${ROOT_DIR}/crates/${package}"

  check "root Cargo.toml missing optional dependency ${package}" \
    contains_fixed "${ROOT_MANIFEST}" "${package} = { path = \"crates/${package}\","

  check "root Cargo.toml dependency ${package} is not optional" \
    grep -Eq "^${package} = \\{ .*optional = true" "${ROOT_MANIFEST}"

  check "root Cargo.toml dependency ${package} must not enable default features" \
    grep -Eq "^${package} = \\{ .*default-features = false" "${ROOT_MANIFEST}"

  check "root Cargo.toml missing feature ${feature}" \
    contains_fixed "${ROOT_MANIFEST}" "${feature} = [\"dep:${package}\"]"

  check "src/lib.rs missing cfg for feature ${feature}" \
    contains_fixed "${ROOT_LIB}" "#[cfg(feature = \"${feature}\")]"

  check "src/lib.rs missing module ${module} for feature ${feature}" \
    contains_fixed "${ROOT_LIB}" "pub mod ${module} {"

  check "src/lib.rs missing re-export for ${crate_name}" \
    contains_fixed "${ROOT_LIB}" "pub use ${crate_name}::*;"
done

if grep -Eq '^default = .*dep:rig-' "${ROOT_MANIFEST}"; then
  printf 'error: root default features must not enable integration crates\n' >&2
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  printf 'Root integration facade validation failed with %d error(s).\n' "${failures}" >&2
  exit 1
fi

echo "Root integration facade is in sync."
