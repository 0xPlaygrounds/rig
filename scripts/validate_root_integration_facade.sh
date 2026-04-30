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

feature_names() {
  local manifest="$1"

  awk '
    /^\[features\]/ {
      in_features = 1
      next
    }

    /^\[/ {
      in_features = 0
    }

    in_features && /^[A-Za-z0-9_-]+[[:space:]]*=/ {
      feature = $1
      sub(/[[:space:]]*=.*/, "", feature)
      print feature
    }
  ' "${manifest}" | sort
}

feature_block() {
  local manifest="$1"
  local feature="$2"

  awk -v target="${feature}" '
    /^\[features\]/ {
      in_features = 1
      next
    }

    /^\[/ {
      in_features = 0
      in_target = 0
    }

    in_features && /^[A-Za-z0-9_-]+[[:space:]]*=/ {
      feature = $1
      sub(/[[:space:]]*=.*/, "", feature)
      in_target = feature == target
    }

    in_features && in_target {
      print
    }
  ' "${manifest}"
}

feature_has_item() {
  local manifest="$1"
  local feature="$2"
  local item="$3"

  feature_block "${manifest}" "${feature}" | grep -Fq "\"${item}\""
}

quoted_feature_items() {
  local manifest="$1"
  local feature="$2"

  feature_block "${manifest}" "${feature}" \
    | grep -Eo '"[^"]+"' \
    | tr -d '"'
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

root_feature_name_for_downstream_feature() {
  local integration_feature="$1"
  local downstream_feature="$2"
  local public_feature="${downstream_feature#rig-}"

  if [[ "${public_feature}" == "${integration_feature}"-* ]]; then
    printf '%s\n' "${public_feature}"
  else
    printf '%s-%s\n' "${integration_feature}" "${public_feature}"
  fi
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
  manifest="${ROOT_DIR}/crates/${package}/Cargo.toml"

  check "missing crate directory: crates/${package}" \
    test -d "${ROOT_DIR}/crates/${package}"

  check "root Cargo.toml missing optional dependency ${package}" \
    contains_fixed "${ROOT_MANIFEST}" "${package} = { path = \"crates/${package}\","

  check "root Cargo.toml dependency ${package} is not optional" \
    grep -Eq "^${package} = \\{ .*optional = true" "${ROOT_MANIFEST}"

  check "root Cargo.toml dependency ${package} must not enable default features" \
    grep -Eq "^${package} = \\{ .*default-features = false" "${ROOT_MANIFEST}"

  check "root Cargo.toml missing feature ${feature}" \
    feature_has_item "${ROOT_MANIFEST}" "${feature}" "dep:${package}"

  check "src/lib.rs missing cfg for feature ${feature}" \
    contains_fixed "${ROOT_LIB}" "feature = \"${feature}\""

  check "src/lib.rs missing module ${module} for feature ${feature}" \
    contains_fixed "${ROOT_LIB}" "pub mod ${module} {"

  check "src/lib.rs missing re-export for ${crate_name}" \
    contains_fixed "${ROOT_LIB}" "pub use ${crate_name}::*;"

  if [[ -f "${manifest}" ]]; then
    mapfile -t downstream_features < <(feature_names "${manifest}")
    mapfile -t downstream_defaults < <(quoted_feature_items "${manifest}" "default")

    for downstream_feature in "${downstream_features[@]}"; do
      if [[ "${downstream_feature}" == "default" ]]; then
        continue
      fi

      root_feature="$(root_feature_name_for_downstream_feature "${feature}" "${downstream_feature}")"

      if [[ "${downstream_feature}" == *rustls* ]]; then
        check "ergonomic root feature ${feature} must enable Rustls via ${package}/${downstream_feature}" \
          feature_has_item "${ROOT_MANIFEST}" "${feature}" "${package}/${downstream_feature}"
        continue
      fi

      check "root Cargo.toml missing downstream feature ${root_feature}" \
        feature_has_item "${ROOT_MANIFEST}" "${root_feature}" "dep:${package}"

      check "root Cargo.toml feature ${root_feature} must forward ${package}/${downstream_feature}" \
        feature_has_item "${ROOT_MANIFEST}" "${root_feature}" "${package}/${downstream_feature}"

      check "src/lib.rs cfg for module ${module} must include feature ${root_feature}" \
        contains_fixed "${ROOT_LIB}" "feature = \"${root_feature}\""
    done

    for downstream_default in "${downstream_defaults[@]}"; do
      check "ergonomic root feature ${feature} must include downstream default ${package}/${downstream_default}" \
        feature_has_item "${ROOT_MANIFEST}" "${feature}" "${package}/${downstream_default}"
    done
  fi
done

mapfile -t core_features < <(feature_names "${ROOT_DIR}/crates/rig-core/Cargo.toml")

for core_feature in "${core_features[@]}"; do
  check "root Cargo.toml missing rig-core feature ${core_feature}" \
    feature_has_item "${ROOT_MANIFEST}" "${core_feature}" "rig-core/${core_feature}"
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
