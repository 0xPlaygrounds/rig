#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/crates/rig-sqlite/tests/fixtures"

skip_clippy=false
skip_integration=false

while (($#)); do
  case "$1" in
    --skip-clippy)
      skip_clippy=true
      ;;
    --skip-integration)
      skip_integration=true
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1

echo "==> Running converter unit tests"
python3 -B -m unittest "$FIXTURE_DIR/fixture_tools_test.py"

echo "==> Running sqlite crate tests"
cargo test -p rig-sqlite

echo "==> Running checked-in vector fixture"
RIG_SQLITE_VECTOR_FIXTURE="$FIXTURE_DIR/vector_fixture.json" \
  cargo test -p rig-sqlite external_vector_fixture_matches_ground_truth -- --ignored

echo "==> Running checked-in retrieval fixture"
RIG_SQLITE_VECTOR_FIXTURE="$FIXTURE_DIR/retrieval_fixture.json" \
  cargo test -p rig-sqlite external_vector_fixture_matches_ground_truth -- --ignored

if [[ "$skip_integration" == false ]]; then
  echo "==> Running sqlite integration tests"
  cargo test --features sqlite --test integrations sqlite
fi

if [[ "$skip_clippy" == false ]]; then
  echo "==> Running sqlite clippy"
  cargo clippy -p rig-sqlite --all-targets
fi

echo "==> SQLite vector validation complete"
