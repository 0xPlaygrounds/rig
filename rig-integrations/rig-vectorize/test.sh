#!/bin/bash
# Run integration tests with credentials from .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    echo "Loaded credentials from .env"
else
    echo "Warning: .env file not found. Tests may be skipped."
fi

cd "$SCRIPT_DIR/../.."
cargo test --package rig-vectorize --test integration_tests -- --nocapture "$@"
