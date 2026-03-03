#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Validating generated agent instruction adapters..."
"${ROOT_DIR}/scripts/sync_agent_instruction_files.sh" --check
