#!/usr/bin/env bash

set -euo pipefail

readonly guide="MIGRATING.md"
readonly start_marker="<!-- MIGRATING-GUIDE-INSTRUCTIONS:START -->"
readonly end_marker="<!-- MIGRATING-GUIDE-INSTRUCTIONS:END -->"
readonly expected_hash="4ee35673f3dc6cd57d77814c9476b49ea63cfab1"

start_matches="$(grep -nFx "${start_marker}" "${guide}" || true)"
end_matches="$(grep -nFx "${end_marker}" "${guide}" || true)"

if [[ "$(printf '%s\n' "${start_matches}" | sed '/^$/d' | wc -l | tr -d ' ')" != "1" ]] ||
    [[ "$(printf '%s\n' "${end_matches}" | sed '/^$/d' | wc -l | tr -d ' ')" != "1" ]]; then
    echo "MIGRATING.md must contain exactly one immutable preamble marker pair." >&2
    exit 1
fi

start_line="${start_matches%%:*}"
end_line="${end_matches%%:*}"

if [[ "${start_line}" != "3" ]] || ((end_line <= start_line)); then
    echo "The immutable MIGRATING.md preamble must start immediately after the title." >&2
    exit 1
fi

actual_hash="$(sed -n "${start_line},${end_line}p" "${guide}" | git hash-object --stdin)"

if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "The immutable MIGRATING.md preamble was modified." >&2
    exit 1
fi

echo "The immutable MIGRATING.md preamble is intact."
