#!/usr/bin/env bash

# CHANGELOG.md, crates/*/CHANGELOG.md, MIGRATING.md and docs/migrations/ are
# release documents: they are written on the release PR, from the merged pull
# requests' `## Changelog` / `## Migration` sections and the public API diff.
# An ordinary PR never edits them — that is what makes several in-flight PRs
# rebase cleanly. This guard fails a PR whose diff against its base touches
# any of them. Release PRs (`release-plz-*` head branch or the `release`
# label) are exempted by the workflow's job condition, not here.

set -euo pipefail

base_ref="${1:-${BASE_REF:-origin/main}}"
readonly base_ref
readonly frozen_pattern='^(CHANGELOG\.md|MIGRATING\.md|crates/[^/]+/CHANGELOG\.md|docs/migrations/.+)$'

if ! merge_base="$(git merge-base "${base_ref}" HEAD)"; then
    echo "Cannot find the merge base of ${base_ref} and HEAD." >&2
    exit 1
fi

touched="$(git diff --name-only "${merge_base}" HEAD | grep -E "${frozen_pattern}" || true)"

if [[ -n "${touched}" ]]; then
    {
        echo "This PR edits release documents that are generated on the release PR:"
        printf '  %s\n' ${touched}
        echo
        echo "Do not edit CHANGELOG.md, crates/*/CHANGELOG.md, MIGRATING.md or docs/migrations/"
        echo "in an ordinary PR. Put changelog bullets under '## Changelog' and migration notes"
        echo "under '## Migration' in the PR description instead. The files are regenerated on"
        echo "the release PR by scripts/release-notes.sh and the editorial pass in"
        echo "scripts/release-notes-prompt.md."
    } >&2
    exit 1
fi

echo "No frozen release docs touched since ${merge_base:0:9}."
