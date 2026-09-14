#!/usr/bin/env bash

# Collect everything the release-time editorial pass needs into one directory:
# the commit log since the previous tag, the merged pull requests with their
# `## Changelog` / `## Migration` sections, and the public API diff per
# publishable package. The editorial pass itself is scripts/release-notes-prompt.md;
# it is the only place CHANGELOG.md and MIGRATING.md get written.
#
# Usage: scripts/release-notes.sh [PREVIOUS_TAG] [OUT_DIR]
#   PREVIOUS_TAG  defaults to the latest v* tag reachable from HEAD
#   OUT_DIR       defaults to target/release-notes/<PREVIOUS_TAG>..HEAD
#
# Requires git, jq and gh (authenticated). cargo-public-api is optional: without
# it the API diff step is skipped with a warning.

set -euo pipefail

for tool in git jq gh; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "${tool} is required." >&2
        exit 1
    fi
done

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

prev_tag="${1:-$(git describe --tags --abbrev=0 --match 'v*')}"
readonly prev_tag
out_dir="${2:-target/release-notes/${prev_tag}..HEAD}"
readonly out_dir

if ! git rev-parse -q --verify "refs/tags/${prev_tag}" >/dev/null; then
    echo "Tag ${prev_tag} does not exist." >&2
    exit 1
fi

mkdir -p "${out_dir}"
head_sha="$(git rev-parse HEAD)"
tag_date="$(git log -1 --format=%cd --date=iso-strict "${prev_tag}")"

echo "Collecting release inputs for ${prev_tag}..${head_sha:0:9} into ${out_dir}"

# 1. Commit log: squash-merge subjects plus their bodies (the PR descriptions).
git log --no-merges --format='- %h %s%n%n%b' "${prev_tag}..HEAD" >"${out_dir}/commits.md"

# 2. Merged pull requests. GitHub's search is by merge date, so it over-fetches;
#    keep only PRs whose number appears in a squash subject within the range.
range_pr_numbers="$(git log --no-merges --format='%s' "${prev_tag}..HEAD" |
    grep -oE '\(#[0-9]+\)$' | tr -dc '0-9\n' | sort -u)"

gh pr list --state merged --base main \
    --search "merged:>=${tag_date%%T*}" \
    --limit 500 \
    --json number,title,body,mergedAt,labels,author,url |
    jq --arg nums "${range_pr_numbers}" '
        ($nums | split("\n") | map(select(length > 0) | tonumber)) as $keep
        | map(select(.number as $n | $keep | index($n)))
        | sort_by(.mergedAt)' >"${out_dir}/prs.json"

pr_count="$(jq length "${out_dir}/prs.json")"

# Render one Markdown file with, per PR, the `## Changelog` and `## Migration`
# sections of its description (the whole body when a PR predates the template).
jq -r '
    def section($name):
        (split("\n") | . as $lines
         | ($lines | to_entries | map(select(.value | test("^## " + $name + "\\s*$"))) | first | .key) as $start
         | if $start == null then null
           else ($lines[$start+1:]
                 | to_entries | map(select(.value | test("^## "))) | first | .key) as $len
                | ($lines[$start+1:] | if $len == null then . else .[:$len] end)
                | join("\n") | gsub("^\\s+|\\s+$"; "")
           end);
    .[] |
    "## #\(.number) \(.title)\n\n" +
    "<\(.url)> · merged \(.mergedAt) · @\(.author.login) · labels: \(.labels | map(.name) | join(", ") | if . == "" then "none" else . end)\n\n" +
    ((.body // "") as $b |
     ($b | section("Changelog")) as $c |
     ($b | section("Migration")) as $m |
     if $c == null and $m == null then
        "### Description (no changelog/migration sections)\n\n" + ($b | if . == "" then "_empty_" else . end) + "\n"
     else
        "### Changelog\n\n" + ($c // "_absent_") + "\n\n### Migration\n\n" + ($m // "_absent_") + "\n"
     end) + "\n---\n"
' "${out_dir}/prs.json" >"${out_dir}/prs.md"

# 3. Public API diff per publishable library / proc-macro package, run from a
#    clean worktree as the MIGRATING.md preamble requires (cargo public-api
#    checks refs out in place, so it must not run in a dirty tree).
api_dir="${out_dir}/public-api"
rm -rf "${api_dir}"
mkdir -p "${api_dir}"

if command -v cargo-public-api >/dev/null 2>&1; then
    worktree="$(mktemp -d "${TMPDIR:-/tmp}/rig-public-api.XXXXXX")"
    rmdir "${worktree}"
    git worktree add --detach -q "${worktree}" "${head_sha}"
    trap 'git worktree remove --force "${worktree}" >/dev/null 2>&1 || true' EXIT

    list_packages() {
        (cd "${worktree}" && git checkout -q "$1" && cargo metadata --no-deps --format-version 1 |
            jq -r '.packages[] | select(.publish != []) | select(any(.targets[]; any(.kind[]; . == "lib" or . == "proc-macro"))) | .name')
    }
    packages="$( { list_packages "${prev_tag}"; list_packages "${head_sha}"; } | sort -u)"

    for pkg in ${packages}; do
        echo "  public-api diff: ${pkg}"
        if ! (cd "${worktree}" && cargo public-api -p "${pkg}" --simplified diff "${prev_tag}..${head_sha}") \
            >"${api_dir}/${pkg}.diff" 2>"${api_dir}/${pkg}.stderr"; then
            echo "    (failed — ${pkg} may exist at only one ref; see ${api_dir}/${pkg}.stderr)"
        else
            rm -f "${api_dir}/${pkg}.stderr"
        fi
    done
else
    echo "warning: cargo-public-api not installed; skipping the API diff." >&2
    echo "         cargo install cargo-public-api --locked" >&2
fi

cat >"${out_dir}/README.md" <<README
# Release inputs: ${prev_tag}..${head_sha:0:9}

\`commits.md\` is the squash-merge log since ${prev_tag} with commit bodies.
\`prs.json\` / \`prs.md\` hold the ${pr_count} pull requests merged in that range, with the
\`## Changelog\` and \`## Migration\` sections of each description extracted.
\`public-api/<package>.diff\` is the \`cargo public-api\` diff per publishable package
(absent when cargo-public-api is not installed).

Next step: on the release PR branch, run the editorial pass in
\`scripts/release-notes-prompt.md\` against this directory. That pass is the only
place \`CHANGELOG.md\` and \`MIGRATING.md\` are written.
README

echo "Done: ${pr_count} PRs, $(ls "${api_dir}" | grep -c '\.diff$' || true) API diffs. See ${out_dir}/README.md"
