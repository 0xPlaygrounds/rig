#!/usr/bin/env bash
# What reaches crates.io must be what we intended to publish, and it must fit.
#
# The root manifest is both the workspace root and the `rig` facade package, so
# the facade's package directory is the whole repository. Cargo ships everything
# not gitignored, so before [package].include existed, `cargo package -p rig`
# swept 1705 files — tests/ (27.3 MiB of recorded cassettes), img/, .github/,
# flake.nix, release-plz.toml — into a 9.51 MiB tarball against a 10,485,760 B
# hard cap, and it had grown 1.54 MiB in one release cycle. It also swept an
# *untracked* scratch file out of a maintainer's working tree, because cargo
# packages untracked-but-unignored files: what shipped depended on what happened
# to be lying around at publish time.
#
# Two guards, because they fail differently:
#
#   1. Contents (facade only). Catches the allowlist being deleted, widened or
#      quietly out-grown. This is the sharp one — a size check alone would not
#      notice tests/ returning until it was already megabytes.
#   2. Size (every publishable crate). The backstop for the crates that have no
#      allowlist at all; rig-core is the realistic next candidate.
#
# ---------------------------------------------------------------------------
# Why this only ever runs `cargo package --list`, and must keep doing so.
#
# `cargo package` (even with --no-verify) builds the tarball's lockfile from the
# *published* manifest, where path deps become registry deps — so it resolves
# every sibling `rig-* = { path = …, version = "X" }` against crates.io. On a
# release-plz PR, X is the version being released and is by definition not
# published yet, so it fails:
#
#   error: failed to select a version for the requirement `rig-agent = "^0.42.0"`
#   candidate versions found which didn't match: 0.41.0
#   location searched: crates.io index
#
# That would be worse than the bug this script exists to prevent: ci.yaml is
# what cd.yaml gates release-plz on, so the check could not go green until the
# version was published and the version could not publish until the check went
# green. Verified both directions against an unpublished version bump —
# `--no-verify` exits 101 for the facade and for siblings, `--list` exits 0 for
# all 22 crates.
#
# `--list` does not build that lockfile, so it is immune. The size below is
# therefore the uncompressed sum of the listed files rather than a .crate size:
# a conservative over-estimate of what the registry weighs, which is the right
# direction for a guard. Compiles nothing either way.
set -euo pipefail

cd "$(dirname "$0")/../.."

# crates.io's documented hard cap, on the *compressed* tarball. Not the gate —
# the thing the gate exists to keep us away from.
readonly CRATES_IO_CAP=10485760

# Gate on UNCOMPRESSED bytes. Compressed is never larger, so a crate under this
# ceiling is guaranteed under the registry cap — the gate is a sufficient
# condition, not an estimate, which is what lets it be trusted without ever
# building a tarball. It is a loose bound: rig-core is 4.32 MiB uncompressed and
# 0.88 MiB compressed, ~5x for source-heavy crates.
#
# 8 MiB leaves rig-core (the largest) ~1.9x room to grow while still failing
# loudly on a structural regression — the facade's pre-fix package was 30.2 MiB
# uncompressed. Raising it is fine, as a reviewed diff, which is the point.
readonly CEILING=$((8 * 1024 * 1024))

# Every path the `rig` facade may publish. Deliberately a SECOND, independent
# statement of [package].include rather than something derived from it: a guard
# that reads the value it is guarding proves only that the file parses. Adding a
# published file therefore takes two edits, here and in Cargo.toml, and that
# friction is the feature.
#
# Cargo.toml.orig, Cargo.lock and .cargo_vcs_info.json are synthesised by cargo
# and are not in the manifest's include list.
readonly RIG_ALLOWED='^(src/|img/(rig-rebranded-logo-(white|black)\.svg|built-by-playgrounds\.svg|ryzome-bg\.png)$|Cargo\.toml$|Cargo\.toml\.orig$|Cargo\.lock$|\.cargo_vcs_info\.json$|README\.md$|LICENSE$|CHANGELOG\.md$|MIGRATING\.md$)'

status=0
listing=$(mktemp)
trap 'rm -f "$listing"' EXIT

# --allow-dirty so this is runnable on a working tree, and so it keeps seeing
# untracked files: those are exactly what the contents check is here to catch.
# CI checks out clean, where the flag is a no-op. Redirected to a file rather
# than piped so `set -e` aborts on a cargo failure, instead of the failure being
# swallowed by a downstream `|| true` and reported as a pass.
cargo package -p rig --list --allow-dirty >"$listing"

# Liveness. Everything below is an absence assertion, and an absence assertion
# over an empty listing passes vacuously — the failure mode where the guard
# reports "ok" precisely because it learned nothing.
if ! grep -qxF 'src/lib.rs' "$listing"; then
  echo "::error::no usable package listing for \`rig\` (src/lib.rs absent), so nothing below would be meaningful"
  exit 1
fi

leaked=$(grep -Ev "$RIG_ALLOWED" "$listing" || true)
if [ -n "$leaked" ]; then
  leaked_count=$(printf '%s\n' "$leaked" | wc -l | tr -d ' ')
  echo "::error::${leaked_count} file(s) would be published inside the \`rig\` crate that are not part of it:"
  # Capped, because dropping [package].include entirely lists ~1,700 files and
  # would bury the error it is reporting. `awk NR<=20` rather than `head -20`:
  # head closes the pipe early, and under `set -o pipefail` that SIGPIPEs the
  # producer and aborts this script with 141 before the size checks run.
  printf '%s\n' "$leaked" | awk 'NR<=20 { print "    " $0 }'
  if [ "$leaked_count" -gt 20 ]; then
    echo "    ... and $((leaked_count - 20)) more (cargo package -p rig --list)"
  fi
  echo "::error::If consumers need them at build time, add them to [package].include in ./Cargo.toml AND to RIG_ALLOWED in this script. That is [package].include near the top of the manifest, not the [workspace].exclude table below it. Otherwise they should not ship."
  status=1
fi

# `--list` prints paths relative to the package's own directory, not the
# workspace root, so each crate's files must be summed from its manifest dir —
# otherwise every crate is measured against the root's src/ and Cargo.lock and
# they all report the same wrong number.
while IFS=$'\t' read -r name dir; do
  # stderr is dropped because cargo emits per-crate manifest-metadata warnings
  # unrelated to this check; the exit status is tested explicitly so a real
  # failure still fails rather than reading as an empty listing, which would
  # score 0 bytes and silently pass.
  if ! cargo package -p "$name" --list --allow-dirty >"$listing" 2>/dev/null; then
    echo "::error::cargo package --list failed for ${name}; cannot check its published size"
    status=1
    continue
  fi
  bytes=$(
    while read -r file; do
      [ -f "$dir/$file" ] && wc -c <"$dir/$file"
    done <"$listing" | awk '{ total += $1 } END { print total + 0 }'
  )
  printf '  %-20s %9d B uncompressed\n' "$name" "$bytes"
  if [ "$bytes" -gt "$CEILING" ]; then
    echo "::error::${name} packages to ${bytes} B uncompressed, over this repo's ${CEILING} B ceiling (crates.io caps the compressed tarball at ${CRATES_IO_CAP} B). Inspect it with: cargo package -p ${name} --list"
    status=1
  fi
done < <(
  cargo metadata --no-deps --format-version 1 | python3 -c '
import json, os, sys

# publish is null when unrestricted and [] when `publish = false`.
for pkg in json.load(sys.stdin)["packages"]:
    if pkg.get("publish") is None:
        print(pkg["name"], os.path.dirname(pkg["manifest_path"]), sep="\t")
'
)

if [ "$status" -eq 0 ]; then
  echo "ok: the rig facade publishes only its allowlist, and every publishable crate is under ${CEILING} B uncompressed"
fi
exit "$status"
