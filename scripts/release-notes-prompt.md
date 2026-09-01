# Release editorial pass: write CHANGELOG.md and MIGRATING.md for this release

This is the one context in which `CHANGELOG.md`, `crates/*/CHANGELOG.md` and
`MIGRATING.md` are edited. Before doing anything, confirm the branch is a
release PR: its name starts with `release-plz-` or the PR carries the `release`
label. If neither holds, stop and say so. The maintainer reviews the resulting
diff before it is pushed onto the release PR.

## Inputs

`scripts/release-notes.sh PREVIOUS_TAG` has been run; its output directory is
`target/release-notes/PREVIOUS_TAG..HEAD/` (the maintainer will name it):

- `commits.md` — the squash-merge log with commit bodies.
- `prs.md` — every merged PR with its `## Changelog` and `## Migration`
  sections extracted; PRs that predate the template show their whole
  description instead.
- `public-api/<package>.diff` — `cargo public-api` diff per publishable package.
- The `## [<version>]` section release-plz already wrote at the top of
  `CHANGELOG.md` (and of each `crates/*/CHANGELOG.md`) from the commit subjects.

## Output 1: the polished changelog section

Rewrite the release-plz section of the root `CHANGELOG.md` in place, keeping
its heading, its `### Added` / `### Changed` / `### Fixed` / `### Removed` /
`### Contributors` structure and its ordering:

- Merge the `## Changelog` bullets from `prs.md` into release-plz's
  commit-derived bullets. Where both describe the same change, keep the PR's
  wording (it is the author's fuller statement) and release-plz's suffixes.
- Keep the `*(scope)* [**breaking**] ...` voice, and the
  `(by [user](https://github.com/user))` and `([#N](...))` / ` - #N` suffixes
  release-plz emits. Every bullet ends with the PR link it came from.
- Dedupe. One bullet per user-visible change; drop pure-internal commits
  (CI, refactors with no public effect) unless a PR listed them explicitly.
- Do not touch older release sections, and do not add an `[Unreleased]` section.
- Leave `crates/*/CHANGELOG.md` as release-plz generated them unless a crate
  section is wrong; the root file is the one users read.

## Output 2: the migration guide section

Follow steps 1 to 7 of the preamble at the top of `MIGRATING.md`, with these
inputs standing in for "the changelogs": the `## Migration` sections in
`prs.md`, the API diffs, and the polished changelog above. Concretely:

- Add a new `## <previous> → <version>` section, newest-first, self-contained:
  every breaking change with old form, new form and the smallest useful
  example; the migration notes from `prs.md` are the raw material, the API
  diffs are the exhaustive spine (check every added, removed and changed item).
- Update **Which sections apply to you** (new row at the top; the previous
  `→ next` row becomes the concrete version).
- Add this release's entries to **Silent behavior changes** for anything that
  compiles and behaves differently: defaults, wire formats, provider behavior,
  feature semantics, error handling.
- Update the old-to-new symbol appendix.
- Verify every named symbol and feature against the tree (`grep`, `cargo doc`),
  and compile snippets where practical; label transcribed ones.
- Never edit the pinned preamble between the `MIGRATING-GUIDE-INSTRUCTIONS`
  markers. CI checks its hash.

## Finish

Run `bash .github/scripts/check-migrating-guide-preamble.sh`, check internal
Markdown links, and present the diff of both files for review. Do not commit
or push unless asked.
