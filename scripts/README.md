# scripts

- `check-dependency-floors.py` — downgrades every direct dependency to its declared floor and checks the workspace still builds.
- `release-notes.sh [PREVIOUS_TAG] [OUT_DIR]` — collects the commit log, merged PR notes and public API diffs since the previous tag into one directory for the release editorial pass.
- `release-notes-prompt.md` — the editorial pass that writes `CHANGELOG.md` and `MIGRATING.md` on the release PR, from the output of `release-notes.sh`.
- `version-switch.sh OLD NEW --work DIR [--expected FILE]` — run on the release PR with the previous release tag as OLD and the release candidate as NEW (git refs or installed `rig` launchers). Switches one used home from OLD to NEW and back with a fake model (`version-switch-model.py`), checks that the session resumes, settings stay, a second start does not rebuild and both versions can run on one home at once, reports each switch's time and disk, and exits non-zero on a failure. `version-switch-expected.txt` lists the checks known to fail for the current pair. Not run in CI: five agent builds, a few minutes each, and crates.io access.
