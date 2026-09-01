# scripts

- `check-dependency-floors.py` — downgrades every direct dependency to its declared floor and checks the workspace still builds.
- `release-notes.sh [PREVIOUS_TAG] [OUT_DIR]` — collects the commit log, merged PR notes and public API diffs since the previous tag into one directory for the release editorial pass.
- `release-notes-prompt.md` — the editorial pass that writes `CHANGELOG.md` and `MIGRATING.md` on the release PR, from the output of `release-notes.sh`.
