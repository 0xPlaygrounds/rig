# rig-tools

Coding tools for [Rig](https://github.com/0xPlaygrounds/rig) agents:

| Tool     | What it does                                                        |
|----------|---------------------------------------------------------------------|
| `read`   | A text file's lines, with an offset and limit                       |
| `write`  | A whole file, written atomically                                    |
| `edit`   | Several exact (or whitespace- and quote-tolerant) replacements, all or nothing, answered with the changed lines' numbers |
| `search` | A regex over the tree as git ignores it, hidden files too when asked or when the glob names one; `find` returns the hits as data, `render` shows them |
| `list`   | A directory's files as git ignores them, down to a depth, optionally matching a glob, hidden files as for `search` |
| `shell`  | `sh -c` in its own process group, as a `Launcher` starts it, with a timeout (at most `max_timeout`, ten minutes by default) and an optional `OutputLimit` of its output |
| `delete`, `move` | A file removed, or moved or renamed |

Tools publish facts beside their text (such as `FilesChanged`), and every tool acts on a
`Workspace`: this machine (`Local`) or another. The pieces the tools are built from, such as
checkpoints, the overlay workspace and process groups, are listed in the crate docs.

The tools run on the file system and processes, so the crate is native only.
The [`rig-harness`](../rig-harness) coding agent registers them.
