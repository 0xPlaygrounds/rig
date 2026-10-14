# rig-tools

Coding tools for [Rig](https://github.com/0xPlaygrounds/rig) agents:

| Tool     | What it does                                                        |
|----------|---------------------------------------------------------------------|
| `read`   | A text file as numbered lines, with an offset and limit             |
| `write`  | A whole file, written atomically                                    |
| `edit`   | Several exact (or whitespace- and quote-tolerant) replacements, all or nothing, answered with a diff |
| `search` | A regex over the tree as git ignores it, hidden files too when asked or when the glob names one; `find` returns the hits as data, `render` shows them |
| `list`   | A directory's files as git ignores them, down to a depth, optionally matching a glob, hidden files as for `search` |
| `shell`  | `sh -c` in its own process group, as a `Launcher` starts it, with a timeout (at most `max_timeout`, ten minutes by default) and an optional `OutputLimit` of its output |
| `delete`, `move` | A file removed, or moved or renamed |

Each tool carries its usage `RULES` for the system prompt, such as "use `read`,
not `cat`", and annotations saying what it does to the world. Beside the text
the model reads, `read` and `search` publish the files they read (`FilesRead`),
`search` what it found (`SearchHits`), `list` what it showed (`FilesListed`),
`write`, `edit`, `delete` and `move` the files they changed with each change's
hunks, the text before and the hashes of the content before and after
(`FilesChanged`), all by
resolved absolute path, `shell` how the command ended (`ExitStatus`), and
`shell`, `read` and `list` where they cut their output (`OutputCut`). The file
tools take an optional `base` directory for relative paths and can be
`confine`d to it, which refuses any path that resolves outside it; a confined
`shell` refuses only a `cwd` outside its directory, not what the command does.
Each call's `footprint` says what it may read and change: `read`, `search` and
`list` read the path they name, the other file tools write it, and `shell` may
change anything. Every tool's arguments are plain data with public fields.

Each tool acts on a `Workspace`: this machine (`Local`) or another, such as one
reached over SSH. It lists files as a `Listing` says (`Visible`, as a person
browsing sees them; `Hidden`, also the hidden files the ignore rules keep;
`All`) and watches a directory the same way (`watch`: system events on this
machine, looking every second elsewhere). The crate also has what the tools are built from: `overlay`
(a workspace that keeps writes from the disk and lists them, to preview a call),
`checkpoint` (a snapshot of the files under a root, ignored paths named in
`Taking::also` included, what changed since an earlier one and each file's
content then, and `Checkpointed`, a handler that reports what any call changed
and keeps the snapshot it took before;
in a git work tree the snapshot is a tree written with a private index and
object directory, never the repository's own, elsewhere a scan keeping content
under `Taking::keep`, a directory each owner keeps for itself), `clipboard` (the system
clipboard's text and images), `context` (the `AGENTS.md`/`CLAUDE.md`
instruction files from the working directory up, read within `Limits`), `fs`
(`write_atomic`, capped reads, `resolve`, `file_change`: a change's hunks
and the text before it), `process` (process
groups and the `Launcher`), `blocking`, and the text helpers `shorten` and `ago`.

The tools run on the file system and processes, so the crate is native only.
The [`rig-harness`](../rig-harness) coding agent registers them.
