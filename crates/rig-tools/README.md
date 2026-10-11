# rig-tools

Coding tools for [Rig](https://github.com/0xPlaygrounds/rig) agents:

| Tool     | What it does                                                        |
|----------|---------------------------------------------------------------------|
| `read`   | A text file as numbered lines, with an offset and limit             |
| `write`  | A whole file, written atomically                                    |
| `edit`   | Several exact (or whitespace- and quote-tolerant) replacements, all or nothing, answered with a diff |
| `search` | A regex over the tree as git ignores it; `find` returns the hits as data |
| `shell`  | `sh -c` in its own process group, as a `Launcher` starts it, with a timeout and an optional `Cut` of its output |

Each tool carries its usage `RULES` for the system prompt, such as "use `read`,
not `cat`", and annotations saying what it does to the world. `write` and `edit`
publish the files they changed (`FilesChanged`) and `shell` how the command
ended (`ExitStatus`), beside the text the model reads. Every tool's arguments
are plain data with public fields. The crate also has what they are built from:
`context` (the `AGENTS.md`/`CLAUDE.md` instruction files from the working
directory up), `fs` (`write_atomic`, capped reads, `file_change`), `process`
(process groups and the `Launcher`) and `blocking`.

The tools run on the file system and processes, so the crate is native only.
The [`rig-harness`](../rig-harness) coding agent registers them.
