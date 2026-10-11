# rig-tools

Coding tools for [Rig](https://github.com/0xPlaygrounds/rig) agents:

| Tool     | What it does                                                        |
|----------|---------------------------------------------------------------------|
| `read`   | A text file as numbered lines, with an offset and limit             |
| `write`  | A whole file, written atomically                                    |
| `edit`   | Several exact (or whitespace- and quote-tolerant) replacements, all or nothing, answered with a diff |
| `search` | A regex over the tree as git ignores it; `find` returns the hits as data, `render` shows them |
| `shell`  | `sh -c` in its own process group, as a `Launcher` starts it, with a timeout and an optional `Cut` of its output |

Each tool carries its usage `RULES` for the system prompt, such as "use `read`,
not `cat`", and annotations saying what it does to the world. Beside the text
the model reads, `read` and `search` publish the files they read (`FilesRead`),
`search` what it found (`SearchHits`), `write` and `edit` the files they changed
with each change's diff (`FilesChanged`), all by resolved absolute path, and
`shell` how the command ended (`ExitStatus`). The file tools take an optional
`base` directory for relative paths and can be `confine`d to it, which refuses
any path that resolves outside it; a confined `shell` refuses only a `cwd`
outside its directory, not what the command does. Every tool's arguments
are plain data with public fields. The crate also has what they are built from:
`context` (the `AGENTS.md`/`CLAUDE.md` instruction files from the working
directory up), `fs` (`write_atomic`, capped reads, `resolve`, `file_change`), `process`
(process groups and the `Launcher`) and `blocking`.

The tools run on the file system and processes, so the crate is native only.
The [`rig-harness`](../rig-harness) coding agent registers them.
