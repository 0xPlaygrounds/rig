# rig-tools

Coding tools for [Rig](https://github.com/0xPlaygrounds/rig) agents, as
`PortableTool`s:

| Tool     | What it does                                                        |
|----------|---------------------------------------------------------------------|
| `read`   | A text file as numbered lines, with an offset and limit             |
| `write`  | A whole file, written atomically                                    |
| `edit`   | Several exact (or whitespace- and quote-tolerant) replacements, all or nothing, answered with a diff |
| `search` | A regex over the tree as git ignores it                             |
| `shell`  | `sh -c` in its own process group, with a timeout and a capped output tail |

Each tool carries its usage `RULES` for the system prompt, such as "use `read`,
not `cat`". The crate also has what they are built from: `context` (the
`AGENTS.md`/`CLAUDE.md` instruction files from the working directory up), `fs`
(`write_atomic`, capped reads), `process` (process groups) and `blocking`.

The tools run on the file system and processes, so the crate is native only.
The [`rig-harness`](../rig-harness) coding agent registers them.
