# rig-tools

Coding tools for [Rig](https://github.com/0xPlaygrounds/rig) agents:

| Tool     | What it does                                                        |
|----------|---------------------------------------------------------------------|
| `read`   | A text file's lines, with an offset and limit                       |
| `write`  | A whole file, written atomically                                    |
| `edit`   | Several exact (or whitespace- and quote-tolerant) replacements, applied in order, all or nothing, answered with the changed lines' numbers |
| `search` | A regex over the tree as git ignores it, hidden files too when asked or when the glob names one, or the names of the files with a match (every file for an empty pattern); `find` returns the hits as data, `render` shows them |
| `list`   | A directory's files as git ignores them, down to a depth, optionally matching a glob, hidden files as for `search` |
| `shell`  | `sh -c` in its own process group, as a `Launcher` starts it, with a timeout (at most `max_timeout`, ten minutes by default), stdout and stderr interleaved line by line without terminal escape codes such as colours, and an optional `OutputLimit` that keeps both ends of a long output and the first and last errors from between |
| `delete`, `move` | A file removed, or moved or renamed |

Tools publish facts beside their text (such as `FilesChanged`), and every tool acts on a
`Workspace`: this machine (`Local`) or another. The pieces the tools are built from, such as
checkpoints, the overlay workspace and process groups, are listed in the crate docs.

`remote` reaches other machines through a transport: a command that runs its last argument,
a shell line, there with the remote user's shell, keeps stdout and stderr apart and ends with
255 when it failed itself, as `ssh HOST` and `koh exec HOST --sh` do. `Remote` is such a
machine's workspace, and `Hosts` wraps one so that a path written `HOST:PATH`, as `scp`
reads it, is that host's and every other path stays the inner workspace's:

```rust
use std::sync::Arc;
use rig_tools::process::Launcher;
use rig_tools::remote::{Hosts, Remote};
use rig_tools::workspace::Local;
use rig_tools::Read;

let koh = ["koh", "exec", "bench", "--sh"];
let ssh = ["ssh", "-o", "BatchMode=yes", "dev@box"];
let hosts = Hosts::new(
    Arc::new(Local),
    [
        Remote::new("bench", koh, Launcher::default())?,
        Remote::new("box", ssh, Launcher::default())?,
    ],
);
// `read` of `bench:~/strata/src/lib.rs` reads that file on bench.
let read = Read { workspace: Arc::new(hosts), ..Read::default() };
```

Each file operation there is one short `sh` script through the transport, at most two at
once per host; a read whose transport failed (exit 255) is sent once more, a write never. An
`edit` of a file read whole, or edited before, is one call that writes only if the file still
holds the text it was made from. A `shell` command there runs as a local transport process:
its timeout stops that process (`SIGTERM`, then `SIGKILL`), and whether the command on the
other machine stops with it is the transport's doing (`ssh` without a terminal leaves it
running until it next writes).

Every tool then takes such paths: `read`, `edit` and `write` a file there, `search` with
`path: "bench:~/strata"`, `shell` with `cwd: "bench:~/strata"`, and the paths they print and
publish keep the `HOST:` form.

The tools run on the file system and processes, so the crate is native only.
The [`rig-harness`](../rig-harness) coding agent registers them.
