# rig-mcp-client

`rig_mcp_client::McpPlugin` gives the rig agent the tools of MCP servers.
Each server's tools are offered to the model as `<server>__<tool>`, with the
server's descriptions, schemas and annotations. They stay in sync with the
server's list and go away when the server does. A server that is lost is
connected again, waiting 1 s, then twice as long each time, up to a minute.

## Turning it on

The plugin list a new `RIG_HOME` gets has it, turned off. Turn it on with:

```sh
rig plugin enable rig_mcp_client::McpPlugin
```

or in `/plugins`. A `RIG_HOME` made before this plugin shipped has no entry
for it: `rig plugin add rig_mcp_client::McpPlugin` adds one, on. The next
start, or `/reload`, builds it in. With it on and
no `mcp.json`, it reads the two missing `mcp.json` files and its approvals at
start, and does nothing else.

## Servers: `mcp.json`

Servers are named in the `mcpServers` format other agents use, in two files:

- `$RIG_HOME/mcp.json` (default `~/.rig/mcp.json`): yours, in every project.
- `<project>/.rig/mcp.json`: the project's, unless the project is on another
  host, whose `.rig` is not on this machine, or its `.rig` is
  `RIG_HOME` itself, as in your home directory, or its path is not UTF-8
  (approvals could not tell it from another such path). Its servers start only once you
  approve them (below). Approved, one replaces a server of yours of the same
  name.

```json
{
  "mcpServers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" }
    }
  }
}
```

| Field | Meaning |
| --- | --- |
| `command`, `args` | A stdio server: the program rig starts, speaking MCP on its stdin and stdout, in a process group of its own that is killed when the server is stopped, removed or replaced, or rig exits (on Linux and macOS). Its stderr is dropped. |
| `env` | Variables added to its environment. The server gets rig's environment without rig's secrets (API keys); name a key here to give it one. |
| `cwd` | Its working directory, relative to the project. |
| `url`, `headers` | A streamable HTTP server, and headers sent with each request. |
| `type` | Optional: `"stdio"`, or `"http"` (also `"streamable-http"`, `"streamableHttp"`). Without it, `url` means HTTP and `command` stdio. |
| `disabled` | `true` keeps it listed and stopped. |

`${VAR}` in `command`, `args`, `env`, `cwd`, `url` and `headers` is the
environment variable `VAR`, and `${VAR:-default}` is `default` when `VAR` is
unset or empty. A server that uses an unset `${VAR}` with no default does not
start, and says which field.

A server's name is letters, digits, `_` and `-`, starting with a letter or
`_` (as Gemini requires of tool names), without `__` and not ending in `_`, so
two servers' tool names never collide. A tool whose full name is
not 1 to 64 of those characters, as model providers require, or that another
tool's name already has, is left out with a notice.

Anything else is refused with a message naming it, and only that server
fails: a field of the wrong type (`` `env.PORT`: invalid type: integer `8080`,
expected a string ``), other fields such as `timeout`, `oauth` or `alwaysAllow`, the
`"sse"` transport (rig speaks stdio and streamable HTTP), or a field of the
other transport. OAuth sign-in for remote servers is not supported; a server
that takes a token in a header works. The files are JSON: one with comments
is refused.

## Project servers need your approval

A cloned repository's `.rig/mcp.json` never starts a program or contacts a
server before you approve it. At start, rig lists the project's servers you
have not answered about, each whole as it is written (its `env`, `cwd` and
`headers` too, no `${VAR}` expanded), and starts none of them:

```
/home/me/repo/.rig/mcp.json names MCP servers you have not approved, so they did not start:
  files: {"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}
/mcp approve <name> starts one and remembers it; /mcp deny <name> stops asking.
```

`/mcp approve <name>` approves the server as the file writes it now and
starts it; `/mcp deny <name>` stops it and keeps it stopped. A project server
that has the name of one of yours is listed too, and yours comes first until you
approve the project's. The answers are
yours, kept in `$RIG_HOME/projects/<project>/mcp/approvals.json`, never in
the repository, and are about this project's path only. When the repository
changes a server (another command, argument, variable or URL), it is not
approved any more and is listed again.

A headless run follows the same rule and asks nothing: `rig -p` never starts
an unapproved server. Approve one from a script with
`rig -p '/mcp approve files'`, or name the server in your own
`$RIG_HOME/mcp.json` instead.

## Servers a client names

A front for a client of rig's, such as `rig acp` for an editor
([rig-acp](https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-acp)), hands rig the MCP servers the client names for the
session, in code:

```rust
use rig_harness::prelude::*;
use rig_mcp_client::{AddMcpServer, McpCorePlugin, ServerSpec};

fn plugin(app: &mut App) {
    if !app.is_plugin_added::<McpCorePlugin>() {
        app.add_plugins(McpCorePlugin);
    }
    // Before `Startup`, when the files' servers start.
    app.add_systems(PreStartup, |mut commands: Commands| {
        commands.trigger(AddMcpServer {
            name: "files".into(),
            spec: Ok(ServerSpec::Http {
                url: "https://example.com/mcp".into(),
                headers: Default::default(),
            }),
        });
    });
}
```

`McpCorePlugin` connects such servers and offers `/mcp`, and reads no
`mcp.json`: the user's servers join only when `McpPlugin`, which adds it,
is on. A client's server is never saved, needs no approval, takes its values
as they are (no `${VAR}`), and shows in `/mcp` as `files (client)`. It takes
its name for the session: a server of either file of that name does not
start, even when the client's fails, and a notice and `/mcp` say so
(`files (user): does not run; the client's of that name does`). A second
server of one name from the client is refused with a notice. `spec` is
`Err(why)` for a declaration that names no way rig can reach (an SSE server):
it is listed invalid, and still holds its name. A run that keeps no session,
such as `rig acp`'s own, starts no server of the files.

## `/mcp`

- `/mcp`: each server, its file (`user` or `project`) or `client`, and its state:
  connected with how many tools, connecting, retrying with the last error,
  not approved, denied, disabled, or invalid with why.
- `/mcp add [--user|--project] <name> <command> [args…]` and
  `/mcp add [--user|--project] <name> <http(s)://url>`: saves the server in
  your `mcp.json` (`--user`, the default), or with `--project` in the project's (approved, since you added
  it), and starts it. The words are split at spaces, with no quoting: for an
  argument with a space in it, or `env`, `cwd` and `headers`, edit the file.
- `/mcp remove [--user|--project] <name>`: takes the server out of that file,
  else out of the file `/mcp` shows for it. The other file's server of that
  name, if there is one, takes its place: it starts if it may, and the reply
  says when it waits for `/mcp approve`.
- `/mcp approve <name>`, `/mcp deny <name>`: above.

A change saves at once and lasts across restarts. It changes only the bytes
of the server it adds or removes: the other keys, their order, the indent,
the line ends and the rest of the layout stay as they were. A file whose
`mcpServers` is on one line, or has none yet, is written whole instead, with
its own indent and line ends; the first server added to a file without
`mcpServers` leaves an empty `"mcpServers": {}` once removed. The file keeps
its permissions and group, and your own `mcp.json`, when it is a link, is changed where
it points. A project's `.rig` that leads out of the project, such as a link
to your own `RIG_HOME`, is never read or written. Each change takes the lock `$RIG_HOME/mcp.lock`, so two rig
processes change a file in turn.

## Settings

The `[mcp]` table of `$RIG_HOME/settings.toml`; a project cannot set it.

```toml
[mcp]
# Seconds before a lost server is connected again; doubles up to a minute.
first_retry_secs = 1
# Seconds an attempt waits for a server to start, answer and list its tools.
connect_timeout_secs = 30
# Servers of $RIG_HOME/mcp.json trusted to say which of their tools only read.
trust_read_only = ["files"]
```

A tool whose `readOnlyHint` says it only reads, of a server in
`trust_read_only`, only reads for rig: its calls run beside others, and again
after a restart in a turn a plugin carries on (a restart stops any other
unfinished turn). Another server's word is a hint: such a call still runs
beside the reply's other readers, but a restart never runs it again. A
project's server is never trusted this way.

## Headless runs

`rig -p` and `rig --json` wait, before the prompt and before any turn, for
each server to connect or fail, each at most `connect_timeout_secs`. A
`/command` prompt waits too, as one may start a turn (a skill's does), so
`rig -p '/mcp approve files'` may wait for a server that does not answer. The
terminal view does not wait: a turn it resumes at once may call a server's
tool before the server is in, and the call is refused as an unknown tool. So
`--tools` can name an MCP tool:

```sh
rig -p --tools files__read_text_file 'Summarize README.md'
```

When a server fails, the run says why on stderr and refuses the name, with
exit code 2:

```
rig: MCP server `files` could not connect: could not start `npx`: No such file or directory (os error 2).
rig: --tools: unknown tool `files__read_text_file`; ...
```

With `--json`, each server's outcome is a `notice` line before the run's
first turn: `"MCP server `files` connected: 14 tools."` (info) or
`"MCP server `files` could not connect: <why>."` (warn). Plain `-p` prints
only the failures, but every notice for a `/command` prompt.

A server that says it offers no tools (only resources or prompts, say)
connects with none. Resources and prompts of MCP servers are not offered.
