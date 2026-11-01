# rig-acp

`rig_acp::AcpPlugin` adds `rig acp`, which serves the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP) on stdin and
stdout. An editor that speaks ACP, such as Zed or a Neovim ACP plugin, then
uses rig: its models with your sign-in, its tools and plugins, its sessions
and its slash commands.

## Turning it on

The plugin list a new `RIG_HOME` gets has it, turned off. Turn it on, and
build:

```sh
rig plugin enable rig_acp::AcpPlugin
rig build
```

A `RIG_HOME` made before this plugin shipped has no entry for it:
`rig plugin add rig_acp::AcpPlugin` adds one, on. Sign in with `rig` first
(`/login` in the terminal view): the editor gets the models you can reach.
`rig acp` builds the agent first when it is out of date, as `rig` does, which
can take minutes the first time; build it beforehand so the editor does not
wait.

## Zed

In Zed's `settings.json`:

```json
{
  "agent_servers": {
    "rig": {
      "type": "custom",
      "command": "/home/me/.cargo/bin/rig",
      "args": ["acp"],
      "env": {}
    }
  }
}
```

`command` is the absolute path of `rig` (`which rig`). Add
`"RIG_HOME": "/path/to/home"` to `env` to use another rig home. Then open the
agent panel and start a thread with rig.

## What it does

Each session runs in a rig process of its own, started in the session's
directory, since rig runs one session per process; `rig acp` routes each
session's messages to its process. Every session of one `rig acp` runs the
build `rig acp` started with: rebuild, then restart the agent in the editor,
to use a new one.

- **Protocol:** ACP v1. A client that offers v2 gets v1, which is the
  fallback ACP defines.
- **Sessions:** new, load (the conversation is shown again, from rig's own
  session, without the commands' output and notices), resume, list, close, cancel. Sessions are rig's: `rig -r <id>`
  opens one in the terminal, and an ACP session is not open in two places at
  once. A session belongs to its directory: loading it from another one is
  refused.
- **Prompts:** text, links to resources, embedded text resources, images
  (for models that take them; rig tells you when one does not), and embedded
  images.
- **Options:** the model, from the models you can reach, each named with
  its provider (one model can come from several), and its reasoning level.
  `/model` and `/effort` change them too, for that session only: a new
  session starts on the terminal's saved default, and nothing chosen in an
  editor changes it. `rig acp`'s own options, such as `--model`, do not
  reach the sessions: the editor's options choose.
- **Tool calls:** each of the main agent's calls, as it is asked for, runs,
  and ends, with its arguments, output and the files it names.
- **Questions:** a tool call a plugin wants approved asks for permission
  (allow or deny, once). Another question is a form, when the editor shows
  forms; else it is closed with a note the model reads, so the turn goes on.
  A list to pick from, such as `/model` with no name, is closed with a note
  that names the commands to type instead; a secret to type, such as an API
  key for `/login`, is typed in rig in a terminal, which hides it, and a
  session opened after it has it.
- **Slash commands:** the editor offers rig's commands, but `/new`,
  `/resume`, `/reload`, `/plugins` and `/quit`, which a session from an
  editor does not need (rig refuses them there; the editor closes the
  session instead of `/quit`), and `/fork` and `/agents`, which switch the
  session to another agent; typed anyway, they work, and the editor
  follows the top-level agent of the one shown. With rig-acp on, `/mcp` is in every rig run,
  the terminal view's too, as rig-acp brings rig-mcp-client's core: it lists
  an editor's servers, and the files' ones only with `McpPlugin` on.
- **MCP servers:** those the editor names for a session (stdio, and
  streamable HTTP with its headers) connect through rig-mcp-client for that
  session only: never saved, no approval, values taken as they are. The
  session opens once each one connected or failed (at most
  `[mcp] connect_timeout_secs`); one that fails is reported and fails
  nothing else. Your own `mcp.json` servers join only when
  `rig_mcp_client::McpPlugin` is on. An editor's server takes its name over
  yours and the project's for that session, which `/mcp` and a notice show.
  SSE servers are not supported.
- **Notices:** what rig tells you (an MCP server that failed, a command's
  output, a message a failed turn took out of the conversation, with its
  text to send again) goes into the reply, or as a notice to an editor that
  shows them.

Not supported: the editor's file system and terminal (rig runs its own
tools), additional directories, session modes, audio, and ACP v2.

## Processes

stdout carries only ACP; logs go to stderr. A line a session's process
writes on stdout that is not ACP, such as a plugin's stray print, goes to
stderr instead. `rig acp` ends when the editor closes its connection, or on
SIGTERM, SIGINT or SIGHUP: it opens no more sessions and closes each one,
which stops its turn and its tools, answers its open prompt `cancelled`
after each cut tool call's closing update, and answers every other open
request; then it kills what a session that does not end in five seconds
left. A line of more than 64 MiB, or one that is not UTF-8, either way,
ends the input it came on, as a closed pipe does. While the editor is
behind reading, the sessions' work waits for it.
A session's process that dies (a crash, or `kill -9`) fails its open prompt;
its tools' processes and MCP servers are killed with it on Linux and macOS,
as the launcher kills what a crashed agent left (on Windows only the
processes that end with their pipes go), and the editor can load the
session again. Closing a session cancels its prompt first, which then
answers `cancelled`.
