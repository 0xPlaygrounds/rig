# rig-bridge

Other coding agents' programs as model providers (`BridgePlugin`), an optional
plugin of the
[rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness),
off by default (`rig plugin enable rig_bridge::BridgePlugin` turns it on). Each
installed program's models are listed under a vendor of its own and answer
with the program's own sign-in, so a subscription you use in that program
works in rig without an API key:

| Vendor | Program | Driven over | Models |
| --- | --- | --- | --- |
| `claude-code/` | Claude Code (`claude`) | `claude --print` with streamed JSON | Anthropic's, by API id |
| `grok-build/` | Grok Build (`grok`) | Agent Client Protocol, `grok agent stdio` | what `grok models` lists |
| `antigravity/` | Antigravity (`agy`) | Agent Client Protocol, `agy acp` | what `agy models` lists |
| `codex/` | Codex (`codex`) | its app server, `codex app-server` | what `codex debug models` lists |
| `muse/` | Meta AI Muse (`muse`) | MSP session host, `muse serve` | what `muse exec --list-models` lists |
| `copilot-cli/` | GitHub Copilot CLI (`copilot`) | Agent Client Protocol, `copilot --acp` | what Copilot's `session/new` advertises |

```bash
rig plugin enable rig_bridge::BridgePlugin
rig --model claude-code/claude-opus-5-5
```

Sign in with the program itself first (`claude`, `grok login`, `agy`,
`codex login`, `muse login`, `copilot login`). The program answers with that
sign-in: an API key in rig's environment (`ANTHROPIC_API_KEY`, `XAI_API_KEY`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `META_API_KEY` and the like) is not passed
on to it. Linux and macOS only. A program that is not installed leaves its
models out of `/model`.

## What the program is used for

Inference, and nothing else. Rig keeps the conversation, the system prompt and
the tools; the program is started with everything of its own turned off:

- **Claude Code**: no built-in tools (`--tools ""`), no settings files, hooks,
  skills, slash commands, `CLAUDE.md` or memory, and no MCP servers but rig's.
  Rig's system prompt replaces Claude Code's.
- **Grok Build**: an empty home directory of its own, so none of your (or
  another agent's) settings, rules, skills, hooks, plugins or MCP servers, with
  memory, subagents, web search and the rest switched off, and rig's system
  prompt in place of Grok's. Grok offers a model MCP tools only through its
  `search_tool` and `use_tool`, so the model sees those two and calls rig's
  tools through them as `rig__<tool>`.
- **Antigravity**: a settings directory of its own holding one agent: rig's
  system prompt, none of Antigravity's tools or customizations, and rig's
  tools as `mcp_rig_<tool>`. Antigravity adds `manage_task`, `list_resources`
  and `read_resource`, which cannot be turned off: rig refuses the first and
  answers the others with nothing.

- **Codex**: a home directory of its own, so none of your settings,
  `AGENTS.md`, hooks or MCP servers, with its shell, patching, web search,
  apps, plugins, subagents, memories and the rest switched off, and rig's
  system prompt as its base instructions. Codex adds `list_mcp_resources`,
  `list_mcp_resource_templates` and `read_mcp_resource`, which cannot be
  turned off and find nothing. The model sees rig's tools as
  `mcp__rig.<tool>`.
- **Meta AI Muse**: private XDG directories (sharing only `auth.json`),
  `--preset empty`, and shell/write/user-input tools disabled. Rig's system prompt is passed as `config.developerPrompt` and
  Rig's MCP server as `config.mcpServers.rig`, so the model sees only
  `mcp__rig.<tool>`.
- **GitHub Copilot CLI**: a private `COPILOT_HOME` holding one custom agent
  (`rig.agent.md`) whose body carries rig's system prompt, `--available-tools
  rig` disabling all built-in tools while keeping Rig's MCP server
  (`rig-<tool>`), and built-in MCPs, custom instructions, auto-update,
  ask-user and telemetry switched off.

Each program runs in a private, disk-backed directory of its conversation's
own under `RIG_HOME/bridge/`, never in your project or RAM-backed `/tmp`, so it
finds no project files to read on its own. After rig's system prompt each model is
told, in a sentence or two, how the program prefixes rig's tool names and,
for programs that report a scratch directory, which directory the project is.

What stays that rig cannot remove: the vendor's own system layer in front of
rig's prompt on Codex; on Claude Code, a sentence in front of rig's prompt
that says what it is, a section after it with the working directory, platform,
model and date and a few rules of Claude Code's own (make independent tool
calls together, treat downloaded files as untrusted), and a note in front of
your first message that holds the signed-in account's email address; on Grok,
the list of skills bundled with the program and a rule of Grok's own on
phrasing (and Grok's own preamble when switching away from Grok's default
model via `session/set_model`); and Codex and Antigravity call tools one at a
time, never several at once.

Rig's tools reach the program as one MCP server the plugin serves on loopback,
named by a bearer token only that process is given. When the model calls a
tool, rig runs it as it runs any model's tool call: approvals, the transcript,
sessions and replay all see it, and the program waits for the result.

## How a conversation maps to a program

One process serves one conversation and stays up between your messages, so the
program's prompt cache is used. Rig remains the owner of the history.

### Esc

Esc stops the program's turn, not the program. The bridge asks the program to
end the turn its own way, and the process waits for what you say next with
everything it had, the part of the reply it was stopped in included:

| Program | How its turn is stopped |
| --- | --- |
| Claude Code | an `interrupt` control request on its input |
| Codex | `turn/interrupt` |
| Meta AI Muse | `turn/interrupt` |
| Grok Build, Antigravity, GitHub Copilot CLI | `session/cancel` |

A program that does not end the turn within ten seconds, or whose turn fails
instead, is ended. So is Claude Code when it is stopped while it holds a
message of yours it has not read (one sent while a tool ran): whether it
would drop the message or answer it later is not known, so the conversation
goes on in a new process, written out, with the message in it. Esc while a
tool runs stops the tool: its result says it was stopped, and the turn goes
on.

### A new process

A new process is needed when none that is running can go on: rig was started
again (`-r`), the process was ended as one idle process too many (`sessions`
below) or is gone, or it cannot take a changed set of tools. Where it can be,
it is started on the session the program itself saved, the program's own way,
so the model has the conversation as it had it, and is sent only what you
said since:

| Program | Started on its saved session with |
| --- | --- |
| Claude Code | `--resume` and the session's id |
| Codex | `thread/resume` |
| Meta AI Muse | `session/resume` |
| Grok Build, Antigravity | `session/resume` |
| GitHub Copilot CLI | `session/load` |

This is done only where rig's history is what the program had when it last
ended a turn or was stopped, under the same model and system prompt. A saved
session is found by that history, whichever conversation left it. In every
other case the new process is given the conversation so far written out as
text, in at most 160 KB: old tool output is cut first, then the oldest turns
are left out. Those cases are `/compact`, a history that was edited, another
model or system prompt, a process that ended in the middle of a turn (a tool
call open, or a message of yours not yet given to it), a saved session the
program does not open, and on Antigravity another set of tools: its saved
session keeps the tools it was made with. In that text a tool's output is
under a heading of its own, apart from what you said.

### What is kept on disk

Each conversation has a directory `RIG_HOME/bridge/rig-bridge-<pid>-<id>/`:
the program's working directory and private home, where the program saves the
conversation in its own format. Beside it the bridge writes `rig.json`: the
program, the model, the program's name for the saved session, and digests of
the history it stands at; of the conversation's text it holds only the part
of a reply that was said before you stopped it. One session works in a
directory at a time, and holds `rig.lock` there locked while it does. Claude
Code is the exception: it writes its session file in its own home, as
`~/.claude/projects/<the directory's path, with - for />/<session id>.jsonl`,
so apart from the sessions of your own projects.

**Nothing here is ever removed by rig.** A directory is kept from the first
turn its program completes, for every conversation, each `rig -p` run and each
subagent included, and it holds your conversation in plain text. One measured
after a three-message conversation was about 15 MB for Grok Build, 11 MB for
Muse, 4 MB each for Codex and Antigravity, 1 MB for Copilot CLI and a few KB
for Claude Code; most of it is files the program unpacks into a new home. Where
the program's sign-in is a file, the directory holds a link to yours, not a
copy.

To free the space, delete directories under `RIG_HOME/bridge/` (for Claude
Code also the folders in `~/.claude/projects/` whose names contain
`rig-bridge-`). `/bridge status` prints the path. Deleting one costs only
this: its conversation, if it goes on in a new process, goes on from text. Do
not delete the directory of a conversation whose process is running.

## A message sent while the model works

A message you send while a tool runs joins the turn: rig gives it to the model
with the tool's result. The bridge hands it to the program as a user message,
the program's own way, before it answers the tool call, so the model reads it
as yours and not as something the tool printed:

| Program | How it takes the message |
| --- | --- |
| Claude Code | a line of input for the running turn (`priority: "next"`), acknowledged by its id |
| Codex | `turn/steer`, naming the running turn |
| Meta AI Muse | `turn/steer`, naming the running turn |
| Grok Build | `_x.ai/interject` (text only) |
| Antigravity 1.5.0 and later | the next `session/prompt`, which ends the running one and carries its work on |
| GitHub Copilot CLI | no way of its own |

A program with no way of its own, one that refuses the message or does not
say within ten seconds that it took it, an older Antigravity, and Grok for a message with an
image get it as their next prompt instead, as soon as the running turn ends,
and rig's reply goes on into the answer. The model then reads it once it has
stopped calling tools, not at its next step. If a program takes the message
after the ten seconds are up, the model reads it twice.

When rig's tools change mid-conversation, a running program is told so over
MCP and lists them again, and the conversation goes on in the same process.
Claude Code does this. A program that keeps no connection open to be told on
(Grok), or that fixes its tools as it starts (Antigravity), gets a new
process instead: Grok's on its saved session, Antigravity's on the
conversation written out, as its saved session keeps the tools it had.

`[bridge]` in `RIG_HOME/settings.toml`:

```toml
[bridge]
claude = "claude"      # each program: a name on PATH or a path
grok = "grok"
antigravity = "agy"
codex = "codex"
muse = "muse"
copilot = "copilot"
sessions = 8           # how many idle processes are kept; the oldest ends first
```

## Commands

- `/bridge` or `/bridge status`: installed harnesses, warm sessions, where
  saved sessions are kept, session token totals, and cache hit rates.
- `/bridge usage`: queries plan/subscription quotas in parallel in the
  background across all 6 harnesses (Claude Code `get_usage`, Codex
  `account/rateLimits/read`, Muse `muse usage --json`, Grok billing endpoint
  with CLI fallback, Antigravity `fetchAvailableModels` quota, and Copilot
  via `gh api /copilot_internal/user`) alongside Rig's session token totals.
- `/bridge reset`: stops all warm idle harness processes without restarting
  rig. What they saved stays: a conversation whose turn had ended goes on
  from it, and one stopped in a tool call from the conversation written out.

## Limits

- Sampling temperature, an output-token limit, a forced tool choice, an output
  schema and a reasoning token budget are not passed on: these programs take a
  model and an effort. Rig reports them as unsupported options.
- Antigravity counts tokens for a whole turn, so a turn with tool calls is
  counted on its last reply. Antigravity's effort is part of its model id.
- Codex counts a model call after the tools it asked for have run, so a
  call's tokens can show on the reply after it.
- Images in your message are sent; audio, video and documents are not.
- No cost is reported: the programs bill your subscription.
- Starting rig discovers models across installed harnesses in parallel
  (bounded by one 5-second timeout).
- When a new process cannot be started on a saved session, rig budgets the
  history it writes out so large prior tool outputs are elided before old
  turns are dropped and the latest user turn is always preserved intact.
