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
  skills, slash commands, `CLAUDE.md`, memory or saved sessions, and no MCP
  servers but rig's. Rig's system prompt replaces Claude Code's; Claude Code
  still opens it with one sentence of its own that says what it is.
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
  `--no-session-log`, `--preset empty`, and shell/write/user-input tools
  disabled. Rig's system prompt is passed as `config.developerPrompt` and
  Rig's MCP server as `config.mcpServers.rig`, so the model sees only
  `mcp__rig.<tool>`.
- **GitHub Copilot CLI**: a private `COPILOT_HOME` holding one custom agent
  (`rig.agent.md`) whose body carries rig's system prompt, `--available-tools
  rig` disabling all built-in tools while keeping Rig's MCP server
  (`rig-<tool>`), and built-in MCPs, custom instructions, auto-update,
  ask-user and telemetry switched off.

Each program runs in a private, disk-backed empty directory under
`RIG_HOME/run/bridge/`, never in your project or RAM-backed `/tmp`, so it finds
no project files to read on its own. After rig's system prompt each model is
told, in a sentence or two, how the program prefixes rig's tool names and,
for programs that report a scratch directory, which directory the project is.

What stays that rig cannot remove: the vendor's own system layer in front of
rig's prompt on Codex; on Grok, the list of skills bundled with the program
(and Grok's own preamble when switching away from Grok's default model via
`session/set_model`); and Codex and Antigravity call tools one at a time,
never several at once.

Rig's tools reach the program as one MCP server the plugin serves on loopback,
named by a bearer token only that process is given. When the model calls a
tool, rig runs it as it runs any model's tool call: approvals, the transcript,
sessions and replay all see it, and the program waits for the result.

## How a conversation maps to a program

One process serves one conversation and stays up between your messages, so the
program's prompt cache is used. Rig remains the owner of the history: when a
request does not continue what a running process answered (after `/compact`, a
restart with `-r`, a changed system prompt, model or effort, or Esc), rig
starts a new process and gives it the conversation so far as text.

When rig's tools change mid-conversation, a running program is told so over
MCP and lists them again, and the conversation goes on in the same process.
Claude Code does this. A program that keeps no connection open to be told on
(Grok), or that fixes its tools as it starts (Antigravity), gets a new
process instead.

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

- `/bridge` or `/bridge status`: installed harnesses, warm sessions, scratch
  directory path, session token totals, and cache hit rates.
- `/bridge usage`: queries plan/subscription quotas in parallel in the
  background across all 6 harnesses (Claude Code `get_usage`, Codex
  `account/rateLimits/read`, Muse `muse usage --json`, Grok billing endpoint
  with CLI fallback, Antigravity `fetchAvailableModels` quota, and Copilot
  via `gh api /copilot_internal/user`) alongside Rig's session token totals.
- `/bridge reset`: stops all warm idle harness processes and removes their
  scratch directories without restarting rig.

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
- When starting a fresh harness mid-conversation, rig budgets the replayed
  history so large prior tool outputs are elided before old turns are dropped
  and the latest user turn is always preserved intact.
