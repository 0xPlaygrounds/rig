# rig-harness

The harness of the rig coding agent: a terminal coding agent that is a
[Bevy](https://bevy.org) app made of plugins, on the [`rig-ecs`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-ecs)
kernel and [Rig](https://github.com/0xPlaygrounds/rig)'s rig-core. The `rig`
launcher (`rig-cli`) generates a small Cargo project for it with the plugins
`RIG_HOME/settings.toml` turns on, builds it and runs it; `/reload` rebuilds it
when the plugins or a plugin crate changed and restarts in the same session. The [agent guide](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md)
describes using it.

## A kernel, a harness and plugins

- **The kernel** is rig-ecs: agents, turns and calls as entities, message
  delivery, tools as entities, the session log and its restore.
- **The harness** is this crate: what the binary needs to run and be
  relaunched, and the services every plugin shares. `HeadlessPlugins` is
  Bevy's `MinimalPlugins` with the log, a clean exit on signals and a loop that
  sleeps until there is work. `RigHarnessPlugins` adds the session and its
  store (`StoreOpener`, which refuses, unchanged, a session of another
  format), the kernel, the launcher protocol
  (`Invoked`, `AppStart`, `Secrets`, `Exit`), slash commands, fronts, and the
  shared services every agent uses: settings, the workspace and child
  processes, file watches, plugin data, questions and pickers, and transcript
  notes. The generated `main.rs` ends with `rig_harness::run`.
- **Everything else is a plugin** in a crate of the repository's
  [`plugins/`](https://github.com/0xPlaygrounds/rig/tree/main/plugins) folder, turned on or off in `settings.toml` and built only
  on public API, exactly as a third-party plugin is: the compiler holds them
  to it. Most are in [`rig-basics`](https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-basics), one module
  per group, with the services only some plugins use: approvals, the preview
  and the checkpoint of tool calls, and the session picker. The launcher's
  generated `main.rs` adds the plugins the settings turn on as one
  `PluginGroup`, each wrapped in `Loaded`, which records it as an entity
  (`PluginSource`) and marks what it adds (`ProvidedBy`). Any of them can be
  turned off.

rig's built-in plugins, in the order the agent adds them (the source of truth
is `built_in_plugins!` in `rig-harness-protocol`'s `config` module). On by
default:

| Crate | Plugin | What it adds |
|---|---|---|
| `rig-basics` | `prompt::DefaultPromptPlugin` | rig's own system prompt for an agent without one: one fixed text, the same on every request |
| `rig-basics` | `agents_md::AgentsMdPlugin` | the `AGENTS.md` files from the working directory up to the project's root, and `RIG_HOME/AGENTS.md`, in the system prompt |
| `rig-basics` | `skills::SkillsPlugin` | the skills (`SKILL.md`) of the project, `RIG_HOME/skills` and `~/.agents/skills`, listed in the system prompt; `/skill:<name>` |
| `rig-basics` | `models::ModelsPlugin` | `/model`, `/effort` and the model picker |
| `rig-basics` | `login::LoginPlugin` | `/login` and `/logout`: the ChatGPT plan, and providers' API keys |
| `rig-basics` | `models::defaults::DefaultsPlugin` | a new session starts on `[model]` of `settings.toml`, which the model and reasoning last picked with `/model` and `/effort` rewrite once it answered |
| `rig-basics` | `sessions::SessionsPlugin` | `/new`, `/resume`, `/name` |
| `rig-basics` | `sessions::agents::AgentsPlugin` | `/fork`, `/agents` and rewind (Alt+Up to an earlier message, then Enter): the agents of this session |
| `rig-basics` | `compaction::CompactionPlugin` | summarizing a conversation near the context window; `/compact` |
| `rig-basics` | `telemetry::usage::UsagePlugin` | what model calls cost, per agent and per tree; `/usage` |
| `rig-traffic` | `TrafficPlugin` | with `--traffic-log` or `[traffic] log = true`, provider HTTP exchanges in the session's `traffic.jsonl` |
| `rig-basics` | `activity::ActivityPlugin` | what each agent is doing, for views |
| `rig-basics` | `coding_tools::ReadTool`, `coding_tools::EditTool`, `coding_tools::WriteTool`, `coding_tools::ShellTool` | the coding tools of [`rig-tools`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-tools), one plugin each, acting in the app's workspace; the library's `search`, `list`, `delete` and `move` are left to `shell` |
| `rig-basics` | `coding_tools::AttachPlugin` | the files the user names as `@path`, from the agent's directory, go with the message |
| `rig-basics` | `coding_tools::remote::RemoteHostsPlugin` | a path written `HOST:PATH` is that host's for every tool, through the transport `[remote.hosts]` in `settings.toml` gives it (such as `ssh` or `koh exec`); nothing without hosts |
| `rig-basics` | `reload::ReloadPlugin` | `/reload`, which rebuilds the agent when a plugin crate changed and restarts it in the session with the plugins `settings.toml` turns on |
| `rig-basics` | `reload::plugins::PluginsPlugin` | `/plugins`: every plugin, on and off, in a picker that turns them on and off in `settings.toml`, then rebuilds and restarts at once |
| `rig-basics` | `BasicCommandsPlugin` | `/help`, `/quit` |
| `rig-basics` | `print::PrintPlugin` | `--print` and `--json`: the fronts of a run with `-p`, with stdin piped in, or driven by JSON lines (`rig help-agents`) |
| `rig-tui` | `TuiPlugin` | the terminal view: the front of a run on an interactive terminal, with the status line and pickers other plugins add to |

Off by default, each turned on by `/plugins`, `rig plugin enable <type>` or
`[plugins] enable` in `settings.toml`:

| Crate | Plugin | What it adds |
|---|---|---|
| `rig-basics` | `approval::ApprovalPlugin` | one question for every effect a plugin wants approved; nothing asks by default, and `SubagentsPlugin` adds it |
| `rig-basics` | `preview::PreviewPlugin` | what a tool call would change on disk, before it runs, for gates that ask |
| `rig-basics` | `checkpoint::CheckpointPlugin` | what each tool call changed on disk, from snapshots before and after |
| `rig-basics` | `redact::RedactPlugin` | API keys, tokens and private keys become placeholders (`‹SECRET_1›`) where they come in (the model's reply as it streams, tool output and results, messages), so the model, the logs, every front, subagents and a resumed session never hold them; a tool call naming one runs with the real value, kept in memory for the run. Without it nothing is redacted |
| `rig-basics` | `telemetry::effect_log::EffectLogPlugin` | every model and tool call in the session's `effects.jsonl` |
| `rig-basics` | `helpers::HelpersPlugin` | the `helper` tool: tasks for `rig -p --json` child processes, one, side by side or in a chain, as `[helpers]` sets; what they spend is the caller's |
| `rig-basics` | `budget::BudgetPlugin` | `[budget] usd` and `tokens`, and `/budget`: at the limit the running turns stop, helpers included, and new model calls are refused |
| `rig-basics` | `subagents::SubagentsPlugin` | the `task`, `message` and `wait` tools, and the agents at work in the status line |
| `rig-basics` | `telemetry::diagnostics::DiagnosticsPlugin` | the process's warnings and errors, Bevy's included, as the `Diagnostics` resource |
| `rig-inspect` | `InspectPlugin` | the `inspect` tool, with which the agent reads its own running world |
| `rig-steel` | `SteelPlugin` | the `run_steel` tool, whose Steel program orchestrates agents and tools, within `[steel]` |
| `rig-mcp-client` | `McpPlugin` | the tools of the MCP servers of `RIG_HOME/mcp.json` and the approved ones of the project's `.rig/mcp.json`, as `<server>__<tool>`; `/mcp` |
| `rig-acp` | `AcpPlugin` | `rig acp`: the Agent Client Protocol on stdin and stdout, for editors such as Zed |
| `rig-claude-cloak` | `ClaudeCloakPlugin` | `claude-cloak/*` models signed in with a Claude subscription (OAuth), and `/claude-cloak login` and `logout` |

The agent builds rig's own crates from crates.io at the launcher's version, or
from the rig checkout it is built from (`RIG_SOURCE`); a crate that is not
rig's is a `[[plugins.extra]]` table of `RIG_HOME/settings.toml`. A plugin crate that adds to the
terminal view (status line items, pickers, transcript render systems) does so
behind a default `tui` cargo feature, so it also builds without `rig-tui`.

The optional [`rig-inspect`](https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-inspect) plugin adds the
`inspect` tool, with which the agent reads its own Bevy world, read-only.
[`rig-steel`](https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-steel) adds code mode.
[`rig-claude-cloak`](https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-claude-cloak) signs in with a Claude subscription instead of an API key.

[`PLUGINS.md`](https://github.com/0xPlaygrounds/rig/blob/main/crates/rig-harness/PLUGINS.md) shows how to write a plugin: an index from task to
section, then one example per extension point.
[`keep_last_compaction`](https://github.com/0xPlaygrounds/rig/blob/main/corpus/ct/src/keep_last_compaction.rs) replaces
a default plugin, the compaction, on public API alone. A plugin's tests run a
small app frame by frame, and `stage_conflicts` checks that its stage systems
are ordered against those of the plugins it is used with
([testing a plugin](https://github.com/0xPlaygrounds/rig/blob/main/crates/rig-harness/PLUGINS.md#testing-a-plugin)).

## Install

```bash
git clone https://github.com/0xPlaygrounds/rig && cd rig
cargo install --path crates/rig-cli
rig
```

It needs Rust 1.97.1 or newer and runs on Linux and macOS.
