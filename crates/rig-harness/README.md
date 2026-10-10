# rig-harness

The rig coding agent: a terminal coding agent that is a [Bevy](https://bevy.org)
app made of plugins, on the [`rig-ecs`](../rig-ecs) agent runtime and
[Rig](https://github.com/0xPlaygrounds/rig)'s rig-core. The `rig` launcher
(`cargo install rig`) generates a small Cargo project for it from
`RIG_HOME/plugins.toml`, builds it and runs it; `/reload` rebuilds it with
changed plugins and restarts in the same session. The
[root README](../../README.md#the-rig-coding-agent) describes using it.

## A kernel and plugins

- **The kernel** is rig-ecs: agents, turns and calls as entities, message
  delivery, tools and slash commands as entities, the session log and its
  restore.
- **The core** is what the binary needs to run and be relaunched.
  `HeadlessPlugins` is Bevy's `MinimalPlugins` with the log, a clean exit on
  signals and a loop that sleeps until there is work; `RigHarnessPlugins`
  adds the session directory, the run mode and what fronts share (`front`),
  the kernel, the launcher protocol and `/reload`.
- **Everything else is a plugin**, listed in `plugins.toml` and built only
  on public API, exactly as a third-party plugin is. The launcher's
  generated `main.rs` adds each with `load`, which records it as an entity
  (`PluginSource`) and marks what it adds (`ProvidedBy`). Any entry can be
  removed.

The default `plugins.toml`, in order (types under `rig_harness::plugins`):

| Plugin | What it adds |
|---|---|
| `project_context::ProjectContextPlugin` | `AGENTS.md`/`CLAUDE.md` and the environment in the system prompt; `/context` |
| `models::ModelsPlugin` | `/model`, `/effort` and the model picker |
| `login_chatgpt::ChatgptLoginPlugin` | `/login` and `/logout` for the ChatGPT plan's models |
| `defaults::DefaultsPlugin` | a new session starts on the last model and reasoning chosen |
| `sessions::SessionsPlugin` | `/new`, `/resume`, `/name` |
| `compaction::CompactionPlugin` | summarizing a conversation near the context window; `/compact` |
| `usage::UsagePlugin` | what model calls cost; `/usage` |
| `activity::ActivityPlugin` | what each agent is doing, for views |
| `effect_log::EffectLogPlugin` | every model and tool call in the session's `effects.jsonl` |
| `tools::{ReadTool, EditTool, WriteTool, SearchTool, ShellTool}` | the coding tools of [`rig-tools`](../rig-tools), one plugin each |
| `reload_tool::ReloadTool` | the `reload` tool, with which the agent rebuilds itself, and the system prompt's section on what the agent is (its plugins, commands and tools) and how it writes plugins |
| `basics::BasicCommandsPlugin` | `/help`, `/retry`, `/agents`, `/quit` |
| `subagents::SubagentsPlugin` | the `task`, `message` and `wait` tools |
| `diagnostics::DiagnosticsPlugin` | the process's warnings and errors, Bevy's included, as the `Diagnostics` resource |
| `print::PrintPlugin` | `--print`, and the front of a run no other front took |
| `rig_harness::tui::TuiPlugin` | the terminal view (feature `tui`) |

Optional plugins live in crates of their own and are not in the default
list: [`rig-inspect`](../rig-inspect)'s `InspectPlugin` adds the `inspect`
tool, with which the agent reads its own Bevy world through Bevy Remote (in
process, read-only). It makes the build heavier, so it is commented out in
the default `plugins.toml`; uncomment its entry, or run
`rig plugin add rig_inspect::InspectPlugin --crate rig-inspect --version 0.44.0`
(your `rig`'s version). [`rig-steel`](../rig-steel) adds code mode.

[`PLUGINS.md`](PLUGINS.md) shows how to write a plugin, with one example per
extension point.

## Install

```bash
cargo install rig
rig
```

It needs Rust 1.97.1 or newer and runs on Linux and macOS. With the
optional `rig-inspect` plugin enabled, building the agent on Linux also
needs the ALSA development headers (`libasound2-dev` on Debian and Ubuntu,
`alsa-lib-devel` on Fedora, `alsa-lib` on Arch): Bevy Remote pulls in
Bevy's audio crate. The agent plays no sound.
