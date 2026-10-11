# rig-harness

The core of the rig coding agent: a terminal coding agent that is a
[Bevy](https://bevy.org) app made of plugins, on the [`rig-ecs`](../rig-ecs)
agent runtime and [Rig](https://github.com/0xPlaygrounds/rig)'s rig-core. The
`rig` launcher generates a small Cargo project for it from
`RIG_HOME/plugins.toml`, builds it and runs it; `/reload` rebuilds it with
changed plugins and restarts in the same session. The
[root README](../../README.md#the-rig-coding-agent) describes using it.

## A kernel, a core and plugins

- **The kernel** is rig-ecs: agents, turns and calls as entities, message
  delivery, tools and slash commands as entities, the session log and its
  restore.
- **The core** is this crate: only what the binary needs to run and be
  relaunched. `HeadlessPlugins` is Bevy's `MinimalPlugins` with the log, a
  clean exit on signals and a loop that sleeps until there is work;
  `RigHarnessPlugins` adds the session directory, the kernel and the
  launcher protocol, with how the agent was started (`Invoked`: its
  arguments, and whether stdin is a terminal). How it was started picks
  the front: `rig-tui` runs on an interactive terminal, `rig-print` with
  `-p` or with stdin piped in; a build without `rig-tui` started on a
  terminal says so and exits.
- **Everything else is a plugin** in a crate of the repository's
  [`plugins/`](../../plugins) folder, listed in `plugins.toml` and built only
  on public API, exactly as a third-party plugin is: the compiler holds them
  to it. The launcher's generated `main.rs` adds each with `load`, which
  records it as an entity (`PluginSource`) and marks what it adds
  (`ProvidedBy`). Any entry can be removed.

The default `plugins.toml`, in order:

| Crate | Plugin | What it adds |
|---|---|---|
| `rig-basics` | `ProjectContextPlugin` | `AGENTS.md`/`CLAUDE.md` and the environment in the system prompt; `/context` |
| `rig-models` | `ModelsPlugin` | `/model`, `/effort` and the model picker |
| `rig-login-chatgpt` | `ChatgptLoginPlugin` | `/login` and `/logout` for the ChatGPT plan's models |
| `rig-models` | `DefaultsPlugin` | a new session starts on the last model and reasoning chosen |
| `rig-sessions` | `SessionsPlugin` | `/new`, `/resume`, `/name` |
| `rig-compaction` | `CompactionPlugin` | summarizing a conversation near the context window; `/compact` |
| `rig-telemetry` | `UsagePlugin` | what model calls cost; `/usage` |
| `rig-activity` | `ActivityPlugin` | what each agent is doing, for views |
| `rig-telemetry` | `EffectLogPlugin` | every model and tool call in the session's `effects.jsonl` |
| `rig-coding-tools` | `ReadTool`, `EditTool`, `WriteTool`, `SearchTool`, `ShellTool` | the coding tools of [`rig-tools`](../rig-tools), one plugin each |
| `rig-coding-tools` | `AttachPlugin` | the files the user names as `@path` go with the message |
| `rig-reload` | `ReloadPlugin` | `/reload` and the `reload` tool, with which the agent rebuilds itself, and the system prompt's section on what the agent is (its plugins, commands and tools) and how it writes plugins |
| `rig-basics` | `BasicCommandsPlugin` | `/help`, `/retry`, `/quit`, `/agents` |
| `rig-subagents` | `SubagentsPlugin` | the `task`, `message` and `wait` tools |
| `rig-telemetry` | `DiagnosticsPlugin` | the process's warnings and errors, Bevy's included, as the `Diagnostics` resource |
| `rig-print` | `PrintPlugin` | `--print`: the front of a run with `-p` or with stdin piped in |
| `rig-tui` | `TuiPlugin` | the terminal view: the front of a run on an interactive terminal, with the status line and pickers other plugins add to |

An entry names its crate, `crate = "rig-tui"`, and no source: the agent
builds rig's own crates from the rig checkout it is built from
(`RIG_SOURCE`). None of rig-harness, rig-ecs, rig-tools and the plugin
crates is published yet. A plugin crate that adds to the terminal view (status
line items, pickers, tool renderers) does so behind a default `tui` cargo
feature, so it also builds without `rig-tui`.

Optional plugins are not in the default list: [`rig-inspect`](../../plugins/rig-inspect)'s
`InspectPlugin` adds the `inspect` tool, with which the agent reads its own
Bevy world through Bevy Remote (in process, read-only). It makes the build
heavier, so it is commented out in the default `plugins.toml`; uncomment its
entry, or run `rig plugin add rig_inspect::InspectPlugin --crate rig-inspect`.
[`rig-steel`](../../plugins/rig-steel) adds code mode.

[`PLUGINS.md`](PLUGINS.md) shows how to write a plugin, with one example per
extension point.

## Install

```bash
git clone https://github.com/0xPlaygrounds/rig && cd rig
cargo install --path .
rig
```

It needs Rust 1.97.1 or newer and runs on Linux and macOS. With the
optional `rig-inspect` plugin enabled, building the agent on Linux also
needs the ALSA development headers (`libasound2-dev` on Debian and Ubuntu,
`alsa-lib-devel` on Fedora, `alsa-lib` on Arch): Bevy Remote pulls in
Bevy's audio crate. The agent plays no sound.
