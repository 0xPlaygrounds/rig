# rig-harness

The harness of the rig coding agent: a terminal coding agent that is a
[Bevy](https://bevy.org) app made of plugins, on the [`rig-ecs`](../rig-ecs)
kernel and [Rig](https://github.com/0xPlaygrounds/rig)'s rig-core. The `rig`
launcher generates a small Cargo project for it from `RIG_HOME/plugins.toml`,
builds it and runs it; `/reload` rebuilds it with changed plugins and restarts
in the same session. The [root README](../../README.md#the-rig-coding-agent)
describes using it.

## A kernel, a harness and plugins

- **The kernel** is rig-ecs: agents, turns and calls as entities, message
  delivery, tools as entities, the session log and its restore.
- **The harness** is this crate: what the binary needs to run and be
  relaunched, and the services every plugin shares. `HeadlessPlugins` is
  Bevy's `MinimalPlugins` with the log, a clean exit on signals and a loop that
  sleeps until there is work. `RigHarnessPlugins` adds the session and its
  store (`StoreOpener`, which refuses, unchanged, a session from before
  `rig-session/2` or from a newer rig), the kernel, the launcher protocol
  (`Invoked`, `AppStart`, `Secrets`, `Exit`), slash commands, fronts, and the
  shared services every agent uses: settings, the workspace and child
  processes, file watches, plugin data, questions and pickers, and transcript
  notes. The services only some plugins use are plugins of this crate that
  those plugins add: approvals (`ApprovalPlugin`), the preview and the
  checkpoint of tool calls (`PreviewPlugin`, `CheckpointPlugin`) and the
  session picker (`SessionPickerPlugin`). The generated `main.rs` ends with
  `rig_harness::run`.
- **Everything else is a plugin** in a crate of the repository's
  [`plugins/`](../../plugins) folder, listed in `plugins.toml` and built only
  on public API, exactly as a third-party plugin is: the compiler holds them
  to it. The launcher's generated `main.rs` adds each with `load`, which
  records it as an entity (`PluginSource`) and marks what it adds
  (`ProvidedBy`). Any entry can be removed.

The default `plugins.toml`, in order:

| Crate | Plugin | What it adds |
|---|---|---|
| `rig-basics` | `DefaultPromptPlugin` | rig's own system prompt for an agent without one: fixed text, so the prompt cache keeps it |
| `rig-models` | `ModelsPlugin` | `/model`, `/effort` and the model picker |
| `rig-login-chatgpt` | `ChatgptLoginPlugin` | `/login` and `/logout` for the ChatGPT plan's models |
| `rig-models` | `DefaultsPlugin` | a new session starts on the last model and reasoning chosen |
| `rig-sessions` | `SessionsPlugin` | `/new`, `/resume`, `/name` |
| `rig-compaction` | `CompactionPlugin` | summarizing a conversation near the context window; `/compact` |
| `rig-telemetry` | `UsagePlugin` | what model calls cost; `/usage` |
| `rig-activity` | `ActivityPlugin` | what each agent is doing, for views |
| `rig-coding-tools` | `ReadTool`, `EditTool`, `WriteTool`, `ShellTool` | the coding tools of [`rig-tools`](../rig-tools), one plugin each, acting in the app's workspace; the library's `search`, `list`, `delete` and `move` are left to `shell` |
| `rig-coding-tools` | `AttachPlugin` | the files the user names as `@path`, from the agent's directory, go with the message |
| `rig-reload` | `ReloadPlugin` | `/reload` and the `reload` tool, with which the agent rebuilds itself, and the system prompt's section on how the agent writes plugins (fixed text naming `plugins.toml`) |
| `rig-basics` | `BasicCommandsPlugin` | `/help`, `/retry`, `/quit` |
| `rig-print` | `PrintPlugin` | `--print`: the front of a run with `-p` or with stdin piped in |
| `rig-tui` | `TuiPlugin` | the terminal view: the front of a run on an interactive terminal, with the status line and pickers other plugins add to |

Not listed by default, each added with its own entry:

| Crate | Plugin | What it adds |
|---|---|---|
| `rig-splits` | `SplitsPlugin` | `/split` and `/splits`: splitting a conversation at any entry into a branch or a new session, moving a range out, switching, comparing and merging back |
| `rig-telemetry` | `EffectLogPlugin` | every model and tool call in the session's `effects.jsonl` |
| `rig-subagents` | `SubagentsPlugin` | the `task`, `message` and `wait` tools, `/agents`, and the agents at work in the status line |
| `rig-telemetry` | `DiagnosticsPlugin` | the process's warnings and errors, Bevy's included, as the `Diagnostics` resource |
| `rig-inspect` | `InspectPlugin` | the `inspect` tool, with which the agent reads its own running world |
| `rig-steel` | `SteelPlugin` | the `run_steel` tool, whose Steel program orchestrates agents and tools |

An entry names its crate, `crate = "rig-tui"`, and no source: the agent
builds rig's own crates from the rig checkout it is built from
(`RIG_SOURCE`). None of rig-harness, rig-ecs, rig-tools and the plugin
crates is published yet. A plugin crate that adds to the terminal view (status
line items, pickers, tool renderers) does so behind a default `tui` cargo
feature, so it also builds without `rig-tui`.

The optional [`rig-inspect`](../../plugins/rig-inspect) plugin adds the
`inspect` tool, with which the agent reads its own Bevy world, read-only.
[`rig-steel`](../../plugins/rig-steel) adds code mode.

[`PLUGINS.md`](PLUGINS.md) shows how to write a plugin, with one example per
extension point. A plugin's tests use
[`rig-harness-test-support`](../../test-support/rig-harness-test-support), and
[`rig-defaults-test-support`](../../test-support/rig-defaults-test-support)'s
`assert_stages_ordered` checks that its stage systems are ordered against the
default plugins'.

## Install

```bash
git clone https://github.com/0xPlaygrounds/rig && cd rig
cargo install --path .
rig
```

It needs Rust 1.97.1 or newer and runs on Linux and macOS. The optional
`rig-inspect` plugin needs more to build on Linux; its README says what.
