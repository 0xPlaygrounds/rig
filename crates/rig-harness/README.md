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
  of tool calls, and the session picker. The launcher's
  generated `main.rs` adds the plugins the settings turn on as one
  `PluginGroup`, each wrapped in `Loaded`, which records it as an entity
  (`PluginSource`) and marks what it adds (`ProvidedBy`). Any of them can be
  turned off.

rig's built-in plugins are listed, in the order the agent adds them, by
`built_in_plugins!` in `rig-harness-protocol`'s `config` module; `rig plugin
list` prints them with what each adds and whether it is on. Most are in
[`rig-basics`](https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-basics),
whose README lists its modules.

The agent builds rig's own crates from crates.io at the launcher's version, or
from the rig checkout it is built from (`RIG_SOURCE`); a crate that is not
rig's is a `[[plugins.extra]]` table of `RIG_HOME/settings.toml`. A plugin crate that adds to the
terminal view (status line items, pickers, transcript render systems) does so
behind a default `tui` cargo feature, so it also builds without `rig-tui`.

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
