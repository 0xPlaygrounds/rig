# rig-tui

The terminal view of the [rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), and the
front of a run on an interactive terminal. `TuiPlugin` is `TuiViewPlugin`
(terminal, keys, frame) plus `DefaultScreenPlugin` (transcript, status line,
input), and the default screen is built from the same public widgets a plugin
places in panels of its own.

The view tells the terminal when a turn of an agent nothing spawned runs, as a
shell does a command: OSC 133 marks (`C` as it starts, `D;1` as it fails or
`D;0` as it answers or stops, then `A` and `B`), a state report
(`133;S;activity=busy` or `reading`) for a multiplexer such as fux, and the
title (`rig · working`, `rig · idle`, `rig · error` only after a failed turn).
When the shell's marks carry a nonce, `mark_nonce_env` under `[tui]` in
`settings.toml` names the variable holding it (for fux, `FUX_MARK_NONCE`).

Plugins add panels, transcript looks, status line items, key bindings and
completion without touching the view. The plugin guide shows each
([PLUGINS.md](https://github.com/0xPlaygrounds/rig/blob/main/crates/rig-harness/PLUGINS.md#the-transcript-notes-looks-and-scrolling)),
and the crate docs list every type. Transcript looks are systems in
`TranscriptRender`, ordered in `RenderSystems::Render`, `Decorate`, and `Edit`.
They select owned inputs from `RenderBatch` and write lines and named
contributions to `RenderOutput`. The batch includes the pane, source agent and
width; exports run the same systems with no pane. A cached item enters the
batch again when its inputs change; increment `RenderRevision.0` when anything
else a look reads changes, or when looks are added or removed at runtime.

A crate that also builds without the view puts its rig-tui code behind a
default `tui` feature. `rig_tui::ratatui` and `rig_tui::crossterm` are
re-exported. Without rig-tui the agent answers `-p` or piped stdin only. Tests
run the view without a terminal by adding the plugin and then `add_view`, put
keys in `PendingInput`, and read a frame from a `TuiTestScreen`.
