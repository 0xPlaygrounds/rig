# rig-tui

The terminal view of the [rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), and the
front of a run on an interactive terminal. `TuiPlugin` is `TuiViewPlugin`
(terminal, keys, frame) plus `DefaultScreenPlugin` (transcript, status line,
input), and the default screen is built from the same public widgets a plugin
places in panels of its own.

Plugins add panels, transcript looks, status line items, key bindings and
completion without touching the view. The plugin guide shows each
([PLUGINS.md](https://github.com/0xPlaygrounds/rig/blob/main/crates/rig-harness/PLUGINS.md#the-transcript-notes-looks-and-scrolling)),
and the crate docs list every type.

A crate that also builds without the view puts its rig-tui code behind a
default `tui` feature. `rig_tui::ratatui` and `rig_tui::crossterm` are
re-exported. Without rig-tui the agent answers `-p` or piped stdin only. Tests
run the view without a terminal with `rig-harness-test-support`'s `tui_view`
and `press`, and read a frame from a `TuiTestScreen`.
