# rig-tui

The terminal view (`TuiPlugin`), the front of a run on an interactive
terminal: the transcript with markdown answers and diffs, a multiline input
with history and `/` and `@` completion, the pickers other plugins ask for
(`PickRequest`), and the status line, which shows the items other plugins set
(`StatusItems`, `AppStatus`). Other plugins add to it without touching it: a
`TuiPanel` beside the transcript or over the screen, drawn by their own
system, and a renderer for their tools' calls (`AppToolRenderersExt`); a crate
that also builds without the view does so behind a default `tui` feature.
ratatui is re-exported as `rig_tui::ratatui`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`; without it the agent answers `-p` or what is piped in, and
says so when started on a terminal. It is not published yet: the agent builds
it from a rig checkout (`RIG_SOURCE`).
