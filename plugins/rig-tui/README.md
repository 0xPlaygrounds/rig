# rig-tui

The terminal view (`TuiPlugin`): the transcript with markdown answers and
diffs, a multiline input with history and `/` and `@` completion, pickers and
the status line. Other plugins add to it without touching it: a `TuiPanel`
beside the transcript or over the screen, drawn by their own system, and a
renderer for their tools' calls (`AppToolRenderersExt`). ratatui is
re-exported as `rig_tui::ratatui`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`; without it the agent answers what is piped in. It is not
published yet: the agent builds it from a rig checkout (`RIG_SOURCE`).
