# rig-tui

The terminal view (`TuiPlugin`), the front of a run on an interactive
terminal: the transcript with markdown answers and diffs, a multiline input
with history and `/` and `@` completion, pickers and the questions plugins
ask (`Ask`), shown one at a time, and the status line, which shows the
items other plugins set (`StatusItems`, `AppStatus`). Other plugins add to
it without touching it: a `TuiPanel` beside the transcript or over the
screen, drawn by their own system, and a renderer for their tools' calls
(`AppToolRenderersExt`). They take part in the input as data: key
bindings (`KeyBinding`, `AppKeyBindingsExt`; the view's own keys are
bindings too), the keys for a panel (`KeyFocus`), the draft (`TuiInput`),
the terminal's events (`TerminalEvent`), a program run with the terminal
(`RunInTerminal`, such as `$EDITOR`), and their own pickers
(rig-harness's `Picker`), which the view shows. A crate
that also builds without the view does so behind a default `tui` feature.
ratatui and crossterm are re-exported as `rig_tui::ratatui` and
`rig_tui::crossterm`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`; without it the agent answers `-p` or what is piped in, and
says so when started on a terminal. It is not published yet: the agent builds
it from a rig checkout (`RIG_SOURCE`).
