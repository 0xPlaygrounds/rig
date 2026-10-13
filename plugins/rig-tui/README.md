# rig-tui

The terminal view (`TuiPlugin`), the front of a run on an interactive
terminal: the transcript with markdown answers and diffs, a multiline input
with history (kept in rig-harness's `PluginData`) and `/` and `@`
completion, pickers and the questions plugins ask (`Ask`), shown one at a
time, with several items ticked (Tab) where they take several and a long
body that scrolls, and the status line, which shows the items other plugins
set (`StatusItems`, `AppStatus`), each by name, so another plugin replaces or
hides it (`StatusSystems::Edit`). Other plugins add to it without touching
it: a `TuiPanel` beside the transcript or over the screen, drawn by their own
system, and a renderer for their tools' calls (`AppToolRenderersExt`), which
also sees the calls a call started through any effect, a resumed session's
too (`ToolCallView::children`, `PastEffects`). They take part
in the input as data: key bindings (`KeyBinding`, `AppKeyBindingsExt`; the
view's own keys are bindings too), the keys for a panel (`KeyFocus`), the
draft (`TuiInput`) and the prompt history (`PromptHistory`, `Sent`), the
frame's keys before any binding sees them, one at a time (`TerminalEvents`,
filtered in `InputSystems::Filter` of the `TuiKeys` schedule, with
`TuiInputState`), bindings while the transcript has the keys
(`KeyScope::Transcript`), a program run with the
terminal (`RunInTerminal`, such as `$EDITOR`), and their own pickers
(rig-harness's `Picker`), which the view shows. They read and steer the
transcript (`TranscriptView`, `ScrollTo`), where a notice is drawn
(`ShownNotice::slot`), the rows a frame drew with the matches it lit and
the items it cut (`DrawnTranscript`), and any agent's transcript laid out
the same way without drawing it, cut or whole (`TranscriptLayout`). The
clipboard goes through rig-tools' `clipboard`. A crate that also builds
without the view does so behind a default `tui` feature. ratatui and
crossterm are re-exported as `rig_tui::ratatui` and `rig_tui::crossterm`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`; without it the agent answers `-p` or what is piped in, and
says so when started on a terminal. It is not published yet: the agent builds
it from a rig checkout (`RIG_SOURCE`). A plugin's tests run the view without
a terminal with `rig-harness-test-support`'s `tui_view` and `press`.
