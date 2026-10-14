# rig-tui

The terminal view (`TuiPlugin`), the front of a run on an interactive
terminal, built from public widgets a plugin also places in panels of its
own: transcript panes over the agent shown, any agent or a saved
conversation (`TranscriptPane`, with its own view, scroll and looks),
editors (`Editor`), status lines (`StatusLine`) and the line producers
(`markdown`, `wrap`, `diff`). The default screen is `DefaultScreenPlugin`,
spawned through that public API only; an app with a screen of its own adds
`TuiViewPlugin` without it. It shows the transcript with markdown answers
and diffs, a multiline input
with history (kept in rig-harness's `PluginData`) and `/` and `@`
completion, pickers and the questions plugins ask (`Ask`), shown one at a
time, with items ticked (Tab) where they take several, a choice ticked with
a note typed on it, a choice's more shown on Right, keys of a question's
own (`ChoiceKeys`), a question changed in place of its predecessor
(`SupersededBy`) keeping the user's ticks, and a long body that scrolls, and the status line, which shows the items other plugins
set (`StatusItems`, `AppStatus`), each by name, so another plugin replaces or
hides it (`StatusSystems::Edit`). Other plugins add to it without touching
it: a `TuiPanel` beside the transcript or over the screen, drawn by their own
system, a renderer for their tools' calls (`AppToolRenderersExt`), which
also sees the calls a call started through any effect, a resumed session's
too (`ToolCallView::children`, `PastEffects`), lines under every call
(`add_call_decorator`, each one's rows an item, `TranscriptItem::Decoration`), a renderer for a kind of message content, such as
images, the user's or the model's (`add_content_renderer`), for a kind of
aside such as a kept branch (`add_aside_renderer`), for the user's, the
model's and its reasoning's text (`add_message_renderer`), lines under every
message (`add_message_decorator`), each also only in one pane when spawned
under it, and candidates
for completion after a marker such as `#` or for a command's argument
(`CompletionSource`, with what is typed in `TuiInputState::typed`). They take part
in the input as data: key bindings (`KeyBinding`, `AppKeyBindingsExt`; the
view's own keys are bindings too), the keys for a panel (`KeyFocus`), the
draft (`MainInput`) and the prompt history (`PromptHistory`, `Sent`), the
frame's keys before any binding sees them, one at a time (`TerminalEvents`,
filtered in `InputSystems::Filter` of the `TuiKeys` schedule, with
`TuiInputState`), bindings while the transcript has the keys
(`KeyScope::Transcript`), a program run with the
terminal (`RunInTerminal`, such as `$EDITOR`), and their own pickers
(rig-harness's `Picker`), which the view shows, filled in place as items
arrive, with what is typed and selected kept (`PickerState`), the item
highlighted (`Highlighted`) and a preview beside it (`PickPreview`). They
read and steer each pane (`TranscriptView`, `ScrollTo`, labels inside replies' text as
`TextMark`s), where a notice is drawn
(`ShownNotice::slot`), the rows a pane laid out with the matches it lit and
the items it cut (`DrawnTranscript`), where the frame drew each panel
(`TuiFrame`, for a plugin that writes to the terminal itself, and
`Repaint` to write cells it drew over again), and any agent's transcript,
or a saved conversation, laid out the same way without drawing it, cut or
whole (`TranscriptLayout`). The
clipboard goes through rig-tools' `clipboard`. A crate that also builds
without the view does so behind a default `tui` feature. ratatui and
crossterm are re-exported as `rig_tui::ratatui` and `rig_tui::crossterm`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`; without it the agent answers `-p` or what is piped in, and
says so when started on a terminal. It is not published yet: the agent builds
it from a rig checkout (`RIG_SOURCE`). A plugin's tests run the view without
a terminal with `rig-harness-test-support`'s `tui_view` and `press`, and
read what a frame drew from a `TuiTestScreen`.
