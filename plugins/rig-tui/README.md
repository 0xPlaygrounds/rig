# rig-tui

The terminal view of the [rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), and the
front of a run on an interactive terminal. `TuiPlugin` is `TuiViewPlugin`
(terminal, keys, frame) plus `DefaultScreenPlugin` (transcript, status line,
input), and the default screen is built from the same public widgets a plugin
places in panels of its own.

## A view of a session

The view is a client of the agent server, `rigd`. It runs in its own process
(the `view` role) with a mirror of one session: the session's events, from the
first, applied to entities that carry the kernel's own components
(`rig_harness::view::MirrorPlugin`), so the transcript, pickers and status line
read a mirrored agent as they read a local one. What you type goes to the
session as its commands: a message is `rig/session/send` (Enter steers a
running turn, Tab queues), Esc is `rig/session/stop {scope: "turn"}`, and an
answer to a question is `rig/session/answer`.

- `rig` starts a session in the current directory and drives it.
- `rig attach S` shows session `S`, whoever started it (a script, `rig start`,
  another agent): it drives `S` when nothing else does, else it watches.
- `rig attach S --takeover`, or `/takeover` in a view, takes the driver lease
  from the view that has it; that view keeps watching and takes it back the
  same way.
- Any number of views show one session, and each sees every event.
- `/quit` (or `/detach`) leaves the view. The session goes on, and its agent
  keeps working; `rig attach S` comes back to it.

A plugin's parts in the view are its view half, named by its `view` key in
`plugins.toml`; its status line items are agent facts
(`rig_harness::status`), which the agent publishes as `status` events.

## The terminal

The view tells the terminal when a turn of an agent nothing spawned runs, as a
shell does a command: OSC 133 marks (`C` as it starts, `D;0` or `D;1` as it
ends, then `A` and `B`), a state report (`133;S;activity=busy` or `reading`)
for a multiplexer or any terminal that reads them, and the title
(`rig · working`, `rig · idle`). A terminal that wants options on the marks,
such as a nonce, names each with the environment variable holding its value:
`[terminal] mark_options = { nonce = "MY_TERMINAL_NONCE" }` in
`settings.toml`; there are none by default.

Plugins add panels, transcript looks, status line items, key bindings and
completion without touching the view. The plugin guide shows each
([PLUGINS.md](https://github.com/0xPlaygrounds/rig/blob/main/crates/rig-harness/PLUGINS.md#the-transcript-notes-looks-and-scrolling)),
and the crate docs list every type.

A crate that also builds without the view puts its rig-tui code behind a
default `tui` feature. `rig_tui::ratatui` and `rig_tui::crossterm` are
re-exported. Tests run the view without a terminal by adding the plugin and
then `add_view`, put keys in `PendingInput`, and read a frame from a
`TuiTestScreen`; a test of a view of a session adds the `MirrorPlugin` with a
`ViewLink::detached()` and applies records with `rig_harness::view::mirror::apply`.
