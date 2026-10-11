# rig-coding-tools

The coding tools of [`rig-tools`](../../crates/rig-tools), one plugin each
(`ReadTool`, `EditTool`, `WriteTool`, `SearchTool`, `ShellTool`), with their
rules on when to pick them and how the terminal view draws their calls (the
default `tui` feature), and `AttachPlugin`: the files the user names as
`@path` go with the message.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugins are in the default
`plugins.toml`, and each can be removed from it. It is not published yet: the
agent builds it from a rig checkout (`RIG_SOURCE`).
