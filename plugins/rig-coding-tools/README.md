# rig-coding-tools

The coding tools of [`rig-tools`](../../crates/rig-tools) for the
[rig coding agent](../../crates/rig-harness), one default plugin each
(`ReadTool`, `EditTool`, `WriteTool`, `ShellTool`, as pi has), with their
rules on when to pick them and, with the default `tui` feature, how the
terminal view draws their calls. `AttachPlugin` sends the files a message
names as `@path` with it.
