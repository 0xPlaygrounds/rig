# rig-compaction

Compaction (`CompactionPlugin`): when a turn's next request nears the model's
context window, older tool outputs are cleared, and when that is not enough
the older messages are summarized by the model into a checkpoint that requests
send in their place. The same happens after a request refused as too long, and
`/compact <focus>` does it now. It uses the kernel's turn hooks only.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
