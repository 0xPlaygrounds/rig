# rig-compaction

Compaction for the [rig coding agent](../../crates/rig-harness)
(`CompactionPlugin`). When a request nears the model's context window, older
tool outputs are cleared, and if that is not enough the older messages are
summarized into a checkpoint. The same happens after an overflow, and
`/compact <focus>` does it now. `/compaction` shows and sets the agent's own
policy over `[compaction]` in `settings.toml`. A plugin can review a summary
before it lands, and `Uncompact` undoes the newest one. See the crate docs.
