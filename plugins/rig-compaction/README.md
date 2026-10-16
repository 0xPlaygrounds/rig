# rig-compaction

Compaction for the [rig coding agent](../../crates/rig-harness)
(`CompactionPlugin`). When a request nears the model's context window, older
tool outputs are cleared down to the newest within a tenth of the room left
below the reserve, and if that is not enough the older messages are
summarized into a checkpoint, asked at the end of the turn's own request so
the provider reads the conversation from its prompt cache; a summary that
would free less than a quarter of the conversation is not asked for. The
same happens after an overflow, and `/compact <focus>` does it now. A request
that does not fit takes one recovery path, carried through retries: cleared,
then compacted, then the turn fails, and a model reply starts it again. The
policy is `[compaction]` in `settings.toml`; other compaction, per agent or
by other rules, is a plugin that replaces this one. A plugin can review a
summary before it lands, and `Uncompact` undoes the newest one. See the crate
docs.
