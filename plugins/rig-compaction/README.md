# rig-compaction

Compaction for the [rig coding agent](../../crates/rig-harness)
(`CompactionPlugin`). When a request nears the model's context window, older
tool outputs are cleared. Before that, the model writes notes on the ones it
couldn't get back, in a request that is the previous one plus an
instruction, so the provider reads it from its prompt cache. The notes stay
in the conversation after the outputs they describe. If clearing would still
leave more than half the room used, the older messages are summarized into a
checkpoint instead, again from the cache; a summary that would free less
than a quarter of the conversation is not asked for. The same happens after
an overflow, and `/compact <focus>` summarizes now, reading every output
whole from the cache. A request that does not fit takes one recovery path,
carried through retries: cleared, then compacted, then the turn fails, and a
model reply starts it again. The policy is `[compaction]` in
`settings.toml`; other compaction, per agent or by other rules, is a plugin
that replaces this one. A plugin can review notes or a summary before they
land, and `Uncompact` undoes the newest summary. See the crate docs.

## Near the window

```toml
[compaction]
# Near the model's context window, old tool outputs are cleared. "summarize" (default): the model first writes notes
# on outputs it couldn't get back (a file still on disk can be read again), in a request read from the prompt cache.
# "clear": cleared with no model call. Either way, when clearing would leave over half the room, the older messages
# are summarized into a checkpoint instead.
near_window = "summarize"
clearing = { keep_tokens = 20000, keep_under = 500 }  # newest output kept, as the provider counts; smaller outputs stay
```
