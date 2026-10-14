# rig-compaction

Compaction (`CompactionPlugin`): when a turn's next request nears the model's
context window, older tool outputs are cleared, and when that is not enough
the older messages are summarized by the model into a checkpoint that requests
send in their place. The same happens after a request refused as too long, and
`/compact <focus>` does it now. A plugin reviews a summary before it replaces
anything (`Summarizing::summary`, in `ReviewSystems::Act`: edit it, hold it,
or take it out to cancel), and `Uncompact` puts back what the newest
compaction replaced, with the messages' ids. It uses the kernel's turn hooks
only.

Each agent has its own policy: its own knobs (`Compaction`, kept with its
conversation's line, so a split inherits it) over the default every agent
starts from (`CompactionDefault`: `[compaction]` in `settings.toml` over the
built-in one). `/compaction` shows the agent's policy and sets or resets its
own knobs; a plugin inserts `Compaction` on an agent, and adds the cuts a
policy may name with `add_compaction_cut`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
