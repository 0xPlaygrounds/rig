# rig-splits

Splits of a conversation (`SplitsPlugin`): `/split` at any entry, into a
branch of the session or into a new session, `/split move` to take a range
out and leave a pointer behind, `/splits` to list and preview the lines of
any saved session, and `/split switch`, `/split compare` and `/split merge`.
Splits nest to any depth and start from any branch of any saved session.
State a plugin keeps with the conversation (`Follows`), such as the
compaction policy, follows each split as of its split point.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
