# rig-sessions

Sessions beyond the one running (`SessionsPlugin`): `/new`, `/resume` and
`/name`, the session's name in the status line, and each session's title and
cost in its metadata, which `/resume` lists.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
