# rig-sessions

Sessions beyond the one running (`SessionsPlugin`): `/new`, `/resume` (by id
or name, or picked in rig-harness's native `SessionPicker`) and `/name`, with
the default `tui` feature the session's name in the terminal view's status
line. A session's name is a plain file in its directory, which
`rig --resume <name>` finds; its title, the `ResumeLabel`s plugins add (such
as rig-telemetry's cost), the main agent's model and message count, and the
values plugins list (`saved_as(..).listed()`) are kept in its rig-memory
`SessionMeta`, which every session listing reads and a session rewrite maps.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
