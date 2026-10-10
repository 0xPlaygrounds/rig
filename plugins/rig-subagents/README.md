# rig-subagents

Subagents (`SubagentsPlugin`): the `task`, `message` and `wait` tools, with
which an agent starts child agents, sends them and its peers requests, and
waits for their reports. Built only on the kernel's public API; leave it out
and there are no subagents.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
