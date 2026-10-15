# rig-subagents

Subagents (`SubagentsPlugin`), one of the
[rig coding agent](../../crates/rig-harness)'s default plugins: the `task`,
`message` and `wait` tools, with which an agent starts child agents, sends
them and its peers requests, and waits for their reports. Every agent is
spawned through a `SpawnSubagent` effect that other plugins edit, refuse or
hold, which is also how a plugin spawns helpers of its own. `[subagents]
max_depth` in the settings caps nesting. Without it there are no subagents.
