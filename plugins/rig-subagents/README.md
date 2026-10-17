# rig-subagents

Subagents (`SubagentsPlugin`), a plugin of the
[rig coding agent](../../crates/rig-harness) that is not listed by default:
the `task`, `message` and `wait` tools, with which an agent starts child
agents, sends them and its peers requests, and waits for their reports; and,
with the default `tui` feature, the agents at work in each agent's status
line (rig-sessions' `/agents` shows one). Every agent is
spawned through a `SpawnSubagent` effect that other plugins edit, refuse or
hold, which is also how a plugin spawns helpers of its own. `[subagents]
max_depth` in the settings caps nesting. Without it there are no subagents.
