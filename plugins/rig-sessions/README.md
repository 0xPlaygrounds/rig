# rig-sessions

Sessions beyond the one running (`SessionsPlugin`), one of the
[rig coding agent](../../crates/rig-harness)'s default plugins: `/new`,
`/resume` (by id or name, or picked in rig-harness's `SessionPicker`) and
`/name`, and, with the default `tui` feature, the session's name in the status
line. It keeps each session's listing (rig-memory's `SessionMeta`) current,
which every session listing reads.

The agents of the session (`AgentsPlugin`), also a default plugin: `/fork`
makes a new agent from the shown one (rig-ecs's `ForkAgent`), at the end, through
a message (`/fork m4`) or one picked (`/fork at`); `/agents` lists the
agents, with the `AgentColumn`s plugins add, and shows the one picked; with
the `tui` feature, rewind: Alt+Up to an earlier message of yours, then
Enter, forks the shown agent just before it, shows the fork and puts the
message back in the input. The shown agent is saved by id, so a resumed
session shows it again.
