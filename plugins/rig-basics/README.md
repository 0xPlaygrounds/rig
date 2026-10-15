# rig-basics

Two of the [rig coding agent](../../crates/rig-harness)'s default plugins:

- `BasicCommandsPlugin`: `/help`, `/retry`, `/quit` and, with the default
  `tui` feature, `/agents` and the counts of agents at work in the status
  line. Plugins add columns to `/agents` with `AgentColumn`s.
- `ProjectContextPlugin`: the instruction files (`AGENTS.md` or `CLAUDE.md`)
  and the environment in every agent's system prompt, re-read when a turn
  starts, cut to `[context]` in `settings.toml` (`ContextLimits`), and
  `/context`, which plugins add to with `ContextEntry`s.
