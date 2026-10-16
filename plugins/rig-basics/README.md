# rig-basics

Two of the [rig coding agent](../../crates/rig-harness)'s default plugins:

- `BasicCommandsPlugin`: `/help`, `/retry` and `/quit`.
- `DefaultPromptPlugin`: rig's own system prompt for every agent without
  one, `DEFAULT_SYSTEM_PROMPT`. It is one fixed text, the same on every
  request, so the provider's prompt cache keeps the whole conversation: no
  date, paths, instruction files or tool rules in it. It tells the model to
  read `AGENTS.md` (or `CLAUDE.md`) itself.
