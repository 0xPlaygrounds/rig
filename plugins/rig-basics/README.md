# rig-basics

Two of the [rig coding agent](../../crates/rig-harness)'s default plugins:

- `BasicCommandsPlugin`: `/help`, `/retry` and `/quit`.
- `DefaultPromptPlugin`: rig's own system prompt for an agent without one.
  It is one fixed text, with no date, working directory or instruction
  files, so the provider's prompt cache keeps it; the model reads
  `AGENTS.md` itself.
