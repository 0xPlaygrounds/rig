# rig-basics

The basic slash commands, `/help`, `/retry`, `/agents` and `/quit`, with the
agents at work in the status line (`BasicCommandsPlugin`), and the project context (`ProjectContextPlugin`): the
instruction files (`AGENTS.md` or `CLAUDE.md`) and the environment in every
agent's system prompt, re-read when a turn starts, and `/context`.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugins are in the default
`plugins.toml`, and each can be removed from it. It is not published yet: the
agent builds it from a rig checkout (`RIG_SOURCE`).
