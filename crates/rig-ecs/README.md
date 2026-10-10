# rig-ecs

A [Bevy](https://bevy.org) agent runtime on [Rig](https://github.com/0xPlaygrounds/rig)'s
rig-core, for building your own agent harness:

- **Agents are entities.** Their components hold the conversation, model,
  reasoning setting, system prompt and tool access. A running turn is an entity
  of its agent, and its model and tool calls are entities of the turn that run
  on Bevy's task pools.
- **One effect path.** Every model and tool call is dispatched through it,
  under an effect id; a plugin that inserts `Effects::recorded_by` its
  recorder sees every effect with rig-core's types, such as rig-harness's
  effect log, which rig-cassette replays.
- **Plugins register tools and commands** (`AppToolsExt`, `AppCommandsExt`),
  save their own components with the session, and re-arm their work after a
  restart on `Restored`.
- **Session journals** go to the `SessionStore` the app inserts, over one of
  rig-cassette's `journal` stores: `MemoryStore`, or `JsonlDirStore` (its
  feature `jsonl`) for JSON-lines files.
- **Turn hooks** for plugins such as compaction: `PrepareRequest` before each
  model request, `ModelFailed` after a call a retry does not fix, and
  `ModelRequest` calls of their own on the agent's model; `Condensed` sends a
  summary in place of older messages. Retries follow rig-core's retry
  policy, and a retry waits on Bevy's clock
  (`bevy_time`'s delayed commands); `AgentPlugin` adds `TimePlugin` when the
  app has none.

Features: `subagents` (default) adds the `task` and `message` tools. The crate
builds for `wasm32-unknown-unknown`.

The [`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
