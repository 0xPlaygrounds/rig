# rig-ecs

A [Bevy](https://bevy.org) agent runtime on [Rig](https://github.com/0xPlaygrounds/rig)'s
rig-core, for building your own agent harness:

- **Agents are entities.** Their components hold the conversation, model,
  reasoning setting, system prompt and tool access. A running turn is an entity
  of its agent, and its model and tool calls are entities of the turn that run
  on Bevy's task pools.
- **One recorded effect path.** Every model and tool call is dispatched through
  rig-core's effect recording, into an effect log rig-cassette can replay.
- **Plugins register tools and commands** (`AppToolsExt`, `AppCommandsExt`),
  save their own components with the session, and re-arm their work after a
  restart on `Restored`.
- **Session journals** go to the `SessionStore` the app inserts: `MemoryStore`,
  or `JsonlDirStore` (feature `fs-journal`) for JSON-lines files.
- **Compaction and retries** from rig-memory and rig-core, set by
  `CompactionPolicy` and the retry policy. A retry waits on Bevy's clock
  (`bevy_time`'s delayed commands); `AgentPlugin` adds `TimePlugin` when the
  app has none.

Features: `subagents` (default) adds the `task` and `message` tools; `fs-journal`
is native-only. Without it the crate builds for `wasm32-unknown-unknown`.

The [`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
