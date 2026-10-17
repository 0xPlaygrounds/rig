# rig-ecs

The kernel of the rig coding agent, on [Bevy](https://bevy.org) and
[Rig](https://github.com/0xPlaygrounds/rig)'s rig-core, for building your own
agent harness. It holds only what every agent needs, whatever its interface.
Subagents, compaction, usage and the effect log are plugins on its public API.

- `agent`: an agent is an entity whose components hold its conversation, model,
  reasoning setting, system prompt and tool access. A running turn is an entity
  `TurnOf` its agent, and its calls are entities `CallOf` the turn.
- `stage`: every step of an agent's work is an effect entity. Each frame the
  `AgentPass` schedule runs its stages (Arrive, Review, Commit, Admit, Prepare,
  Gate, Start) until nothing moves, and a plugin steps in with an ordinary
  system in a stage.
- `turn`: the turn loop: model requests, tool calls, retries, stops, and how a
  turn ends (`TurnOutcome`).
- `tools`, `prompt`: tools and system-prompt sections are entities plugins
  register.
- `inbox`: a message to an agent is an effect too (`Incoming`, sent by
  `Deliver`).
- `model`: the agent's catalog model and how it is connected.
- `tasks`, `effects`: work off the main thread on Bevy's task pools, and the
  one path every effect is dispatched and recorded through.
- `session_log`, `restore`: the session log in any rig-memory `Store`, the
  restore after a restart, and state kept with the session (`Saved`).

The rules a plugin follows, with an example per extension point, are in
rig-harness's [plugin guide](../rig-harness/PLUGINS.md).

`AgentPlugin` adds Bevy's `TimePlugin` when the app has none. The crate has no
features and builds for `wasm32-unknown-unknown`. The
[`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
