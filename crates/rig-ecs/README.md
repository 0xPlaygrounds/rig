# rig-ecs

A [Bevy](https://bevy.org) agent runtime on [Rig](https://github.com/0xPlaygrounds/rig)'s
rig-core, for building your own agent harness. It is a small kernel: only
what every agent needs, whatever its interface. Features such as
subagents, compaction, usage, activity views and the effect log are
plugins built on its public API (rig-harness's are examples).

What it holds, by module:

- `agent`: an agent is an entity. Its components hold the conversation,
  model, reasoning setting, system prompt, tool access and last usage; a
  running turn is an entity `TurnOf` its agent, and its model and tool calls
  are entities `CallOf` the turn. Agents a plugin spawns are `SpawnedBy`
  their parent.
- `turn`: the turn loop (model request, tool calls, results, the next
  request or the end), steering, interrupts, and retries that wait on
  Bevy's clock. Plugins extend a turn with `PrepareRequest` (before each
  request: its system prompt, messages, tools and options), `ModelFailed`
  (after a call a retry does not fix) and `ModelRequest` calls of their
  own; a `Connection` on a turn or call sends it to another model than the
  agent's. `Condensed` sends a summary in place of older messages.
- `calls`: off-thread work on Bevy's task pools, and `Wake`, which wakes a
  loop that sleeps while nothing happens; `KeepAwake` keeps it running for a
  timer.
- `tools`, `commands`, `prompt`: tools, slash commands and system prompt
  sections are entities that plugins register (`AppToolsExt`,
  `AppCommandsExt`, `PromptSection`, for every agent or, `SectionOf` one,
  that agent alone); tools can also come and go while the app runs
  (`WorldToolsExt`).
- `inbox`: `Deliver`, the one way a message reaches an agent.
- `model`: the agent's catalog model and how it is connected, through
  rig-core's `catalog::Connector`.
- `effects`: one dispatch path for every model and tool call, which a
  plugin may record.
- `journal`, `restore`: the append-only session log and its restore. Every
  conversation change goes through `Commit` and is seen as a `Committed`
  message; a component that says `#[reflect(Component, Saved)]`, or a
  resource that says `#[reflect(Resource, Saved)]`, is logged and restored
  by reflection. The store is one of rig-cassette's `journal`
  stores (`MemoryStore`, or `JsonlDirStore` with its feature `jsonl`).

`AgentPlugin` adds Bevy's `TimePlugin` when the app has none. The crate has
no features and builds for `wasm32-unknown-unknown`.

The [`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
