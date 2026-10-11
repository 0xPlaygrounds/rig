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
  Bevy's clock. A model call is an effect (`ModelRequest`): the turn's own
  (`TurnCall`) and a side call built from an agent (`BuildFrom`) have
  their request built by the kernel, a plugin edits it in Prepare (system
  prompt, messages, tools, options, the model it goes to), and Start fits
  it to that model. A failed call carries its `RetryVerdict`, which Review
  may act on by superseding the call, such as on another model. A turn
  ends with a `TurnOutcome` on it: the kernel proposes one, Review may
  replace it, hold it or carry the turn on, a plugin may fail or stop the
  turn, and once its calls settled the turn ends, `Settled`.
  `Condensed` sends a summary in place of older messages.
- `stage`: effects as entities. An effect's components are its request
  and its outcome, and each frame the `AgentPass` schedule runs its stages
  (Arrive, Review, Commit, Admit, Prepare, Gate, Start) until nothing
  moves. A plugin steps in with a system in a stage: it edits a request in
  Prepare, holds (`HoldOf`) or answers (`Outcome`) it in Gate, and reads or
  replaces the outcome in Review. A tool call is such an effect
  (`ToolRequest`): Gate sees a reply's call when it is about to run, once
  every earlier call it must not run beside settled (`Waiting`). Every
  settled effect is `Recorded`, with its raw and reviewed outcome. A
  failing Prepare, Gate or Review system closes its stage for that pass.
- `calls`: off-thread work on Bevy's task pools, and `Wake`, which wakes a
  loop that sleeps while nothing happens.
- `tools`, `commands`, `prompt`: tools, slash commands and system prompt
  sections are entities that plugins register (`AppToolsExt`,
  `AppCommandsExt`, `PromptSection`, for every agent or, `SectionOf` one,
  that agent alone); tools can also come and go while the app runs
  (`WorldToolsExt`).
- `inbox`: a message to an agent is an effect too (`Incoming`, spawned by
  `Deliver`): plugins edit it in Prepare and take it over in Gate; the
  kernel accepts it at Start, runs a user's `/` line as a slash command,
  and an agent's accepted, unanswered messages are its inbox.
- `model`: the agent's catalog model and how it is connected, through
  rig-core's `catalog::Connector`.
- `effects`: the one dispatch path, recorded into the kernel's own buffer
  and written as `Recorded` when each effect settles.
- `journal`, `restore`: the append-only session log and its restore. Every
  conversation change goes through `Commit` (add, cut back, halt, retire
  the agent) and is seen as a `Committed` message; a component that says
  `#[reflect(Component, Saved)]`, or a resource that says
  `#[reflect(Resource, Saved)]`, is logged and restored by reflection.
  `read_session` reads a session's logs without restoring or writing, and
  `fork_session` copies a session as of one of its records. The work a
  restart carries on is marked `Restored` and gated like any. The store is
  one of rig-cassette's `journal` stores (`MemoryStore`, or `JsonlDirStore`
  with its feature `jsonl`).

`AgentPlugin` adds Bevy's `TimePlugin` when the app has none. The crate has
no features and builds for `wasm32-unknown-unknown`.

The [`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
