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
  prompt, messages, tools, options, the model it goes to) and pushes its
  `Adapters` (an `outermost()` one wraps all the others), and Start fits
  it to that model. Its stream is written as `StreamDelta`s (text,
  reasoning and tool-call argument fragments) and stamped on the call
  (`Streamed`). A failed call carries its `RetryVerdict`, which Review
  may act on by superseding the call, such as on another model. A turn
  ends with a `TurnOutcome` on it: the kernel proposes one, Review may
  edit it in place, hold it or carry the turn on, a plugin may fail or stop the
  turn, and once its calls settled the turn ends, `Settled`.
- `stage`: effects as entities. An effect's components are its request
  and its outcome, and each frame the `AgentPass` schedule runs its stages
  (Arrive, Review, Commit, Admit, Prepare, Gate, Start) until nothing
  moves. A plugin steps in with a system in a stage: it edits a request in
  Prepare, in the phase of what it does (`PrepareSystems`: Input, Compose,
  Edit, Fit), holds (`HoldOf`) or decides it (`Decision`, the first decision
  wins) in Gate, where an answer from a record of an earlier run, such as a
  session replay's, goes first (`AnswerFromRecord`), and reads or edits the `Outcome` in place in Review
  (`ReviewSystems`: Edit, then Act), so Bevy reports two unordered
  deciders or reviewers. Kind filters
  (`ToolCalls`, `ModelCalls`, `IncomingMessages`, `PluginEffects`) tell
  Bevy which effects two systems can never share. A tool call is such an
  effect (`ToolRequest`): as Gate opens, the kernel gives it the
  `Footprint` of what it reads and writes (rig-core's `Touches`), and Gate
  sees it when it is about to run, once every earlier call of its reply
  it conflicts with settled (`Waiting`). A running effect's owner keeps
  `IfStopped` current, the answer the kernel gives if it stops the effect
  first. Every settled effect is `Recorded`, with its raw and reviewed
  outcome. A failing Prepare, Gate or Review system closes its stage for
  that pass.
- `calls`: off-thread work on Bevy's task pools, and `Wake`, which wakes a
  loop that sleeps while nothing happens.
- `tools`, `prompt`: tools and system prompt sections are entities that
  plugins register (`AppToolsExt`, `PromptSection`, for every agent or,
  `SectionOf` one, that agent alone); tools can also come and go while the
  app runs (`WorldToolsExt`).
- `inbox`: a message to an agent is an effect too (`Incoming`, spawned by
  `Deliver`): plugins edit it in Prepare and take it over in Gate, as
  rig-harness does to run a user's `/` line as a slash command; the kernel
  accepts it at Start, and an agent's accepted, unanswered messages are its
  inbox.
- `model`: the agent's catalog model and how it is connected, through
  rig-core's `catalog::Connector`.
- `effects`: the one dispatch path, recorded into the kernel's own buffer
  and written as `Recorded` when each effect settles.
- `journal`, `restore`: the append-only session log and its restore. Every
  conversation change goes through `Commit` (add a message, splice
  messages after the fact, halt, retire the agent) and is seen as a
  `Committed` message. A splice replaces a range of messages: a cut, a
  compaction's summary or an edit in place; each message keeps its
  `MessageId` across splices and restores; a component that says
  `#[reflect(Component, Saved)]`, or a resource that says
  `#[reflect(Resource, Saved)]`, is logged and restored by reflection.
  `read_session` reads a session's logs without restoring or writing, and
  `fork_session` copies a session as it was when one of its messages was
  made. The work a
  restart carries on is marked `Restored` and gated like any. The store is
  one of rig-cassette's `journal` stores (`MemoryStore`, or `JsonlDirStore`
  with its feature `jsonl`).

`AgentPlugin` adds Bevy's `TimePlugin` when the app has none. The crate has
no features and builds for `wasm32-unknown-unknown`.

The [`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
