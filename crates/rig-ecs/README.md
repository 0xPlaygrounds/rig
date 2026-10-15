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
  (`TurnCall`) and a side call built from an agent (`SideCall`) have
  their request built by the kernel, with the system prompt in parts
  (rig-core's `SystemPrompt`: the role and tagged sections), a plugin
  edits it in Prepare (prompt, tools, options, the model it goes to, and
  its messages as a view of the agent's conversation, `history`, each
  edit attributed to its author and never saved) and pushes its
  `Adapters` (an `outermost()` one wraps all the others), and Start
  renders the prompt and fits the request to that model, putting on it
  the options left out (`DroppedOptions`); `ModelStarts::sent` gives what Start
  will send, `ModelStarts::dropped` what it leaves out, and
  `ModelStarts::prompt` its prompt in parts. Its stream is
  written as `StreamDelta`s (rig-core's `StreamEvent`s: a part's start,
  its fragments, its end) and stamped on the call (`StreamTiming`). A failed
  call carries its `RetryVerdict`, which Review may act on by superseding
  the call, such as on another model (any number of effects may be
  `SupersededBy` one successor, as a race's losers are by its winner). A
  turn ends with a `TurnOutcome` on it: the kernel proposes one, Review
  may edit it in place, hold it or carry the turn on, a plugin may fail
  or stop the turn, and once its calls settled the turn ends, `Settled`.
  Each turn is recorded in its agent's conversation (a `rig.turn` aside,
  rig-core's `TurnRecord`: where it began and how it ended), and a reply
  an interrupt cut short is kept as an aborted reply no model reads again.
- `stage`: effects as entities. An effect's components are its request and
  its outcome, and each frame the `AgentPass` schedule runs its stages
  (Arrive, Review, Commit, Admit, Prepare, Gate, Start) until nothing
  moves. A plugin steps in with a system in a stage: it edits a request in
  Prepare, in the phase of what it does (`PrepareSystems`: Input, Compose,
  Edit, Fit), holds (`HoldOf`) or decides it in Gate (`Decision`, naming
  who decided, rig-core's `Decider`; the first decision wins), where an
  answer from a record of an earlier run, such as a session replay's, goes
  first and an answer standing in for running it goes after every gate
  (`GateSystems`: FromRecord, After, then Fallback for the kernel's own
  answers), and reads or edits the `Outcome` in
  place in Review (`ReviewSystems`: Edit, then Act), so Bevy reports two
  unordered deciders or reviewers. Kind filters (`ToolCalls`,
  `ModelCalls`, `IncomingMessages`, `PluginEffects`) tell Bevy which
  effects two systems can never share. A tool call is such an effect
  (`ToolRequest`): as Gate opens, the kernel gives it the `Footprint` of
  what it reads and writes (rig-core's `Touches`; one a plugin `DeclaredFootprint`
  on the call in Prepare stands), and Gate sees it when
  it is about to run, once every earlier call of its reply it conflicts
  with settled (`Waiting`). A running effect's owner keeps `IfStopped`
  current, the answer the kernel gives if it stops the effect first. Every
  settled effect is `EffectRecorded`, with its raw and reviewed outcome. A
  failing Prepare, Gate or Review system closes its stage for that pass.
- `calls`: off-thread work on Bevy's task pools, and `Wake`, which wakes a
  loop that sleeps while nothing happens.
- `tools`, `prompt`: tools and system prompt sections are entities that
  plugins register (`AppToolsExt`, `PromptSection`, for every agent or,
  `SectionOf` one, that agent alone); tools can also come and go while the
  app runs (`WorldToolsExt`).
- `inbox`: a message to an agent is an effect too (`Incoming`, spawned by
  `Deliver`), its text and attachments in its author's order
  (`MessagePart`): plugins edit it in Prepare and take it over in Gate, as
  rig-harness does to run a user's `/` line as a slash command; the kernel
  accepts it at Start, and an agent's accepted, unanswered messages are its
  inbox, saved with the conversation (a `rig.inbox` aside each). A message
  taken back leaves it at once. The messages a turn ended without sending
  are `Unsent`: a plugin keeps one in the inbox or takes it over in Gate,
  and otherwise the kernel hands the user's back to the views (`Recalled`)
  and puts the others in the conversation, in `GateSystems::Fallback`.
- `model`: the agent's catalog model and how it is connected, through
  rig-core's `catalog::Connector`.
- `effects`: the one dispatch path, recorded into the kernel's own buffer
  and written as `EffectRecorded` when each effect settles.
- `journal`, `restore`: the session log, appended to while the session
  runs, and its restore. An
  agent's `Conversation` component is rig-core's conversation (it derefs
  to it): a plugin changes it by calling its methods (push, splice with
  its own `Author`, halt, asides), and each change waits in the
  conversation's outbox until the kernel saves it with rig-memory's
  `Writer`, between the stages of every pass, at the end of the frame and
  before a tool that may change something, and announces it as a
  `ConversationChanged`. `Retire` retires an agent for good. Nothing is rewritten
  except through rig-memory: a `RewriteSession` maps every saved record in
  place, keeping each record's place and stamp, the agents' conversations
  and saved values become the rewritten ones, and `SessionRewritten` tells
  the owners of other state (the effect log, the listing, scratch files).
  A component that says
  `#[reflect(Component, Saved)]`, or a resource that says
  `#[reflect(Resource, Saved)]`, is logged and restored by reflection,
  under its type path (Bevy's `#[type_path]` makes it stable, as the
  model's is `rig::model`, so a program reads it without Bevy); one
  reflected opaquely with serde (`#[reflect(opaque, Component, Saved,
  Clone, Serialize, Deserialize)]`) is saved through serde, the way to
  save rig-core data such as a `MessageId`. The store (`SessionStore`) is
  any rig-memory `Store`; `open_session` reads it strictly, refusing an old
  or broken session before anything is written, and sets a torn last line
  aside (`SessionRestored::repairs`). The work a restart carries on is
  marked `Restored` and gated like any, a turn keeping its index and first
  message from its `rig.turn` record, and the inbox comes back. A tool call
  that may change something is marked in the
  conversation before it starts (a `rig.call` aside), and any call that
  settles while its reply runs on is marked with its result, so a restart
  can tell those from calls that never started: such a call is restored
  `Interrupted`, which no holding gate sees, a plugin may answer or hold
  in Gate or let run again in Prepare, and the kernel answers it in
  `GateSystems::Fallback` once nothing holds it (an error, or the result
  it had). Nothing is decided before plugins run.

`AgentPlugin` adds Bevy's `TimePlugin` when the app has none. The crate has
no features and builds for `wasm32-unknown-unknown`.

The [`rig-harness`](../rig-harness) terminal coding agent is built on it.

This crate replaces an older, unrelated `rig-ecs` (0.43–0.44) on crates.io.
