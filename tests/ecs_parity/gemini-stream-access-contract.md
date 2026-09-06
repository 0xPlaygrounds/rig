# Gemini streamed invalid-call access and repair

Baseline: `805fb18e6135c9050ee7ca4295d96ddf08cb223f`. Original source is
`tests/providers/gemini/cassette/agent_run_streamed.rs`; the native counterparts
are in `ecs_agent_run_streamed.rs`. The batch now covers all six original
streamed-run cases: four explicit streamed-machine scenarios and the two
built-in diagnostic cases described in `gemini-stream-diagnostics-contract.md`.

Both paths use Gemini 2.5 Flash, `FORCE_TOOLS_PREAMBLE`, the same public neutral
Add/Sum definitions and strict ordered cassette wrapper. No legacy runner,
assembler or invalid-call policy executes inside native ECS. The native path
uses the actual provider adapter hosted by existing RuntimeHandler, ECS bus
collection and native agent systems. `ToolAccess` changes execution and policy
sets without altering provider advertisements; assembly snapshots it on the
turn before completion dispatch.

## Original obligation closure

`run_streamed_turn` opens the provider stream and expects successful items,
feeds the legacy block assembler, checks streaming context and invalid name,
resolves an actual invalid call, and either continues assembly or drains on
failure. Successful completion requires no pending assembler error, records a
completion call while a real model call is pending, and supplies the finished
turn to the agent machine. Native bus collection assembles real events; the
success helper rejects item errors; a Judge system observes a native InvalidCall
with stream event offset and verifies the run's streaming setting and name.
Failure uses `drain_completion` to keep updating the real bus until the issued
completion has an outcome, corresponding to the original error branch's drain.
A controlled synthetic gated suite separately proves before-EOF timing; the
transport cassette alone does not guarantee scheduler grouping.

- **fails_fast_mid_stream**: preserve required tool choice, only add advertised
  and executable, empty allowed set, prompt and max-turn limit 2. Assert actual
  native UnknownToolCall/add, turn snapshot executable=[add], allowed=[], and
  real assistant-tool history. Original failure's available/allowed fields map
  to that durable turn snapshot; no legacy error is synthesized. Judge observes
  one invalid event, and failure drains the actual completion.
- **repair_continues_the_same_stream**: preserve no explicit tool choice,
  advertisements add+sum, only sum executable/allowed, prompt and max-turn limit
  3. Judge requires add and resolves it to sum. Assert nonempty actual dispatched
  tool effects, all named sum, final output mentions 5, and actual assistant
  history contains no unrepaired add. Neutral assistant_tool_call_names and
  assert_mentions_expected_number preserve the original predicates. Native
  repair is retained until final core identity assembly and real tool dispatch.

The baseline helper's completion-record legality is a runtime state-machine
contract; native equivalents are completion effects parented to actual turns
and materialisation only from those effects. Broader phantom-record and exact
per-call accounting assertions belong to the multi-turn counterpart below,
not an inferred assertion added to the fail/repair originals.

- **skip_abandons_the_turn_and_recovers**: preserve add advertisements and
  execution, no explicit tool choice, first allowed set empty, later allowed
  set add, exact reason/prompt and limit 3. Judge requires the first and only
  restricted call, writes Skip and changes the run policy for subsequent
  turns. The current turn snapshot remains unchanged. Actual subsequent
  completion requests must include abandoned add history and a tool-result
  prompt. Inspect the first retry's synthetic result: generated correlation,
  no provider identity and exact reason. Require nonempty final answer and at
  least two actual completion outcomes. An additional assertion sums actual
  usage including the abandoned turn; it strengthens the original count check.
- **hand_driven_multi_turn_run_completes**: preserve add/subtract advertisements,
  executable defaults, no explicit tool choice, exact arithmetic prompt and
  limit 5. Read real bus text/events and completion outcomes associated with
  actual turn entities. Assert streamed and final output mention 9, at least
  two turns, one completion outcome per cursor turn, total usage equals the
  per-completion sum and is positive, wire block/provider call correlation,
  add+subtract history, and neutral canonical assistant ordering. The original
  phantom-record rejection remains exercised in baseline; ECS has no analogous
  record-completion method. Native fresh-run assertions establish no completion
  effect and zero usage before scheduling. This is an explicit API-contract
  mapping, not a claim that calling a nonexistent native method was tested.

## Evidence and limits

`batches/gemini-stream-access.json` selects exact original/native IDs and fixture
paths. Pair-run artifacts bind results to their own source indexes. A successful
pair result is execution evidence, not by itself an approved semantic verdict.
Final independent family review and manifest integration remain required.
Subsequent source changes must not be attributed to an earlier
pair run. No network-barrier claim follows from credential removal alone.
