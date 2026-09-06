# OpenAI Chat tool-lifecycle matrix

Scope: all24 original registrations at baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f, default root features/native host.
Twelve Agent cells migrate to native ECS. Twelve Model cells remain unchanged
shared provider coverage, executed in both revisions but credited zero migrated
agent cases. `batches/chat-tools.json` and `evidence/chat-tools-report.json` retain
that full24-cell pair. No original effect golden exists for this family.

Both transports, gpt-4o-mini/gpt-4.1-mini and Zero/Nested/Parallel shapes retain
original literal fixtures, Chat Completions route, preamble/prompts, unset
temperature, max_tokens128, required tool choice and parallel_tool_calls only
for Parallel. Original schemas remain shared. Agent default_max_turns1 remains
explicit; streamed agent also sets run max1 with empty history, blocking has no
run override. Bus/task-pool/tool concurrency defaults are unchanged.

The native producer owns App/plugins and uses the real provider adapter and
ToolAdapter. Four tiny typed tools duplicate the original neutral macro's exact
argument types, note-to-shared-log call and name result. They reuse the original
tool_definition helper, without changing original macro visibility or executing
original run_agent/AgentBuilder/agent runner. Invocation order is the actual order
of Tool::call entering its mutex-protected log, not reordered after the fact.

Original shared execute/assert_cell runs unchanged on the native Observation:
request model/cap/absence of max_completion_tokens/choice/tool count/parallel and
stream flags; recorded finish tool_calls; ordered wire names, nonempty distinct
IDs, parsed exact JSON arguments (empty, Unicode/nested arrays, or parallel red
and blue); actual agent exact-once invocation names/order and absence of
ProviderResponseError in collected errors. Wire streaming fragments are
reassembled by the original helper. These wire checks are not misrepresented as
original agent runtime normalized-data assertions. Model cells separately assert
actual normalized finish/names/IDs/arguments/errors through original run_model.

The agent fixtures intentionally exhaust their one-turn budget after executing
required tools. Native wait_for_outcome must return actual MaxTurns{limit:1};
public Streamed.errors must be empty. The actual native diagnostic populates
Observation.errors. This stronger native failure-kind check does not invent a
legacy error string or relabel the run as a successful final answer. Native
supplemental graph checks inspect actual ToolCallSlot IDs/names and PendingEffect
arguments in call-index order; they do not replace the original invocation log.

Only sibling visibility of original enums, structs/fields, constants/functions
changes; all bodies/variants/schema/macro tokens remain frozen. The batch guard
now allows pub(super) enum visibility alongside existing const/fn/struct rules.
Its new control permits only that visibility and rejects changed variants,
payload types, attributes, discriminants and broader visibility. Existing guard
controls preserve all other source assertions and fixtures. The change does not
allow arbitrary original-source edits or broaden visibility normalization into
macro bodies.

Helper source anchors retain whole selected shared functions. assert_cell's
common branch plus Agent or Model branch is the applicable obligation, not every
branch for every cell. Only selected transport helpers are in each call chain.
Default-feature passing is scoped separately from the inventory's other feature
obligations. Source/compiled/full-feature inventory, network/interruption/WASM,
capability comparisons/performance/aggregate and publication/CI remain required.
Observed parallel start order is not a proof for arbitrary task scheduling.
No production rig-ecs runtime change, provider call or cassette recapture needed.
