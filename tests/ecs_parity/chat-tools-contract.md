# OpenAI Chat tool-lifecycle matrix

Scope: 24 original registrations. Twelve agent cases have native counterparts;
twelve model cases remain shared provider coverage and count as zero migrated
agent cases. No original effect golden exists for this family.

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

Helper source anchors retain whole selected shared functions. assert_cell's
common branch plus Agent or Model branch is the applicable obligation, not every
branch for every cell. Only selected transport helpers are in each call chain.
Observed parallel start order does not establish arbitrary task scheduling
guarantees. The shared wire assertions remain distinct from actual runtime
invocation assertions.
