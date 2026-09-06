# Gemini main stress parity

Scope: all seven executable hook_stress tests at baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f, root default features, native host.
Six genuine provider cassettes and the original gemini_tool_call_turns effect
golden are frozen. The golden producer and streaming taxonomy test share one
cassette but remain separate executed IDs. Original producer/support bodies are
unchanged; copied neutral constants retain their exact original string values.

Ordinary cases use real Gemini2.5Flash and original CountingAdd/CountingSubtract,
their schemas, counters and output conversion. Name stress-agent, preambles,
prompts, tool order, empty starting history and run limits4/5/6 are retained.
DefaultMaxTurns remains undeclared/effective1. Temperature0 is set on the builder
except the context case, where it remains unset until the per-turn patch sets0.
No original AgentHook methods or agent orchestration execute.

LifecycleRecorder maps to the shared native EventTap, whose breadcrumbs read
actual run Entity identity, Cursor, stream-mode component and configured name.
Actual tool issuance increments run-owned Tally; a separate reader samples it at
model-turn completion. Original stable identity/name/mode, nondecreasing turn
indices reaching>=2, exact call/result pairing and actual-counter equality,
both-tools-execute, nonempty/monotone/final-tally assertions remain. Per-turn
pairing uses actual observed breadcrumbs, not expected event counts.

This batch improves the shared context observer: CompletionResponse now observes
published successful outcomes after bus Collect/before bus Judge; a separate
ModelTurnFinished system observes actual done Outputs after agent Fold/before
agent Judge and marks the turn once. It also exposes read-only breadcrumbs and
an explicit stream-mode argument. All twelve earlier context/patch cases receive
paired revalidation in stress-main-context-regression; their historical evidence
is retained rather than rewritten. These are valid accepted turns, not a proof
of invalid-call/retry lifecycle timing.

Context steering installs a native RequestPatch with the exact vault Document,
active_tools=[add] and temperature0 on every Fresh turn. Original vault code in
the answer, subtractcount0 and addcount>=1 remain. The chained-tool case uses two
ordered Gate systems (replace whole args, then independent observation) and two
ordered Judge systems (observe raw result, then redact output). The original
neutral recorder stores actual pending args/outcome text. Callslen1, valid JSON,
exact{x7,y8}, resultslen1/raw15 and final marker present/raw15 absent all remain.
Skip supplies actual structured ToolResult::skipped before issuance; original
subtractcount0, addcount>=1 and nonempty response remain. Strict outbound requests
cover the applied context, tool restrictions, rewritten arguments/results and
skip reason through the original cassette wrapper and exhaustion checks.

The streamed taxonomy records actual published text/tool deltas and new tool
intents. Execution commits are observed from the real user utterance written by
native land_batch after all tool results succeed; they are not inferred merely
from Issued. Each result is correlated with the preceding native Turn by Order,
then its ToolCallSlot ID within that turn, and a real issued slot/outcome. Gemini
generated block IDs can repeat across turns: an initial run-wide lookup failed
on this corpus and was corrected to retain turn scope. Correlation guards are
supplemental native checks, not invented original assertions.

The observer emits the execution-commit tag for an issued slot, then its actual
committed result tag; Settled/RunResult independently supplies final_response.
The original trace assertions remain: all required tags exist and first
tool_call < execution_committed <= tool_result < final_response. Full drainage
and rejection of every stream error (including after Final), final presence,
nonempty answer, actual identity/name/mode and tool counters are preserved.
This is application-visible native event observation, not a legacy stream facade
or identical transport grouping claim. Output-tool settlement in the same batch,
arbitrary simultaneous runs, aborted batches and interruption remain separate.

The effect-golden producer configures native registration at its source:
stress-agent/model:default, tool:add#0 and tool:subtract#1, matching the original
registration order, owner, default bus policy and builder fingerprint. It uses
RuntimeHandler to enter Tokio on each poll while ECS owns the actual future.
No recorded answers or old agent runner are used. Original record_effects does
not retain stream events, and its collector ignores item errors while requiring
a final response; the named native host preserves that limited collector claim.
Original nonempty tool-ID list and every ID is_generated assertions remain.

The unchanged shared full comparator now also compiles in the Gemini target.
It preserves every original request/outcome/handler/fingerprint/error/event field
under its existing nominal effect-ID and native scope/program/delivery comparison
rules. Complete native logs retain scope/program/delivery metadata separately,
and the ECS golden is generated only by this real producer after original full
comparison passes. Original effect golden and transport cassettes are unchanged.

Negative control surfaces the actual committed tool result before its execution
commit. Exactly the original ordering assertion must reject this fault; exact
runtime source is restored before paired replay. This is one ordering control,
not a universal concurrency/interrupt/delivery proof. Earlier context tally and
stream delta controls remain source-bound historical evidence.

No production ECS change, paid provider calls or cassette recapture. All31 Gemini
stress cases now have native implementations, subject to their respective scoped
execution/review records. Full inventory/feature matrix, capability/supplemental,
network/interruption/performance/aggregate/review/publication and CI remain open.
