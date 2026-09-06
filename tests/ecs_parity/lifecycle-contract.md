# Native lifecycle provider parity

Scope: fifteen lifecycle_matrix cells (five each Anthropic/OpenAI/Gemini) at
805fb18e6135c9050ee7ca4295d96ddf08cb223f, default root features/native host.
Original producer/support files are unchanged. `batches/lifecycle.json` freezes
fifteen genuine provider fixtures, pairs exact IDs and preserves source identity.
These cells have no original effect golden; the oracle is strict original
provider cassette matching/exhaustion plus every original direct/helper assertion.

Each literal original wrapper still constructs BoxedHttpClient with the original
WireProbe middleware. The actual Anthropic Sonnet4.6, OpenAI GPT4O and Gemini2.5Flash
adapters execute through RuntimeHandler. Temperature remains unset. Original
undeclared effective one-turn default is preserved; tool cases override max3 on
the run. Original preambles/prompts and Adder/schema are unchanged. Native test
owner `parity` replaces unasserted original generated identity. Native recording
is debug observation and does not claim equivalence to an original effect log.

| Case per provider | Retained obligations |
| --- | --- |
| middleware_unary | Successful nonempty text; actual shared WireProbe asserts one header/body/response phase, status200 and body length>0. Hidden body middleware assertion requires the earlier header mutation. |
| middleware_streaming | Successful final text, actual last typed provider Final, usage.total_tokens>0; same single-exchange assertions; one native start and one actual response settlement. |
| run_start_rewrite | Actual startup system replaces sole prompt Utterance before Advance; provider sees the PINEAPPLE request and answer contains marker; one start and one response settlement. |
| entry_log_order | Actual streamed add9+16, final answer25 and typed provider final. Run-owned phase entries project into unchanged original EntryLogProbe::assert_phases: start0, then every consecutive one-based completion turn, at least two calls, append order preserved. |
| run_settled_tool_run | Actual unary add7+15, final answer22, one start and response settlement; exported last completion counter exists and is >=2. |

The shared native helper is ordinary application machinery, not an agent engine:
start queries Added<Run> before Advance, asserts Cursor0 and creates Entries on
that run; completion observes actual model PendingEffects after Assemble/before
Patch, follows their Turn parent to the run, and appends either phase or counter
using the actual Cursor. The counter reads the preceding stored snapshot, not a
separate reconstructed count. Settlement observes Added<Settled> with actual
RunResult and exports the run's stored entries after a serde roundtrip.
Original AgentHook methods never execute. Only neutral RunEntry data is used when
projecting observations into the original assertion helper.

This maps rig-agent's built-in append log to explicit native application-owned
serializable Entries and three systems. That host code is part of the comparison,
not evidence that ECS has an identical built-in scratchpad API. The JSON roundtrip
proves entry data serialization, not fresh-world continuation or registration with
native Scene. Successful single-run no-memory/no-history cases only; failure
settlement, repeated runs, memory-load startup ordering and concurrent shared
probes remain separate obligations. The startup rewrite asserts exactly one
utterance rather than silently rewriting arbitrary history.

The shared EcsAgent consumer awaits native success/EOF and rejects every stream
item error. The typed provider final comes from actual Streamed.events, selecting
the last by dispatch Seq as the original collector selects the last delivered
provider Final. Only middleware_streaming requires usage>0; the entry-log case
requires Final existence. WireProbe's named test asserts final phase counts; it
does not itself timestamp response-vs-consumption, so no extra timing guarantee
is claimed from that test name.

`evidence/lifecycle-turn-mutation.json` temporarily increments completion-entry
turn stamps. One real Anthropic entry-log replay must fail the unchanged original
per-entry assertion; exact source bytes are restored before final paired replay.
This protects the shared three-provider observer without claiming independent
mutation execution for each provider. Existing stream-error controls also apply.

No provider calls/recapture or production runtime changes are required. Full
inventory/feature/WASM/network/interruption/capability/performance/aggregate and
publication/CI obligations remain unfinished.
