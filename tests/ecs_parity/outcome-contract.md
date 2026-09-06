# Outcome family contract

Seven Anthropic corpus_outcome scenarios use native ECS execution with the real
provider adapters and original HTTP cassettes. Original constants and the neutral
tool_outcome assertion helper are shared through visibility-only changes.
FailingAdd and WriteNote remain the original tool implementations. No legacy
agent runner, hook implementation or recorded answer drives the native path.

## Configuration and assertions

All cases preserve Sonnet4.6, owner golden, temperature zero, preamble and prompts,
tool descriptions and registration order, and event recording choices. The
cancel/tool-error runs override max turns to3; exhausted overrides to1; model
errors keep default1. default_max_turns sets both declared DefaultMaxTurns(Some3)
and effective MaxTurns3, with no run override. It reuses effect_corpus/tool_call_turn;
the other six use their corpus_outcome recordings.

Every direct original assertion and complete golden comparison is retained.
Unary tool failure retains the nonempty answer, three-effect sequence, is_error
and BROKEN_ADD output assertion; streaming retains its original nonempty answer,
three effects, event presence and is_error. The shared native success consumer
requires a final RunResult after stream closure and rejects every stream error,
mapping final_output's full EOF drainage, per-item expect and required terminal.

Provider failures are observed as actual native Failure::Provider. Unary retains
ProviderResponse and HTTP401 checks. Streaming counts the recorder's actual error
items, including positions and full reports, and requires exactly one. It does
not fabricate an item from the run's Failure. Additional assertions reject a
successful settlement or RunResult on that failed run. MaxTurns1 is checked as
actual Failure::MaxTurns{limit:1}, with completion/tool records preserved.

## Consumer drop

The application observes bus::Streamed after Collect and before Fold. The first
published ToolName or ToolArguments delta triggers despawn of that issued effect
entity. This drops its native owned stream task; runtime observers record the
cancel and end its owning run. The application does not write a desired outcome,
truncate an event list, inspect a transport inbox, or invoke legacy cancellation.
There is only one run in this application. Original completion-only family and
Cancelled record assertions remain. Additional checks require actual native
Failure::Cancelled and no Settled/RunResult. The full original golden pins the
partial event prefix. Scheduling-dependent delivery batches are not compared.
These fixtures do not prove arbitrary transport scheduling equivalence.

The cancellation case now applies test-owned backpressure after the first real
tool delta. The provider stream remains owned and suspended until the ECS
consumer despawns the dispatch; the gate neither polls later items nor creates
EOF. This removes a CI race where Collect drained an additional empty argument
delta before the consumer could cancel. A releasable synthetic control preserves
every event and error through the boundary and resumes the unchanged tail;
another control verifies that cancelling the paused stream drops its provider.
The request, provider capabilities, golden assertions and recorded traffic are
unchanged. This is cancellation under controlled backpressure, not a guarantee
of identical prefixes under unrestricted producer scheduling.

## Full log comparison

The existing nominal dispatch-ID mapping now also maps header.stream_errors keys
through the same bijection. References outside the recorded effects fail. Error
item positions and all report fields remain unchanged and compare exactly.
Separate negative controls change an item's position and attach it to a missing
effect, requiring rejection. Native scope identities and delivery normalization
follow the existing request-shape contract; no new payload normalization exists.

The comprehensive cancellation matrix, network isolation and exhaustive
functional-superset comparison remain outside this family.
