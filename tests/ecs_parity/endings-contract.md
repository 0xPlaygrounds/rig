# Hook-ended runs contract

Ten Anthropic corpus_endings scenarios execute through native provider adapters,
agent systems and ordinary application policies. Original neutral inputs,
last_outcome_kind, real Adder/WriteNote implementations and assertion tails remain.
Original hook implementations are read as contracts and never execute natively.

## Policies and timing

CancelAddDispatch observes unissued add work in BusSet::Gate and writes Cancelled
on its actual owning run. Native cancellation prunes that work before dispatch,
so only the completion is recorded. CancelAddOutcome requires an actual add tool
result in BusSet::Judge; CancelAnswer requires an actual completion without tool
calls there. Both stop the native run after recording its real successful effect
outcome, preserving the original producer's record-versus-policy distinction.

StopAfterTurn observes completed Outputs after Fold and before Judge and stops
unconditionally. StopAtAnswer stops only when completed content has no tool call.
They preserve the original completed-turn predicates and prevent materialisation
of the cancelled turn. Tool-bearing prior turns remain in the record and graph.

Delta policies observe published bus::Streamed after Collect and before Fold.
They stop on the first published Text or ToolName/ToolArguments delta respectively.
The policy first writes Cancelled with the exact original reason and flushes
native observers, then despawns the issued effect to drop its owned stream task.
This ordering matters: run cancellation itself leaves issued effects to handlers;
consumer/task cancellation produces the cancelled effect record. A direct check
requires the native Failed reason before effect drop. No input inbox is inspected,
no expected prefix selects execution, and no recorded event or outcome is patched.
Full original goldens compare the exact partial event prefix for both fixtures.
The tool-delta fixture shares the outcome corpus's test-owned FirstToolDelta
backpressure gate: the real first tool delta is delivered, then the provider
stream stays owned until the native policy cancels it. No events are filtered
or rewritten. Exact prefix equality is scoped to this controlled boundary.
The application has one run; arbitrary transport scheduling is not proven.

The Ending enum selects concrete system installation only. It is neither an
agent hook dispatcher nor an effect interpreter. PolicyVersion names native
policy composition; original semantic hook names and RecordSettled are explicitly
declared for legacy header interoperability. A policy name alone is not proof of its behavior.

## Terminal observations and collectors

Actual On<Add,Failed> and On<Add,Settled> observers populate a run-keyed observation
map. The original RecordSettled assertion requires an observed error prefix; this
is asserted on that map, independently of the run wait result. No expected reason
is used to populate the observation. The actual native Failure must separately be
Cancelled with the exact original reason. Any other failure is rejected. Native
Settled and RunResult must be absent, preserving streamed_cancel's rejection of a
successful final response. No terminal-uniqueness claim is attributed to the
original collector, which selects the last cancellation and does not count it.

All retained native effects must have EffectOutcome before returning; completed
streams acquire it only at channel closure. Delta-cancelled effects are despawned
and their native task is dropped. Recorded provider stream errors must be absent,
preserving the original rejection of non-prompt stream errors. The original
streamed helper's yield64 and subsequent terminal observation are retained in
meaning: terminal observation is required before return; yields follow actual
termination. A timeout is a failure, not a cancellation result.

## Inputs and assertions

Owner golden, Sonnet4.6, temperature zero, original prompts/preambles and tools,
handler registration order and recorder event choices remain unchanged. Unary
helper and streamed programs override max turns to3; answer_outcome_cancelled
retains default1 with no override. Original helper thinking flag is always false
in these ten cells; its budget1024/unset-temperature branch is preserved but is
not counted as an executed variation. text_delta_stop reuses the original
effect_corpus/cancelled_stream fixture; other cells use corpus_endings fixtures.

All direct and helper assertions remain: expected cancellation and exact reason,
required RecordSettled error observation, exact effect families, successful real
outcome records where required, cancelled partial-stream records, and full original
goldens. Neutral helper expect/unwrap obligations and original wrapper matching,
interaction exhaustion and teardown are retained and independently reviewed.
Comparison uses the existing full-log contract; no additional normalization.
Native goldens keep scopes/program identities; delivery grouping is not compared.

Network isolation, supplemental empty-effect endings and comprehensive
interruption guarantees remain outside these scenarios.
