# Hook family contract

Scope: all eleven Anthropic corpus_hooks scenarios. Native tests share original prompts and
request/tool-record/transcript assertion helpers via visibility-only edits. They
use real provider and tool adapters, ordinary native systems and unchanged
original HTTP cassettes. No legacy hook executes in the native runner.

## Policy boundaries and configuration

All cases preserve Sonnet4.6, temperature zero, owner golden, original preamble,
Adder configuration and recording mode. Agent default budget is None/effective1;
all runs override to3 except replace_answer and preamble_override, which retain1.
Streamed patch/deny cases retain events and the original EOF/error/final-response
obligations through the native success consumer.

Dispatch patches act in BusSet::Gate before Issued, preserving the model's original
call in history while changing actual dispatch arguments. Denial supplies the
native Denied outcome before dispatch: no tool effect record, and the same denial
text in the next request. Result and completion replacements act in BusSet::Judge,
after recorder publication and before native folding/materialisation. Thus raw
records retain original answers while the transcript/final answer changes.
Preamble patches act on Fresh turns after Select/before Assemble; agent Preamble
remains unchanged. DemandDone reads completed Outputs after Fold/before Judge and
inserts native Retry with the exact original feedback when text lacks DONE.
The two-policy stack keeps both original semantic names in registration order.

ObserveEverything registers real InMemoryConversationMemory through MemoryAdapter
before model/tool registration, matching the original descriptor table order.
Native Remembers/Conversation drive real load and append effects. An observation
system accepts every dispatch family without mutation, and the test additionally
compares observed families to the complete actual effect log.

LookupBeforeRun installs an On<Add,Run> observer that issues the real add(1,2)
effect under the original key, ChildOf the run. A native schedule run condition
holds Advance until the actual effect outcome arrives. It preserves the hidden
original hook assertion: the actual tool result must render as "3" before any
model turn advances. This is a single-run application policy; the condition is
not claimed as per-run gating for arbitrary multi-run worlds. It does not await
an invented answer, replay an effect, or create a second agent interpreter.

Semantic hook names accompany installed native systems in stamp_header. Native
PolicyVersion identifies the ordered policy declaration. Builder identity is computed from the
actual graph; it is not claimed to hash application code.

## Memory response boundary

Native Settled publishes an answer before memory append must have acknowledged.
The legacy response waits for append. The parity consumer therefore also waits
for the actual run-owned MemoryOp::Append outcome, requires exactly one append,
and retains its record before returning. The subsequent memory family establishes
that an append error also permits the legacy answer; the consumer accepts that
recorded terminal error, while these successful-backend hook cases receive Appended.
It neither changes native phases nor manufactures outcomes. A gated synthetic
memory handler demonstrates that an unreleased append keeps the response pending
and that release permits success. These are consumer
boundary controls, not a claim that native Settled itself means durable append.

## Assertions and evidence

Every original explicit assertion and helper obligation remains, including
original call arguments versus patched dispatch, original tool result versus
replacement transcript, original model text versus replacement answer, the
unchanged builder preamble and the awaited startup result. Original full goldens
compare with the unchanged request-shape-contract.md normalization: nominal effect
IDs/parents and native-only scope/program/delivery representation only. No original
request, outcome, usage, event, header or causal relationship is removed.
Native stable goldens retain scoped identity; scheduling-dependent poll delivery
traces are excluded from stable equality.
