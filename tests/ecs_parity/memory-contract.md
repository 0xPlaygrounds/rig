# Memory family contract

Scope: all eleven Anthropic corpus_memory producers using the original scenarios. Native execution uses the original HTTP
cassettes, real provider adapters and real ConversationMemory backends through
MemoryAdapter. Original prompts, bypass history, memory_ops and loaded_lengths
are shared through visibility-only edits. No recorded memory answers or legacy
agent hooks participate in execution.

## Configuration and repeated runs

Sonnet4.6, temperature zero, golden owner, golden-conversation, original preambles
and handler order are preserved. Memory registers before model/tools for owned
programs. Native Remembers/Conversation drive load and append; repeated prompts
use one backend and world, so subsequent loads see actual prior mutations.
Unary helper runs override budget to3, streamed runs to8; history bypass and host
bus retain default1. Serial-two-tools enables serial_per_handler and uses original
AlphaSignal/BetaSignal implementations. Its full append preserves all four
messages and the original effect-family sequence.

Explicit history is converted from the exact original messages and passed to
spawn_run. It bypasses load/append without removing memory from the agent's
required row. Native program identities are stamped for every run, including
both scopes in two-run logs. The comparator now requires each record's scope to
exist in the program map, not merely a nonempty map; a negative control removes
only the first identity and demonstrates rejection.

## Clear policies and response boundaries

AtStart observes the actual successful Load outcome, dispatches a real Clear and
holds Advance until its acknowledgement. AtSettled observes completion of the
Append effect, then dispatches Clear. The native helper awaits that run's actual
Clear result before returning or starting the next prompt. It preserves the
original clear_conversation helper's successful dispatch and Cleared assertion.
Sequential-run scope is explicit: this application run condition is not a general
per-run scheduling primitive for concurrent active runs.

The corpus establishes that append errors are recorded while the prompt still
returns its answer. This corrects the earlier shared consumer's overly strict
success-only acknowledgement check: it now awaits either Appended or an actual
recorded error. It still cannot return while append is pending; the existing
gated-memory control remains applicable. FailingMemory::append_fails is the same
original backend, not a synthetic substitute, and both unary/streamed cases retain
original nonempty-answer, memory-op sequence and MemoryBackend error assertions.
This is not a durability guarantee for successful answers. Native Settled remains
separate from append completion; the consumer explicitly waits for the latter.

## Host-owned bus

The native host owns the App/World and its bus independently of the agent entity.
It registers the model first, then memory, matching the original host registrar.
The original host producer leaves header.bus undeclared, so this test explicitly
sets declare_bus_policy=false for the interoperability stamp. Actual native
Policy remains at its ordinary defaults; no runtime policy is changed to make
header comparison pass.
The consumer waits for append, requires no unfinished PendingEffect before host
teardown, then drops the App. There is no detached legacy driver to await.

## Assertions

All original direct assertions and helper obligations remain, including nonempty
outputs, exact memory operation ordering, first/second loaded lengths, event
retention, required-row inclusion despite bypass, host policy declaration, and
serial append message count. Full original golden comparison retains every
original header/request/outcome/usage/event/tool-output field under the documented
request-shape-contract.md nominal-ID and native-only representation rules. Stable native goldens retain both scopes
and program identities; scheduling-dependent delivery traces are not compared. No new normalization masks memory contents or errors.
