# Host custom-effect family contract

Ten corpus_host Anthropic scenarios use native ECS execution, real provider
adapters and the original neutral Note, NoteAck, NoteTaker and Adder implementations.
Original prompts, note_ats helper and all assertion tails remain. No AgentHook,
AgentBuilder, legacy driver or recorded answer executes on the native path.

## Host ownership and custom effects

The App owns its bus/registry independently of the agent entity. It registers
Sonnet4.6 under golden/model:default, then host/note when served, then Adder when
requested. Original serial policy, event choice, temperature0, preamble, prompt,
owner golden and run max3 remain. Host policy is not agent-owned header metadata:
declare_bus_policy=false maps original header.bus=None without altering actual
native Policy. Full original golden comparison retains handlers and ordered
semantic hook names; PolicyVersion names the native composition without fingerprinting implementation code.

A NotePending component identifies each actual custom-effect entity and its
expected acknowledgement. Payload is serialized from the original Note type;
NoteTaker executes through the normal native task-owned Serve bridge. Checks decode
the actual Outcome::Custom payload as NoteAck. Single take_note equivalents require
successful acknowledgement, accepted=true and matching at. Twice checks successful
acknowledgements and each associated at (the original does not separately assert
accepted there); full golden comparison preserves all original payload fields.

## Lifecycle and awaited acknowledgements

AtStart uses actual On<Add,Run>. Twice creates both requests together in original
first/second order; both acknowledgements are required before continuation. Serial
and concurrent cases set actual native policy, but original assertions do not prove
overlap or reverse completion.

AtCompletionCall observes each fresh turn after Select/before Assemble, creates
one note and marks that turn to prevent duplicate notes while waiting. This places
the awaited note before the completion effect is created; the original callback's
prepared-request data is unused by these tests, so no request-observation claim is
made for this mapping. AtOutcome observes actual tool EffectOutcome publication and
creates a run-owned note before tool result materialisation/next completion.
AtSettled observes actual On<Add,Settled>, then the consumer waits for its note's
acknowledgement before returning the response or dropping the host.

The ordinary run conditions gate Advance, Assemble and Materialise while notes
remain unanswered. They do not gate the bus, which must continue serving notes.
These gates intentionally apply to a single-active-run application, not an
arbitrary concurrent multi-agent world. Ready outcomes are checked rather than
merely counted. The consumer waits for actual native success/stream closure,
rejects stream item errors and requires a final RunResult. It then awaits any
settled note and checks that every retained effect has finished before host drop.
This preserves original final_output EOF/expect/terminal and driver-join obligations
without a detached legacy driver. Deadlines fail tests rather than count as success.

## Unserved note mapping

The original startup hook synchronously fails bind with HandlerUnavailable and
never dispatches. Native ECS has no equivalent borrowed bind operation here: it
submits an intent to the absent key, and the native dispatcher refuses it before
issuance. The application observes the actual HandlerUnavailable, requires no
Issued component, and gates agent continuation until the refusal is checked.
No handler runs and no custom effect record exists; full original golden contains
only the completion and NoteUnserved declaration. This is an explicit API/timing
mapping, not a claim that an unissued native intent and a failed legacy bind are
identical internal operations. No expected refusal is synthesized in the test.

## Evidence and controls

All direct and helper assertions remain, including note_ats kind/at checks, real
acknowledgements inside original hook methods, answer42 for tool runs, exact effect
families, event retention, unserved error and complete original logs. No new golden normalization is introduced.

A separate synthetic control replaces the note handler with a gated wrapper around
the real NoteTaker. Across six policy configurations (startup, completion-call,
outcome, settled, combined startup/settled and twice), it holds acknowledgements
and keeps polling the same native consumer. The response must remain pending and
an actual Issued observer counts model dispatches: zero before startup/completion,
one after a tool outcome or settlement. Releasing the handler then permits the
expected response. This tests continuation timing independently of fast cassette
responses; it is separately counted from the ten provider cells.
The combined startup/settled configuration releases two permits together; its
second acknowledgement is independently held only by the standalone settled
configuration.

Scope remains default root features/native host/replay. These cases do not establish complete feature coverage, network isolation,
arbitrary multi-run scheduling or interruption/capability guarantees.

RuntimeHandler enters the supplied Tokio runtime for initial handler polls and returned stream polls. The host keeps ticking on Pending; stream work is owned by its effect through EOF.
