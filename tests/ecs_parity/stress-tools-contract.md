# Gemini tool stress parity

Scope: all six hook_stress_tools cases at baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f, root default features, native host.
The batch freezes original producers, support, tools and six provider cassettes.
Original source and fixtures are unchanged. There are no original effect goldens
for these cases: strict requests, interaction exhaustion and original assertions
are the oracle. Thirteen other Gemini stress cases remain outside this batch.

Native execution uses the real Gemini2.5Flash adapter, original tools and counters,
original prompts/preambles, owner stress-agent, temperature0, empty starting
history, undeclared effective default1 and per-run limit4 (recovery limit5).
Blocking success requires actual settlement; cancellation requires actual native
Failure::Cancelled. Original tools retain their schemas, output conversion and
CodewordLookup error mapping with explicit model feedback.

Separate ordered DispatchSlot systems run in native BusSet::Gate. Each SetArg
system parses the actual pending tool arguments, changes one key and preserves
the rest. A later observer writes actual name/arguments to the original neutral
ToolEventRecorder data, whose getters and assertions are reused. No original
AgentHook method executes. New-intent filtering prevents duplicate observations.
These cases have one tool per turn; no arbitrary concurrent recorder ordering
claim is made.

Ordered OutcomeSlot systems run in native BusSet::Judge before agent Fold.
Each reads the current actual ToolResult, preserves its non-output fields and
changes only output text. Replace then Wrap therefore composes; Truncate counts
Unicode characters. A later observer records actual outcome text. Original
argument JSON/object/nonempty checks, exact composed arguments/result15, response
contains/excludes assertions, helper expectations and execution counters remain.
The original single-key case asserts that y exists, not that y equals its prior
value. The original redaction validator receives literal true for secret
production; this remains literal and is not promoted to an observed assertion.
Strict outbound request matching independently covers the rewritten tool result.

Termination observes a newly published outcome of the named actual tool and
inserts Cancelled on its owning run. The original counter proves execution came
first. Native Failure report.message and actual run-owned Utterance/Parts ordered
by Order are projected to PromptError only to call the original neutral validator.
The validator checks exact reason AND retained assistant add call; the original
reason match is also preserved. Neither expected reason nor expected history is
an input to that projection. This maps observable diagnostics; it does not claim
identical public error types.

Recovery uses no intervention: the real CodewordLookup rejects the red team with
its original corrective model feedback, then the provider calls again for blue.
Original actual call count>=2 and case-insensitive recovered codeword assertions
remain. This is a model-driven extra turn, not policy retry of a tool-bearing turn.

The application systems are scoped to these single-active-run test Apps. They
are not a compatibility runtime or a general HookStack replacement. No streaming,
invalid-call or general cancellation/drain guarantees are inferred from these six
blocking cases. No production ECS changes, paid calls or cassette recaptures.

Negative control: remove retained utterances from the cancellation projection.
The original shared cancellation validator must reject the missing assistant
call despite an otherwise correct reason and execution count. Restore exact
source before final paired replay; retain the failed run and hashes as evidence.
