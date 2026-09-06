# Gemini streamed failure diagnostics

The two `gemini-stream-diagnostics` batch cells cover the original built-in
streaming max-turn and tool-dispatch cancellation cases. Four other cases in
`agent_run_streamed.rs` remain unported, including three requiring early
invalid-call decisions. They must not inherit these verdicts.

## Configuration and observations

Both use the original GEMINI_2_5_FLASH, FORCE_TOOLS_PREAMBLE, Add implementation
and schema, required tool choice, unset default turn override (effective one),
and actual per-run max_turns 2. Temperature remains unset. Native execution
uses EcsAgent's real adapters and systems, but calls wait_for_outcome rather
than a success-only wait.

Native Failure maps the error category and budget/reason. Diagnostic messages
come from actual ordered Utterances belonging to the run. No legacy PromptError
is constructed to supply otherwise absent diagnostics.

For max-turn failure, advance refuses a new turn after land_batch has appended
the pending tool-result message. The last actual utterance is therefore the
pending prompt; preceding utterances are diagnostic chat history. The test
retains the portable validator's exact limit, nonempty history, nonempty user
prompt predicates, plus original tool-result prompt and assistant-add checks.

For cancellation, an application system in the bus Gate set observes the
unissued tool effect and writes native Cancelled with the original reason.
The assistant turn has already materialised. The test retains the portable
validator's exact reason and assistant-add history predicates, the original
reason substring check, and a schedule observer that latches any final
publication. No final response is allowed. This intervention changes actual
execution before tool dispatch; it does not merely relabel a successful run.

The original loops stop at the expected prompt error and reject other errors;
they do not require draining the remainder after that error. Both paths retain
the original strict ordered cassette wrapper and consumption teardown.

## Historical timing gap and current production work

`crates/rig-ecs/tests/run_stream_boundary.rs` began as a failing regression
for a required native behavior. It publishes a tool block start and invalid
name, then waits on a live oneshot gate. The test observes the delivered name,
allows a subsequent full schedule pass, and requires zero EffectOutcome before
checking that invalid policy can act. Its failure is not a timeout and cannot
be explained by the provider having already finished.

At that evidence revision, fold exposed only text before outcome. Materialise created
InvalidCall after Outputs.done. `partial_turn_at` reconstructs diagnostics from
completed content and retained events; that cannot establish early intervention.
Injecting an early repair alone is also insufficient: final folding would
overwrite the edit. Skip needs an actual retained prefix and abandoned usage
accounting. A production fix and further timing/identity/history tests remain
required; this two-case batch does not resolve that gap.

The subsequent native implementation publishes early invalid names, retains
their delivered prefixes and resolves final identities through core assembly.
Six gated regressions now cover early policy access, persistent repair, skip
prefix/usage, exhausted retry, later failure after repair, and Ignore with block
reuse. These are synthetic native evidence, not provider cassette coverage.
The original failure artifacts remain historical evidence. The four remaining
Gemini streamed-run scenarios still need native counterparts, including an
independent allowed-tool set. This batch's historical paired results do not
verify the later production edits; affected provider batches must be rerun.
