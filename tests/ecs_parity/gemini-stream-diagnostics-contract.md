# Gemini streamed failure diagnostics

The native streamed max-turn and tool-dispatch cancellation scenarios preserve
the diagnostic observations below. The other streamed-run cases are described
in `gemini-stream-access-contract.md`.

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

Synthetic gated stream-boundary tests separately check policy intervention before
EOF, persistent repair, skipped prefixes and usage, retries, later failures and
block reuse. Cassette scheduling alone does not establish before-EOF timing.
