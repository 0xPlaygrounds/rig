# Recorded extraction and provider validation

`batches/extractor-smoke.json` pairs ten original/native cases across nine
providers: Gemini contributes two; OpenAI, llama.cpp, Copilot, DeepSeek, xAI,
OpenRouter, Doubleword, and Venice contribute one each. Original fixtures,
provider wrappers and assertions remain shared. Only Gemini's Person type and
fields widen to pub(super); other original files remain unchanged.

## Native execution and configuration

All counterparts use the existing native EcsExtractor described in
`extractor-usage-contract.md`. Every call creates an independent one-turn run,
uses the real provider adapter, and requires actual submit output, successful
native settlement, deserialization, and reported run Usage. This helper imports
no legacy extractor/runner. The four `rig_agent::test_utils` imports are the
same pure field validator already used by the baseline, not agent execution.

Original model expressions, EXTRACTOR_TEXT, SmokePerson schema, and all original
success `expect` calls remain. SmokePerson has required nullable schema fields.
Gemini's second case uses its separate Person schema with optional fields and
its original John Doe prompt. Both Gemini cases preserve typed
AdditionalParameters/GenerationConfig construction and serde_json serialization
`expect` boundaries. All cases retain the default extractor preamble and submit
configuration, without additional instructions or output augmentation.

## Assertion obligations

| Cases | Preserved observations |
| --- | --- |
| Copilot, llama.cpp, xAI, OpenRouter | All three Option fields must exist, each is trim-nonempty, total_tokens > 0 |
| DeepSeek | All three fields must exist and be trim-nonempty; no original usage assertion |
| OpenAI | Portable validator plus trim-nonempty on each field and total_tokens > 0 |
| Gemini smoke | Portable validator plus each field's presence and trim-nonempty; no separate total_tokens requirement |
| Doubleword, Venice | Portable validator only after successful extraction |
| Gemini additional parameters | Exact John/Doe first/last names; job.unwrap_or_default is trim-nonempty |

`validate_extraction_fields` requires case-insensitive Ada and Lovelace, a
case-insensitive mathematician substring in job, and Usage::has_values(). The
last condition means Usage differs from Usage::new(); it does not require every
counter or the total counter to be positive. Its error-returning condition is
an assertion obligation despite containing no assert macro. The native cases
call the exact same validator with actual observed data. No expected response
is supplied to execution.

Both runtimes use identical strict ordered cassette wrappers. Successful test
closures await finish_after_test and ReplayServer::assert_consumed; a mismatch,
miss, error, or unconsumed interaction fails. Full option/expect/helper/teardown
closure is reviewed separately from direct macro source anchors.

## Evidence and limits

`evidence/extractor-smoke-report.json` retains exact original/native IDs, run
results, source identity, fixture hashes and raw logs. Shared helper usage-loss
fault injection from `evidence/extractor-usage-mutation.json` remains applicable
to the unchanged helper; it is not a new mutation execution for these ten cases.
These scenarios establish their original provider request, typed result and
usage obligations. They do not establish extraction retry, streaming, effect-log
equality, or policy timing fidelity. Ignored/live cases and the full programme
remain separate work.
