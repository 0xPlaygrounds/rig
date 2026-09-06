# Extractor usage parity

The batch `batches/extractor-usage.json` pairs 30 recorded tests across Copilot,
DeepSeek, llama.cpp, OpenAI, OpenRouter, and xAI. Each provider contributes five
scenarios and seven extraction calls. These are blocking, zero-retry cases;
they do not prove retry usage accumulation or streaming extraction.

## Independent execution

Original tests run from the immutable baseline. Native counterparts use the
same provider cassette wrappers, models, literal prompts, and neutral Person
and Address schemas and profession validators. `EcsExtractor` constructs native
components and invokes `spawn_run`; it never invokes a legacy extractor or
runner. Each extraction gets a separate run, including repeated calls on the
same helper. The actual native run must settle successfully, contain an actual
submit tool call, expose Usage, and deserialize its RunResult into the shared
type. The existing host bridge enters Tokio on every handler poll while ECS
owns the future. Public stream errors remain failures.

The exact baseline extractor preamble is a Rust string constant, including its
trailing whitespace, and participates in source hashing. llama.cpp retains its
required schema fields, original extra instructions (including the two-newline
append separator), and additional JSON temperature 0.0. The other providers
use their original default preamble and model. All use required tool choice,
submit, the original extraction-specific description, no output augmentation,
zero invalid-call retries with unhandled calls ignored, and an explicit run
limit of one. History messages are converted without dropping unsupported
system messages: the helper rejects those rather than silently changing them.
These recorded histories contain ordinary user messages.

## Preserved assertions

Every original anyhow::ensure expression and compatible-profession validator
is retained in its native counterpart. Original tests only widen visibility
of shared data definitions and validators. The batch rejects other AST changes.

| Scenario in each provider | Required observations |
| --- | --- |
| extract_backward_compatibility | Successful extraction; original John name/age/profession checks |
| extract_with_usage_returns_data_and_usage | Jane data checks; input, output, and total usage all positive |
| extract_with_chat_history_with_usage_works | Original history and exact address fields; positive input and total usage |
| extract_and_extract_with_usage_return_same_data | Two independent successful calls; original Bob/equality/profession checks; second call total usage positive |
| usage_tracking_works_for_different_schemas | Independent Person and Address extraction; each total usage positive; no invented data-field assertions |

Copilot, OpenAI, and llama.cpp require exact John/Jane profession strings and
compare the two Bob profession options with their shared compatibility helper.
DeepSeek, OpenRouter, and xAI preserve their original compatibility-with-expected
profession checks. Case folding, trimming, optional-value requirements, and
bidirectional substring acceptance remain exactly those of each original helper.

Success also includes original `?` propagation and cassette result-wrapper
teardown. Both runners use the same strict ordered HTTP replay with all recorded
interactions consumed. Missing/mismatched requests and unconsumed interactions
fail. This proves the existing provider request/response and typed result/usage
obligations; there is no effect-golden or policy-delivery equality claim.
The helper requires presence of an output call, not terminal uniqueness.

## Native configuration coverage

`OutputToolConfig` supplies optional reserved name/description and preamble
augmentation. Run settings override agent settings. With a schema a reserved
name commits Tool mode; an already minted name survives later configuration
changes. Real-tool collisions fail before dispatch. Without a schema no tool
is created. Description-only configuration retains collision-safe automatic
naming. Scene persistence and replay identity include the configuration.
The native tests exercise actual requests, results, refusal, schedule-boundary
changes, and restoration into a fresh world.

## Evidence limits

`evidence/extractor-usage-report.json` indexes exact original/native run records,
source indices, logs, fixture hashes, and execution outcomes. Execution alone is
not an assertion-review verdict. Ignored live extractor cases remain supplemental
and unexecuted. The 246-pair regression passed after the production change; 24 older mapped
cases lack batch definitions (the 246-pair run also includes 12 shared-provider cells) and remain outside that run. Global feature coverage,
network isolation, full inventory, performance and publication remain programme work.
