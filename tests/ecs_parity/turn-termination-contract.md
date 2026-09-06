# Turn-termination family contract

Scope: eight existing scenarios for each of Anthropic, OpenAI Chat Completions,
and Gemini. The catalog lists original/native correspondences and fixtures.

## Independent execution and observation

The original tests keep their runners and hooks. Native tests use the existing
ECS model/tool adapters, schedule, request assembly, fold, judgment, materialization,
and settlement systems. The new shared test observer reads completed `Outputs`
and the child completion effect's actual request and outcome after `Fold`, before
`Judge`. It applies the core finish-reason reconciliation with actual tool content;
this matters for Gemini's wire `STOP` on a tool turn. It never reads the legacy
probe or a saved effect log to produce a native observation.

The cap-escalating policy is the application policy from the original test,
expressed through existing native extension points: a per-turn `RequestPatch`
after `Advance`, before `Assemble`, and one tool-free truncation `Retry` before
`Judge`. It does not assemble requests, execute retries, or run tools itself.
The probe records the rejected attempt before the retry decision. Each test has
one fresh app/run; these resources do not claim multi-run isolation.

## Preserved obligations

Every original post-execution assertion and fixture-premise helper invocation is
retained in the corresponding native test. The original source files change only
sibling visibility and formatting. Provider constants and fixture readers are
shared by import; their bodies and string values are unchanged. Native tests
retain literal wrapper calls, so the cassette registration scanner can discover
them. Existing wrapper request matching and exhaustion still apply.

| Cases per provider | Native observable | Retained fixture checks |
| --- | --- | --- |
| Two truncated cases | First `Length`, actual tiny cap, original portable predicate where asserted | Original wire reason and request cap |
| Two completed cases | First `Stop`, actual roomy cap, original non-truncation predicate where asserted | Original wire reason checks |
| Two tool cases | First `ToolCalls`, actual roomy cap, original non-truncation predicate where asserted | Original provider wire tool reason |
| Two escalating cases | Ordered `(Length, tiny)` then `(Stop, roomy)`; one grown cap; retry count where originally asserted | Original ordered request caps and ordered reasons where asserted |

The tests intentionally preserve differences between blocking and streaming
assertions; no inferred assertion is substituted for an original one. Native
settlement additionally requires a successful result and no recorded stream
errors, including errors after a provider final. This is stronger than originals
that discard the stream collector's returned error.

## Configuration

All cases use temperature zero. Anthropic uses Haiku 4.5 with tiny cap 3;
OpenAI uses the Chat Completions route with GPT-4o mini and tiny cap 16;
Gemini uses Flash 2.5 with tiny cap 24 and the exact `no_thinking()` parameters.
The roomy cap is 512. Original prompt and preamble constants are imported.
Tool cases install the original `Adder` and use run budget 3. Retry cases keep
agent cap 64 and apply tiny/512 per-turn patches with run budget 2. Other cases
use the original default budget 1. No agent configuration is overwritten to
simulate a run override.

## Evidence and limits

This establishes the scoped family only. It does not establish all hook semantics,
all finish-reason variants, every provider, multi-run resource isolation, WASM,
or full-programme parity. A replay environment with credential variables removed
is not by itself an operating-system network-denial proof. These cases do not establish a network barrier or comprehensive architecture
equivalence.
