# Native smoke observation contract

The nine native smoke cases for OpenAI, Anthropic and Gemini retain the following
original observations. The scenario catalog records their correspondences; current
tests establish their results.

| Original case, in each provider's cassette module | Original observable obligations | Native observation |
| --- | --- | --- |
| `agent::completion_smoke` | Completion succeeds; trimmed answer is nonempty | `EcsAgent::prompt` requires no `Failed`, then `Settled` and `RunResult`; the original `assert_nonempty_response` checks the answer |
| `streaming::streaming_smoke` | Drain to EOF without an item error; require a run final and provider final; nonempty answer; positive terminal usage; provider identity for OpenAI/Anthropic | Native Collect closes the channel before publishing `EffectOutcome`; success helper checks recorded stream errors; `RunResult` supplies the answer; the case selects the last `StreamEvent::Final` and checks the original fields |
| `streaming_tools::streaming_tools_smoke` | Drain to EOF without an item error; require a run final; nonempty answer mentioning -3 under the shared helper's accepted spellings | Same success/error observation; original `assert_mentions_expected_number` checks the answer after native model/tool execution |

The stream collectors in `tests/common/support.rs` use `item?` and `expect`,
not just assertion macros. They keep the last final value; they do not
themselves prove exactly one terminal. The ECS cases likewise select the
last provider final. The single-stream query is specific to the one-turn
streaming smoke cases, not a general multi-turn collector.

`Streamed.outcome` retains the first folded terminal/error. Consequently,
checking only successful settlement would miss an error after a terminal.
The native helper checks public `Streamed.errors` at settlement, including when
event retention is disabled. `collect_streams` records every
received error and publishes the outcome only after channel closure. The
recorder is an observer of native execution, not an input oracle. This
success-only helper checks the whole fresh application's record; expected
failure and independent multi-run support will need scoped observations.

Two synthetic controls in `tests/common/ecs_agent/tests.rs` exercise clean
success and a final followed by an error. The latter requires the specific
stream-error assertion panic. Disabling that assertion makes the control fail
because the expected panic does not occur. These controls do not establish
delayed EOF, backpressure, cancellation, or policy-visible timing fidelity.

Provider model IDs, prompts, tool order, Gemini thinking parameters and
turn limits remain case-specific. The tools smoke originals assert final
arithmetic content; they do not independently assert exact tool execution
counts. Strict cassette interaction checks supply additional request evidence,
whose recorder boundary and normalization limitations still apply. A final
answer alone must not be presented as independent tool-count evidence.

The catalog's `configuration_mapping` records each smoke case's settings.
Gemini's tools case leaves the agent's default at one turn and overrides
only the run to three turns. `prompt_with_max_turns` forwards that override
to native `spawn_run`; it does not change the agent. A third synthetic
control completes a two-turn unary tool workflow with a run override and
checks that both agent budget components remain at one. That control does
not exercise a subsequent default-budget run.
