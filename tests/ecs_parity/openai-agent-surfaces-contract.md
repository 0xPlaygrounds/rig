# OpenAI agent-surface families (agent, not yet ported)

Scope: the `tests/providers/openai/` registrations that construct rig-agent
orchestration and are not yet mapped to a native `ecs_*` counterpart. They
are classified `agent` with `ecs: null`; each later batch ports one family
and replaces the `ecs` field of its rows. The catalog references one of the
rules below in every row's `classification_reason`, followed by the positive
evidence of what that test constructs and drives (builder, model, preamble,
tools, budgets, hooks, prompt surface and collector).

## Rules

- **A1 — rig-agent orchestration on a recording.** The test builds
  `client.agent(..)`, `client.completions_api().agent(..)`,
  `CompletionsClient.completion_model(..).into_agent_builder()`,
  `AgentBuilder::new(model)` or `AgentBuilder::over_bus(..)` and drives
  `prompt`, `prompt_typed`, `chat`, `stream_prompt`, `stream_chat` or
  `runner(..).run()`. Streams are drained to EOF by
  `collect_stream_observation`, `collect_stream_final_response(_and_provider_final)`,
  `collect_stream_stats`, `stream_to_stdout` or the module's own loop. Hooks
  (`IdentityProbe`, `RawProbe`, `RetryOnce*`, `PermissionHook`,
  `SessionIdHook`, `ReasoningDeltaHookRecorder`, corpus hooks), memory,
  retrieval and effect-log goldens are part of the driven path. Evidence is
  `cassette_agent` when the cell replays a recording; a cell over an
  in-memory `RecordingHttpClient` is `supplemental_synthetic`.
- **A2 — Extractor orchestration.** The test builds `client.extractor::<T>(..)`
  or `ExtractorBuilder::new(model)` and drives `extract(..)`. Extractors run
  the agent loop with an output tool, so they follow the extractor families
  already ported (`ecs_extractor`, `ecs_extractor_usage`).
- **A3 — live-only agent case.** An `#[ignore]` test that builds
  `openai::Client::from_env()` (or `Client::new(key)`) and drives an A1/A2
  surface against the real API with no recording. Classified by its path
  with `evidence_scope: supplemental_live` and `baseline_ignored: true`;
  counted as discovered only, never as ported or executed.

## Modules and cells

| Module | Rule | Cells |
|---|---|---|
| `chat_tool_truncation_matrix` | T-agent (ported) | 12, see `openai-chat-truncation-contract.md` |
| `raw_capture_agent_matrix` | A1 | 9 (`RawProbe` hook, chat and Responses routes, blocking/streamed, tool runs, one retried run) |
| `responses_sessions` | A1 | 6 (sequential/parallel tool sessions, reasoning session, usage accumulation) |
| `refusal_matrix` | A1 | 5 (`chat_blocking_agent_prompt_surfaces_refusal`, `chat_streaming_agent_surfaces_refusal`, `chat_refusal_turn_survives_into_history`, `responses_agent_blocking_refusal_surfaces`, `responses_agent_streaming_refusal_surfaces`) |
| `max_completion_tokens_matrix` | A1 / A2 | 4 agent cap cells; `capped_extractor_turn_on_reasoning_model` is A2 |
| `truncated_turn_matrix` | A1 | 1 (`chat_blocking_agent_reports_the_truncation`, expects an error naming `finish_reason=Length`) |
| `completions_api` | A1 | 3 (`completions_api_agent_prompt`, `completions_api_streams_two_tool_calls_before_final_answer`, `completions_api_stream_emits_tool_call_before_later_text`) |
| `corpus_breadth`, `corpus_retrieval`, `corpus_host`, `corpus_output`, `corpus_delta`, `corpus_serving`, `effect_corpus` | A1 | 6 + 5 + 2 + 2 + 1 + 1 + 2 effect-log golden runs (`record_effects*`, `golden_effects`) |
| `responses_tool_args` | A1 | 1 (`nested_arguments_roundtrip_nonstreaming`) |
| `responses_behaviors` | A1 | 1 (`system_messages_as_input_items_mid_conversation`) |
| `response_identity` | A1 | 2 (`IdentityProbe` hook) |
| `response_identity_edge` | A1 | 1 (`blocking_hook_retry_uses_second_attempts_id`) |
| `response_retry` | A1 | 1 (`RetryOnceOnMarker` through `runner().run()`) |
| `structured_output` | A1 | 3 (`classic_tool_mode_maps_through_openai_responses` is over `RecordingHttpClient`, synthetic evidence) |
| `reasoning_roundtrip` | A1 | 1 (`reasoning_delta_hook_streaming`; the two roundtrip cells are P1) |
| `reasoning_tool_roundtrip`, `chat_history`, `openai_compatible_reasoning_content` | A1 | 2 + 1 + 1 (reasoning tool sessions; the last records a local axum Responses server) |
| `request_hook`, `regression_suite`, `permission_control`, `typed_prompt_tools`, `url_pdf_document`, `boxed_transport` | A1 | 1 + 1 + 2 + 1 + 1 + 2 |
| `multi_extract` | A2 | 1 (nine concurrent extractions, unordered replay) |
| `error_identity_edge` | A2 | 1 (`extractor_failure_surfaces_provider_error_context`) |
| `regressions` (root) | A2 | 1 (extractor over `RecordingHttpClient`, synthetic evidence) |
| `live::gpt_5_5` | A3 | 16 agent/extractor smokes (3 raw cells are P4) |
| `live::document_file_id`, `live::streaming_tools_reasoning` | A3 | 2 + 1 |

## Porting order

By size: `raw_capture_agent_matrix` (9), the effect-corpus goldens (19 across
seven modules, sharing `crate::goldens`), `responses_sessions` (6),
`refusal_matrix` agent cells (5), `max_completion_tokens_matrix` agent cells
(5 incl. the extractor), then the singletons. Hook-driven cells (`RawProbe`,
`IdentityProbe`, `RetryOnce*`, `PermissionHook`) need the native hook
observation points before they can close every obligation; cells whose
obligation is the legacy error text (`truncated_turn_matrix` agent cell) need
an explicit contract decision on the native failure kind.

## Limits

Classification records the driven path only; no row here is an execution
result or a parity claim. A3 rows have no recording and are never counted as
ported. The 39 previously mapped agent rows (`agent`, `chat_tool_lifecycle_matrix`
agent cells, `extractor`, `extractor_usage`, `lifecycle_matrix`,
`prompt_caching` agent loops, `streaming`, `streaming_tools`,
`turn_termination_matrix`) keep their existing family contracts.
