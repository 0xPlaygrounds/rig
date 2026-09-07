# Copilot provider-only family

## Rules

- **embeddings** — `client.embedding_model(live_embedding_model()).embed_texts(..)`.
- **listing** — `client.list_models()`.
- **model-raw** — `completion(request)` / `raw_completion[_with_request_id](request)`
  round trips and parity comparisons on both routes (nine `unrecorded` cells,
  no fixture on disk).
- **model-stream** — `client.completion_model(..).stream(request)` drained by
  `terminal_of` (four `unrecorded` cells) or, for the recorded
  `streaming_tools::raw_*` cells, by `collect_raw_stream_observation` /
  `assert_stream_contains_zero_arg_tool_call_named`.
- **reasoning-roundtrip-driver** — `reasoning::run_reasoning_roundtrip_*` over
  `client.completion_model(live_responses_model())` (the streaming cell wraps
  it in a `CapturingProviderFinals` `CompletionModel` impl); the driver builds
  `CompletionRequest`s by hand and calls `model.completion` / `model.stream`;
  no `AgentBuilder`.

## Phase 2

Not ported; the unrecorded cells cannot be recorded without Copilot
credentials and are not recaptured by this lane.

## Fixture kind

Cassette cells are **hosted GitHub Copilot recordings** replayed through
`with_copilot_cassette` / `with_copilot_cassette_result` (strict ordered
`ProviderCassette`, placeholder `GITHUB_COPILOT_API_KEY`, `finish_after_test[_result]`
exhaustion). Chat models (`GPT_4O`, `GPT_4O_MINI`) route through Chat
Completions, codex models (`live_responses_model()`, default `GPT_5_3_CODEX`)
through the Responses route. `#[ignore]` cells are live-only: Copilot needs an
API key, a GitHub access token or an OAuth device-flow cache, none configured
here; they stay `supplemental_live` (OAuth), not counted as ported. The
`unrecorded` raw matrices have complete bodies but no fixture on disk.

## What a classification here claims

A row classified `agent` means the original test constructs a rig-agent
builder (`client.agent(..)`, `AgentBuilder::new(..)`, `client.extractor(..)`,
`into_agent_builder()`) and drives its orchestration (`prompt`, `prompt_typed`,
`chat`, `stream_prompt`, `stream_chat`, `runner(..).run()`, `extract`). A row
classified `shared_provider` means the reviewed body calls a provider
capability directly (`CompletionModel::{completion,raw_completion,stream}`,
`completion_request(..).send()/.stream()`, embeddings, rerank, model listing,
`verify`, transcription, image generation) or a shared driver that does so, and
never constructs an agent. A row classified `infrastructure` executes no rig
provider client and no agent at all. Classification is from source reading and
the base listing `superset-lanes/ecs-tests-a7eb63a91.json`; it is not an
execution result. Ignored rows are discovered, never executed; nothing in this
contract is a parity or superset verdict.

## Cells

| Cell | Class | Rule | Fixture(s) | Ignored |
| --- | --- | --- | --- | --- |
| `embeddings::embeddings_smoke` | shared_provider (provider) | embeddings | `copilot/embeddings/embeddings_smoke.yaml` | no |
| `models::list_models_smoke` | shared_provider (provider) | listing | `copilot/models/list_models_smoke.yaml` | no |
| `raw_capture_matrix::chat_normalized_fields_equal_raw_renormalized` | shared_provider (provider) | model-raw | none | yes |
| `raw_capture_matrix::chat_raw_exposes_system_fingerprint` | shared_provider (provider) | model-raw | none | yes |
| `raw_capture_matrix::chat_raw_round_trips_provider_type` | shared_provider (provider) | model-raw | none | yes |
| `raw_capture_matrix::responses_normalized_fields_equal_raw_renormalized` | shared_provider (provider) | model-raw | none | yes |
| `raw_capture_matrix::responses_raw_exposes_envelope` | shared_provider (provider) | model-raw | none | yes |
| `raw_capture_matrix::responses_raw_round_trips_provider_type` | shared_provider (provider) | model-raw | none | yes |
| `raw_completion_parity_matrix::chat_plain_raw_completion_lacks_request_id` | shared_provider (provider) | model-raw | none | yes |
| `raw_completion_parity_matrix::chat_raw_with_request_id_reproduces_completion` | shared_provider (provider) | model-raw | none | yes |
| `raw_completion_parity_matrix::responses_raw_completion_carries_request_id` | shared_provider (provider) | model-raw | none | yes |
| `raw_stream_capture_matrix::chat_stream_raw_exposes_copilot_usage` | shared_provider (provider) | model-stream | none | yes |
| `raw_stream_capture_matrix::chat_stream_raw_terminal_round_trips_provider_type` | shared_provider (provider) | model-stream | none | yes |
| `raw_stream_capture_matrix::responses_stream_raw_exposes_terminal_status` | shared_provider (provider) | model-stream | none | yes |
| `raw_stream_capture_matrix::responses_stream_raw_terminal_round_trips_provider_type` | shared_provider (provider) | model-stream | none | yes |
| `reasoning_roundtrip::nonstreaming` | shared_provider (provider) | reasoning-roundtrip-driver | `copilot/reasoning_roundtrip/nonstreaming.yaml` | no |
| `reasoning_roundtrip::streaming` | shared_provider (provider) | reasoning-roundtrip-driver | `copilot/reasoning_roundtrip/streaming.yaml` | no |
| `streaming_tools::raw_followup_uses_tool_result_without_new_tool_calls` | shared_provider (provider) | model-stream | `copilot/streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls.yaml` | no |
| `streaming_tools::raw_stream_emits_required_zero_arg_tool_call` | shared_provider (provider) | model-stream | `copilot/streaming_tools/raw_stream_emits_required_zero_arg_tool_call.yaml` | no |
| `streaming_tools::raw_stream_surfaces_two_distinct_tool_calls_before_text` | shared_provider (provider) | model-stream | `copilot/streaming_tools/raw_stream_surfaces_two_distinct_tool_calls_before_text.yaml` | no |
