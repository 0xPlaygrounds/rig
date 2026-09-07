# Ollama provider-only family

Cells that call a provider capability directly and never construct an agent.

## Rules

- **model-raw** — `client.completion_model(MODEL).completion(request)` with
  `CompletionResponse::raw` round-trip / `try_into` comparisons against the
  recorded body.
- **model-stream** — `client.completion_model(MODEL).stream(request)` drained by
  `terminal_of` or `drain_stream` (with `assert_valid_event_stream` and
  `snapshot()`), or live `pause()`/`resume()` in `pause_control`.
- **listing** — `client.list_models()` (GET `/api/tags`), cassette and live.
- **reasoning-roundtrip-driver** — `reasoning::run_reasoning_roundtrip_{nonstreaming,streaming}
  (ReasoningRoundtripAgent::new(model, think))`: despite the type name the
  driver builds `CompletionRequest`s by hand and calls `model.completion` /
  `model.stream` for two turns, replaying the turn-1 assistant content; no
  `AgentBuilder` is constructed.

## Phase 2

Not ported; these rows remain shared-provider evidence.

## Fixture kind

Cassette cells are **local-model recordings**: recorded against a local Ollama
daemon serving `qwen3:4b` (no API key; `ollama::Client::builder().api_key(Nothing)`)
and replayed through `with_ollama_cassette` (strict ordered `ProviderCassette`,
`finish_after_test` exhaustion); replay needs no daemon. The `#[ignore]` cells
under `tests/providers/ollama/*.rs` are live-only: they require a local Ollama
server, none is configured in this environment, no fixture exists, and they
stay `supplemental_live` (not counted as ported).

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
| `cassette::models::list_models_smoke` | shared_provider (provider) | listing | `ollama/models/list_models_smoke.yaml` | no |
| `cassette::raw_capture_matrix::normalized_fields_equal_raw_renormalized` | shared_provider (provider) | model-raw | `ollama/raw_capture_matrix/normalized_fields_equal_raw_renormalized.yaml` | no |
| `cassette::raw_capture_matrix::raw_exposes_ollama_durations` | shared_provider (provider) | model-raw | `ollama/raw_capture_matrix/raw_exposes_ollama_durations.yaml` | no |
| `cassette::raw_capture_matrix::raw_round_trips_provider_type` | shared_provider (provider) | model-raw | `ollama/raw_capture_matrix/raw_round_trips_provider_type.yaml` | no |
| `cassette::raw_stream_capture_matrix::stream_raw_exposes_terminal_durations` | shared_provider (provider) | model-stream | `ollama/raw_stream_capture_matrix/stream_raw_exposes_terminal_durations.yaml` | no |
| `cassette::raw_stream_capture_matrix::stream_raw_terminal_round_trips_provider_type` | shared_provider (provider) | model-stream | `ollama/raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type.yaml` | no |
| `cassette::reasoning_roundtrip::nonstreaming` | shared_provider (provider) | reasoning-roundtrip-driver | `ollama/reasoning_roundtrip/nonstreaming.yaml` | no |
| `cassette::reasoning_roundtrip::streaming` | shared_provider (provider) | reasoning-roundtrip-driver | `ollama/reasoning_roundtrip/streaming.yaml` | no |
| `cassette::streaming_grammar::chat_sourced_history_replays_the_tool_name_not_the_identifier` | shared_provider (provider) | model-stream | `ollama/streaming_grammar/chat_sourced_history_replay.yaml` | no |
| `cassette::streaming_grammar::parallel_id_less_tool_calls_stay_distinct` | shared_provider (provider) | model-stream | `ollama/streaming_grammar/parallel_tool_calls.yaml` | no |
| `cassette::streaming_grammar::same_tool_called_twice_in_one_turn_stays_distinct` | shared_provider (provider) | model-stream | `ollama/streaming_grammar/same_tool_twice.yaml` | no |
| `cassette::streaming_grammar::thinking_and_tool_call_in_one_stream` | shared_provider (provider) | model-stream | `ollama/streaming_grammar/thinking_and_tool_call.yaml` | no |
| `models::list_models_smoke` | shared_provider (provider) | listing | none | yes |
| `pause_control::streaming_pause_and_resume` | shared_provider (provider) | model-stream | none | yes |
