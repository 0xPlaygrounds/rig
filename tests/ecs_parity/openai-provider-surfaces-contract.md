# OpenAI provider-surface families (shared provider)

Scope: the `tests/providers/openai/` registrations that drive a provider API
directly through rig-core client/model types and never construct rig-agent
orchestration (no `agent(..)`/`AgentBuilder`/`into_agent_builder()`, no
`prompt`/`chat`/`stream_prompt`/`stream_chat`/`runner().run()`, no
`Extractor`). They stay `shared_provider` and count as zero migrated agent
cases. The catalog references one of the rules below in every row's
`classification_reason`, followed by the positive evidence of what that test
constructs and drives.

## Rules

- **P1 — raw completion model call.** The test builds
  `client.completion_model(..)` (Responses route),
  `client.completions_api().completion_model(..)` or
  `CompletionsClient.completion_model(..)` (Chat Completions route) and drives
  `completion(..)`, `raw_completion(..)` (+ `normalize("openai")`),
  `raw_completion_with_request_id(..)`, `completion_request(..).send()` or
  `stream(..)`. Streams are drained to EOF by the module's own collector or by
  `collect_raw_stream_observation` / `collect_text_and_terminal`, except the
  error-shape cells that stop at the first error item by design. Multi-turn
  cells hand-build history messages and issue further model calls; helper
  wrappers such as `ReasoningRoundtripAgent` and `run_cache_probe` are raw
  model drivers. Any post-closure assertion re-reads the recorded fixture
  (request body, SSE frames, headers) and compares it with an observation
  moved out through `Arc<Mutex<Option<_>>>`.
- **P2 — non-completion provider endpoint.** Embeddings (`embedding_model`),
  transcription (`transcription_model`), speech (`audio_generation_model`,
  through the direct-recording client), image generation
  (`image_generation_model`), model listing (`list_models`), `verify`, and the
  websocket upgrade handshake (`responses_websocket`). No completion loop.
- **P3 — provider-type unit check without HTTP.** A `#[test]` converting
  messages, requests, output items or tool definitions through the provider's
  serde types (`InputItem`, `OpenAIResponsesRequest`, `AssistantContent`,
  `ResponsesToolDefinition`), or comparing model constants. No client, no
  cassette, no runtime.
- **P4 — live-only provider case.** An `#[ignore]` test that builds
  `openai::Client::from_env()` and drives a P1/P2 surface against the real
  API with no recording. Classified by its path with
  `evidence_scope: supplemental_live` and `baseline_ignored: true`; counted as
  discovered only, never as executed.

## Modules and cells

| Module | Rule | Cells |
|---|---|---|
| `chat_history_roundtrip_matrix` | P1 | 24 (`run_cell` → Raw/Normalized × blocking/streaming, caller history) |
| `chat_streaming_logprobs_matrix` | P1 | 24 (`run_cell` → raw_completion / stream, logprobs) |
| `chat_terminal_metadata_matrix` | P1 | 24 (`run_cell` → raw_completion / stream, terminal metadata) |
| `chat_tool_truncation_matrix` | T-model | 12 (see `openai-chat-truncation-contract.md`) |
| `streaming_grammar` | P1 | 10 (Responses `stream` with the conformance validator) |
| `streaming_grammar_chat` | P1 | 4 (Chat Completions `stream` with the conformance validator) |
| `raw_capture_matrix`, `raw_stream_capture_matrix`, `raw_completion_parity_matrix` | P1 | 7 + 6 + 5 (raw payload capture; parity issues each request twice) |
| `max_completion_tokens_matrix` | P1 | 17 model cells (4 agent cells and 1 extractor cell are A1/A2) |
| `refusal_matrix` | P1 | 13 model cells (5 agent cells are A1) |
| `truncated_turn_matrix` | P1 | 15 model cells (1 agent cell is A1) |
| `completions_api` | P1 | 5 model cells (3 agent cells are A1) |
| `responses_sessions` | P1 | 1 (`long_history_replay_nonstreaming`; 6 agent cells are A1) |
| `gpt_5_6_reasoning` | P1 / P3 | 6 model cells; `model_constants` is P3 |
| `responses_tool_args` | P1 | 4 model cells (1 agent cell is A1) |
| `responses_tool_choice` | P1 | 4 |
| `responses_behaviors` | P1 | 2 (1 agent cell is A1) |
| `response_identity` | P1 | 4 (2 agent cells are A1) |
| `response_identity_edge` | P1 | 4 (1 agent cell is A1) |
| `error_identity_edge` | P1 / P2 | 4 completion errors; 3 endpoint errors (embeddings, list_models, verify); 1 extractor cell is A2 |
| `error_envelope`, `document_ordering`, `additional_params_tools`, `vllm` | P1 | 2 + 2 + 1 + 1 |
| `prompt_caching` | P1 / P4 | 5 probe cells (three raw turns each); `live_cache_economics` is P4 |
| `responses_input_item`, `response_schema` | P3 | 18 + 3 |
| `embedding_matrix` | P2 | 6 |
| `transcription_usage_matrix` | P2 | 8 |
| `audio_params_matrix` (`audio`) | P2 | 7 |
| `image_params_matrix` (`image`) | P2 | 17 (post-closure fixture re-reads) |
| `models` | P2 | 2 |
| `websocket_error_identity_matrix` (`websocket`) | P2 | 2 |
| `live::gpt_5_5` | P4 | 3 (`responses_reasoning_{nonstreaming,streaming}_smoke`, `responses_websocket_smoke`) |
| `live::image_generation`, `live::audio_generation`, `live::transcription`, `live::websocket` | P4 | 2 + 1 + 1 + 1 |

## Fixture disposition

Every P1/P2 cassette cell replays its own recording under strict ordered
matching (the `multi_extract` unordered spec is an A2 cell) and the wrapper's
`finish_after_test` exhaustion check; `boxed_transport` (A1) and
`corpus_serving` (A1) re-use fixtures owned by other modules. P3 and P4 rows
carry no fixture. Feature-gated modules (`audio`, `image`, `websocket`) are
outside the pinned `--features bedrock` listing and are recorded as gated,
never as executed.

## Limits

These rows establish the provider wire/normalization behavior both runtimes
share. They do not establish agent orchestration parity and are not
candidates for native porting; a native counterpart of a P1 cell would only
re-drive the same adapter. Where a module mixes surfaces, the split is by
cell, and the agent cells are listed in `openai-agent-surfaces-contract.md`.
