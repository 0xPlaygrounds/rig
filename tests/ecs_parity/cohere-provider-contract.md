# Cohere provider-only family

## Rules

- **model-completion** — `client.completion_model(model).completion(request)` /
  `completion_request(..).send()` with documents, tool definitions, `tool_choice`
  or `strict_tools`, `.expect` or `.expect_err` on the preserved status/body.
- **model-raw** — `raw_completion(request)` / `CompletionResponse::raw`
  round trips and `try_into` parity against the recorded bodies.
- **model-stream** — `client.completion_model(..).stream(request)` /
  `completion_request(..).stream()` drained by `stream_to_terminal`,
  `drain_stream` (with `assert_valid_event_stream`) or a manual loop.
- **cache-probe** — `cache_conformance::run_cache_probe[_streaming]` over
  `client.completion_model(CACHE_MODEL)` (direct model turns), then
  `assert_cache_warms_over_turns`, `assert_prefix_stable`,
  `assert_breakpoints_match_support`.
- **embeddings** — `client.embedding_model(..)` / `client.image_embedding_model()`
  with the normalized, raw and convenience routes.

The five `cassette::agent::*` provider-only rows classified at the base keep
their original reason and are not restated here.

## Phase 2

Not ported; shared-provider evidence only.

## Fixture kind

Cassette cells are **hosted Cohere recordings** (`command-a-03-2025`, and
`command-a-reasoning-08-2025` for two streaming-grammar cells) replayed through
`with_cohere_cassette` / `with_cohere_prompt_caching_cassette` (strict ordered
`ProviderCassette`, placeholder `COHERE_API_KEY`, `finish_after_test`
exhaustion). The `#[ignore]` cells under `tests/providers/cohere/*.rs` need a
real `COHERE_API_KEY`, which is configured in this environment; only
`agent::completion_smoke` is in the programme's 12-case capture set.

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
| `cassette::context::document_metadata_and_multiple_documents_are_accepted` | shared_provider (provider) | model-completion | `cohere/context/document_metadata_and_multiple_documents_are_accepted.yaml` | no |
| `cassette::embedding_matrix::error_preserves_provider_body` | shared_provider (provider) | embeddings | `cohere/embedding_matrix/error_preserves_provider_body.yaml` | no |
| `cassette::embedding_matrix::image_normalized_and_raw_round_trip` | shared_provider (provider) | embeddings | `cohere/embedding_matrix/image_normalized_and_raw_round_trip.yaml` | no |
| `cassette::embedding_matrix::normalized_response_is_complete` | shared_provider (provider) | embeddings | `cohere/embedding_matrix/normalized_response_is_complete.yaml` | no |
| `cassette::embedding_matrix::raw_round_trips` | shared_provider (provider) | embeddings | `cohere/embedding_matrix/raw_round_trips.yaml` | no |
| `cassette::embedding_matrix::raw_route_parity` | shared_provider (provider) | embeddings | `cohere/embedding_matrix/raw_route_parity.yaml` | no |
| `cassette::embedding_matrix::single_text_convenience` | shared_provider (provider) | embeddings | `cohere/embedding_matrix/single_text_convenience.yaml` | no |
| `cassette::embeddings::embed_classification_smoke` | shared_provider (provider) | embeddings | `cohere/embeddings/embed_classification_smoke.yaml` | no |
| `cassette::embeddings::embed_image_smoke` | shared_provider (provider) | embeddings | `cohere/embeddings/embed_image_smoke.yaml` | no |
| `cassette::embeddings::embed_images_preserves_batch_order` | shared_provider (provider) | embeddings | `cohere/embeddings/embed_images_preserves_batch_order.yaml` | no |
| `cassette::embeddings::embed_search_query_smoke` | shared_provider (provider) | embeddings | `cohere/embeddings/embed_search_query_smoke.yaml` | no |
| `cassette::embeddings::embed_texts_smoke` | shared_provider (provider) | embeddings | `cohere/embeddings/embed_texts_smoke.yaml` | no |
| `cassette::errors::completion_error_preserves_status_and_body` | shared_provider (provider) | model-completion | `cohere/errors/completion_error_preserves_status_and_body.yaml` | no |
| `cassette::prompt_caching::blocking_probe_warms_to_a_full_cache_hit_over_three_turns` | shared_provider (provider) | cache-probe | `cohere/prompt_caching/blocking_probe.yaml` | no |
| `cassette::prompt_caching::streaming_probe_warms_to_a_full_cache_hit_over_three_turns` | shared_provider (provider) | cache-probe | `cohere/prompt_caching/streaming_probe.yaml` | no |
| `cassette::raw_capture_matrix::raw_exposes_billing_metadata` | shared_provider (provider) | model-raw | `cohere/raw_capture_matrix/raw_exposes_billing_metadata.yaml` | no |
| `cassette::raw_capture_matrix::raw_roundtrips_cohere_completion_response` | shared_provider (provider) | model-raw | `cohere/raw_capture_matrix/raw_roundtrips_cohere_completion_response.yaml` | no |
| `cassette::raw_completion_parity_matrix::raw_try_into_matches_completion` | shared_provider (provider) | model-raw | `cohere/raw_completion_parity_matrix/raw_try_into_matches_completion.yaml` | no |
| `cassette::raw_stream_capture_matrix::raw_exposes_terminal_only_fields` | shared_provider (provider) | model-stream | `cohere/raw_stream_capture_matrix/raw_exposes_terminal_only_fields.yaml` | no |
| `cassette::raw_stream_capture_matrix::raw_roundtrips_streaming_completion_response` | shared_provider (provider) | model-stream | `cohere/raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response.yaml` | no |
| `cassette::response_identity::nonstreaming_request_id_is_none_by_design` | shared_provider (provider) | model-completion | `cohere/response_identity/nonstreaming_request_id_is_none_by_design.yaml` | no |
| `cassette::response_identity::streaming_request_id_is_none_by_design` | shared_provider (provider) | model-completion | `cohere/response_identity/streaming_request_id_is_none_by_design.yaml` | no |
| `cassette::streaming_grammar::none_tool_choice_streams_text` | shared_provider (provider) | model-stream | `cohere/streaming_grammar/none_tool_choice_streams_text.yaml` | no |
| `cassette::streaming_grammar::reasoning_then_tool_call_closes_reasoning_before_the_call` | shared_provider (provider) | model-stream | `cohere/streaming_grammar/reasoning_then_tool_call.yaml` | no |
| `cassette::streaming_grammar::required_tool_choice_streams_tool_call` | shared_provider (provider) | model-stream | `cohere/streaming_grammar/required_tool_choice_streams_tool_call.yaml` | no |
| `cassette::streaming_grammar::thinking_stream_keeps_reasoning_and_text_discrete` | shared_provider (provider) | model-stream | `cohere/streaming_grammar/thinking_stream.yaml` | no |
| `cassette::tools::none_tool_choice_with_tools_returns_text` | shared_provider (provider) | model-completion | `cohere/tools/none_tool_choice_with_tools_returns_text.yaml` | no |
| `cassette::tools::none_tool_choice_without_tools_returns_text` | shared_provider (provider) | model-completion | `cohere/tools/none_tool_choice_without_tools_returns_text.yaml` | no |
| `cassette::tools::required_tool_choice_is_accepted` | shared_provider (provider) | model-completion | `cohere/tools/required_tool_choice_is_accepted.yaml` | no |
| `cassette::tools::required_tool_choice_selects_from_multiple_tools` | shared_provider (provider) | model-completion | `cohere/tools/required_tool_choice_selects_from_multiple_tools.yaml` | no |
| `cassette::tools::strict_required_tool_choice_is_accepted` | shared_provider (provider) | model-completion | `cohere/tools/strict_required_tool_choice_is_accepted.yaml` | no |
