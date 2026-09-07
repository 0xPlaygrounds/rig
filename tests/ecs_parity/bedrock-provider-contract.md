# Bedrock provider-only and corpus-guard families

## Provider rules (`families: ["provider"]`)

- **model-completion** — `client.completion_model(model).completion(request)` /
  `completion_request(..).send()` with documents, tool definitions or
  `tool_choice`, `.expect` or `.expect_err` on the preserved provider error
  body; the recorded request order re-read where the cell says so.
- **model-raw** — `raw_completion(request)` (optionally `.with_guardrail(..)`)
  with `AwsConverseOutput` trace / request id / `text_response()` assertions,
  and the three `unrecorded` `raw_capture_matrix` round-trip cells.
- **model-stream** — `client.completion_model(AMAZON_NOVA_LITE).stream(request)`
  drained by `terminal_of`, `collect_raw_stream_observation`,
  `assert_stream_contains_zero_arg_tool_call_named` or a manual loop (two
  `unrecorded` `raw_stream_capture_matrix` cells and the `#[ignore]` streamed
  request-id cell have no fixture).
- **embeddings** — `client.embedding_model_with_ndims(AMAZON_TITAN_EMBED_TEXT_V2_0, 256)
  .embed_texts(..)`.

The twelve macro-generated `streaming_conformance` cells were classified
shared-provider at the base and keep their reason.

## Corpus rule (`families: ["corpus"]`, classification `infrastructure`)

- **registry-guard** — `streaming_conformance::suite_families_are_registered_wire_families`
  is a plain `#[test]` comparing `SUITE_FAMILIES` with
  `rig_core::test_utils::streaming_conformance::WIRE_FAMILIES`; no client, no
  fixture, no agent.

## Phase 2

Not ported; shared-provider evidence only. The unrecorded cells are not
recaptured.
## Fixture kind

Cassette cells are **hosted AWS Bedrock recordings** (`AMAZON_NOVA_LITE`, and
the Claude Haiku 4.5 / DeepSeek R1 inference profiles where the cell says so)
captured by the direct recorder and replayed through `with_bedrock_cassette`
(direct-recording `ProviderCassette`, dummy AWS credentials, fixed `us-east-1`
region, loopback endpoint, `finish_after_test` exhaustion). Every row is behind
`cfg(feature = "bedrock")` and only exists in the root-bedrock listing. The six
`#[ignore]` cells are `unrecorded` (no valid AWS credentials); lane rule: do
not recapture anything for bedrock.

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
| `cassette::document_ordering::documents_are_prepended_before_history` | shared_provider (provider) | model-completion | `bedrock/document_ordering/documents_are_prepended_before_history.yaml` | no |
| `cassette::embeddings::embeddings_batch_smoke` | shared_provider (provider) | embeddings | `bedrock/embeddings/embeddings_batch_smoke.yaml` | no |
| `cassette::embeddings::embeddings_smoke` | shared_provider (provider) | embeddings | `bedrock/embeddings/embeddings_smoke.yaml` | no |
| `cassette::model_ids::bare_profile_only_model_id_is_rejected` | shared_provider (provider) | model-completion | `bedrock/model_ids/bare_profile_only_model_id_is_rejected.yaml` | no |
| `cassette::model_ids::cross_region_profile_id_completes` | shared_provider (provider) | model-completion | `bedrock/model_ids/cross_region_profile_id_completes.yaml` | no |
| `cassette::model_ids::retired_model_id_preserves_provider_error` | shared_provider (provider) | model-completion | `bedrock/model_ids/retired_model_id_preserves_provider_error.yaml` | no |
| `cassette::raw_capture_matrix::normalized_fields_equal_raw_renormalized` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_capture_matrix::raw_exposes_latency_metrics` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_capture_matrix::raw_round_trips_provider_type` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_completion::raw_response_text_matches_normalized_choice_text` | shared_provider (provider) | model-raw | `bedrock/raw_completion/raw_response_text_matches_normalized_choice_text.yaml` | no |
| `cassette::raw_provider_data::guardrail_trace_survives_into_raw_completion` | shared_provider (provider) | model-raw | `bedrock/raw_provider_data/guardrail_trace_survives_into_raw_completion.yaml` | no |
| `cassette::raw_provider_data::request_id_survives_into_raw_completion` | shared_provider (provider) | model-raw | `bedrock/raw_provider_data/request_id_survives_into_raw_completion.yaml` | no |
| `cassette::raw_provider_data::request_id_survives_into_streamed_terminal` | shared_provider (provider) | model-stream | none | yes |
| `cassette::raw_stream_capture_matrix::stream_raw_exposes_bedrock_stop_reason` | shared_provider (provider) | model-stream | none | yes |
| `cassette::raw_stream_capture_matrix::stream_raw_terminal_round_trips_provider_type` | shared_provider (provider) | model-stream | none | yes |
| `cassette::raw_streaming::raw_stream_emits_required_zero_arg_tool_call` | shared_provider (provider) | model-stream | `bedrock/raw_streaming/raw_stream_emits_required_zero_arg_tool_call.yaml` | no |
| `cassette::raw_streaming::raw_stream_emits_tool_call_before_text` | shared_provider (provider) | model-stream | `bedrock/raw_streaming/raw_stream_emits_tool_call_before_text.yaml` | no |
| `cassette::raw_streaming::raw_stream_surfaces_two_distinct_tool_calls` | shared_provider (provider) | model-stream | `bedrock/raw_streaming/raw_stream_surfaces_two_distinct_tool_calls.yaml` | no |
| `cassette::raw_streaming::raw_stream_text_response_smoke` | shared_provider (provider) | model-stream | `bedrock/raw_streaming/raw_stream_text_response_smoke.yaml` | no |
| `cassette::tool_choice::required_forces_function_call` | shared_provider (provider) | model-completion | `bedrock/tool_choice/required_forces_function_call.yaml` | no |
| `cassette::tool_choice::specific_add_raw_nonstreaming_allows_only_add` | shared_provider (provider) | model-completion | `bedrock/tool_choice/specific_add_raw_nonstreaming.yaml` | no |
| `cassette::tool_choice::specific_add_raw_streaming_allows_only_add` | shared_provider (provider) | model-completion | `bedrock/tool_choice/specific_add_raw_streaming.yaml` | no |
| `streaming_conformance::suite_families_are_registered_wire_families` | infrastructure (corpus) | registry-guard | none | no |
