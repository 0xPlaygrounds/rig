# llama.cpp provider-only and corpus-guard families

Every llama.cpp cell that never constructs an agent. `provider` rows exercise a
provider capability directly against a recorded local `llama-server`; `corpus`
rows execute no rig client at all. These rows stay shared-provider /
infrastructure and never inflate agent counts.

## Provider rules (`families: ["provider"]`)

- **model-completion** — `client.completion_model(model).completion(request)`
  (or `completion_request(..).send()`), with hand-built histories, tool
  definitions, `tool_choice`, `output_schema`, `additional_params` or image
  parts as the cell states, `.expect` or `.expect_err` on the preserved status
  and body; the recorded request/response is re-read after closure for the
  cell's premise.
- **model-stream** — `client.completion_model(model).stream(request)` drained by
  the module's own collector (`terminal_of`, `drain_stream`,
  `collect_raw_stream_observation`, `assert_stream_contains_zero_arg_tool_call_named`
  or a manual loop over `StreamEvent`).
- **model-raw** — `raw_completion(request)` / `CompletionResponse::raw` round
  trips through the provider wire type and `normalize`.
- **cache-probe** — `cache_conformance::run_cache_probe` /
  `run_cache_probe_streaming` (direct `model.completion` / `model.stream` turns
  with hand-built history) or explicit `model.completion` turns, then
  `assert_cache_conformance` / `assert_prefix_stable` over the fixture.
- **embeddings** — `client.embedding_model[_with_ndims](..).embed_texts(..)` or
  `client.embeddings(..).document(..).build()`.
- **rerank** — `client.rerank_model(CASSETTE_RERANK_MODEL)[.top_n(n)].rerank(..)`.
- **listing** — `client.list_models()`; **verify** — `client.verify()` (GET `/props`).
- **client-refusal** — `llamacpp::Client::from_url_with` at a dead loopback
  port; `ToolChoice::Specific` is rejected in-process by
  `Llamacpp::prepare_request` before any request; no cassette.
- **request-serialization** — `Llamacpp.build_completion_request(..)` serialized
  in-process; no server.

Server configurations per module (from `tests/providers/llamacpp/mod.rs`):
default 8080 for most modules; embeddings 8081 (`Qwen3-Embedding-0.6B`);
vision 8082 (`Qwen3-VL-2B`); small context 8083 (`-c 512`); no-jinja 8084;
rerank 8085 (`bge-reranker-v2-m3`); pooling-none 8086; causal embeddings 8087;
competent 8088 (`Qwen3-8B`, tool/truncation/turn-termination/structured cells
that say so); api-key 8089; Llama/Mistral/Gemma family 8090-8092; large vision
8093 (`Qwen2.5-VL-7B`).

## Corpus rules (`families: ["corpus"]`, classification `infrastructure`)

- **source-guard** — `matrix_index` reads `tests/providers/llamacpp/**/*.rs`
  from disk and checks tables and `mod` declarations.
- **fixture-guard** — a plain `#[test]` that re-reads sibling fixtures with the
  `crate::cassettes::recorded_*` readers (finish-reason sweep, cache counters,
  zero-argument fragments, the clamped cap, the smoke-tier round trip).
- **raw-http** — `with_llamacpp_raw_http_cassette` hands the cassette base URL
  to a bare `reqwest` multipart POST; no rig client or provider type is used.

## Phase 2

Nothing in this file is ported. The rows are the provider request/response
evidence the agent families rely on and remain shared-provider.
## Fixture kind

Every fixture in this family is a **local-model recording**: it was recorded
against a local `llama-server` b10499 (commit 6d05498) built from source, with
generation pinned (`--seed 42 --temp 0`), and is replayed through the
provider's own wrapper (`with_llamacpp_cassette` and the per-configuration
wrappers in `tests/providers/llamacpp/cassette_support.rs`: strict ordered
`ProviderCassette`, `finish_after_test` exhaustion, credential-free
`llamacpp::Client::from_url_with`). No hosted provider and no credential are
involved, and replay needs no local server. The default smoke tier is
`unsloth/Qwen3-1.7B-GGUF` Q4_K_M (`--jinja -c 4096`); cells that name the
competent tier (`Qwen3-8B` Q4_K_M, `-c 8192`) or another server say so in
their rule.

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
| `cassette::content_matrix::a_system_message_plus_history_keeps_its_order` | shared_provider (provider) | model-completion | `llamacpp/content_matrix/system_plus_history.yaml` | no |
| `cassette::content_matrix::a_very_long_tool_output_survives_the_round_trip` | shared_provider (provider) | model-completion | `llamacpp/content_matrix/long_tool_output.yaml` | no |
| `cassette::content_matrix::an_answer_fully_consumed_by_a_stop_sequence_surfaces_as_an_empty_response` | shared_provider (provider) | model-completion | `llamacpp/content_matrix/empty_answer_with_stop.yaml` | no |
| `cassette::content_matrix::consecutive_same_role_messages_are_sent_as_sent` | shared_provider (provider) | model-completion | `llamacpp/content_matrix/consecutive_same_role.yaml` | no |
| `cassette::embedding_matrix::a_declared_width_llamacpp_cannot_honour_is_refused` | shared_provider (provider) | embeddings | `llamacpp/embedding_matrix/declared_width_mismatches.yaml` | no |
| `cassette::embedding_matrix::a_declared_width_that_matches_is_accepted` | shared_provider (provider) | embeddings | `llamacpp/embedding_matrix/declared_width_matches.yaml` | no |
| `cassette::embedding_matrix::several_inputs_come_back_in_order_at_one_width` | shared_provider (provider) | embeddings | `llamacpp/embedding_matrix/batch.yaml` | no |
| `cassette::embedding_matrix::the_native_width_comes_back_when_none_is_declared` | shared_provider (provider) | embeddings | `llamacpp/embedding_matrix/native_width.yaml` | no |
| `cassette::embeddings::derive_document_embeddings` | shared_provider (provider) | embeddings | `llamacpp/embeddings/derive_document_embeddings.yaml` | no |
| `cassette::embeddings::embeddings_smoke` | shared_provider (provider) | embeddings | `llamacpp/embeddings/embeddings_smoke.yaml` | no |
| `cassette::error_matrix::a_malformed_body_keeps_its_parse_error` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/malformed_request_field.yaml` | no |
| `cassette::error_matrix::a_missing_api_key_is_a_401_the_caller_can_read` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/missing_api_key_is_401.yaml` | no |
| `cassette::error_matrix::an_embeddings_input_past_the_batch_size_is_a_500` | shared_provider (provider) | embeddings | `llamacpp/error_matrix/embeddings_input_past_the_batch.yaml` | no |
| `cassette::error_matrix::an_oversized_output_cap_is_clamped_not_rejected` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/oversized_output_cap.yaml` | no |
| `cassette::error_matrix::an_unknown_model_is_ignored_rather_than_rejected` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/unknown_model_is_ignored.yaml` | no |
| `cassette::error_matrix::context_overflow_preserves_the_token_counts` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/context_overflow_blocking.yaml` | no |
| `cassette::error_matrix::embeddings_on_a_causal_lm_return_pooled_numbers` | shared_provider (provider) | embeddings | `llamacpp/error_matrix/embeddings_on_a_causal_lm.yaml` | no |
| `cassette::error_matrix::embeddings_with_pooling_none_are_a_400` | shared_provider (provider) | embeddings | `llamacpp/error_matrix/embeddings_with_pooling_none.yaml` | no |
| `cassette::error_matrix::embeddings_without_the_flag_are_a_501` | shared_provider (provider) | embeddings | `llamacpp/error_matrix/embeddings_without_the_flag.yaml` | no |
| `cassette::error_matrix::rerank_with_an_empty_document_list_is_a_400` | shared_provider (provider) | rerank | `llamacpp/error_matrix/rerank_empty_documents.yaml` | no |
| `cassette::error_matrix::rerank_without_a_reranker_is_a_501` | shared_provider (provider) | rerank | `llamacpp/error_matrix/rerank_without_a_reranker.yaml` | no |
| `cassette::error_matrix::streaming_context_overflow_matches_the_blocking_envelope` | shared_provider (provider) | model-stream | `llamacpp/error_matrix/context_overflow_streaming.yaml`<br>`llamacpp/error_matrix/context_overflow_blocking.yaml` | no |
| `cassette::error_matrix::the_api_key_the_provider_sends_is_accepted` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/api_key_is_accepted.yaml` | no |
| `cassette::error_matrix::the_model_listing_is_public_even_on_a_keyed_server` | shared_provider (provider) | listing | `llamacpp/error_matrix/model_listing_is_public.yaml`<br>`llamacpp/error_matrix/missing_api_key_is_401.yaml` | no |
| `cassette::error_matrix::tools_without_jinja_are_a_500` | shared_provider (provider) | model-completion | `llamacpp/error_matrix/tools_without_jinja.yaml` | no |
| `cassette::error_matrix::verify_fails_without_the_key_and_succeeds_with_it` | shared_provider (provider) | verify | `llamacpp/error_matrix/verify_rejects_a_missing_key.yaml`<br>`llamacpp/error_matrix/verify_accepts_the_key.yaml` | no |
| `cassette::image_tool_result::a_tool_result_image_is_read_by_the_model` | shared_provider (provider) | model-completion | `llamacpp/image_tool_result/a_tool_result_image_is_read_by_the_model.yaml` | no |
| `cassette::image_tool_result::the_same_image_in_a_user_message_is_read_too` | shared_provider (provider) | model-completion | `llamacpp/image_tool_result/the_same_image_in_a_user_message_is_read_too.yaml` | no |
| `cassette::matrix_index::every_matrix_module_names_all_of_its_cells_in_its_table` | infrastructure (corpus) | source-guard | none | no |
| `cassette::matrix_index::the_suite_index_mentions_every_cassette_module` | infrastructure (corpus) | source-guard | none | no |
| `cassette::matrix_index::the_untabulated_list_has_no_stale_entries` | infrastructure (corpus) | source-guard | none | no |
| `cassette::model_family_matrix::even_a_zero_argument_call_streams_as_two_fragments` | infrastructure (corpus) | fixture-guard | `llamacpp/streaming_tools/raw_stream_emits_required_zero_arg_tool_call.yaml` | no |
| `cassette::model_family_matrix::gemma_family_has_no_tool_calling_in_its_template` | shared_provider (provider) | model-completion | `llamacpp/model_family_matrix/gemma_tool_request_degrades_to_text.yaml` | no |
| `cassette::model_family_matrix::llama_family_calls_a_tool` | shared_provider (provider) | model-completion | `llamacpp/model_family_matrix/llama_blocking_tool_call.yaml` | no |
| `cassette::model_family_matrix::llama_family_streams_tool_call_arguments_as_deltas` | shared_provider (provider) | model-completion | `llamacpp/model_family_matrix/llama_streaming_tool_call.yaml` | no |
| `cassette::model_family_matrix::mistral_family_calls_a_tool` | shared_provider (provider) | model-completion | `llamacpp/model_family_matrix/mistral_blocking_tool_call.yaml` | no |
| `cassette::model_family_matrix::mistral_family_streams_tool_call_arguments_as_deltas` | shared_provider (provider) | model-completion | `llamacpp/model_family_matrix/mistral_streaming_tool_call.yaml` | no |
| `cassette::models::list_models_smoke` | shared_provider (provider) | listing | `llamacpp/models/list_models_smoke.yaml` | no |
| `cassette::multimodal_matrix::a_malformed_data_uri_is_a_400` | shared_provider (provider) | model-completion | `llamacpp/multimodal_matrix/malformed_data_uri.yaml` | no |
| `cassette::multimodal_matrix::a_url_the_server_cannot_fetch_is_a_500` | shared_provider (provider) | model-completion | `llamacpp/multimodal_matrix/unfetchable_image_url.yaml` | no |
| `cassette::multimodal_matrix::a_video_part_is_refused_even_though_props_advertises_video` | shared_provider (provider) | model-completion | `llamacpp/multimodal_matrix/video_part_is_refused.yaml`<br>`llamacpp/unmapped_surface/props.yaml` | no |
| `cassette::multimodal_matrix::an_image_and_a_tool_reach_the_model_together` | shared_provider (provider) | model-completion | `llamacpp/multimodal_matrix/image_plus_tools.yaml` | no |
| `cassette::multimodal_matrix::an_image_to_a_text_only_server_names_the_missing_mmproj` | shared_provider (provider) | model-completion | `llamacpp/multimodal_matrix/image_without_mmproj.yaml` | no |
| `cassette::multimodal_matrix::two_images_in_one_turn_keep_their_order` | shared_provider (provider) | model-completion | `llamacpp/multimodal_matrix/two_images_keep_their_order.yaml` | no |
| `cassette::prompt_caching::blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows` | shared_provider (provider) | cache-probe | `llamacpp/prompt_caching/blocking_probe.yaml` | no |
| `cassette::prompt_caching::cache_prompt_false_turns_the_cache_off_for_that_turn_only` | shared_provider (provider) | cache-probe | `llamacpp/prompt_caching/cache_prompt_disabled.yaml` | no |
| `cassette::prompt_caching::streaming_probe_survives_the_streaming_accumulator` | shared_provider (provider) | cache-probe | `llamacpp/prompt_caching/streaming_probe.yaml` | no |
| `cassette::prompt_caching::timings_cache_n_agrees_with_the_normalized_cached_tokens` | infrastructure (corpus) | fixture-guard | `llamacpp/prompt_caching/blocking_probe.yaml`<br>`llamacpp/prompt_caching/cache_prompt_disabled.yaml`<br>`llamacpp/prompt_caching/agent_loop.yaml` | no |
| `cassette::raw_capture_matrix::normalized_fields_equal_raw_renormalized` | shared_provider (provider) | model-raw | `llamacpp/raw_capture_matrix/normalized_fields_equal_raw_renormalized.yaml` | no |
| `cassette::raw_capture_matrix::raw_exposes_envelope_fields` | shared_provider (provider) | model-raw | `llamacpp/raw_capture_matrix/raw_exposes_envelope_fields.yaml` | no |
| `cassette::raw_capture_matrix::raw_preserves_the_timings_the_openai_type_drops` | shared_provider (provider) | model-raw | `llamacpp/raw_capture_matrix/raw_preserves_timings.yaml` | no |
| `cassette::raw_capture_matrix::raw_round_trips_provider_type` | shared_provider (provider) | model-raw | `llamacpp/raw_capture_matrix/raw_round_trips_provider_type.yaml` | no |
| `cassette::raw_stream_capture_matrix::stream_raw_exposes_envelope_fields` | shared_provider (provider) | model-stream | `llamacpp/raw_stream_capture_matrix/stream_raw_exposes_envelope_fields.yaml` | no |
| `cassette::raw_stream_capture_matrix::stream_raw_preserves_llamacpp_timings` | shared_provider (provider) | model-stream | `llamacpp/raw_stream_capture_matrix/stream_raw_preserves_llamacpp_timings.yaml` | no |
| `cassette::raw_stream_capture_matrix::stream_raw_terminal_round_trips_provider_type` | shared_provider (provider) | model-stream | `llamacpp/raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type.yaml` | no |
| `cassette::rerank_matrix::a_single_document_is_still_a_ranking` | shared_provider (provider) | rerank | `llamacpp/rerank_matrix/single_document.yaml` | no |
| `cassette::rerank_matrix::multiple_documents_come_back_ranked` | shared_provider (provider) | rerank | `llamacpp/rerank_matrix/multiple_documents.yaml` | no |
| `cassette::rerank_matrix::scores_are_raw_logits_and_may_be_negative` | shared_provider (provider) | rerank | `llamacpp/rerank_matrix/negative_scores.yaml` | no |
| `cassette::rerank_matrix::top_n_below_the_document_count_truncates` | shared_provider (provider) | rerank | `llamacpp/rerank_matrix/top_n_truncates.yaml` | no |
| `cassette::rerank_matrix::top_n_beyond_the_document_count_is_clamped` | shared_provider (provider) | rerank | `llamacpp/rerank_matrix/top_n_beyond_count.yaml` | no |
| `cassette::rerank_matrix::top_n_zero_returns_an_empty_ranking` | shared_provider (provider) | rerank | `llamacpp/rerank_matrix/top_n_zero.yaml` | no |
| `cassette::response_identity_matrix::the_response_id_reaches_the_caller_on_both_transports` | shared_provider (provider) | model-completion | `llamacpp/response_identity_matrix/blocking_response_id.yaml`<br>`llamacpp/response_identity_matrix/streaming_response_id.yaml` | no |
| `cassette::response_identity_matrix::the_transport_request_id_is_absent_because_the_server_sends_none` | shared_provider (provider) | model-completion | `llamacpp/response_identity_matrix/blocking_identity.yaml`<br>`llamacpp/response_identity_matrix/streaming_identity.yaml` | no |
| `cassette::response_identity_matrix::the_typed_route_reproduces_the_normalized_one` | shared_provider (provider) | model-completion | `llamacpp/response_identity_matrix/typed_route_parity.yaml` | no |
| `cassette::response_shape_matrix::logprobs_survive_into_the_raw_response` | shared_provider (provider) | model-completion | `llamacpp/response_shape_matrix/logprobs.yaml` | no |
| `cassette::response_shape_matrix::n_greater_than_one_answers_from_candidate_zero_on_both_transports` | shared_provider (provider) | model-completion | `llamacpp/response_shape_matrix/two_candidates_blocking.yaml`<br>`llamacpp/response_shape_matrix/two_candidates_streaming.yaml` | no |
| `cassette::response_shape_matrix::reasoning_content_reaches_the_caller_on_both_transports` | shared_provider (provider) | model-completion | `llamacpp/response_shape_matrix/reasoning_blocking.yaml`<br>`llamacpp/response_shape_matrix/reasoning_streaming.yaml` | no |
| `cassette::response_shape_matrix::the_finish_reason_vocabulary_is_covered_end_to_end` | infrastructure (corpus) | fixture-guard | none | no |
| `cassette::sampling_matrix::a_cap_past_the_context_is_clamped` | infrastructure (corpus) | fixture-guard | `llamacpp/error_matrix/oversized_output_cap.yaml` | no |
| `cassette::sampling_matrix::a_fixed_seed_and_an_absent_seed_are_both_accepted` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/seed_fixed.yaml`<br>`llamacpp/sampling_matrix/seed_absent.yaml` | no |
| `cassette::sampling_matrix::a_normal_cap_lets_the_turn_stop_on_its_own` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/max_tokens_normal.yaml` | no |
| `cassette::sampling_matrix::a_one_token_cap_truncates_with_finish_reason_length` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/max_tokens_one.yaml` | no |
| `cassette::sampling_matrix::a_single_stop_sequence_truncates_the_answer` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/stop_single.yaml` | no |
| `cassette::sampling_matrix::a_stop_sequence_that_never_matches_changes_nothing` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/stop_never_fires.yaml` | no |
| `cassette::sampling_matrix::additional_params_wins_over_the_typed_field_it_collides_with` | shared_provider (provider) | request-serialization | none | no |
| `cassette::sampling_matrix::several_stop_sequences_fire_on_whichever_comes_first` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/stop_multiple.yaml` | no |
| `cassette::sampling_matrix::stop_matching_is_case_sensitive` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/stop_case_sensitive.yaml` | no |
| `cassette::sampling_matrix::temperature_zero_and_nonzero_both_reach_the_wire` | shared_provider (provider) | model-completion | `llamacpp/sampling_matrix/temperature_zero.yaml`<br>`llamacpp/sampling_matrix/temperature_nonzero.yaml` | no |
| `cassette::streaming_tools::raw_followup_uses_tool_result_without_new_tool_calls` | shared_provider (provider) | model-stream | `llamacpp/streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls.yaml` | no |
| `cassette::streaming_tools::raw_stream_emits_required_zero_arg_tool_call` | shared_provider (provider) | model-stream | `llamacpp/streaming_tools/raw_stream_emits_required_zero_arg_tool_call.yaml` | no |
| `cassette::streaming_tools::raw_stream_surfaces_two_distinct_tool_calls_before_text` | shared_provider (provider) | model-stream | `llamacpp/streaming_tools/raw_stream_surfaces_two_distinct_tool_calls_before_text.yaml` | no |
| `cassette::structured_output_matrix::a_gbnf_grammar_through_additional_params_is_enforced` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/gbnf_grammar_is_enforced.yaml` | no |
| `cassette::structured_output_matrix::a_schema_alongside_tools_is_deferred_so_the_tool_stays_reachable` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/schema_alongside_tools.yaml` | no |
| `cassette::structured_output_matrix::a_schema_and_a_grammar_together_are_rejected` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/schema_and_grammar_conflict.yaml` | no |
| `cassette::structured_output_matrix::a_schema_the_smoke_tier_cannot_hold_is_still_held_by_the_server` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/smoke_tier_cannot_escape_the_grammar.yaml` | no |
| `cassette::structured_output_matrix::json_object_response_format_constrains_nothing` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/json_object_is_a_no_op.yaml` | no |
| `cassette::structured_output_matrix::json_schema_response_format_is_enforced_by_the_server` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/json_schema_is_enforced.yaml` | no |
| `cassette::structured_output_matrix::response_format_and_a_grammar_silently_let_the_schema_win` | shared_provider (provider) | model-completion | `llamacpp/structured_output_matrix/response_format_beats_grammar_silently.yaml` | no |
| `cassette::tool_matrix::a_tool_result_carrying_json_reaches_the_model` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/tool_result_json.yaml` | no |
| `cassette::tool_matrix::a_tool_result_carrying_text_reaches_the_model` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/tool_result_text.yaml` | no |
| `cassette::tool_matrix::a_zero_argument_tool_is_called_with_an_empty_object` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/zero_argument_tool.yaml` | no |
| `cassette::tool_matrix::the_smoke_tier_round_trip_is_covered_elsewhere` | infrastructure (corpus) | fixture-guard | `llamacpp/tools/tools_roundtrip.yaml` | no |
| `cassette::tool_matrix::three_tools_are_all_advertised_and_the_right_one_is_chosen` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/three_tools.yaml` | no |
| `cassette::tool_matrix::tool_choice_auto_lets_the_model_decide` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/tool_choice_auto.yaml` | no |
| `cassette::tool_matrix::tool_choice_none_suppresses_the_parsed_call` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/tool_choice_none.yaml` | no |
| `cassette::tool_matrix::tool_choice_required_forces_a_call` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/tool_choice_required.yaml` | no |
| `cassette::tool_matrix::tool_choice_specific_is_refused_before_the_request_is_sent` | shared_provider (provider) | client-refusal | `llamacpp/tool_matrix/tool_choice_required.yaml` | no |
| `cassette::tool_matrix::tool_choice_specific_is_refused_on_the_streaming_path_too` | shared_provider (provider) | client-refusal | none | no |
| `cassette::tool_matrix::two_independent_calls_arrive_in_one_turn` | shared_provider (provider) | model-completion | `llamacpp/tool_matrix/parallel_calls.yaml` | no |
| `cassette::truncation_matrix::a_complete_call_under_the_same_cap_survives` | shared_provider (provider) | model-completion | `llamacpp/truncation_matrix/complete_call_control.yaml` | no |
| `cassette::truncation_matrix::a_tool_call_cut_mid_arguments_does_not_destroy_the_turn` | shared_provider (provider) | model-completion | `llamacpp/truncation_matrix/tool_call_cut_mid_arguments.yaml` | no |
| `cassette::truncation_matrix::the_streaming_path_drops_the_same_cut_call` | shared_provider (provider) | model-completion | `llamacpp/truncation_matrix/streaming_tool_call_cut_mid_arguments.yaml` | no |
| `cassette::unmapped_surface::props_states_which_model_and_modalities_produced_this_corpus` | shared_provider (provider) | verify | `llamacpp/unmapped_surface/props.yaml` | no |
| `cassette::unmapped_surface::the_model_listing_reads_the_openai_half_of_a_hybrid_body` | shared_provider (provider) | listing | `llamacpp/unmapped_surface/models_envelope.yaml` | no |
| `cassette::unmapped_surface::the_responses_api_is_reachable_but_rig_does_not_route_to_it` | shared_provider (provider) | model-completion | `llamacpp/unmapped_surface/responses_api.yaml` | no |
| `cassette::unmapped_surface::transcription_is_a_501_unless_the_loaded_model_hears` | infrastructure (corpus) | raw-http | `llamacpp/unmapped_surface/transcription_not_supported.yaml` | no |
