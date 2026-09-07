# ChatGPT provider-only family

## Rules

- **model-completion** — `client.completion_model(GPT_5_4)[.with_strict_tools()]
  .completion(request)` with tool definitions / `tool_choice` / hand-built
  long histories, `.expect` or `.expect_err` on the preserved status and body.
- **model-raw** — `raw_completion(request)` and `CompletionResponse::raw`
  round trips against the recorded terminal `response.completed` frame (six
  `unrecorded` cells).
- **model-stream** — `client.completion_model(..).stream(request)` drained by
  `terminal_of`, `collect_raw_stream_observation`,
  `assert_stream_contains_zero_arg_tool_call_named` or a manual loop; the live
  `completion::system_messages_are_lifted_into_instructions` cell drains text
  deltas by hand.
- **reasoning-roundtrip-driver** — `reasoning::run_reasoning_roundtrip_streaming`
  over `live_client().completion_model(LIVE_MODEL)`; hand-built requests and
  direct `model.stream`; no `AgentBuilder`.

## Phase 2

Not ported; the unrecorded cells cannot be recorded without ChatGPT OAuth and
are not recaptured by this lane.
## Fixture kind

Cassette cells are **hosted ChatGPT Codex backend recordings**
(`https://chatgpt.com/backend-api/codex`, model `GPT_5_4`) replayed through
`with_chatgpt_cassette` / `with_chatgpt_cassette_default_instructions` (strict
ordered `ProviderCassette`, placeholder `CHATGPT_ACCESS_TOKEN` /
`CHATGPT_ACCOUNT_ID`, `finish_after_test` exhaustion). The non-interactive
OAuth cell seeds a temp `auth.json` from the cassette keys with
`allow_device_flow(false)`. Every `tests/providers/chatgpt/*.rs` cell outside
`cassette/` is `#[ignore]` live-only: ChatGPT needs an OAuth device flow or a
cached `auth.json`, neither available here; they stay `supplemental_live`
(OAuth), not counted as ported. The `unrecorded` raw matrices have complete
bodies but no fixture.

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
| `cassette::codex_behaviors::store_false_and_prompt_cache_fields_roundtrip` | shared_provider (provider) | model-raw | `chatgpt/codex_behaviors/store_false_and_prompt_cache_fields_roundtrip.yaml` | no |
| `cassette::codex_behaviors::strict_tools_opt_in_roundtrip` | shared_provider (provider) | model-completion | `chatgpt/codex_behaviors/strict_tools_opt_in_roundtrip.yaml` | no |
| `cassette::codex_sessions::long_history_replay_nonstreaming` | shared_provider (provider) | model-completion | `chatgpt/codex_sessions/long_history_replay_nonstreaming.yaml` | no |
| `cassette::codex_tool_args::nested_arguments_streaming` | shared_provider (provider) | model-stream | `chatgpt/codex_tool_args/nested_arguments_streaming.yaml` | no |
| `cassette::codex_tool_args::unicode_arguments_streaming` | shared_provider (provider) | model-stream | `chatgpt/codex_tool_args/unicode_arguments_streaming.yaml` | no |
| `cassette::codex_tool_args::zero_argument_tool_call_nonstreaming` | shared_provider (provider) | model-completion | `chatgpt/codex_tool_args/zero_argument_tool_call_nonstreaming.yaml` | no |
| `cassette::codex_tool_args::zero_argument_tool_call_streaming` | shared_provider (provider) | model-stream | `chatgpt/codex_tool_args/zero_argument_tool_call_streaming.yaml` | no |
| `cassette::codex_tool_choice::none_suppresses_tool_calls` | shared_provider (provider) | model-completion | `chatgpt/codex_tool_choice/none_suppresses_tool_calls.yaml` | no |
| `cassette::codex_tool_choice::required_forces_a_tool_call` | shared_provider (provider) | model-completion | `chatgpt/codex_tool_choice/required_forces_a_tool_call.yaml` | no |
| `cassette::codex_tool_choice::specific_multiple_functions_use_allowed_tools` | shared_provider (provider) | model-completion | `chatgpt/codex_tool_choice/specific_multiple_functions_use_allowed_tools.yaml` | no |
| `cassette::codex_tool_choice::specific_single_function_targets_named_tool` | shared_provider (provider) | model-completion | `chatgpt/codex_tool_choice/specific_single_function_targets_named_tool.yaml` | no |
| `cassette::http_errors::nonstreaming_unauthorized_preserves_status_and_body` | shared_provider (provider) | model-completion | `chatgpt/http_errors/nonstreaming_unauthorized_preserves_status_and_body.yaml` | no |
| `cassette::raw_capture_matrix::normalized_fields_equal_raw_renormalized` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_capture_matrix::raw_exposes_response_envelope` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_capture_matrix::raw_round_trips_provider_type` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_completion_parity_matrix::empty_output_fallback_still_carries_raw` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_completion_parity_matrix::raw_normalize_reproduces_completion` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_completion_parity_matrix::raw_normalize_reproduces_completion_with_tool_call` | shared_provider (provider) | model-raw | none | yes |
| `cassette::raw_stream_capture_matrix::stream_raw_exposes_terminal_status` | shared_provider (provider) | model-stream | none | yes |
| `cassette::raw_stream_capture_matrix::stream_raw_terminal_round_trips_provider_type` | shared_provider (provider) | model-stream | none | yes |
| `cassette::streaming_tools::nonstreaming_tool_call_completed_response_without_output` | shared_provider (provider) | model-completion | `chatgpt/streaming_tools/tool_call_completed_response_without_output.yaml` | no |
| `cassette::streaming_tools::stream_tool_call_completed_response_without_output` | shared_provider (provider) | model-stream | `chatgpt/streaming_tools/tool_call_completed_response_without_output.yaml` | no |
| `completion::system_messages_are_lifted_into_instructions` | shared_provider (provider) | model-stream | none | yes |
| `reasoning_roundtrip::streaming` | shared_provider (provider) | reasoning-roundtrip-driver | none | yes |
