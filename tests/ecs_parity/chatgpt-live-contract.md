# ChatGPT live-only agent family

Every `#[ignore]` agent cell in `tests/providers/chatgpt/*.rs`: auth flows,
default instructions, extractor smoke/usage/multi-extract, permission control,
reasoning tool round trip, request hook, streaming and streaming tools, plus
`agent::completion_smoke`.

## Rule

- **live-agent** — `live_client()` / `live_builder()` (OAuth cache or
  `CHATGPT_ACCESS_TOKEN`) builds `client.agent(LIVE_MODEL)` or
  `client.extractor::<T>(LIVE_MODEL)` and drives `prompt`, `stream_prompt`,
  `stream_chat`, `extract` or a hook-instrumented run exactly as the cell's
  reason states; `#[ignore]`, no fixture on disk.

## Credential disposition

ChatGPT requires OAuth (device flow or a cached `auth.json`); no credential is
configured here and none of these cells can be captured. They stay
`supplemental_live`, are counted as discovered only, and are not ported. The
agent semantics they exercise (extractor usage, permission control, request
hook, streaming tools) are covered by the cassette-backed counterparts in other
trees.

## Exception

`completion::system_messages_are_lifted_into_instructions` and
`reasoning_roundtrip::streaming` are live but provider-only
(`chatgpt-provider-contract.md`).

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
| `agent::completion_smoke` | agent (smoke) | live-agent | none | yes |
| `auth::oauth_device_flow_authorize_and_cached_completion_smoke` | agent (live) | live-agent | none | yes |
| `auth::refresh_token_cache_authorize_and_completion_smoke` | agent (live) | live-agent | none | yes |
| `completion::default_instructions_fill_required_instructions` | agent (live) | live-agent | none | yes |
| `extractor::extractor_smoke` | agent (live) | live-agent | none | yes |
| `extractor_usage::extract_and_extract_with_usage_return_same_data` | agent (live) | live-agent | none | yes |
| `extractor_usage::extract_backward_compatibility` | agent (live) | live-agent | none | yes |
| `extractor_usage::extract_with_chat_history_with_usage_works` | agent (live) | live-agent | none | yes |
| `extractor_usage::extract_with_usage_returns_data_and_usage` | agent (live) | live-agent | none | yes |
| `extractor_usage::usage_tracking_works_for_different_schemas` | agent (live) | live-agent | none | yes |
| `multi_extract::batch_multi_extract_chain` | agent (live) | live-agent | none | yes |
| `permission_control::permission_control_prompt_example` | agent (live) | live-agent | none | yes |
| `permission_control::permission_control_streaming_example` | agent (live) | live-agent | none | yes |
| `reasoning_tool_roundtrip::streaming` | agent (live) | live-agent | none | yes |
| `request_hook::request_hook_records_prompt_and_response` | agent (live) | live-agent | none | yes |
| `streaming::example_streaming_prompt` | agent (live) | live-agent | none | yes |
| `streaming::streaming_smoke` | agent (live) | live-agent | none | yes |
| `streaming_tools::example_streaming_with_tools` | agent (live) | live-agent | none | yes |
| `streaming_tools::streaming_tools_smoke` | agent (live) | live-agent | none | yes |
