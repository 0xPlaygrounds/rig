# Copilot agent tools family

## Rules

- **agent-stream-tools** — `client.agent(LIVE_MODEL).preamble(..).tool(..)
  [.default_max_turns(2)].build().stream_prompt(..)[.max_turns(8)].stream().await`
  drained by `collect_stream_final_response` (`assert_mentions_expected_number(-3)`)
  or `collect_stream_observation` (`assert_two_tool_roundtrip_contract`).
- **agent-typed-tools** — `client.agent(live_responses_model()).tool(WeatherTool)
  .default_max_turns(2).build().prompt_typed::<WeatherResponse>(..).await?`;
  `ensure!(call_count >= 1)` and `assert_weather_tool_roundtrip_response`.
- **agent-chat-tools** — `client.agent(live_responses_model()).max_tokens(4096)
  .tool(WeatherTool).additional_params({"reasoning": {"effort": "high"}})
  .default_max_turns(2).build().chat(.., &mut Vec::new()).await.expect(..)`
  then `reasoning::assert_nonstreaming_universal`.
- **live-agent** — the streaming reasoning tool round trip against
  `live_client()`.

## Exceptions

`streaming_tools::raw_*` cells and `reasoning_roundtrip::*` are provider-only
(`copilot-provider-contract.md`): the former call `model.stream` directly, the
latter use the shared driver that builds requests by hand.

## Phase 2 obligations

Real `WeatherTool` counters through native `ToolAdapter`s; reasoning effort in
`additional_params` preserved; Responses-route adapters for codex models.
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
| `reasoning_tool_roundtrip::nonstreaming` | agent (tools) | agent-chat-tools | `copilot/reasoning_tool_roundtrip/nonstreaming.yaml` | no |
| `reasoning_tool_roundtrip::streaming` | agent (tools) | live-agent | none | yes |
| `streaming_tools::example_streaming_with_tools` | agent (tools) | agent-stream-tools | `copilot/streaming_tools/example_streaming_with_tools.yaml` | no |
| `streaming_tools::streaming_tools_smoke` | agent (tools) | agent-stream-tools | `copilot/streaming_tools/streaming_tools_smoke.yaml` | no |
| `streaming_tools::streaming_tools_surface_two_distinct_tool_calls_before_final_answer` | agent (tools) | agent-stream-tools | `copilot/streaming_tools/streaming_tools_surface_two_distinct_tool_calls_before_final_answer.yaml` | no |
| `typed_prompt_tools::prompt_typed_with_tool_call_roundtrip` | agent (tools) | agent-typed-tools | `copilot/typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml` | no |
