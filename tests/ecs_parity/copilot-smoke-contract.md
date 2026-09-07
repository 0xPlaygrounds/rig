# Copilot agent smoke family

Text-only agent cells: the two recorded streaming smokes, the recorded codex
routing cell, the recorded non-interactive OAuth cell, and the live auth /
routing / all-models cells. `agent::completion_smoke` is already mapped under
`provider-completions`.

## Rules

- **agent-prompt** — `client.agent(model).preamble(BASIC_PREAMBLE).build()
  .prompt(..).await.expect(..)` then `assert_nonempty_response`; the OAuth cell
  first builds the client with `.oauth().allow_device_flow(false).token_dir(temp)`
  seeded with the cassette api-key record and calls `client.authorize().await.expect`.
- **agent-stream** — `client.agent(LIVE_MODEL).preamble(..)[.temperature(0.5)]
  .build().stream_prompt(..).stream().await` drained by
  `collect_stream_final_response` then `assert_nonempty_response`.
- **live-agent** — API-key, access-token, device-flow and bootstrap-refresh
  clients call `authorize()` then `client.agent(LIVE_MODEL).prompt(BASIC_PROMPT)`;
  `all_models_completion_smoke` iterates every listed model live.

## Credential disposition

`agent::all_models_completion_smoke` (live): requires Copilot credentials or an
OAuth cache; no capture possible; stays `supplemental_live`. The cassette-backed
`agent::completion_smoke` is the mapped counterpart.

## Phase 2 obligations

The non-interactive OAuth cell must keep its client construction (temp token
dir seeded from the cassette key, `allow_device_flow(false)`, `authorize()`
before the run); the codex routing cell must go through the Responses route.
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
| `agent::all_models_completion_smoke` | agent (smoke) | live-agent | none | yes |
| `auth::access_token_bootstrap_refresh_and_completion_smoke` | agent (smoke) | live-agent | none | yes |
| `auth::api_key_completion_smoke` | agent (smoke) | live-agent | none | yes |
| `auth::github_access_token_completion_smoke` | agent (smoke) | live-agent | none | yes |
| `auth::oauth_device_flow_authorize_and_cached_completion_smoke` | agent (smoke) | live-agent | none | yes |
| `noninteractive_oauth_cassette::cached_oauth_allows_noninteractive_completion` | agent (smoke) | agent-prompt | `copilot/noninteractive_oauth/cached_oauth_allows_noninteractive_completion.yaml` | no |
| `routing::chat_models_route_through_chat_completions` | agent (smoke) | live-agent | none | yes |
| `routing::codex_models_route_through_responses` | agent (smoke) | agent-prompt | `copilot/routing/codex_models_route_through_responses.yaml` | no |
| `streaming::example_streaming_prompt` | agent (smoke) | agent-stream | `copilot/streaming/example_streaming_prompt.yaml` | no |
| `streaming::streaming_smoke` | agent (smoke) | agent-stream | `copilot/streaming/streaming_smoke.yaml` | no |
