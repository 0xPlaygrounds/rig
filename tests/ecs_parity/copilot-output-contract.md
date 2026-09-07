# Copilot agent output family

## Rules

- **agent-typed** — `client.agent(LIVE_MODEL)[.preamble(..)].build()
  .prompt_typed::<T>(..).await.expect(..)`; `assert_smoke_structured_output` /
  `assert_weather_forecast`, `usage.total_tokens > 0` for the extended call, and
  a second agent with `.output_schema::<WeatherForecast>().prompt(..)` parsed
  with `serde_json`.
- **agent-extractor** — three `client.extractor::<T>(LIVE_LIGHT_MODEL).preamble(..)
  .retries(2).build()` extractors driven concurrently under
  `CassetteSpec::unordered`; `ensure!` on three nonempty responses.

The extractor smoke and usage cells are already mapped under the runtime-owned
`extractor-smoke` / `extractor-usage` contracts.

## Phase 2 obligations

Same as `llamacpp-output-contract.md`: reserved output tool, `retries(2)`,
unordered matching with full exhaustion, typed deserialization on the actual
run result.

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
| `multi_extract::batch_multi_extract_chain` | agent (output) | agent-extractor | `copilot/multi_extract/batch_multi_extract_chain.yaml` | no |
| `structured_output::prompt_typed_and_output_schema` | agent (output) | agent-typed | `copilot/structured_output/prompt_typed_and_output_schema.yaml` | no |
| `structured_output::structured_output_smoke` | agent (output) | agent-typed | `copilot/structured_output/structured_output_smoke.yaml` | no |
