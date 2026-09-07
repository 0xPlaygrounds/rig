# Ollama agent smoke family

Text-only agent cells. `cassette::agent::{completion_smoke,completion_respects_max_tokens}`
are already mapped under `provider-completions`; the cassette streaming smoke
is the one portable cell here, the rest are live examples.

## Rules

- **agent-stream** — `client.agent(MODEL).preamble(STREAMING_PREAMBLE)
  .additional_params({"think": false}).build().stream_prompt(STREAMING_PROMPT)
  .stream().await` drained by `collect_stream_final_response` then
  `assert_nonempty_response`.
- **live-agent** — `ollama::Client::from_env()` / `Client::new(Nothing)` agents
  driven by `.prompt(..)` (text or `Image`) or `.stream_prompt(..)` against a
  live local server; `#[ignore]`, no fixture.

## Credential disposition

`agent::completion_smoke` (live): no local Ollama server is configured here, no
capture is possible, the row stays `supplemental_live`.

## Phase 2 obligations

The streaming smoke ports like the other `streaming_smoke` cells (drain to EOF,
final response required, nonempty answer) with `think: false` kept in
`additional_params` so the recorded request body matches.
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
| `agent::completion_smoke` | agent (smoke) | live-agent | none | yes |
| `cassette::streaming::streaming_smoke` | agent (smoke) | agent-stream | `ollama/streaming/streaming_smoke.yaml` | no |
| `multimodal::multimodal_image_prompt` | agent (smoke) | live-agent | none | yes |
| `streaming::example_streaming_prompt` | agent (smoke) | live-agent | none | yes |
