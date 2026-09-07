# llama.cpp agent output family

Agent cells whose observable is a typed or schema-constrained output. Native
counterparts belong in `tests/providers/llamacpp/cassette/ecs_output.rs`; the
extractor cells of this tree are already mapped under the runtime-owned
`extractor-smoke` / `extractor-usage` contracts.

## Rules

- **agent-typed** — `client.agent(..)[.preamble(WEATHER_PREAMBLE).temperature(0)]
  .build().prompt_typed::<T>(..).await.expect(..)`; the typed output is
  checked with `assert_smoke_structured_output` or `assert_weather_forecast`,
  and the extended-details cell also requires `usage.total_tokens > 0`.
- **agent-schema** — `client.agent(..).output_schema::<WeatherForecast>().build()
  .prompt(..).await.expect(..)`; `response.output` is parsed with `serde_json`
  and checked with `assert_weather_forecast`.
- **agent-extractor** — three `client.extractor::<T>(model).preamble(..).retries(2)
  .build()` extractors driven concurrently (`futures::try_join!` inside
  `.buffered(4)`) under `CassetteSpec::unordered`, with `anyhow::ensure!` on
  three nonempty formatted responses.

## Phase 2 obligations

Reserved output-tool naming, `retries(2)` and the unordered cassette matching
must be preserved; the batch cell's exhaustion means every recorded
interaction is consumed by some extraction. Typed deserialization happens on
the actual run result; `usage` comes from the run's Usage component.

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
| `cassette::multi_extract::batch_multi_extract_chain` | agent (output) | agent-extractor | `llamacpp/multi_extract/batch_multi_extract_chain.yaml` | no |
| `cassette::structured_output::output_schema_structured_output` | agent (output) | agent-schema | `llamacpp/structured_output/output_schema_structured_output.yaml` | no |
| `cassette::structured_output::prompt_typed_extended_details_structured_output` | agent (output) | agent-typed | `llamacpp/structured_output/prompt_typed_extended_details_structured_output.yaml` | no |
| `cassette::structured_output::prompt_typed_structured_output` | agent (output) | agent-typed | `llamacpp/structured_output/prompt_typed_structured_output.yaml` | no |
| `cassette::structured_output::structured_output_smoke` | agent (output) | agent-typed | `llamacpp/structured_output/structured_output_smoke.yaml` | no |
