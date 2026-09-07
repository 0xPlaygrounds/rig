# Ollama agent output family

Agent cells whose observable is schema-constrained output, including the three
`OutputMode`s and the tool-plus-schema interaction (#1928).

## Rules

- **agent-schema** — `client.agent(MODEL).output_schema::<T>()` or
  `.output_schema_raw(schema)[.output_mode(OutputMode::Prompted)]`
  `.additional_params(think).build().prompt(..).await.expect(..)`; the output
  (or its first JSON object in Prompted mode) is parsed and required keys are
  nonempty.
- **agent-schema-tools** — `client.agent(MODEL).tool(WeatherTool).output_schema_raw(schema)
  [.output_mode(OutputMode::Native)].additional_params(think).default_max_turns(n)
  .build().prompt(..)` or `.stream_prompt(..).max_turns(5).stream()` drained by
  `collect_stream_final_response`; schema JSON parsed and, where the cell says
  so, the real tool's `call_count >= 1`.
- **live-agent** — `structured_output_prompt` against a live server.

## Phase 2 obligations

Output mode selection (Tool default, Native, Prompted) is configuration to
preserve exactly; the synthetic output tool's arguments becoming the final
response string on the streaming path is an original obligation, not a
normalization.
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
| `cassette::agentic::native_mode_emits_structured_output` | agent (output) | agent-schema-tools | `ollama/agentic/native_mode.yaml` | no |
| `cassette::agentic::prompted_mode_returns_parseable_json` | agent (output) | agent-schema | `ollama/agentic/prompted_mode.yaml` | no |
| `cassette::agentic::streaming_structured_output_with_tools` | agent (output) | agent-schema-tools | `ollama/agentic/streaming_structured_output_with_tools.yaml` | no |
| `cassette::agentic::structured_output_raw_with_thinking` | agent (output) | agent-schema | `ollama/structured_output/raw_with_thinking.yaml` | no |
| `cassette::agentic::structured_output_with_tools_and_thinking` | agent (output) | agent-schema-tools | `ollama/agentic/structured_output_with_tools.yaml` | no |
| `cassette::structured_output::structured_output_smoke` | agent (output) | agent-schema | `ollama/structured_output/structured_output_smoke.yaml` | no |
| `structured_output::structured_output_prompt` | agent (output) | live-agent | none | yes |
