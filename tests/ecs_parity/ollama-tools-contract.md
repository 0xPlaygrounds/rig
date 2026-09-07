# Ollama agent tools family

Agent cells with real tool round trips, including the thinking-enabled ones.

## Rules

- **agent-conformance** — `rig_agent::test_utils::model_conformance::{optional_argument,
  sequential_tools}(client.completion_model(MODEL), configure)` build
  `AgentBuilder::new(model)..tool(..).default_max_turns(4).build().prompt(..)`,
  count real tool invocations, require a tool round trip in `result.messages`
  and return a `ScenarioReport` (`?`-propagated `.expect`).
- **agent-stream-tools** — `client.agent(MODEL).tool(Adder).tool(Subtract)
  .additional_params({"think": false}).build().stream_prompt(..).max_turns(3)
  .stream().await` drained by `collect_stream_final_response`, or the
  reasoning twin `.tool(WeatherTool).additional_params({"think": true})
  .stream_chat(.., Vec::new()).max_turns(3)` collected by
  `reasoning::collect_stream_stats` then `assert_universal`.
- **agent-chat-tools** — `client.agent(MODEL).tool(WeatherTool)
  .additional_params({"think": true}).default_max_turns(2).build()
  .chat(.., &mut history).await.expect(..)`; `assert_nonstreaming_universal`,
  `assert_chat_history_preserves_reasoning_tool_roundtrip` and an assistant
  `Reasoning` part in the history are required (#1926).
- **live-agent** — `example_streaming_with_tools` against a live server.

## Exceptions

`cassette::reasoning_roundtrip::*` are not agent cells: the shared driver builds
`CompletionRequest`s by hand and calls `model.completion` / `model.stream`
directly (see `ollama-provider-contract.md`).

## Phase 2 obligations

The conformance cells may reuse only neutral tool definitions and validators
from `rig_agent::test_utils` (as `gemini-tools-contract.md` did), never the
runner. Reasoning preservation in history is a level-1/2 obligation on the
actual run-child utterances; tool counts come from the real `WeatherTool`
counter through native `ToolAdapter`s.

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
| `cassette::reasoning_tool_roundtrip::nonstreaming` | agent (tools) | agent-chat-tools | `ollama/reasoning_tool_roundtrip/nonstreaming.yaml` | no |
| `cassette::reasoning_tool_roundtrip::streaming` | agent (tools) | agent-stream-tools | `ollama/reasoning_tool_roundtrip/streaming.yaml` | no |
| `cassette::streaming_tools::streaming_tools_smoke` | agent (tools) | agent-stream-tools | `ollama/streaming_tools/streaming_tools_smoke.yaml` | no |
| `cassette::tools::tool_with_optional_argument` | agent (tools) | agent-conformance | `ollama/tools/optional_argument.yaml` | no |
| `cassette::tools::two_tools_nonstreaming_chain` | agent (tools) | agent-conformance | `ollama/tools/two_tools_nonstreaming.yaml` | no |
| `streaming_tools::example_streaming_with_tools` | agent (tools) | live-agent | none | yes |
