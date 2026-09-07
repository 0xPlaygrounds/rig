# Ollama agent hooks family

The raw-capture-through-the-agent matrix: hook events must see each attempt's
own raw provider payload on both surfaces.

## Rules

- **agent-raw-hook** — `client.agent(MODEL)[.preamble(TOOLS_PREAMBLE).tool(Adder)]
  .max_tokens(64).additional_params({"think": false}).add_hook(RawProbe).build()`
  then `.prompt(..)[.max_turns(3)].await.expect(..)` or
  `.stream_prompt(..)[.max_turns(3)].stream().await` drained for
  `CompletionCall` / `StreamEvent::Final` / `FinalResponse` items; after closure
  the probe's `CompletionResponse` and `ModelTurnFinished` raw payloads,
  `HookContext::is_streaming` flags, `PromptResponse::completion_calls` raws and
  the last streamed `Final.raw` are compared with the fixture fingerprints
  (`eval_count`, `done_reason`, `total_duration`, `eval_duration`).

## Phase 2 obligations

Raw responses per attempt are a fidelity requirement; the native observation
must expose each successful completion outcome's raw payload in turn order and
distinguish blocking from streamed delivery. The multi-turn cells require two
attempts with distinct payloads and the tool turn first.

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
| `cassette::raw_capture_agent_matrix::hooks_observe_raw_blocking` | agent (hooks) | agent-raw-hook | `ollama/raw_capture_agent_matrix/hooks_observe_raw_blocking.yaml` | no |
| `cassette::raw_capture_agent_matrix::hooks_observe_raw_streamed` | agent (hooks) | agent-raw-hook | `ollama/raw_capture_agent_matrix/hooks_observe_raw_streamed.yaml` | no |
| `cassette::raw_capture_agent_matrix::multi_turn_tool_run_records_distinct_raw_blocking` | agent (hooks) | agent-raw-hook | `ollama/raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking.yaml` | no |
| `cassette::raw_capture_agent_matrix::multi_turn_tool_run_records_distinct_raw_streamed` | agent (hooks) | agent-raw-hook | `ollama/raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed.yaml` | no |
