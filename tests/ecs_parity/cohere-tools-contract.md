# Cohere agent tools family

## Rules

- **agent-prompt-tools** — `client.agent(CASSETTE_MODEL).preamble(..)
  .tool(IntegerAdder).tool(IntegerSubtract).default_max_turns(2).build()
  .prompt(TOOLS_PROMPT).await.expect(..)` then `assert_mentions_expected_number(-3)`;
  the cache cell instead registers `CacheProbeLookupTool`, uses
  `.prompt(AGENT_CACHE_PROMPT).max_turns(6)`, asserts
  `completion_calls().len() >= 2` and after closure `assert_prefix_stable` and
  `assert_breakpoints_match_support` over the recorded requests.
- **agent-stream-tools** — `client.agent(CASSETTE_MODEL).preamble(..)
  .tool(IntegerAdder).tool(IntegerSubtract).default_max_turns(2).build()
  .stream_prompt(STREAMING_TOOLS_PROMPT).stream().await` collected by
  `collect_stream_observation`; no errors, `tool_calls == ["subtract"]`,
  `tool_results == 1`, `assert_mentions_expected_number(-3)`.
- **live-agent** — `cohere::Client::from_env().agent(..).tool(Adder).tool(Subtract)`
  driven by `prompt` / `stream_prompt` against the live API.

## Exceptions

The five other `cassette::tools` cells and both prompt-caching probes call the
model directly (`cohere-provider-contract.md`). Cohere's cassette tools are the
integer-typed `IntegerAdder` / `IntegerSubtract` copies from
`tests/providers/cohere/support.rs`, not the shared `Adder` / `Subtract`.

## Phase 2 obligations

Exact tool-call sequence and result count from the native observation; the
cache cell's prefix-stability post-closure assertions kept verbatim.
## Fixture kind

Cassette cells are **hosted Cohere recordings** (`command-a-03-2025`, and
`command-a-reasoning-08-2025` for two streaming-grammar cells) replayed through
`with_cohere_cassette` / `with_cohere_prompt_caching_cassette` (strict ordered
`ProviderCassette`, placeholder `COHERE_API_KEY`, `finish_after_test`
exhaustion). The `#[ignore]` cells under `tests/providers/cohere/*.rs` need a
real `COHERE_API_KEY`, which is configured in this environment; only
`agent::completion_smoke` is in the programme's 12-case capture set.

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
| `cassette::prompt_caching::agent_loop_does_not_move_its_own_prefix` | agent (tools) | agent-prompt-tools | `cohere/prompt_caching/agent_loop.yaml` | no |
| `cassette::streaming_tools::streaming_tool_call_roundtrip` | agent (tools) | agent-stream-tools | `cohere/streaming_tools/streaming_tool_call_roundtrip.yaml` | no |
| `cassette::tools::tool_call_roundtrip` | agent (tools) | agent-prompt-tools | `cohere/tools/tool_call_roundtrip.yaml` | no |
| `streaming_tools::streaming_tools_smoke` | agent (tools) | live-agent | none | yes |
| `tools::tools_smoke` | agent (tools) | live-agent | none | yes |
