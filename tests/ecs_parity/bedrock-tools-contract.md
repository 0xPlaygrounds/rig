# Bedrock agent tools family

## Rules

- **agent-stream-tools** — `client.agent(AMAZON_NOVA_LITE).preamble(STREAMING_TOOLS_PREAMBLE)
  .max_tokens(1024).tool(Subtract).default_max_turns(2).build()
  .stream_prompt(STREAMING_TOOLS_PROMPT).stream().await` drained by
  `collect_stream_final_response` then `assert_mentions_expected_number(-3)`;
  the `tool_choice::none_streaming_does_not_emit_tool_calls` cell registers
  `Adder` and `Subtract` with `.tool_choice(ToolChoice::None)` and is collected
  by `collect_stream_observation` (no errors, a final response, no tool calls
  or results, `assert_mentions_expected_number(42)`).
- **agent-chat-tools** — `client.agent(AMAZON_NOVA_LITE).preamble(..).temperature(0)
  .tool(Adder).tool(Subtract).tool_choice(ToolChoice::None).build()
  .chat(.., &mut history).await.expect(..)`; `assert_mentions_expected_number(42)`
  and no tool call in the caller-owned history.

## Exceptions

`tool_choice::{required_forces_function_call,specific_add_raw_nonstreaming_allows_only_add,
specific_add_raw_streaming_allows_only_add}` call the model directly
(`bedrock-provider-contract.md`).

## Phase 2 obligations

`ToolChoice::None` is a negative guarantee: no tool call may be scheduled or
recorded in the run's utterances; the streaming twin asserts the same on the
observed stream. Tool advertisement must still reach the request.

## Fixture kind

Cassette cells are **hosted AWS Bedrock recordings** (`AMAZON_NOVA_LITE`, and
the Claude Haiku 4.5 / DeepSeek R1 inference profiles where the cell says so)
captured by the direct recorder and replayed through `with_bedrock_cassette`
(direct-recording `ProviderCassette`, dummy AWS credentials, fixed `us-east-1`
region, loopback endpoint, `finish_after_test` exhaustion). Every row is behind
`cfg(feature = "bedrock")` and only exists in the root-bedrock listing. The six
`#[ignore]` cells are `unrecorded` (no valid AWS credentials); lane rule: do
not recapture anything for bedrock.

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
| `cassette::streaming::streaming_tools_smoke` | agent (tools) | agent-stream-tools | `bedrock/streaming/streaming_tools_smoke.yaml` | no |
| `cassette::tool_choice::none_nonstreaming_does_not_emit_tool_calls` | agent (tools) | agent-chat-tools | `bedrock/tool_choice/none_nonstreaming_no_tools.yaml` | no |
| `cassette::tool_choice::none_streaming_does_not_emit_tool_calls` | agent (tools) | agent-stream-tools | `bedrock/tool_choice/none_streaming_no_tools.yaml` | no |
