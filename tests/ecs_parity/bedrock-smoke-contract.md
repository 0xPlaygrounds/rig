# Bedrock agent smoke family

`cassette::agent::{completion_smoke,completion_with_context_smoke,
prompt_caching_completion_smoke,tool_roundtrip_smoke}` are already mapped under
`provider-completions`; the two cells here are the remaining text-only agent
cells.

## Rules

- **agent-stream** — `client.agent(AMAZON_NOVA_LITE).preamble(STREAMING_PREAMBLE)
  .build().stream_prompt(STREAMING_PROMPT).stream().await` drained by
  `collect_stream_final_response` then `assert_nonempty_response`.
- **agent-prompt** — `client.agent(ANTHROPIC_CLAUDE_HAIKU_4_5).preamble("You are concise.")
  .build().prompt(..).await.expect(..)` then `assert_nonempty_response`
  (`model_ids::claude_profile_constant_completes`; its three siblings call the
  model directly).

## Phase 2 obligations

Same shape as the other `streaming_smoke` ports (drain to EOF, final response
required) through the unchanged Bedrock wrapper; the Claude profile cell ports
like `completion_smoke` with the profile model id.
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
| `cassette::model_ids::claude_profile_constant_completes` | agent (smoke) | agent-prompt | `bedrock/model_ids/claude_profile_constant_completes.yaml` | no |
| `cassette::streaming::streaming_smoke` | agent (smoke) | agent-stream | `bedrock/streaming/streaming_smoke.yaml` | no |
