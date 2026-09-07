# Mira live-only family

All three rows are `#[ignore = "requires MIRA_API_KEY"]`; no cassette wrapper,
no fixture.

## Rules

- **live-agent** — `mira::Client::from_env().agent(..)` with `TOOLS_PREAMBLE`,
  `Adder` and `Subtract`, `.prompt(TOOLS_PROMPT).await.expect(..)` then
  `assert_mentions_expected_number(-3)` (family `tools`); `agent::completion_smoke`
  (family `smoke`) is the plain prompt.
- **live-model** — `client.list_models().await.expect(..)`; no agent.

## Credential disposition

`MIRA_API_KEY` is not configured here; no capture is possible; the agent rows
stay `supplemental_live`, discovered only, not ported.

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
| `models::list_models_smoke` | shared_provider (provider) | live-model | none | yes |
| `tools::tools_smoke` | agent (tools) | live-agent | none | yes |
