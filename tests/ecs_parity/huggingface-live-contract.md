# Hugging Face live-only family

All nine rows are `#[ignore = "requires HUGGINGFACE_API_KEY"]`; there is no
cassette wrapper in this tree and no fixture on disk.

## Rules

- **live-agent** — `huggingface::Client::from_env()` (or `Client::builder()
  .api_key(..).subprovider(..)`) builds `client.agent(model)` with contexts,
  tools (`Adder` / `Subtract`) or a preamble and drives `prompt` /
  `stream_prompt` drained by `collect_stream_final_response`; families `smoke`
  (context, loaders, streaming, together streaming) and `tools` (tools,
  subproviders across Together / HFInference / SambaNova).
- **live-model** — `client.image_generation_model(..).image_generation_request()
  ..send()` and `client.transcription_model(..).transcription_request()..send()`;
  no agent.

## Credential disposition

`HUGGINGFACE_API_KEY` is not configured here; no capture is possible; every row
stays `supplemental_live` (agent rows) or shared-provider, discovered only, not
ported. `image_generation::image_generation_smoke` is additionally behind
`cfg(feature = "image")` and is not in the root-bedrock compiled listing.

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
| `context::context_smoke` | agent (smoke) | live-agent | none | yes |
| `image_generation::image_generation_smoke` | shared_provider (provider) | live-model | none | yes |
| `loaders::loaders_smoke` | agent (smoke) | live-agent | none | yes |
| `streaming::streaming_smoke` | agent (smoke) | live-agent | none | yes |
| `streaming::together_subprovider_streaming` | agent (smoke) | live-agent | none | yes |
| `subproviders::tool_prompt_across_subproviders` | agent (tools) | live-agent | none | yes |
| `tools::tools_smoke` | agent (tools) | live-agent | none | yes |
| `transcription::transcription_smoke` | shared_provider (provider) | live-model | none | yes |
