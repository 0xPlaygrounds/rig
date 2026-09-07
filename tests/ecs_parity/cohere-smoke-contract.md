# Cohere agent smoke family

## Rules

- **agent-prompt** — `CONTEXT_DOCS` folded into `client.agent(CASSETTE_MODEL)
  .context(..)`, `.preamble(..).build().prompt(CONTEXT_PROMPT).await.expect(..)`,
  `assert_contains_any_case_insensitive`.
- **agent-stream** — `client.agent(CASSETTE_MODEL).preamble(STREAMING_PREAMBLE)
  .max_tokens(64).build().stream_prompt(STREAMING_PROMPT).stream().await`
  drained by `collect_stream_final_response_and_provider_final`;
  `assert_nonempty_response` and the provider final's exact usage
  (`input 553`, `output 64`, `total = input + output`, `cached 480`).
- **live-agent** — `cohere::Client::from_env().agent(COMMAND_A_03_2025)` driven
  by `stream_prompt` against the live API.

## Credential disposition

`agent::completion_smoke` (live): `COHERE_API_KEY` is configured. Phase 2
captures one cassette for this case in an isolated checkout with the recording
command restricted to it, inspects the diff, promotes it and ports it natively;
until then it stays `supplemental_live` and is not counted as ported.
`streaming::streaming_smoke` (live) is outside the 12-case set and stays
`supplemental_live` unless the integrator schedules a capture.

## Phase 2 obligations

The cassette streaming smoke's exact usage assertion is a level-1 obligation on
the provider final (`tokens`, not `billed_units`) and must be observed from the
native stream terminal, not synthesized.
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
| `agent::completion_smoke` | agent (smoke) | live-agent | none | yes |
| `cassette::context::context_documents_are_accepted` | agent (smoke) | agent-prompt | `cohere/context/context_documents_are_accepted.yaml` | no |
| `cassette::streaming::streaming_smoke` | agent (smoke) | agent-stream | `cohere/streaming/streaming_smoke.yaml` | no |
| `streaming::streaming_smoke` | agent (smoke) | live-agent | none | yes |
