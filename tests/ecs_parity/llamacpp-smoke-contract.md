# llama.cpp agent smoke family

Text-only agent cells (no tools, no hooks, no schema) that the llama.cpp suite
inherited from the migrated examples. Native counterparts belong in a
`tests/providers/llamacpp/cassette/ecs_smoke.rs` module on the same fixtures;
`agent::completion_smoke` is already mapped under `provider-completions`.

## Rules

- **agent-prompt** — `client.agent(CASSETTE_MODEL)` (with `AgentBuilder::context`
  folds for the context and loaders cells) `.build().prompt(..).await.expect(..)`;
  the output is checked by the shared `tests/common/support.rs` assertion the
  cell names (`assert_contains_any_case_insensitive`,
  `assert_loader_answer_is_relevant`).
- **agent-stream** — `client.agent(..).build().stream_prompt(..).stream().await`
  drained to EOF by `collect_stream_final_response` (`item?`, requires a final
  response) then `assert_nonempty_response` or a content assertion.

## Exceptions

`content_matrix::unicode_split_across_stream_chunks_reassembles` is the only
agent cell in `content_matrix` (its four siblings call `model.completion`
directly and sit in `llamacpp-provider-contract.md`). After the wrapper returns
it re-reads the recorded SSE frames for the chunking premise; a port keeps that
post-closure assertion.

## Phase 2 obligations

Every `expect`, the drain-to-EOF and final-response requirement, the exact
preamble / context order / model / temperature / max_tokens configuration, the
post-closure fixture re-read, and strict matching plus exhaustion through the
unchanged wrapper. `loaders_smoke` folds `FileLoader::with_glob(LOADERS_GLOB)`
contexts in read order; the native port must build the same context list.
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
| `cassette::content_matrix::unicode_split_across_stream_chunks_reassembles` | agent (smoke) | agent-stream | `llamacpp/content_matrix/unicode_across_chunks.yaml` | no |
| `cassette::context::context_smoke` | agent (smoke) | agent-prompt | `llamacpp/context/context_smoke.yaml` | no |
| `cassette::loaders::loaders_smoke` | agent (smoke) | agent-prompt | `llamacpp/loaders/loaders_smoke.yaml` | no |
| `cassette::streaming::example_streaming_prompt` | agent (smoke) | agent-stream | `llamacpp/streaming/example_streaming_prompt.yaml` | no |
| `cassette::streaming::streaming_smoke` | agent (smoke) | agent-stream | `llamacpp/streaming/streaming_smoke.yaml` | no |
