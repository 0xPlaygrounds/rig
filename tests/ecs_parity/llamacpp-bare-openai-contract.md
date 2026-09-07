# llama.cpp bare `openai::Client` family

The deliberately small path that reaches the local server through a bare
`openai::Client` (`with_llamacpp_bare_openai_cassette`: base URL plus `/v1`,
placeholder key) instead of `providers::llamacpp`. Three cells drive an agent,
two are provider-only; all five share this contract because the native
counterpart must use the **openai** provider's adapters over llamacpp fixtures.

## Rules

- **agent-prompt** — `openai::Client.completions_api().agent(CASSETTE_MODEL)
  .build().prompt(..).await.expect(..)` (or `completion_model(..).completions_api()
  .into_agent_builder()...`); `assert_nonempty_response`, and for the `/v1`
  cell the recorded request path is re-read after closure.
- **agent-stream-tools** — `openai::Client.completions_api().agent(..).tool(Adder)
  .tool(Subtract).build().stream_prompt(..).max_turns(4).stream().await`
  drained by `collect_stream_final_response`; `assert_mentions_expected_number(-3)`
  and the recorded SSE fragmentation premise re-read after closure.
- **model-completion** / **model-raw** — the header cell compares an in-process
  `RecordingHttpClient` `embed_texts` through both clients, then calls
  `completions_api().completion_model(..).completion(request)` over the
  cassette; the raw cell calls `raw_completion(request)` and compares
  `text_response()` with `.normalize("openai")`. Neither constructs an agent.

## Phase 2 obligations

The agent cells port with the openai `CompletionAdapter` pointed at the
cassette base URL with the caller-supplied `/v1`, the placeholder bearer key,
and the same fixtures; the fragmentation and path premises stay post-closure
assertions. The two provider-only cells stay shared-provider.

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
| `cassette::bare_openai_client::a_fragmented_tool_call_stream_reassembles_without_the_provider_consts` | agent (bare-openai) | agent-stream-tools | `llamacpp/bare_openai_client/tool_call_stream_without_the_single_chunk_const.yaml` | no |
| `cassette::bare_openai_client::agent_prompt_through_completions_api` | agent (bare-openai) | agent-prompt | `llamacpp/bare_openai_client/agent_prompt_through_completions_api.yaml` | no |
| `cassette::bare_openai_client::bare_openai_client_always_sends_an_authorization_header` | shared_provider (bare-openai) | model-completion | `llamacpp/bare_openai_client/authorization_header_is_always_sent.yaml` | no |
| `cassette::bare_openai_client::caller_supplies_the_v1_prefix_the_provider_would_add` | agent (bare-openai) | agent-prompt | `llamacpp/bare_openai_client/caller_supplies_the_v1_prefix.yaml` | no |
| `cassette::bare_openai_client::raw_response_text_matches_normalized_choice_text` | shared_provider (bare-openai) | model-raw | `llamacpp/bare_openai_client/raw_response_text_matches_normalized_choice_text.yaml` | no |
