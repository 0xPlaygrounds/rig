# OpenAI-compatible lane: classification rules

Scope: the fourteen provider trees of the `openai-compatible` lane (openrouter,
xai, venice, groq, doubleword, perplexity, together, moonshot, hyperbolic,
minimax, xiaomimimo, zai, azure, voyageai), 520 catalog rows at base
`a7eb63a91`. This batch classifies the 476 rows that were `unclassified` and
restates the five live `agent::completion_smoke` rows under the same rules; the
39 other rows that were already classified (32 with native counterparts, 7
shared-provider tool-session rows) are untouched. Every rule below
is stated once; each catalog row cites its rule id and the concrete execution
path that row's test constructs and drives. Nothing here is an execution or
parity claim: classification comes from reading the source and the compiled
root listing, and a classified row has no result until its native counterpart
runs.

## What counts as evidence

A row is `agent` only when its test builds a rig-agent agent and drives a run
through it. The positive markers are `client.agent(..)`/`AnthropicClient.agent(..)`
(the `CompletionClient` blanket for `AgentBuilder`), a literal
`AgentBuilder::new(model)`, `client.extractor::<T>(..)` (the rig-agent
`ExtractorBuilder`), and the run surfaces `prompt`, `prompt_typed`, `chat`,
`stream_prompt`, `stream_chat`, `extract`. A row is `shared_provider` when the
test's only entry point into a completion is `client.completion_model(..)` (or
another capability model) followed by `completion`, `raw_completion`,
`raw_completion_with_request_id`, `completion_request(..).send()`/`.stream()`,
`stream`, `embed_texts*`, `list_models`, `transcription_request`, and so on:
those calls execute rig-core provider code that both runtimes share, so the
test is coverage of the provider, not of an agent runtime. Tool
*definitions* passed through `completion_request(..).tool(..)` or
`tools(vec![..])` never execute a tool and do not make a row `agent`. The
absence of an agent symbol is never the reason; the presence of a model-direct
call is.

Ignored (`#[ignore]`) originals keep their rule and are recorded as
discovered, never executed: agent rows become `supplemental_live`, shared
rows keep `shared_provider` with `baseline_ignored: true`. Feature-gated rows
(`#[cfg(feature = "audio"|"image")]`) are outside the root-bedrock compiled
listing and are classified from source with that limit noted.

## Rules

### OC-A1 agent blocking run

`client.agent(MODEL)…build()` then `agent.prompt(..)`, `agent.prompt_typed::<T>(..)`
or `agent.chat(.., &mut history)`; assertions on the returned output,
usage, tool call counters, hook state or the caller-owned history. Covers the
smoke families (context, loaders, tools, multimodal, structured output, typed
prompt with tools, provider selection, chat history, document file_data,
Anthropic-compatible transports on moonshot/minimax/xiaomimimo/zai) and the
blocking agent cell of the refusal matrix, whose body is a plain `prompt`.

### OC-A2 agent streaming run

`client.agent(MODEL)…build()` then `stream_prompt(..)`/`stream_chat(.., history)`
`.stream()`, drained by `collect_stream_final_response`,
`collect_stream_final_response_and_provider_final`, `collect_stream_observation`,
`stream_to_stdout` or a local loop over `MultiTurnStreamItem`; assertions on
the final response, streamed tool calls/results, ordering helpers or the
provider final. Covers streaming smoke, streaming tools, streaming reasoning,
the streaming halves of the reasoning-tool roundtrips and the streaming agent
cell of the refusal matrix. The reasoning-usage agent cells are OC-A6.

### OC-A3 agent run with hooks

OC-A1/OC-A2 plus an `AgentHook` implementation attached through
`.add_hook(..)` on the request or on the builder: `SessionIdHook`/`ObservingHook`
(`on_completion_call` + `on_outcome`), `PermissionHook` (`on_dispatch` skip +
`on_outcome`), `IdentityProbe` (`on_outcome` + `on_model_turn_finished`) and
`ReasoningDeltaHookRecorder` (through `reasoning::run_reasoning_delta_hook_streaming`,
which builds `AgentBuilder::new(model)` itself). The hook observations are
part of the row's obligations (policy-visible level).

### OC-A4 rig-agent Extractor

`client.extractor::<T>(MODEL)…build()` then `.extract(..)` (optionally
`.history(..)`), including three extractors driven concurrently in
`batch_multi_extract_chain` and the conformance scenario
`structured_extraction`, which builds `ExtractorBuilder::<ExtractedPerson>::new(model)`
inside `rig_agent::test_utils`. The extractor is rig-agent orchestration (one
synthetic output tool turn), so these rows are `agent`, matching the existing
`extractor-smoke`/`extractor-usage` contracts.

### OC-A5 rig-agent conformance scenario

`rig_agent::test_utils::{zero_argument_tool, parallel_tools,
cancellation_and_max_turns, tool_output_serialization, invalid_tool_recovery,
hook_rewrites_and_request_patch, streaming_tool, structured_after_tool,
streaming_structured_after_tool, optional_argument, sequential_tools}` called
with `client.completion_model(..)` and an identity `configure` closure. Each
scenario constructs `AgentBuilder::new(model)…build()` and drives
`prompt`/`stream_prompt` with its own typed tools and hooks
(`crates/rig-agent/src/test_utils/model_conformance.rs`), then validates a
`ScenarioReport`; the provider test only `.expect`s success. These rows are
`agent`. A phase-2 port must reproduce each scenario's obligations natively
(tools, hooks, cancellation, max-turn diagnostics, structured output after a
tool) and must not call the legacy helper. `tool_choice_modes` is the one
scenario that never builds an agent: see OC-M5.

### OC-A6 matrix agent surface

Matrix cells whose runner takes the `Surface::Agent` branch or a dedicated
agent helper: `run_agent` in `tool_lifecycle_matrix` and
`tool_truncation_contract_matrix` (`client.agent(..).tool(..).default_max_turns(1)`,
blocking `prompt` or `stream_chat(..).max_turns(1)`), `run_signed_agent` in
`reasoning_tool_order_matrix` (`default_max_turns(2)`, blocking `prompt` or
`stream_chat(..).max_turns(2)`), and the two `*_agent_*` cells of
`reasoning_usage_matrix`. Their obligations are exact-once invocation
order, zero invocations plus an error for malformed calls, signed reasoning
replay in the second recorded request, or reasoning tokens on the agent's
aggregated usage. The sibling `*_model` cells of the same matrices are OC-M1/
OC-M2 and stay shared-provider; the split is by cell body, not by file.

### OC-A7 live agent completion smoke

An OC-A1..A4-shaped test that is `#[ignore = "requires <KEY>"]` and reads its
client from `Client::from_env()`: `agent::completion_smoke` on groq,
perplexity, together, moonshot and hyperbolic, every ignored groq agent test,
and the small live trees (together, moonshot, minimax, xiaomimimo, zai, azure).
Classified `agent`, `evidence_scope: supplemental_live`, `baseline_ignored: true`,
no fixture; discovered, never executed, never counted as ported. Phase 2
captures one cassette each for the groq and perplexity `completion_smoke`
rows (credentials configured) and leaves together/moonshot/hyperbolic
(no credential) as `supplemental_live` with that reason.

### OC-M1 direct model completion

`client.completion_model(MODEL)` then `model.completion(request)`,
`model.raw_completion(request)` (optionally `.normalize(provider)`),
`model.raw_completion_with_request_id(..)`, or
`completion_request(..).send()`; assertions on the normalized response, the
raw provider type, usage, identity, finish reason, error classification, or
the recorded request/response bytes. No `AgentBuilder`. Covers raw-capture,
raw-completion-parity, response-identity, history-roundtrip (blocking cells),
streaming-logprobs and terminal-metadata (blocking cells), the `*_model`
blocking cells of tool-lifecycle/tool-truncation/reasoning-tool-order, the
non-agent cells of reasoning-usage and refusal, doubleword's error/finish-
reason/model-family/reasoning/request-parameter matrices, venice's
error-envelope and venice-parameters, perplexity's migration pain points,
document ordering, moonshot's reasoning history, and the model-direct cells
of groq's agent_tool_sessions (same partition as the runtime-owned
tool-sessions contract for openrouter/xai/deepseek).

### OC-M2 direct model stream

`client.completion_model(MODEL)` then `model.stream(request)` or
`completion_request(..).stream()`, drained by `collect_text_and_terminal`,
`collect_raw_stream_observation`, `assert_stream_contains_zero_arg_tool_call_named`,
`stream.snapshot()` or a local loop to `StreamEvent::Final`; assertions on
deltas, terminal record, `terminal.raw`, in-band errors or recorded frames.
Two-turn variants replay a hand-built assistant tool call and tool result
message between the streams (openrouter/xai/groq `raw_followup_*`,
openrouter `stream_encrypted_reasoning_survives_into_the_next_turn`). No
`AgentBuilder`, tools never executed.

### OC-M3 cache probe

`cache_conformance::run_cache_probe(&model, ..)` / `run_cache_probe_streaming`:
three `model.completion`/`model.stream` turns over a hand-built history,
followed by `assert_cache_conformance`, `assert_no_meaningful_prefix_cache`,
`assert_cache_read_is_surfaced`, `assert_cache_key_stable` and
`assert_prefix_stable` (the last three re-read the recorded requests). The
`live_cache_economics` cells run the same probe against `Client::from_env()`
and are ignored. The `agent_loop` cells in the same files are OC-A1 (doubleword
in this batch; openrouter and venice were already mapped under
`agent-cache-growth`).

### OC-M4 reasoning roundtrip helper

`reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(model, ..))`
and `run_reasoning_roundtrip_nonstreaming`. Despite the type name the helper
holds only the model: it builds `CompletionRequest` values by hand and calls
`agent.model.stream(request)` / `agent.model.completion(request)` for two
turns (`tests/common/reasoning.rs`). Shared-provider. The
`reasoning_delta_hook_streaming` case in the same openrouter file is OC-A3.

### OC-M5 conformance tool_choice_modes

`rig_agent::test_utils::tool_choice_modes(model)`: three direct
`model.completion(completion_request(..).tools(..).tool_choice(None|Required|Specific))`
calls with assertions on `choice`; no agent is built. Shared-provider.

### OC-C1 embeddings, OC-C2 model listing, OC-C3 transcription, OC-C4 audio generation, OC-C5 image generation, OC-C6 rerank

Non-completion capabilities: `embedding_model[_with_ndims]` + `embed_texts*`/
`raw_embed_texts`/`EmbeddingsBuilder`; `list_models()`; `transcription_model`
+ `transcription_request().load_file(..).send()`; `audio_generation_model` /
`image_generation_model` + their request builders; `rerank_model(..).rerank(..)`.
No completion runtime is involved on either side; every row is
`shared_provider`. Doubleword's `embedding_dimensions` cells additionally
re-read their recorded `/v1/embeddings` interactions through
`with_doubleword_embedding_cassette`. Venice's audio cell records through the
binary-body `DirectRecordingHttpClient`; xai's and venice's image cells are
cassette-backed but feature-gated.

### OC-W1 wire-conversion unit test

Synchronous `#[test]`s over `rig::providers::openrouter::messages_from_rig_message`
(root `document_file_data.rs`, `file_id.rs`): no HTTP, no cassette, no runtime;
they pin the provider's message serialization, which both runtimes share.
Shared-provider.

## Per-row fields

`classification_reason` = rule id, rule title, this file, then the execution
path of that test (models, builders, run surface, collectors, assertions).
`fixtures` lists every `tests/cassettes/<provider>/<scenario>.yaml` named by a
string literal inside the test body that exists on disk (sibling scenarios a
parity cell re-reads are included). `assertion_sources` lists the test, the
provider wrapper it uses and the shared helper module(s) its body calls.
`mapping_limits` records the live/feature-gated status, the shared-provider
"never an agent cell" statement, and whether the row also re-reads its own
cassette bytes for premise assertions. `ecs` stays `null` and
`assertion_mappings` stays empty for every row in this batch.

## Resulting partition

| tree | rows | agent (mapped / cassette, unmapped / live) | shared |
|---|---|---|---|
| openrouter | 237 | 64 (13 / 51 / 0) | 173 |
| xai | 51 | 25 (12 / 13 / 0) | 26 |
| venice | 49 | 20 (3 / 17 / 0) | 29 |
| groq | 48 | 23 (0 / 4 / 19) | 25 |
| doubleword | 90 | 20 (2 / 18 / 0) | 70 |
| perplexity | 17 | 6 (2 / 3 / 1) | 11 |
| together | 6 | 5 (0 / 0 / 5) | 1 |
| moonshot | 6 | 4 (0 / 0 / 4) | 2 |
| hyperbolic | 3 | 1 (0 / 0 / 1) | 2 |
| minimax | 3 | 2 (0 / 0 / 2) | 1 |
| xiaomimimo | 3 | 2 (0 / 0 / 2) | 1 |
| zai | 3 | 3 (0 / 0 / 3) | 0 |
| azure | 2 | 1 (0 / 0 / 1) | 1 |
| voyageai | 2 | 0 | 2 |
| total | 520 | 176 (32 / 106 / 38) | 344 |

Agent cells to port in phase 2, cassette-backed and unmapped, by family:

| tree | families (cells) |
|---|---|
| openrouter (51) | tool-lifecycle 12, tool-truncation 12, multimodal 5 + document-file-data 2, hooks 4 (request_hook 1, permission_control 2, reasoning_delta_hook 1), streaming-smoke 2, responses-compat 2 (openai Responses client against the OpenRouter cassette), reasoning-tool-roundtrip 2, reasoning-tool-order signed 2, reasoning-usage agent 2, refusal agent 2, streaming-tools 1, typed-prompt-tools 1, extractor-multi 1, provider-selection 1 |
| doubleword (18) | conformance 12 (10 `conformance.rs` incl. structured_extraction + 2 `tools.rs`), agent-cache-growth 1, streaming-smoke 1, streaming-tools 1, typed-prompt-tools 1, structured-output 1, hooks 1 |
| venice (17) | conformance 12, streaming-smoke 1, streaming-tools 1, typed-prompt-tools 1, structured-output 1, hooks 1 |
| xai (13) | hooks 4 (request_hook 1, permission_control 2, response_identity streamed agent 1), reasoning-tool-roundtrip 2, context 1, loaders 1, streaming-smoke 1, tools-smoke 1, streaming-tools 1, typed-prompt-tools 1, extractor-multi 1 |
| groq (4) | tool-sessions 4 (sequential/parallel, blocking/streaming) |
| perplexity (3) | chat-history 1, context 1, streaming-smoke 1 |

Live agent rows (38) are scheduled separately: groq 19 (completion_smoke plus
18 ignored agent tests; GROQ_API_KEY configured, only `completion_smoke` is
captured in phase 2), perplexity 1 (PERPLEXITY_API_KEY configured, captured in
phase 2), together 5, moonshot 4, hyperbolic 1, minimax 2, xiaomimimo 2, zai
3, azure 1 (no credential; stay `supplemental_live`).

## Limits

- Fixture lists are derived from the scenario literals in each test body;
  a test whose scenario comes from a `const` outside the function body would
  list nothing (none in these trees).
- The shared-provider matrices that re-read their own cassette bytes mix a
  provider path with fixture inspection; the row stays shared-provider because
  the runtime path is model-direct, and the inspection is noted in
  `mapping_limits`.
- Feature-gated rows (audio/image) are absent from the root-bedrock listing;
  they are classified but cannot be checked against the compiled listing until
  the integrator regenerates one with those features.
- The five live `completion_smoke` rows were already `agent`; this batch only
  restates their reason, families, contract and limits under OC-A7.
