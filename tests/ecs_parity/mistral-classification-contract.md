# Mistral, DeepSeek and mistral.rs classification rules

Scope: the 208 catalog rows that were `unclassified` at the base (`a7eb63a91`)
in `scenarios/mistral.json` (109), `scenarios/deepseek.json` (86) and
`scenarios/mistralrs.json` (13). Each rule below was written once from the
test body and the helper it calls; every row's `classification_reason` names
the rule and the family's concrete call chain. Owner: the `mistral-deepseek`
lane. The file is referenced only by that lane's three catalog files.

Classification is decided by the execution path the test drives, never by a
file or test name. `agent` means rig-agent orchestration executes (turn loop,
tool dispatch, hooks, history, extractor submit tool) and a native counterpart
is owed. `shared_provider` means the provider seam is driven directly and the
same code runs unchanged in both runtimes, so no native counterpart is
invented and the row never counts as agent coverage.

## Rules

| Rule | Class | Positive evidence required in the test body |
| --- | --- | --- |
| R1 agent-builder | `agent` | `client.agent(<model>)…build()` (or `with_system_instructions_as_messages().agent(..)`) driven by `prompt`, `chat(&mut history)`, `stream_prompt`, `stream_chat` or `prompt_typed`; any `.add_hook`, `.max_turns`, `default_max_turns`, tools and history are consumed by that runner. |
| R2 extractor | `agent` | `client.extractor::<T>(<model>)…build()` driven by `.extract(..)` (optionally `.history(..)`); the extractor is a typed rig-agent run with the submit tool, so it is agent orchestration. |
| R3 direct-model | `shared_provider` | `client.completion_model(<model>)` driven by `.completion(req)`, `.raw_completion(req)` (with `.normalize(..)`), `.stream(req)`, or `completion_request(..).send()/.stream()`. `Tool` values, when present, contribute only `tool_definition(&tool)` schemas or history messages; no builder and no runner. The shared helpers `cache_conformance::run_cache_probe{,_streaming}(&model, ..)` and `reasoning::run_reasoning_roundtrip_{streaming,nonstreaming}(ReasoningRoundtripAgent::new(model, ..))` belong here: both build `CompletionRequest`s by hand and call `model.completion`/`model.stream`; `ReasoningRoundtripAgent` is a plain struct holding the model, not an `AgentBuilder`. |
| R4 non-completion capability | `shared_provider` | `embedding_model(..).embed_*`, `EmbeddingsBuilder` over an embedding model plus `InMemoryVectorStore::top_n`, `list_models()`, `verify()`, or `transcription_model(..).transcription_request()…send()`. No completion runner exists on these paths. |
| R5 raw-http | `shared_provider` | A hand-built `reqwest::Client::new().post(<cassette base>/chat/completions)` with a literal JSON body; rig's provider code is bypassed and only the recorded wire exchange is asserted. |

Evidence scope, applied to every row the rules touch:

- E1 cassette-backed: the test passes a literal scenario to one of the tree's
  `with_<provider>_*cassette*` wrappers and `tests/cassettes/<provider>/<scenario>.yaml`
  exists at the base. Agent rows get `evidence_scope: cassette_agent` and a
  `fixtures` entry naming that recording. Shared-provider rows record the same
  fixture and no evidence scope.
- E2 ignored: the test is `#[ignore]` in the base listing. The row gets
  `baseline_ignored: true`; agent rows get `evidence_scope: supplemental_live`.
  Live rows are discovered, never executed; the mistral.rs `raw_*_capture_matrix`
  cells are ignored because no fixture was ever recorded, so they carry no
  fixture.

## Families in `mistral.json`

| Family (source module) | Rule | agent | shared | Call chain read |
| --- | --- | ---: | ---: | --- |
| agent_tool_sessions | R1 / R3 | 4 | 5 | Sequential and parallel cases build `client.agent(SESSION_MODEL)` with the five complex tools or `AlphaSignal`+`BetaSignal` and drive `agent.chat(.., &mut history)`, `stream_chat(..).max_turns(10)` or `stream_prompt(..).max_turns(5)` through `collect_stream_observation`. The raw-stream, long-history, tool-choice, JSON-object and JSON-schema cases call `model.stream`, `raw_completion` (via `raw_and_normalized_completion`) or `model.completion` on hand-built requests. |
| capability_edges | R4 / R3 | 0 | 5 | `embed_texts` over the batch cap, `embed_text` against `ndims()`, `list_models()`; `completion_request(..).stream()` with `n: 2`; `model.completion` with a hand-built turn-one history and `ToolChoice::Required`. |
| embedding_matrix | R4 | 0 | 7 | `embed_texts_response`, `raw_embed_texts`, `embed_text_response`, `embed_text`, `embedding_model_with_ndims`. |
| embeddings | R4 (E2) | 0 | 1 | `EmbeddingsBuilder::new(embedding_model)…build()` then `InMemoryVectorStore::from_documents(..).index(..).top_n`. |
| extractor, extractor_usage, multi_extract | R2 (E2) | 7 | 0 | `client.extractor::<T>(DEFAULT_MODEL).build()` and `.extract(..)`; multi_extract joins three extractors with `retries(2)`. |
| logprobs_rejection_matrix | R3 | 0 | 4 | `run_cell` calls `raw_completion` or `model.stream` and captures the 3051 rejection string. |
| models | R4 | 0 | 2 | `list_models()` with the real and the bogus key. |
| multimodal_content | R1 / R3 | 21 | 3 | The three `blocking_raw_model_*` cells call `model.completion(completion_request(Message::User{parts}))`. Every other cell, including `streaming_raw_model_sends_a_base64_image`, builds `client.agent(VISION_MODEL / SECOND_VISION_MODEL / AUDIO_MODEL)` and drives `agent.prompt(Message)`, `stream_prompt(Message)`, `chat(&str, &mut history)` or `stream_chat(&str, history)`; the two `*_with_a_tool_configured` cells add `RecordColour` and `default_max_turns(3)`. The `assert_recorded_*` calls after the wrapper read the committed cassette bytes and are fixture premises, not runtime observations. |
| permission_control | R1 (E2) | 2 | 0 | Agent with `ReadFileHead`/`ReadFileTail`, `PermissionHook` via `.add_hook`, `prompt(..).max_turns(5)` or `stream_prompt(..).max_turns(5)` plus `stream_to_stdout`. |
| prompt_caching | R3 | 0 | 2 | `run_cache_probe(&model, ..)` and `run_cache_probe_streaming(&model, ..)` call `model.completion`/`model.stream` three times on the probe's hand-built history; `assert_prefix_stable` reads the fixture afterwards. |
| raw_capture_matrix, raw_stream_capture_matrix | R3 | 0 | 7 | `model.completion(request/tool_request)` and `model.stream(..)` with post-wrapper fixture reads. |
| request_hook | R1 (E2) | 1 | 0 | `agent.prompt("Entertain me!").add_hook(SessionIdHook)`. |
| request_shape_matrix | R3 | 0 | 24 | `run_cell` calls `model.completion` or `model.stream` per `Transport` and `assert_cell` compares with the recorded body. |
| response_identity_edge | R3 / R4 / R1 | 1 | 5 | Blocking, error, unauthorized and streaming-error cells use `completion_request(..).send()/.stream()`; `verify_succeeds_against_the_versioned_models_route` calls `client.verify()`; `streaming_terminal_carries_the_correlation_id` builds `client.agent(MISTRAL_SMALL)` and drives `stream_prompt` through `collect_stream_final_response_and_provider_final`. |
| streaming, streaming_tools, typed_prompt_tools | R1 (E2) | 7 | 0 | `agent.stream_prompt`/`stream_chat` with the shared `Adder`/`Subtract`/`AlphaSignal` tools, and `agent.prompt_typed::<WeatherResponse>` with `WeatherTool`. |
| transcription | R4 (E2) | 0 | 1 | `transcription_model(VOXTRAL_MINI).transcription_request().load_file(..).send()`. |

Totals: 43 agent (26 cassette-backed, 17 ignored live), 66 shared_provider (2 ignored).

## Families in `deepseek.json`

| Family (source module) | Rule | agent | shared | Call chain read |
| --- | --- | ---: | ---: | --- |
| document_ordering | R3 | 0 | 1 | `completion_request(PROMPT).message(..).document(..).send()`; post-wrapper request-order fixture read. |
| followup_hunt_matrix | R3 | 0 | 4 | `model.completion`, `model.stream` through `collect_raw_stream_outcome`, `raw_completion(..).normalize`. |
| models | R4 | 0 | 2 | `list_models()` with the real and the bogus key. |
| multi_extract | R2 | 1 | 0 | Three `client.extractor::<T>(DEEPSEEK_V4_FLASH)…retries(2).build()` joined per input under an `unordered()` cassette spec. |
| permission_control | R1 | 2 | 0 | Agent with `ReadFileHead`/`ReadFileTail`, `PermissionHook` via `.add_hook`, `prompt(..).max_turns(5)` or `stream_prompt(..).max_turns(5)` plus `stream_to_stdout`. |
| prompt_caching | R3 | 0 | 3 | `run_cache_probe{,_streaming}(&model, ..)`; `live_cache_economics` is the same probe against the real API and is ignored (E2). |
| raw_capture_matrix, raw_stream_capture_matrix | R3 | 0 | 7 | `model.completion(request/reasoning_request)` and `model.stream(..)`. |
| reasoning_block_order | R3 / R1 | 2 | 8 | Eight cells call `raw_completion(..).normalize` or `model.stream` through `collect_raw_stream_outcome`. The two `agent_*` cells build `client.agent(MODEL)` with `WeatherTool` and drive `agent.chat(.., &mut history)` with `default_max_turns(3)` or `stream_chat(..).max_turns(3)` through `collect_stream_observation`. |
| reasoning_roundtrip | R3 | 0 | 2 | `run_reasoning_roundtrip_{streaming,nonstreaming}(ReasoningRoundtripAgent::new(client.completion_model(..), thinking))`: the helper hand-builds two `CompletionRequest`s and calls `agent.model.stream`/`.completion`; no `AgentBuilder`. |
| reasoning_tool_roundtrip | R1 | 2 | 0 | `client.agent(..).tool(WeatherTool)…build()`, `stream_chat(..).max_turns(3)` through `collect_stream_stats`, or `chat` with `default_max_turns(2)`. |
| request_hook | R1 | 1 | 0 | `agent.prompt("Entertain me!").add_hook(SessionIdHook)`. |
| response_identity_edge | R3 | 0 | 1 | `completion_request(..).send()`. |
| streaming | R1 | 1 | 0 | `agent.stream_prompt("Tell me a joke")` through `collect_stream_final_response`. |
| streaming_logprobs_matrix | R3 | 0 | 24 | `run_cell` calls `raw_completion` or `model.stream` and reads the terminal `raw`. |
| streaming_tools | R1 / R3 | 3 | 4 | `streaming_chat_with_tools`, `streaming_chat_surfaces_two_distinct_tool_calls_before_final_answer` and `streaming_chat_emits_tool_call_before_later_text` build `client.agent(..)` with `Adder`/`Subtract` or `AlphaSignal`/`BetaSignal` and drive `stream_chat`. The four `raw_*` cells call `model.stream` with `tool_definition` schemas; the follow-up cell issues two such streams. |
| tools | R1 | 1 | 0 | `agent.prompt(TOOLS_PROMPT)` with `Adder`/`Subtract` and `default_max_turns(2)`. |
| wire_shape_matrix | R3 / R5 | 0 | 17 | Sixteen cells call `model.completion`, `model.stream` or `raw_completion` (rejections, flattening controls, tool-choice suppression, error envelopes, cache split). `forced_tool_choice_under_thinking_is_rejected_upstream` posts two hand-built bodies with `reqwest` (R5). |

Totals: 13 agent (all cassette-backed), 73 shared_provider (1 ignored).

## Families in `mistralrs.json`

| Family (source module) | Rule | agent | shared | Call chain read |
| --- | --- | ---: | ---: | --- |
| cassette::chat_completions | R3 / R1 | 1 | 1 | `raw_chat_completion_surfaces_reasoning_or_text` calls `raw_completion(..)` and `normalize("openai")` on the OpenAI-compatible completions client; `chat_completions_agent_prompt_completes` builds `client.agent(model_name())` and drives `agent.prompt`. |
| cassette::raw_capture_matrix, cassette::raw_stream_capture_matrix | R3 (E2) | 0 | 5 | `model.completion` / `model.stream`; every cell is `#[ignore]` because no mistral.rs recording exists. |
| cassette::responses_api | R1 / R3 | 2 | 1 | `responses_api_no_think_returns_text` and `responses_api_multi_turn_replays_history` build `client.with_system_instructions_as_messages().agent(..)` and drive `prompt` or two `chat(.., &mut history)` calls; `responses_api_reasoning_plus_answer_completes` calls `raw_completion(..).normalize("openai")`. |
| cassette::streaming | R1 | 1 | 0 | `agent.stream_prompt(..)` through `collect_stream_observation`. |
| cassette::tools | R5 | 0 | 1 | `reqwest::Client::new().post(<base>/chat/completions)` with a literal tools body. |
| cassette::usage | R3 | 0 | 1 | `raw_completion(..).normalize("openai")`. |

Totals: 4 agent (all cassette-backed), 9 shared_provider (5 ignored, unrecorded).

## Exceptions the names would have hidden

- `mistral::multimodal_content::streaming_raw_model_sends_a_base64_image` says
  "raw model" but builds `client.agent(VISION_MODEL)` and streams through it:
  agent (R1).
- `deepseek::reasoning_roundtrip::*` pass a `ReasoningRoundtripAgent`, which is
  a model holder, not an agent: shared_provider (R3).
- `mistral::response_identity_edge::streaming_terminal_carries_the_correlation_id`
  is the one agent cell in an otherwise direct-model file: agent (R1).
- `deepseek::wire_shape_matrix::forced_tool_choice_under_thinking_is_rejected_upstream`
  and `mistralrs::cassette::tools::raw_chat_completion_emits_requested_tool_call`
  bypass rig entirely with `reqwest`: shared_provider (R5), and they are not
  candidates for a native counterpart at all.

## What this file does not decide

The 24 `tool_lifecycle_matrix` and `tool_truncation_matrix` agent cells, the
credential-gated `agent::completion_smoke`, and every row that was already
classified at the base are untouched here. Native counterparts, assertion
mappings and configuration mappings for the agent rows above belong to later
batches and their own family contracts; this file only fixes which rows owe
one. Shared-provider rows here never inflate agent counts.
