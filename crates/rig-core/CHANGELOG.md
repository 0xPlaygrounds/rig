# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.42.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.41.0...rig-core-v0.42.0) - 2026-08-17

### Added

- the provider's own response on every completion: `raw: serde_json::Value` on `CompletionResponse` and `StreamFinal` — the value `raw_completion` / `raw_stream` would have returned, serialized — populated at every provider seam and the shared `normalize_stream` seam; `openai::GenericCompletionModel::raw_completion_with_request_id` and `copilot::CompletionModel::raw_completion_with_request_id` are public so the typed route reproduces `completion()` ([#2366](https://github.com/0xPlaygrounds/rig/issues/2366)) - #2367
- *(voyageai)* expose embedding request options ([#2343](https://github.com/0xPlaygrounds/rig/pull/2343)) (by [sergiomeneses](https://github.com/sergiomeneses))
- carry the provider transport request id on completion errors ([#2314](https://github.com/0xPlaygrounds/rig/pull/2314)) ([#2315](https://github.com/0xPlaygrounds/rig/pull/2315)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2315
- response identity metadata — native response id + provider transport request id, to every completion observer ([#2265](https://github.com/0xPlaygrounds/rig/pull/2265)) ([#2313](https://github.com/0xPlaygrounds/rig/pull/2313)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2313
- *(anthropic)* per-breakpoint cache TTL — static prefix independent of conversation tail ([#2266](https://github.com/0xPlaygrounds/rig/pull/2266)) ([#2312](https://github.com/0xPlaygrounds/rig/pull/2312)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(venice)* add Venice AI provider with live-recorded cassette coverage ([#2306](https://github.com/0xPlaygrounds/rig/pull/2306)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cohere)* add image embeddings ([#2304](https://github.com/0xPlaygrounds/rig/pull/2304)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(anthropic)* support strict tool use ([#2296](https://github.com/0xPlaygrounds/rig/pull/2296)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(doubleword)* report the embedding width Doubleword actually returns ([#2356](https://github.com/0xPlaygrounds/rig/pull/2356)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openrouter)* surface chat-completions refusals, and map the reasoning share of usage ([#2358](https://github.com/0xPlaygrounds/rig/pull/2358)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(embeddings)* keep a document's embeddings in text order across batch boundaries ([#2348](https://github.com/0xPlaygrounds/rig/pull/2348)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(embeddings)* builder input order ([#2344](https://github.com/0xPlaygrounds/rig/pull/2344)) (by [sergiomeneses](https://github.com/sergiomeneses))
- *(mistral)* eight more provider bugs found by live cassette recording ([#2337](https://github.com/0xPlaygrounds/rig/pull/2337)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openai)* preserve the provider's response when a websocket upgrade is rejected ([#2338](https://github.com/0xPlaygrounds/rig/pull/2338)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] preserve response headers on non-success HTTP errors ([#2333](https://github.com/0xPlaygrounds/rig/pull/2333)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- three model-listing and usage bugs found by live cassette recording (anthropic, gemini) ([#2334](https://github.com/0xPlaygrounds/rig/pull/2334)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2334
- *(openai)* six wire-level defects found by live cassette recording ([#2332](https://github.com/0xPlaygrounds/rig/pull/2332)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mistral)* four provider bugs found by live cassette recording ([#2331](https://github.com/0xPlaygrounds/rig/pull/2331)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(anthropic)* two stop_sequence bugs found by live cassette recording ([#2329](https://github.com/0xPlaygrounds/rig/pull/2329)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini)* four response-mapping bugs found by live cassette recording ([#2328](https://github.com/0xPlaygrounds/rig/pull/2328)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini, agent)* close the whole output-budget truncation chain, not just the 4096 cap ([#2324](https://github.com/0xPlaygrounds/rig/pull/2324)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* enable all-feature wasm builds ([#2319](https://github.com/0xPlaygrounds/rig/pull/2319)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cohere)* validate required tool choice ([#2302](https://github.com/0xPlaygrounds/rig/pull/2302)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cohere)* Cohere provider sends request shapes the v2 chat API rejects, and ships removed model IDs ([#2263](https://github.com/0xPlaygrounds/rig/pull/2263)) (by [rleisti](https://github.com/rleisti))
- *(openai)* preserve Responses message phase across stateless replay ([#2269](https://github.com/0xPlaygrounds/rig/pull/2269)) ([#2295](https://github.com/0xPlaygrounds/rig/pull/2295)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openai)* merge additional_params tools into chat completions tool list ([#1890](https://github.com/0xPlaygrounds/rig/pull/1890)) ([#2294](https://github.com/0xPlaygrounds/rig/pull/2294)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini)* send temperature and max_tokens; add live regression cassettes and a cache-prefix guard ([#2283](https://github.com/0xPlaygrounds/rig/pull/2283)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(anthropic)* prefer `message_delta` usage.input_tokens when the provider sends it there ([#2279](https://github.com/0xPlaygrounds/rig/pull/2279)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(release)* isolate macro hygiene fixture ([#2227](https://github.com/0xPlaygrounds/rig/pull/2227)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- reconcile the changelogs and the migration guide with what actually merged ([#2353](https://github.com/0xPlaygrounds/rig/pull/2353)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2353
- *(rig-core)* make a live tracing capture provable, not assumed ([#2347](https://github.com/0xPlaygrounds/rig/pull/2347)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(providers)* share the paginated model-listing loop, and add Groq listing ([#2339](https://github.com/0xPlaygrounds/rig/pull/2339)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- remove #[non_exhaustive] from the workspace ([#2335](https://github.com/0xPlaygrounds/rig/pull/2335)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2335
- workspace-wide LOC consolidation pass 8 (net −1,353 production lines) ([#2320](https://github.com/0xPlaygrounds/rig/pull/2320)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2320
- *(rig-core)* consolidate provider boilerplate ([#2317](https://github.com/0xPlaygrounds/rig/pull/2317)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- share xAI Responses and audio drivers ([#2316](https://github.com/0xPlaygrounds/rig/pull/2316)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2316
- workspace-wide LOC consolidation pass 7 (net −366 production lines) ([#2310](https://github.com/0xPlaygrounds/rig/pull/2310)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2310
- workspace-wide LOC consolidation pass 6 (net −3,424 lines) ([#2308](https://github.com/0xPlaygrounds/rig/pull/2308)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2308
- consolidate provider/loader/agent plumbing (net −566 production LOC) ([#2305](https://github.com/0xPlaygrounds/rig/pull/2305)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2305
- remove dead API surface and consolidate provider/agent plumbing (net −794 production LOC) ([#2301](https://github.com/0xPlaygrounds/rig/pull/2301)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2301
- Revert "fix(openai): preserve Responses message phase across stateless replay ([#2269](https://github.com/0xPlaygrounds/rig/pull/2269)) ([#2295](https://github.com/0xPlaygrounds/rig/pull/2295))" ([#2300](https://github.com/0xPlaygrounds/rig/pull/2300)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2300
- consolidate provider schema/normalization and agent plumbing (net −365 production LOC) ([#2299](https://github.com/0xPlaygrounds/rig/pull/2299)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2299
- consolidate provider scaffolding and agent-runner plumbing (net −439 production LOC) ([#2289](https://github.com/0xPlaygrounds/rig/pull/2289)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2289
- consolidate provider and agent plumbing ([#2288](https://github.com/0xPlaygrounds/rig/pull/2288)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2288
- Consolidate provider and agent boilerplate ([#2285](https://github.com/0xPlaygrounds/rig/pull/2285)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2285
- consolidate provider plumbing and agent boilerplate (−365 production LOC, 5 defect fixes) ([#2286](https://github.com/0xPlaygrounds/rig/pull/2286)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2286
- [**breaking**] assistant content is tagged and provider extras are a named field ([#2277](https://github.com/0xPlaygrounds/rig/pull/2277)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2277
- post-Vec-migration precision and the pre-Vec serde accommodations go ([#2276](https://github.com/0xPlaygrounds/rig/pull/2276)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2276
- [**breaking**] `OneOrMany<T>` becomes `Vec<T>` — the fake is deleted, the enforcement moves ([#2273](https://github.com/0xPlaygrounds/rig/pull/2273)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2273
- Tool identity holds at every boundary: legacy lift, honest constructors, and the drains the siblings already had (2262 round-7 follow-up) ([#2267](https://github.com/0xPlaygrounds/rig/pull/2267)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2267
- Stream parts become entities: lifecycle grammar, opaque keys, and tool names as data (the 84a43e9e C→B→A program) ([#2262](https://github.com/0xPlaygrounds/rig/pull/2262)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2262
- Canonical stream grammar: mandatory identity, one accumulator, decode-then-validate, and a wire-conformance corpus ([#2258](https://github.com/0xPlaygrounds/rig/pull/2258)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2258
- Normalize completion responses at the provider boundary and erase the model type at agent construction ([#2257](https://github.com/0xPlaygrounds/rig/pull/2257)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2257

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
* [sergiomeneses](https://github.com/sergiomeneses)
* [rleisti](https://github.com/rleisti)

### Changed

- *(deps)* dependency requirements are now floors — the lowest version rig's own code needs (a bare major, or the version that introduced an API rig relies on) — instead of the latest patch at the time of release; Dependabot only moves `Cargo.lock` for in-range releases, and `scripts/check-dependency-floors.py` (CI `dependency-floors`) builds the workspace against the declared floors. The `deranged = "=0.5.8"` exact pin is gone. Downstream users no longer have to `cargo update` unrelated crates to take a rig release ([#2195](https://github.com/0xPlaygrounds/rig/issues/2195)) - #2369
- *(anthropic)* [**breaking**] `completion::Citation`'s five locator variants become newtypes over five new public payload structs: `CharLocation(CharLocationCitation)`, `PageLocation(PageLocationCitation)`, `ContentBlockLocation(ContentBlockLocationCitation)`, `SearchResultLocation(SearchResultLocationCitation)`, `WebSearchResultLocation(WebSearchResultLocationCitation)` — the crate-private `*CitationFields` DTOs the hand-written `Deserialize` already decoded into, made public as the variant payload itself and renamed without the `Fields` suffix. Field names, types and optionality are carried over verbatim and the `type`-tagged wire shape is untouched (including `web_search_result_location`'s `title`, still written even when absent), so persisted citations load and a serialized one carries the same keys and values; only source that spells the fields breaks — `Citation::CharLocation { cited_text, .. }` becomes `Citation::CharLocation(CharLocationCitation { cited_text, .. })`. `Citation::Unknown(serde_json::Value)` is unchanged. Both routes to the type are provider-native: `Content::Text { citations, .. }` and `streaming::ContentDelta::CitationsDelta`

- *(image, audio)* `image_generation::ImageGenerationModel` and `audio_generation::AudioGenerationModel` state their bounds as `WasmCompatSend`/`WasmCompatSync` rather than `Send`/`Sync`: `ImageGenerationModel`'s own supertraits, plus the associated `Response` and the returned future on both. That is the shape `CompletionModel`, `EmbeddingModel` and `TranscriptionModel` already had. On native targets nothing changes — `WasmCompatSend: Send`, `WasmCompatSync: Sync`, blanket-implemented for every qualifying type — so existing implementors compile unchanged and generic code still gets `Send`/`Sync` from the bound; on `wasm32-unknown-unknown` both markers are empty, so a model whose HTTP future is not `Send` can implement either trait, which the old `+ Send` future bound ruled out

- *(providers)* [**breaking**] xAI completion and streaming now use the shared OpenAI-compatible Responses driver while retaining xAI's request conversion, `/v1/responses` path, 2xx error envelope, request-id capture, and immediate streamed-tool-call behavior. `xai::completion::CompletionResponse` is the shared Responses wire type; `ResponseStatus::Other` preserves unknown compatible-provider statuses. xAI and OpenRouter audio generation now route through the shared raw-audio request driver.

- *(providers)* workspace-wide consolidation pass 7 (net −366 production LOC): every provider's unary completion tail (send → decode → telemetry → error preservation) routes through one `internal::completion_send::send_completion` driver (10 sites: openai chat/responses, anthropic, gemini + interactions, cohere, ollama, xai, copilot chat/responses) — decode failures now log the error and offending body for **all** providers (previously gemini-only), and the openai/copilot tails read the response body once instead of twice; six SSE stream-open preambles collapse into `internal::sse_transport::open_wire_stream`; TRACE request/response dumps go through one infallible `trace_json` helper; the copilot/chatgpt device-flow OAuth file/prompt/expiry helpers are shared in `internal::device_auth`. [**breaking**] rig-candle drops seven dead public items (`from_artifacts{,_async}`, `from_gguf_async`, `from_gguf_bytes_async`, the `LlamaModelBuilder` alias, both `model_family` accessors). [**behavior**] the agent runner's sequential tool path is now the concurrent path at `buffer_unordered(1)` — identical ordering, fail-fast, and history, with per-call instrumentation now covering the whole call block; milvus `top_n` no longer requires the unused `embeddedText` field in search responses. Vector-store crates share `vector_store::flatten_embedded` for insert flattening, lancedb's Arrow deserializer collapses 30 mechanical match arms into a macro + a generic run-end decoder, and rig-memory's four in-flight release sites share one reservation-checked `release_in_flight`

- *(completion)* [**breaking**] `AssistantContent` serializes with a `"type"` tag, exactly like `UserContent` — the tag is required on deserialize and there is no untagged fallback — **0.41-persisted assistant content (serialized untagged) does not load**; insert the tag per MIGRATING's recipe. `additional_params` on every content block (`Text`/`Image`/`Audio`/`Video`/`Document`) is a **named** serde field instead of a flatten, typed `Option<message::AdditionalParams>` — a newtype that is a non-empty JSON object by construction (the `non_empty_params`/`params_carry_data` helper family is gone; plain `is_none()`/`is_some()` are always correct). A stray key can no longer be silently captured and replayed to providers, an absent field round-trips as `None` instead of the flatten's `Some({})` artifact, and a non-object `additional_params` value is a loud decode error while being unrepresentable in memory. The unknown-key policy is uniform and tolerant across every content block: unknown keys are ignored on load (0.41 flattened extras that were never re-nested load silently minus those keys — verify with `message::keys_lost_in_round_trip`, per MIGRATING's recipe), and an unknown content-block *tag* stays a loud error. A 0.41-serialized *stream* item with flattened text extras decodes as stream *text* (stray keys dropped, text assembled); a replayed **tagged** assistant block decodes as `StreamedAssistantContent::Unknown` and is excluded from assembly — the agent assembler counts those exclusions and logs one warning per turn. A streamed `TextStart` whose metadata is the empty object no longer opens (or position-fixes) a text block: `null` and `{}` are canonicalized to "no metadata" before accumulation, so a `{}`-only start yields no empty text part and no longer pins part order. The OpenAI Responses wire type `responses_api::AssistantContent::OutputText` carries its own `OutputText` wire struct (preserving `annotations`/siblings verbatim), and on the blocking response path its extras now also survive conversion into generic history under `additional_params["openai_responses"]`, replayed by the Responses serializer alone (the streaming adapter does not yet capture annotation events into params)

- *(completion)* [**breaking**] pre-provider-split `ToolCall` JSON is no longer migrated on load: the `ToolCallWire` legacy lift is deleted, `ToolCall` deserializes against the current schema only, and a legacy `call_id` key is ignored as an unknown field — migrate persisted JSON by hand (`call_id` → `provider.call_id`, and for dual-identifier payloads also `id` → the `call_…` correlator, with the `fc_…` handle as `provider.item_id`) if you need those identifiers; see MIGRATING

- *(tool)* [**breaking**] `ToolOutput::as_content` returns `&[ToolResultContent]` instead of `&Vec<ToolResultContent>`; `message::EMPTY_RESPONSE_ERROR` (via `message::require_non_empty_response`) is the one home for the shared empty-response wording every provider decode rejects with. [**behavior**] Four wires' empty-response error text changes to the shared wording: ollama (was "No content provided"), xai (was "Response contained no output"), vertexai (was `ProviderError("No text or tool call content found in response")` — the variant changes to `ResponseError` too), and bedrock's assistant-message conversion (was "Bedrock returned an assistant message with no content")

- *(completion)* [**breaking**] `OneOrMany<T>` and `EmptyListError` are removed; message content, `CompletionResponse::choice`, `CompletionRequest::chat_history`, `ToolResult::content` and `EmbeddingsBuilder` output are `Vec<T>`. The serialized form is unchanged (the container already wrote a plain sequence), so persisted histories and stored embeddings need no migration and no recorded fixture changes. Decoding widens in two places that used to be parse errors: `[]`, which the container rejected outright, and `null` on the fields that moved onto `json_utils::string_or_vec` (its `visit_none`/`visit_unit` arms are load-bearing for OpenAI's tool-calls-only `"content": null`). A tool with `type Output = Vec<ToolResultContent>` also compiles unchanged and now takes `IntoToolOutput`'s rich-content path — N ordered blocks instead of one JSON array — because the guard that used to name `OneOrMany<ToolResultContent>` now names `Vec`; see MIGRATING. `one_or_many::string_or_one_or_many` folds into `json_utils::string_or_vec`; the orphan rule turns three list conversions into the `pub` free functions `openai::completion::{user_content_to_messages, assistant_content_to_messages}` and `openai::responses_api::reasoning_summaries` — see MIGRATING

- *(completion)* [**behavior**] an assistant turn that carried no text and no tool calls is now empty instead of a fabricated `AssistantContent::text("")`. Six production sites pushed that part solely to satisfy the non-empty container — including anthropic's documented empty `end_turn` follow-up — and it reached history and the wire indistinguishable from a real empty text block; `is_empty_assistant_turn` recognises both spellings of an empty turn — zero parts, or one empty unannotated text block (a shape a blocking wire can still deliver, and the shape old histories encode) — and the agent loop keeps such turns out of history; caller-supplied history is never filtered. Three guards that were unreachable only because of the padding are removed rather than made live: reachable, each would have failed a run that previously succeeded

- *(completion)* the container's two enforcement directions separate: `CompletionRequest::validate_message_content` rejects an empty `chat_history`, a content-less user/assistant message, or a tool result whose own block list is empty (named by tool, since `ToolResult::content` carried the same by-construction guarantee; a single empty-string block still passes — the rule is on block count, not block content) once at the request boundary (called by `CompletionRequestBuilder::send`/`stream`, which is also how both agent surfaces issue their requests; System content exempt — it is a `String` and was never constrained), while inbound per-wire guards route through the new `message::require_non_empty(items, || error)` — most via `require_non_empty_response`, which pairs the guard with the shared `EMPTY_RESPONSE_ERROR` wording (see the dedicated entry); guards rejecting a different state keep their own text; its `Option` sibling `message::non_empty(items)` is the one home for the "empty list means absent" rule (the replacement for `OneOrMany::from_iter_optional`). The OpenAI Responses websocket session validates in `send_with_options` — it takes a raw `CompletionRequest`, so it is a direct-to-model surface under the validator's own contract

- *(tool)* [**breaking**] an empty tool output is rejected at construction: `ToolOutput::content` is fallible (`Vec<ToolResultContent> -> Result<ToolOutput, ToolExecutionError>`) and `From<Vec<ToolResultContent>>` becomes `TryFrom` — on 0.41 the `OneOrMany` argument made the empty case unrepresentable, and the `Vec` argument moves that guarantee into the return type. Every construction route is covered (rich-content tool returns, tools returning `ToolOutput` directly, hook rewrites), so the failure surfaces as an ordinary tool error fed back to the model instead of a zero-block result entering history and aborting the run at the next request's boundary check. Deliberately not normalized to an empty text block (that would fabricate content the tool never produced); an empty **MCP** result still normalizes to one empty text block, because that outcome is protocol-legal and outside the tool author's control. `text`/`json`/`one` stay infallible

- *(ollama)* [**behavior**] converting an assistant history message with empty `content` no longer mints the legacy `vec![AssistantContent::text("")]` sentinel — the text block is pushed only when non-empty, matching the response decode path. Consequence: such a converted message cannot be replayed through the request boundary (`validate_message_content` rejects a content-less assistant message); callers ingesting raw Ollama history filter empty assistant messages rather than rig inventing content for them. The agent loop never produces this shape — it drops empty turns before history

- *(completion)* [**breaking**] `UserContent::tool_result`/`Message::tool_result` record their string as the correlation handle only (echo `ToolCall::id`), never as a provider-issued id — a bare string cannot prove provider provenance, and stamping an echoed minted handle sent it upstream on optional-id wires as an asymmetric functionCall/functionResponse pair. Wire output on required-id wires is unchanged (the handle is the fallback). Callers holding a wire-issued id use the new `tool_result_from_wire` (the `ToolCall::from_wire` mirror), `tool_result_with_call_id`, or `tool_result_for`
- *(streaming)* [**breaking**] `RawStreamingChoice::ReasoningEnd` gains `wire_sent`; the driver yields the completed `Reasoning` block for a payload-carrying end OR a bare end frame the wire itself sent (anthropic `content_block_stop` on an unsigned thinking block, restoring its pre-lifecycle completed event) — only adapter-synthesized bare ends stay silent. The chat-compat adapter now synthesizes the reasoning end before tool calls as well as text, matching the ollama adapter

- *(ollama)* `ToolCall` gains `id: Option<String>`: modern daemons issue `"id":"call_..."` and rig now preserves it as the durable tool-call id (streaming key + blocking history) instead of discarding it; absent ids still mint. Never serialized back — request shapes are unchanged

- *(openrouter)* id-less encrypted reasoning details key by a dedicated `MintKind::EncryptedReasoning`, so a whole encrypted block can no longer replace reasoning text accumulating under the shared compat `Reasoning` mint key

- *(gemini)* [**behavior**] streamed function calls carry a single-wire identity: the wire's one id travels as the part id only, so `provider` is `{call_id, item_id: None}` — filling both slots fabricated a dual identity whose fake item id could pass the foreign-id guard on cross-provider replay, and made `stream()` and `completion()` disagree on byte-identical wire content

- *(cohere)* [**behavior**] an id-less tool call mints its correlation handle and records no provider id, instead of adopting the tool *name* as a provider-issued id — a name-as-id is fake provenance and collided two parallel same-tool calls in one turn

- *(xai)* [**behavior**] the request conversion guards its converted input with `require_non_empty` (id-less reasoning has no xAI representation and drops — now with a warning — so rig-level non-empty content can convert to zero wire items), failing locally with a named error instead of shipping `input: []` for a remote 400

- *(openai)* [**behavior**] a Responses unary response with `status: incomplete` and an empty choice normalizes with its finish reason (e.g. `Length` after the documented truncated-function-call drop) instead of being rejected as "contained no message or tool call" — matching the streaming path, which already surfaced truncation this way

- *(completion)* [**breaking**] tool-call identity is typed: `message::ToolCall` is `{ id: ToolCallId, provider: Option<ProviderCallId>, .. }` and `message::ToolResult` is `{ call: ToolCallId, provider: Option<ProviderCallId>, name: String, .. }`. `ToolCallId` is non-empty by construction and minted at the provider boundary when the wire issued no id (`ToolCall::from_wire` / `from_dual_wire`); `ProviderCallId` carries the wire's `call_id` plus the dual-wire item id (OpenAI Responses `fc_*`). Serializers send `provider.call_id` when the provider issued one, else the minted handle on wires that require an id; optional-id wires (Gemini REST/gRPC) omit minted handles entirely, and the Interactions/Responses/xAI "requires `call_id`" request errors are gone. `ToolResult::name` is required and read directly by every name-keyed serializer; the `resolve_tool_result_names` back-compat pairing shim (and its name-in-id legacy encodings) is deleted. Persisted-history serde is breaking — see MIGRATING

- *(streaming)* [**breaking**] the raw grammar is a part lifecycle: `ReasoningStart`/`ReasoningEnd`/`TextEnd` join the vocabulary, `ReasoningSignature` is deleted (a trailing signature is an `End` arriving late), and the accumulator becomes open-maps into an arrival-ordered part list with entity-owned idempotence — a repeated `ToolInputEnd` finalizes nothing even with an authoritative payload (review 84a43e9e #1), and one end primitive replaces the per-adapter signature/boundary branches (#2). Boundary-less wires synthesize their ends in the adapter; the ordinal machinery, `closed_by_full_call`, and every adapter-side thought/restatement buffer are deleted

- *(streaming)* [**breaking**] the raw-event identity is the opaque `StreamPartId` (no `Serialize`, no rendering, no durable accessor), with the durable provider handle carried separately as `WireId` (`provider_id` on reasoning events, `tool_id` on `RawStreamingToolCall`/`ToolInputEnd`); `WireId::new` rejects the empty string so absence is `None` — the fabricated `Wire("")` class and its per-serializer empty-string filters are gone (review 84a43e9e #3/#4). Public delta ids are rig-generated correlators (`ReasoningDelta` gains `provider_id`; `ToolCallDelta` drops `id`)

- *(completion)* [**breaking**] `message::ToolResult` carries the executed tool's name as required data (`name: String`) — name-requiring wires (Gemini `functionResponse.name`, Ollama tool messages, Vertex, gemini-grpc) read it directly, and an identifier is never replayed as a name (review 84a43e9e #5, pinned by live cross-provider replay cassettes)
- *(gemini)* [**behavior**] Interactions function-call steps assemble their `arguments_delta` fragments through the shared accumulator — previously the wire's fragmented tool-call arguments were dropped entirely (the call aggregated with `{}` args) and an unmodeled `arguments_delta` frame errored the stream; recorded live in `interactions_same_tool_twice`

- *(streaming)* [**breaking**] `RawStreamingChoice`'s part ids (`TextStart`/`ToolCallDelta`/`Reasoning`/`ReasoningDelta`, `ToolInputEnd::id`, `RawStreamingToolCall::id`) are the opaque `streaming::StreamPartId`; `SyntheticIds` lives in `rig_core::streaming` and mints it. `ToolCallDelta` lost `internal_call_id` (the shared accumulator mints it at assembly open; read it from `StreamedAssistantContent::ToolCallDelta`); exhaustive matches over `RawStreamingChoice` need arms for the lifecycle variants (`ToolInputEnd`, `ReasoningStart`/`ReasoningEnd`, `TextEnd`) — `RawStreamingChoice` is not `#[non_exhaustive]`. `MINTED_ID_NAMESPACES`/`is_boundary_minted_id` and the request-side provenance gate are deleted (unrepresentable by construction)
- *(providers)* [**breaking**] `OpenAICompatibleProvider::decorate_streaming_tool_call` returns `Option<ToolCallDecoration>` instead of mutating a `&mut HashMap<usize, RawStreamingToolCall>`; `OutputFunctionCall::arguments` is a `FunctionCallArguments` newtype over the raw string (`.parse()`/`.as_str()`)
- *(providers)* [**breaking**] the Anthropic and OpenAI Responses streaming event enums no longer carry `#[serde(other)]`: unrecognized events triage as `Unknown` in the classify layer, and a known tag with a defective payload is a decode error instead of a silent absorb
- *(providers)* [**behavior**] gemini (REST/interactions/gRPC), vertex and ollama no longer fabricate durable tool-call ids — not from an index and not from the tool name, so two calls to the same tool in one turn stay distinct; an id-less call carries a minted `ToolCallId` with `provider: None`, replays with the wire id absent on optional-id wires, and the function name a replayed tool result needs travels as the required `ToolResult::name`

- *(providers)* provider plumbing consolidation: `GET /models` listing, OpenAI-wire multipart transcription (openai/groq/azure), image generation, audio generation, OpenAI-wire embeddings (azure/doubleword), and the tolerant provider-error envelope now share `providers::internal` drivers, and copilot's duplicated chat-completions streaming profile/wire types and unary response conversion are deleted in favor of openai's shared path. Wire shapes are preserved and pinned by new form/body tests (azure transcription still omits `model` and posts to its deployment path; it does send the request's `language` — see the Fixed entry below); copilot's chat route gains openai's tolerant streaming dialect (defaulted tool-call `index`, object-or-string `arguments`, array-of-parts deltas) and `reasoning`/`reasoning_details` handling on both surfaces

- *(huggingface)* [**breaking**] `transcription::TranscriptionResponse` is a re-export of `openai::TranscriptionResponse` rather than HuggingFace's own `{ text: String }` copy of it: the model behind it is the shared OpenAI-wire transcription model now, and the two types decoded the same body. Nothing on the wire moves, but the type identity does — an out-of-tree impl written for both paths becomes a conflicting implementation, and a struct literal must also supply the `usage` field the OpenAI type carries (`#[serde(default)]`, so decoding a response without it is unaffected)

### Added

- *(vector-store)* `vector_store::request::SqlCondition<P>` — a rendered SQL-style condition together with its positional bind parameters, shared by the SQL-flavoured stores whose filter algebra differs only in the parameter type `P` and in the placeholder token their driver expects. The leaf constructors `binary` and `list` take that token from the caller instead of baking one in (`raw` renders a fragment that carries none), `and`/`or`/`not` compose, and `condition()`/`params()`/`into_parts()` read the result back with the parameters in placeholder order. rig-postgres' `PgSearchFilter` and rig-scylladb's `ScyllaSearchFilter` are newtypes over it
- *(providers)* `providers::internal::chunk_lifecycle` is public, joining `adapter`, `wire` and `tool_call_bridge`, so an out-of-tree provider over a boundary-less wire (ollama's `thinking`, cohere's `thinking` content, gemini REST's `thought` parts) inherits the reasoning-lifecycle derivation instead of hand-rolling it: the adapter declares what one wire chunk carried as a `ChunkParts` and `MintedReasoningLifecycle::emit_chunk` derives the canonical event sequence, so "forgot to close the open reasoning block before another part class" is not expressible through the interface
- *(streaming)* `StreamingCompletionResponse::identity()` returns the stream's `completion::ResponseIdentity` as one carrier, alongside the existing `CompletionResponse::identity()`/`StreamFinal::identity()`: the message id comes from the stream (an explicit `MessageId` event outranks the terminal record, which backfills the field when the stream never saw one), while the response-scoped and transport ids exist only on the terminal record and stay `None` for a stream that ended without one
- *(completion)* the provider's transport request id now survives onto **errors** (#2314): `ProviderResponseError` gains `provider_request_id` (a public field; the type is not `#[non_exhaustive]` — #2335 removed the attribute workspace-wide — so a full struct literal must name it, and `headers`, alongside `status`/`body`; the `new`/`without_status` constructors plus the `with_provider_request_id`/`with_headers` setters are the shape that does not have to be revisited every time transport metadata grows, and the id appears in the Display message as ` (request id: …)`), read via the new `provider_request_id()` accessor on every capability error enum and forwarded through rig-agent's `PromptError`/`StructuredOutputError`. Capture points: the unary driver reads the id off failed responses via the new header-preserving transport variant `http_client::Error::InvalidStatusCodeWithDetails` [**breaking**: new variant on the exhaustive `Error` enum; its Display matches `InvalidStatusCodeWithMessage`]; in-band SSE provider error envelopes are stamped with the delivering connection's id; Bedrock attaches its SDK error metadata id to preserved provider bodies. [**behavior**] providers with a request-id contract (anthropic, openai chat + responses, xai, groq, copilot) now preserve **non-success HTTP responses as `ProviderResponse`** instead of `HttpError` — status and body stay recoverable through the same `provider_response_*` accessors, and classification follows the provider's contract, never a particular response's headers; contract-less providers (gemini, cohere, ollama, compat defaults) keep the exact previous shape. Census notes recorded live: groq sends `x-request-id` on errors; **xAI sends it on successes but omits it on 4xx responses** (`None` by design); errors with no HTTP response (connect failures, timeouts) have nothing to capture and stay `None`


- *(completion)* `CompletionResponse.provider_request_id` and `StreamFinal.provider_request_id` carry the provider's transport-level request id — the id provider support asks for when investigating a request — captured from each provider's request-id response header on both the unary path (`send_completion` reads it before consuming the body) and the streaming path (from the SSE connection's response headers; every successful (re)connect *replaces* the captured value, including with `None` when that connection omits the header, so the terminal record always names the connection that delivered it). The stream→response conversion (`From<StreamingCompletionResponse> for CompletionResponse`) carries it like every other terminal field. `ResponseIdentity` is the shared carrier for the three distinct id axes (message-scoped, response-scoped, transport), built by `CompletionResponse::identity()` / `StreamFinal::identity()`. Capture is a per-provider contract, not a header allowlist: Anthropic `request-id` (Ext-defaulted for compatible gateways), OpenAI chat + Responses, xAI, Groq (verified live to send it), and Copilot chat + Responses (all four route/surface combinations) `x-request-id` (compat-provider default `None`); Bedrock maps its SDK-captured `x-amzn-RequestId` on the unary *and* converse-stream surfaces. Gemini, Cohere, OpenRouter, DeepSeek, and Mistral report no request-id header (verified live; Cohere's `x-debug-trace-id` and OpenRouter's `x-generation-id` deliberately not adopted as transport request ids) and yield `None` — a documented outcome, never an error. Anthropic/OpenAI/xAI/Bedrock raw wire responses and streaming terminals expose the same field for raw-surface callers, stamped by the transport since it is never part of a response body (#2265)
- *(anthropic)* `CompletionModel::with_static_prefix_cache_ttl(CacheTtl)` sets the cache TTL for the static prefix (tool definitions + system prompt) independently of the moving conversation-tail breakpoint, enabling the mixed configuration Anthropic's pricing rewards: `1h` on the prefix that is byte-identical across sessions, the 5-minute default on the tail that changes every turn (a 1h cache write costs ~2x base input where a 5m write costs ~1.25x, so caching the tail at `1h` pays the premium for retention nothing consumes). Composes with `with_prompt_caching`, `with_automatic_caching`, and raw top-level `cache_control`; on its own it marks just the prefix. Unset, every existing constructor's request bytes are unchanged. Setting the prefix to `FiveMinutes` under a 1h top-level TTL fails client-side with an error naming both knobs (#2266)
- *(anthropic)* [**breaking**] `Usage` and streaming terminal usage parse the per-TTL `cache_creation` breakdown (`CacheCreation { ephemeral_5m_input_tokens, ephemeral_1h_input_tokens }`) alongside the preserved `cache_creation_input_tokens` aggregate, so mixed-TTL configurations are observable; the streaming adapter carries the split from `message_start` (the only frame Anthropic reports it on) onto the terminal record. `anthropic::completion::Usage` and `anthropic::streaming::PartialUsage` each gain a public `cache_creation` field and neither carries `#[non_exhaustive]`, so code building either with a full struct literal must add it — `PartialUsage` derives `Default` so `..Default::default()` absorbs it, `Usage` does not. The field is `#[serde(default, skip_serializing_if = "Option::is_none")]`, so usage JSON persisted by 0.41 still deserializes and an absent breakdown stays off the wire
- *(venice)* new provider for the [Venice](https://docs.venice.ai) API (`providers::venice`), covering chat completions and streaming (tools, vision, structured output, reasoning) through the shared OpenAI-compatible path, embeddings, `GET /models` listing, Venice's native `POST /image/generate` (feature `image`), text-to-speech (feature `audio`), and transcription. Configure via `VENICE_API_KEY` (and optional `VENICE_BASE_URL`). Venice's own `venice_parameters` request block — web search and citations, thinking control, characters, prompt-cache hints — is a serializable `VeniceParameters` helper that callers merge through `additional_params`, and `venice::CompletionResponse` preserves Venice's response-only blocks (the resolved parameter echo with `web_search_citations`, plus per-request `cost`) that an OpenAI-shaped decode would drop. Venice's video, image-editing, music, web-augmentation, crypto-RPC, character, API-key and billing endpoints have no Rig trait and are not wrapped
- *(streaming)* `wire::classify_typed_event` extends the decode-then-validate policy to typed-transport wires (bedrock, candle, gemini-grpc): modeled variants are `Known`, the SDK's non-exhaustive/unrecognized variants are `Unknown`, SDK decode errors are `Corrupt` — a typed transport earns no policy exemption
- *(streaming)* `WireAdapter` gains an associated `Frame` type so typed-event wires implement the same contract over their SDK events; `classify` now takes the frame by value
- *(streaming)* the conformance corpus accepts typed-event input (`WireInput::{Bytes, Event}`), so typed wires run the shared scenarios events-first with no mock transport; frame-level scenarios a typed wire cannot spell report visible skips
- *(streaming)* wire-sequence conformance corpus (`test_utils::streaming_conformance` + `tests/core`) driving raw bytes through each provider's full pipeline; recorded `streaming_grammar` cassette suites for openai (reasoning summaries, encrypted multi-part reasoning, parallel tool calls, incomplete) and gemini (max-tokens truncation, tool calls, thinking, interactions requires_action)
- *(streaming)* `OpenAICompatibleProvider::streaming_detail_reasoning` — a defaulted hook letting an OpenAI-compatible provider map a per-chunk streaming detail onto a complete reasoning block instead of a tool-call decoration
- *(completion)* `ToolCall::wire_call_id()`/`ToolResult::wire_call_id()` — the one derivation for a required-id wire's call-id slot (provider-issued when it exists, else the minted handle), replacing the expression every serializer hand-rolled — and `ToolCallId::for_provider`, the shared correlation-handle derivation
- *(completion)* add typed `raw_completion`/`raw_stream` escape hatches on every provider model
- *(completion)* add public `ProviderCapabilities`, replacing `CompletionModel::composes_native_output_with_tools`

### Fixed

- *(openrouter)* `ProviderResponseExt::get_text_response` applies openai's whole-message refusal rule instead of its own. OpenRouter kept a private copy of `assistant_message_text_response` that appended a non-empty top-level `refusal` unconditionally, so a message carrying both content text and a refusal read back as `"text\nrefusal"`; it now routes through openai's shared `assistant_refusal_fallback`, which uses the field only when no content part carries text, and reads back as `"text"` — the same answer every other OpenAI-compatible provider gives for those bytes. A refusal-only turn is unaffected, and nothing on the wire or in the normalized response changes

- *(gemini)* [**breaking**] `gemini::completion::GenerationConfig`'s `Default` is now all-`None` (was `temperature: Some(1.0)`, `max_output_tokens: Some(4096)`), so a default config puts nothing on the wire and Gemini applies each model's own documented limit. The hardcoded values were injected into two request paths that seeded themselves from `Default`: **native structured output** (any `output_schema` turn) and **image generation**. Both silently capped output at 4096 tokens and pinned temperature to 1.0 regardless of the caller's budget — a 16k-token structured-output request was truncated at 4096, and because the streaming path reports a `MAX_TOKENS` turn with no content as a normal completion, the truncation surfaced as an unexplained empty response rather than an error. Only callers who relied on the model default were affected: an explicit `max_tokens` was applied afterwards and overwrote the injected value. Callers who *want* the old values must now set `temperature`/`max_tokens` explicitly. A recorded matrix pins all four corners (schema without `max_tokens`, schema with `max_tokens`, `temperature` alone, image generation) at the request boundary. See #2322

- *(gemini, model listing)* [**breaking**] `Model` gains `max_output_tokens`, and Gemini's listing stops discarding it (#2322). Gemini reports `outputTokenLimit` for every model — 65,536 for `gemini-2.5-flash`, i.e. ~16x the hardcoded 4096 cap above — and rig dropped the field during conversion, so nothing in the library ever knew the real per-model limit. The new field is distinct from `context_length` (input window vs output ceiling) and `None` when a provider's listing does not report one, never a rig-invented default. Rig deliberately does **not** send this value on requests: omitting an output limit is what lets the provider apply its own per-model default, and populating `maxOutputTokens` from the listing would reintroduce a rig-chosen cap by another route. It is for callers and diagnostics. OpenRouter reports an equivalent under `top_provider.max_completion_tokens`; its listing entry did not parse it when this landed and now does, so the ceiling is reported there too — read off the wire, never guessed, and still `None` for the entries that omit it

- *(loaders)* the `pdf` feature builds for `wasm32-unknown-unknown`. `lopdf` reaches `getrandom` through its PDF-encryption support, and `getrandom` refuses to compile for browser wasm until a backend is selected — the target triple alone cannot pick one — so any browser build that enabled `pdf` failed outright with getrandom's "not supported by default" error. `lopdf`'s `wasm_js` backend is now enabled under `cfg(all(target_arch = "wasm32", target_os = "unknown"))` only, leaving the native and WASI dependency graphs untouched; the runtime condition is that the host provides the Web Crypto API's `Crypto.getRandomValues`, as browsers, Web Workers and Node.js 19+ do, and the README's WASM target support section says so

- *(azure)* text-to-speech reaches the deployment it names: the model passed the literal `"/audio/speech"` where `post_audio_generation` expects a deployment id, so every Azure TTS request went to `{endpoint}/openai/deployments/audio/speech/audio/speech?api-version=…` — a deployment that cannot exist, which made the declared `AudioGeneration` capability fail for every caller and every key. The model name is now the deployment segment, the request body drops the redundant `"model"` key (Azure names the model in the path), and Azure text-to-speech carries its own API version — `2025-04-01-preview`, the first deployment-scoped Azure release that exposes the route, overridable with the new `ClientBuilder::audio_api_version` — rather than the GA `api_version` (`2024-10-21`) the other Azure routes share

- *(azure, doubleword)* [**breaking**] embeddings report real token usage — both providers implemented only `embed_texts` and fell through to the zero-usage `embed_texts_with_usage` default; they now ride the shared OpenAI-compatible embeddings path, which parses `usage` (including `prompt_tokens_details.cached_tokens`). Azure's deployment-URL request shape (no `model` field, `dimensions` still sent) and doubleword's never-sends-dimensions wire are pinned by tests. Breaking for doubleword: `EmbeddingModel` becomes a type alias for `openai::embedding::GenericEmbeddingModel<DoublewordExt, T>`, and the hand-rolled `doubleword::{EmbeddingResponse, EmbeddingData, Usage}` response types — re-exported from `providers::doubleword` by `pub use embedding::*` — are deleted; decode a Doubleword embeddings body with `openai::embedding::EmbeddingResponse`/`EmbeddingData` instead. `EmbeddingModel::new(client, model, ndims)` keeps its signature, so building and using the model needs no edits

- *(openrouter)* `max_tokens` reaches the wire — the request builder hardcoded `max_tokens: None`, silently dropping the caller's configured value on every request; a regression test asserts the serialized body carries it

- *(providers)* error envelopes in azure, groq, hyperbolic, cohere, voyageai, anthropic, and openai tolerate an OpenAI-style nested `{"error":{...}}` body AND a body carrying both `message` and `error` keys — previously a nested body failed both untagged arms (and a dual-key body was a serde duplicate-field error) and surfaced as a `JsonError` instead of a classified provider error; the non-null `error` key wins as the canonical provider error object, and the raw body still flows through `from_http_response` unchanged

- *(azure)* [**behavior**] transcription sends the request's `language` form field — the hand-rolled request silently dropped a caller's `.language(..)` while the public builder exposed it, leaving Azure Whisper to auto-detect

- *(transcription)* [**behavior**] string-valued `additional_params` go onto the multipart form verbatim for openai/groq/azure — they were serialized with `Value::to_string`, so `{"response_format": "verbose_json"}` reached the wire JSON-quoted (`"verbose_json"`) and was rejected or ignored; non-string values stay JSON-encoded

- *(telemetry)* [**behavior**] the openai responses API, ollama, chatgpt, xai, and copilot unary paths record usage through the shared span helpers, so `cache_creation.input_tokens`, `tool_use_prompt_tokens`, and `reasoning_tokens` are now recorded and all-zero usage is suppressed per `Usage::has_values()` (previously hand-rolled records wrote literal zeros and missed those fields)

- *(model-listing)* [**behavior**] every `GET /models` implementation — including copilot's auth-derived listing, via the shared `map_transport_error` — pre-maps a transport-level `InvalidStatusCodeWithMessage` into `ModelListingError::api_error_with_context` (provider/path/status/body preserved); previously only deepseek and xiaomimimo did, and the other providers lost that context

- *(model-listing)* [**behavior**] an entry that omits `created` or `owned_by` no longer fails the entire listing. OpenAI, Mistral, DeepSeek and Xiaomi MiMo each hand-wrote a near-identical entry DTO with those fields required (DeepSeek's and Xiaomi MiMo's modeled `id` and `owned_by` only), and the `{"data": [...]}` envelope decodes as one value, so a single incomplete entry took `list_models()` down with a serde error instead of listing what the response did describe. One shared entry replaces the four, with `id` as its only required field, pinned by `minimal_entry_decodes_with_id_alone`. It also models `name` and `created`, which DeepSeek's and Xiaomi MiMo's entries did not, so `Model::name`/`Model::created_at` carry those keys when a listing sends them rather than being unconditionally `None` — DeepSeek's live listing sends neither, so its own models read the same. Mistral has since taken its entry back, under the same all-optional rule, for the `description`/`max_context_length`/`type` keys its listing carries

- *(anthropic)* a streamed turn's `Usage::input_tokens` prefers the terminal `message_delta` and falls back to `message_start`, instead of always reading `message_start`. Anthropic proper reports the count on both frames and they agree, so nothing changes there; Anthropic-*compatible* gateways need not, and OpenRouter's Messages endpoint can send `input_tokens: 0` on `message_start` with the real count on `message_delta` (observed when it routes to an Amazon Bedrock upstream) — which silently surfaced as `Usage { input_tokens: 0 }`, worse than a missing value for a consumer sizing a context window from it. A zero on the delta is read as "not reported" so a gateway with the inverse split cannot erase a count `message_start` got right

- *(providers)* name-keyed wires (Gemini REST/Interactions, Ollama, Vertex AI, gemini-grpc) fill a cross-provider ingested result's empty `name` from its paired call at request assembly (`providers::internal::resolve_empty_tool_result_names`, matched by identifier only) — rig's own inbound converters stamp `""` because the Anthropic/OpenAI-chat/Cohere/Bedrock wires carry no name, and replaying it raw was INVALID_ARGUMENT
- *(openai)* a Responses tool call whose `output_item.done` frame was lost survives a healthy `response.completed`: the terminal closes every open slot (with the announced dual-wire identity) and the call finalizes from its streamed fragments instead of being discarded as truncation — the same terminal-proof drain Interactions and chat-compat ship
- *(gemini)* an Interactions streamed call carries a single-wire identity: its `fc_*` id lands in `provider.call_id` with `item_id` empty, instead of a fabricated Responses-shaped dual id whose fake item id passed the foreign-id guard on cross-provider replay
- *(streaming)* whole-call adoption accepts the two gateway shapes it rejected: a nameless args-only assembly (empty name is no evidence against the restatement) and a buffer still holding the literal `null` placeholder (covered vacuously) — both used to publish the call under a fresh minted id and finalize the assembly as a duplicate
- *(openai)* a multi-block Responses reasoning done item (summaries + `encrypted_content`) aggregates as ONE part: the adapter emits one wire-sent end restatement carrying every block in wire order, so history replays exactly one `rs_*` input item instead of same-id siblings that duplicated the reasoning input on the next request (xai shares the fix)
- *(openai)* Responses reasoning keys are slot-scoped like the tool path: a slot mixing id-bearing and id-less frames (gateways, ChatGPT envelope-less replays) keeps one assembly key fixed at the slot's first event, and the done item resolves through the slot map — no more orphaned partial part beside the superseded one
- *(openai)* an unparseable restated done-item argument string is re-emitted into the assembly buffer only when no `function_call_arguments.delta` fragment preceded it — fragments already streamed those bytes, and the re-emit doubled them against consumers and the accumulation bound
- *(streaming)* a second signature under a per-stream constant reasoning key (gemini, cohere, ollama) records a distinct signed part instead of overwriting the first — signatures cannot merge, and the overwrite left only the last one to replay (`MISSING_THOUGHT_SIGNATURE`)
- *(streaming)* a wire-keyed whole tool call adopts an open minted assembly only when it evidently restates it (same tool name, arguments covering the streamed fragments) — an unrelated id-bearing call could steal the single open assembly and silently drop its streamed arguments
- *(gemini)* Interactions closes function-call assemblies still open at `interaction.completed` (a lost or reordered `step.stop` no longer loses the announced call), and `step.start` announce arguments are replace-if-no-deltas instead of concatenating with `arguments_delta` fragments into unparseable `{..}{..}`
- *(gemini)* an Interactions `model_output` step yields every convertible content item in wire order — a `function_call` following text in the same step no longer vanishes
- *(gemini)* a trailing `thoughtSignature` arriving after the answer text now signs the reasoning block that carries the chain-of-thought (a signature-bearing `ReasoningEnd`) instead of appending an empty signed sibling and leaving the real thinking to replay unsigned; the gRPC surface no longer drops a signature carried on a non-thought part
- *(streaming)* non-object JSON frames (a gateway keep-alive `null`, a bare array or scalar) classify `Unknown` (warn-and-skip) on every classifier instead of `Corrupt` (a fatal in-band error); conversely an Anthropic `content_block_delta` whose `delta` omits `type` is `Corrupt` instead of a silent skip that yielded a successful empty completion
- *(streaming)* an OpenAI-compatible error body that also carries `"choices":[]` (or `null`) is detected as an error — previously the mere presence of a `choices` key masked it and a following `[DONE]` committed the failed turn as a successful zero-usage completion (introduced in #1944)
- *(streaming)* cancelling a paused stream terminates instead of deadlocking (`cancel()` also resumes); a streamed tool call's accumulated argument bytes are bounded, with overflow finalizing through the wire's unparseable-input policy instead of growing memory without bound
- *(openrouter)* [**data loss**] encrypted reasoning (`reasoning_details` of type `reasoning.encrypted`) was dropped on every streaming turn and could not be replayed on the next request — the decoration key never matched (reasoning ids are `rs_*`, tool ids `call_*`) and the detail arrived before any tool slot existed. It now reaches the aggregated choice as `ReasoningContent::Encrypted`, matching the non-streaming path. Two committed cassettes had recorded the loss into their turn-2 request bodies; both were re-recorded live and the provider accepts the replayed blob
- *(anthropic)* signature-only thinking blocks are no longer dropped: a block whose text is empty but which carries a signature survives into chat history and replays, matching the non-streaming path (Anthropic rejects a replayed adaptive-thinking turn missing it)
- *(openai)* `response.reasoning_text.done` is a modeled Responses event; it previously logged an "unknown event type" warning and passed through as `Unknown` on every raw-reasoning block, across the SSE, buffered and websocket surfaces
- *(streaming)* a wire that streams a tool call's input as fragments and then restates it as a complete `ToolCall` now publishes the completed call under the `internal_call_id` its deltas already used, and a trailing `ToolInputEnd` for that id no longer produces a duplicate call in the aggregated choice
- *(streaming)* re-polling a drained `StreamingCompletionResponse` no longer re-runs the destructive aggregation, which replaced the aggregated choice with an empty text part
- *(streaming)* a paused stream parks on the pause channel instead of busy-polling its executor task
- *(streaming)* [**breaking**] a completed reasoning event restates the correlator its deltas carried (including through a synthesized silent end followed by trailing signature metadata), and streamed-turn assembly keeps distinct reasoning parts distinct instead of merging them into one buffer
- *(streaming)* [**breaking**] the raw unknown-frame passthrough carries `streaming::UnknownPayload` instead of a bare `serde_json::Value`: serialization is transparent, `Debug` is redacted by the type (structural byte count only), and consumers opt into the content via `.value()`

### Removed

- *(telemetry)* [**breaking**] `ProviderResponseExt::get_output_messages` and its `type OutputMessage` — nothing read them across 14 impls (`SpanCombinator::record_response_metadata` records only the response id and model name), so an out-of-tree impl that still defines either now fails with E0437/E0407 and should delete both; `get_text_response` is deliberately kept
- *(client)* [**breaking**] `ImageGenerationClient::custom_image_generation_model` (feature `image`) — a defaulted trait method whose whole body was `Self::ImageGenerationModel::make(self, model)`, which is exactly what the blanket `ImageGenerationClient` impl resolves the trait's own `image_generation_model` to. `client.custom_image_generation_model(m)` becomes `client.image_generation_model(m)`; the model it hands back is built the same way
- *(json)* [**breaking**] `json_utils::null_or_vec` — `null_or_default` is the drop-in for a `Vec<T>` field
- *(anthropic)* [**breaking**] `completion::apply_cache_control`, with no public successor: its replacement `apply_prompt_cache_control` is `pub(super)` and the provider applies the breakpoints on the way out
- *(gemini)* [**breaking**] the fifteen `interactions_api::interactions_api_types::*Delta` structs (`ImageDelta`, `AudioDelta`, `DocumentDelta`, `VideoDelta`, `FunctionCallDelta`, `FunctionResultDelta`, `CodeExecutionCallDelta`, `CodeExecutionResultDelta`, `UrlContextCallDelta`, `UrlContextResultDelta`, `GoogleSearchCallDelta`, `GoogleSearchResultDelta`, `McpServerToolCallDelta`, `McpServerToolResultDelta`, `FileSearchResultDelta`): `ContentDelta` now carries the identically-shaped `*Content` payloads directly, so the JSON is unchanged and only a name in a `match` or a type annotation breaks. `TextDelta`, `ThoughtSummaryDelta` and `ThoughtSignatureDelta` stay — their payloads genuinely differ from their content counterparts — as does `ArgumentsDelta`, which is not a survivor at all: it is new this cycle, and it has no `*Content` counterpart to fold into
- *(streaming)* [**breaking**] the `"aborted"`-substring special case in the stream error path. A `CompletionError::ProviderError` whose message merely *contained* `"aborted"` used to terminate the stream as a clean end-of-stream — silently discarding both the error and every item streamed before it. Such errors now reach the consumer like any other. Real cancellation is unaffected: `StreamingCompletionResponse::cancel()` goes through `Abortable` and still ends the stream normally. Nothing in-tree produced the sentinel

### Changed

- *(streaming)* choice aggregation lives in one `PartsAccumulator` driven by the lifecycle events, keyed by opaque `StreamPartId`s into an arrival-ordered part list with entity-owned idempotence — OpenAI multi-part reasoning items keep every part; the aggregation heuristics in `poll_next` are gone
- *(streaming)* stream parse policy is decode-then-validate, stated once per wire family: known event with a defective payload surfaces an `Err`; unknown event types warn and skip; corrupt frames surface and the stream continues
- *(providers)* copilot's Responses route and the ChatGPT buffered SSE path delegate to the shared Responses interpreter — seven latent behavioral divergences resolved toward the canonical path
- *(streaming)* [**breaking**] reasoning stream events carry mandatory identity: `RawStreamingChoice::{Reasoning, ReasoningDelta}` take `id: StreamPartId` — the opaque accumulation key, with the provider's own item id carried beside it as `provider_id: Option<WireId>` — and the public `StreamedAssistantContent::ReasoningDelta` carries a rig-generated correlator `id: String` plus `provider_id: Option<String>`; providers propagate the wire's identity or mint a stream-stable key, and aggregation keys by exact key — OpenAI Responses summary-delta streams no longer duplicate reasoning content
- *(streaming)* [**breaking**] text-block stream events carry mandatory identity: `RawStreamingChoice::TextStart` takes `id: StreamPartId` and aggregation keys text blocks by it — two OpenAI Responses `message` output items now aggregate as two distinct text parts instead of concatenating; wires that never announce text boundaries need no `TextStart` (a bare `Message` opens a block under a key minted from `MintKind::Text`, preserving single-block aggregation exactly)
- *(streaming)* the public wire-adapter surface (`WireAdapter`, `run_wire_stream`, `run_wire_buffered`, `SyntheticIds`, `ToolCallBridge`) is documented as a contract for out-of-tree provider authors: classify delegation, driver-owns-policy, mandatory part identity (the wire's own key via `StreamPartId::wire`, else a `SyntheticIds` mint — provenance lives in the type, so there is no reserved string namespace to steer clear of and no request-side gate), and the finish/flush obligations (see MIGRATING)

- *(completion)* [**breaking**] normalize completion responses at the provider boundary — `CompletionResponse` and `StreamingCompletionResponse` are concrete and carry normalized `finish_reason`/`provider`/`model`/`message_id`
- *(completion)* [**breaking**] `CompletionModel` no longer requires `Clone`; generic code that cloned models must bound `+ Clone` explicitly, and `completion_request` now gates on `Self: Clone` (a relaxation for implementors — derives kept only for the old bound can be dropped)
- *(completion)* implement `CompletionModel` for `Arc<M>` by forwarding, making the "wrap it in an `Arc`" guidance work through the generic APIs
- *(completion)* [**breaking**] `CompletionResponse::finish_reason` is now a private field with a `finish_reason()` getter, so the `Stop` → `ToolCalls` reconciliation cannot be bypassed by direct assignment
- *(completion)* identifier and model setters on `CompletionResponse`/`StreamFinal` treat empty strings as absent
- *(streaming)* [**breaking**] corrupt stream frames (invalid JSON) are surfaced as `Err` items (stream continues; a later genuine terminal still completes it) instead of being logged and skipped; valid-JSON events with unrecognized shapes are still skipped for forward compatibility — openai responses, copilot, cohere, ollama
- *(streaming)* a bare `[DONE]` after only unparseable frames no longer fabricates a zero-usage terminal record
- *(streaming)* a full reasoning block now supersedes its accumulated deltas in the aggregated choice — correlated strictly by reasoning item id (matching ids or both absent replace; an id on only one side appends), with a by-id fallback scan so interleaved output (reasoning → tool call → completed block) also replaces
- *(streaming)* on a terminal stream error, fully-delivered tool calls are yielded before the terminal `Err` on every path (shared compat, openai responses, copilot) — previously the three paths disagreed
- *(streaming)* stream parse policy discriminates on the known event `type`: known event with a schema defect surfaces an `Err`; unknown event types are skipped for forward compatibility (openai chat default profile included — its silent `Ok(None)` swallow is gone)
- *(completion)* `CompletionResponse` and `StreamFinal` deserialize through a wire-shape mirror that funnels the validating setters, so finish-reason reconciliation and empty-string filtering also hold for persisted values (wire format unchanged)
- *(providers)* the ChatGPT buffered SSE fallback fails the completion on corrupt known frames instead of returning silently partial content; the openai responses websocket path merges terminal-body message text absent from deltas
- *(providers)* gemini interactions `InteractionStatus::is_terminal` enumerates the known in-flight states, so unknown statuses read as terminal instead of spinning a future poll loop forever
- *(providers)* xai `response.reasoning_summary_text.done` events (which carry `text` rather than `delta`) now decode
- *(providers)* delta-less streamed choices (e.g. Azure's `prompt_filter_results` content-filter prelude) parse as no-op frames instead of surfacing a spurious error on every stream — openai-compatible and copilot chat chunk models
- *(providers)* unmodeled Responses `content_part` shapes (`refusal`, `reasoning_text` parts) parse as no-ops instead of erroring refusal/reasoning-text turns; refusal text flows via `response.refusal.delta` as before
- *(providers)* gemini interactions `RequiresAction` is terminal for a poll loop (it never advances without submitted tool results); callers branch on it as a distinct resumable outcome
- *(providers)* copilot Responses streaming treats `response.incomplete` as a genuine terminal (partial content + `Length` finish reason) instead of an error; the WebSocket session preserves streamed partial output on incomplete terminals
- *(providers)* errored streams flush fully-delivered tool calls before ending (shared OpenAI-compatible path and copilot Responses route)
- *(providers)* gemini REST and Interactions wire enums preserve unknown values verbatim (`FinishReason`/`BlockReason`/`InteractionStatus` gained untagged catch-alls), matching the gRPC mapper
- *(providers)* cohere `message-end` without a `delta` still emits the terminal record

### Removed

- *(completion)* [**breaking**] remove `CompletionModel::{Response, StreamingResponse, Client, make}`; model construction moves to the required `CompletionClient::completion_model`
- *(completion)* [**breaking**] remove the `GetTokenUsage` trait — read `StreamFinal::usage`
- *(completion)* [**breaking**] remove `CompletionResponse::raw_response` — use a provider model's `raw_completion`/`raw_stream`

## [0.41.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.40.0...rig-core-v0.41.0) - 2026-07-28

### Added

- *(agent)* restore dynamic context helper ([#2219](https://github.com/0xPlaygrounds/rig/pull/2219)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] split rig-core and rig-agent behind the rig facade ([#2197](https://github.com/0xPlaygrounds/rig/pull/2197)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2197
- *(agent)* add response retry hooks ([#2182](https://github.com/0xPlaygrounds/rig/pull/2182)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(doubleword)* add provider with cassette coverage ([#2163](https://github.com/0xPlaygrounds/rig/pull/2163)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(telemetry)* make sensitive span content opt-in ([#2151](https://github.com/0xPlaygrounds/rig/pull/2151)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openai)* expose complete Responses reasoning metadata ([#2112](https://github.com/0xPlaygrounds/rig/pull/2112)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openai)* support GPT-5.6 models and reasoning controls ([#2106](https://github.com/0xPlaygrounds/rig/pull/2106)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(ollama)* send max_tokens as options.num_predict in native requests ([#2185](https://github.com/0xPlaygrounds/rig/pull/2185)) (by [bugprone](https://github.com/bugprone))
- *(openai)* omit filename for URL-backed PDFs in Responses API requests ([#2166](https://github.com/0xPlaygrounds/rig/pull/2166)) (by [dgrijalva](https://github.com/dgrijalva))
- *(openai)* preserve multipart tool result content ([#2217](https://github.com/0xPlaygrounds/rig/pull/2217)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(anthropic)* support URL-backed PDF documents in requests ([#2215](https://github.com/0xPlaygrounds/rig/pull/2215)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openai)* omit empty non-streaming encrypted reasoning ([#2209](https://github.com/0xPlaygrounds/rig/pull/2209)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openai)* accept nullable strict tool definitions ([#2178](https://github.com/0xPlaygrounds/rig/pull/2178)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(anthropic)* support code execution tool results ([#2158](https://github.com/0xPlaygrounds/rig/pull/2158)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* prevent structured-output tools from shadowing real tools ([#2146](https://github.com/0xPlaygrounds/rig/pull/2146)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini)* preserve image generation error envelopes ([#2147](https://github.com/0xPlaygrounds/rig/pull/2147)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* make Extractor usage accounting match its docs, drop per-attempt clones, fix retry log ([#2109](https://github.com/0xPlaygrounds/rig/pull/2109)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(core,agent)* [**breaking**] make the WASM support matrix explicit and true ([#2213](https://github.com/0xPlaygrounds/rig/pull/2213)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(telemetry)* single declarative completion-parent contract ([#2208](https://github.com/0xPlaygrounds/rig/pull/2208)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- openai Responses API: Filter empty encrypted reasoning content to prevent duplicate reasoning events from being emitted ([#2196](https://github.com/0xPlaygrounds/rig/pull/2196)) (by [boondocklabs](https://github.com/boondocklabs)) - #2196
- *(derive)* [**breaking**] single resolution authority, coherent required semantics, dependency hygiene ([#2207](https://github.com/0xPlaygrounds/rig/pull/2207)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Make managed agent hooks provider-independent ([#2176](https://github.com/0xPlaygrounds/rig/pull/2176)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2176
- Remove built-in agent dynamic context ([#2174](https://github.com/0xPlaygrounds/rig/pull/2174)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2174
- Make AgentRunner the only Agent execution path ([#2161](https://github.com/0xPlaygrounds/rig/pull/2161)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2161
- Add rig-candle local inference and WASM chat ([#2155](https://github.com/0xPlaygrounds/rig/pull/2155)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2155
- *(providers)* share embedding transport ([#2157](https://github.com/0xPlaygrounds/rig/pull/2157)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Simplify tool execution and hook APIs ([#2132](https://github.com/0xPlaygrounds/rig/pull/2132)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2132
- *(telemetry)* centralize completion span lifecycle ([#2115](https://github.com/0xPlaygrounds/rig/pull/2115)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] make core errors non-exhaustive ([#2114](https://github.com/0xPlaygrounds/rig/pull/2114)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* deduplicate HttpClientExt implementations ([#2113](https://github.com/0xPlaygrounds/rig/pull/2113)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- bump rmcp depency to latest ([#2103](https://github.com/0xPlaygrounds/rig/pull/2103)) (by [ThomasMarches](https://github.com/ThomasMarches)) - #2103
- *(core)* collapse Extractor's four retry loops into one private helper ([#2107](https://github.com/0xPlaygrounds/rig/pull/2107)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [bugprone](https://github.com/bugprone)
* [dgrijalva](https://github.com/dgrijalva)
* [gold-silver-copper](https://github.com/gold-silver-copper)
* [boondocklabs](https://github.com/boondocklabs)
* [ThomasMarches](https://github.com/ThomasMarches)

### Added

- *(core)* `rig_core::telemetry::Empty` re-exports `tracing::field::Empty`, so a
  runtime can declare a completion-parent field as not-yet-valued without taking
  a direct `tracing` dependency.

### Changed

- *(core)* The telemetry completion-parent contract has one declarative
  source: the new `rig_core::telemetry::completion_parent_span!` macro
  declares the adoption marker and every required `gen_ai.*` field. `tracing`
  bakes a span's field set into static metadata and `Span::record` silently
  no-ops on undeclared fields, so a hand-mirrored field list that drops one
  field loses that telemetry with no error. Exact-set tests now pin the macro
  against `COMPLETION_PARENT_REQUIRED_FIELDS` and against the span the
  completion builder itself creates, so the two can no longer drift. A
  completion parent that carries the marker but omits a required field
  triggers a `warn!` naming the missing fields — once per offending span
  callsite, so two broken runtimes are both reported — before it degrades to
  a fresh `rig::completions` child span. The macro accepts an optional
  `parent:` argument (default: the current span), and its expansion resolves
  `tracing` through `rig-core`, so downstream crates do not need a direct
  `tracing` dependency merely to invoke it (see the `Empty` re-export above).
  Nothing is breaking: the marker field and the required field set are
  unchanged.

- *(agent)* [**breaking**] Remove the completion-model parameter from
  `AgentHook`, `HookStack`, and the erased hook interface. Managed response
  hooks now receive canonical Rig lifecycle fields (`prompt`, `content`,
  `usage`, and `message_id`) through non-generic `CompletionResponse` and
  `StreamResponseFinish` events. This lets one concrete hook attach to agents
  backed by different providers. Typed raw provider responses remain available
  from direct `CompletionModel` completion and streaming APIs.

  ```rust
  // Before
  impl<M: CompletionModel> AgentHook<M> for TelemetryHook { /* ... */ }

  // After
  impl AgentHook for TelemetryHook { /* ... */ }
  ```

- *(agent)* [**breaking**] Make `AgentRunner` the only execution path for configured agents: remove the raw `Completion` and `StreamingCompletion` traits and their `Agent` implementations, make agent execution state private, add runner-backed per-request overrides, and route `Extractor` through the full hook lifecycle. Raw hook-free requests remain available explicitly through `CompletionModel`.

  Migration examples:

  ```rust
  // Before: built a one-shot request from configured Agent state, but bypassed
  // the AgentRunner lifecycle.
  agent.completion(prompt, history).await?.send().await?;

  // After, for managed Agent execution: hooks, tools, retrieval, memory, and
  // turn accounting all run. Budget enough calls for tool follow-ups.
  agent
      .runner(prompt)
      .history(history)
      .max_turns(3)
      .run()
      .await?;
  ```

  Streaming follows the same boundary:

  ```rust
  // Before
  agent.stream_completion(prompt, history).await?.stream().await?;

  // After
  let stream = agent
      .runner(prompt)
      .history(history)
      .max_turns(3)
      .stream()
      .await;
  ```

  The runner consumes tool calls instead of returning the first raw model
  response. If the old caller handled that response itself, or for any other
  intentionally hook-free provider transport, start from the model rather than
  an `Agent`:

  ```rust
  model
      .completion_request(prompt)
      .messages(history)
      .send()
      .await?;

  let stream = model
      .completion_request(prompt)
      .messages(history)
      .stream()
      .await?;
  ```

  `AgentRun::new(prompt).with_history(history)` remains the public sans-I/O
  state machine for custom drivers. It contains no model, tools, memory, or hook
  stack: callers must handle every `AgentRunStep`, perform provider/tool IO, and
  feed results back explicitly. It is not a way to execute a configured
  `Agent`; use `AgentRunner` for that.

  An `Agent` also keeps its configured model private and fixed. Applications
  that previously called `.model(...)` or `.model_opt(...)` on the returned raw
  request builder should retain the provider `CompletionModel` and use its raw
  request API, or construct a separate `Agent` for that model selection.
- *(providers)* [**breaking**] Move Together, OpenRouter, and Mistral embeddings onto the shared `GenericEmbeddingModel`, with provider-specific endpoint and typed request-shaping hooks. Together now forwards configured embedding dimensions, Mistral maps Codestral Embed dimensions to `output_dimension` while rejecting dimensions for fixed-size models, compatible providers may omit usage without weakening OpenAI's public response type, and Base64 response encoding is rejected before sending because the shared parser accepts numeric vectors. Remove the superseded provider-specific embedding response/data types, Together's API envelope module, and OpenRouter's duplicate `EncodingFormat`.
- *(tool)* [**breaking**] Replace the parallel tool-execution APIs with one structured path. Typed tools now implement only `Tool::call(&mut ToolContext, Args) -> Result<Output, Error>`; author-facing errors remain typed until private runtime erasure normalizes them into `ToolExecutionError`, `ToolContext` carries inbound values and host-only result metadata, `ToolResult` is the single runtime observation, and `ToolSet::execute` / `ToolServerHandle::execute` are the dispatch surfaces. Event-specific hook action types make invalid event/action combinations unrepresentable.
  - Tool implementations: retain one typed `type Error` for ordinary `?` propagation and direct-call tests; remove `classify_error`, `call_with_extensions`, and `call_structured`. The optional `map_error` method classifies domain failures at the erased boundary, while its default preserves the source as `Other`. Return refusals through `map_error` with `ToolExecutionError::refused`, and attach host-only result metadata with `ToolContext::insert_result`.
  - Context: replace `ToolCallExtensions` and `ToolResultExtensions` with `ToolContext`; replace request/runner `.tool_extensions(...)` with `.tool_context(...)`. Each dispatch snapshots inbound context exactly once, isolates tool-local mutations, and publishes only result metadata back to the caller and hooks.
  - Dynamic tools: `ToolDyn` is removed from the public API; use `DynamicTool` for runtime-defined tools. Rig's erased dispatch trait is private. Typed tools use `Tool::NAME` as their sole identity; runtime-named agents convert explicitly with `Agent::into_tool()`.
  - Registration vocabulary: `AgentBuilder::tools(Vec<Box<dyn ToolDyn>>)` is removed; use repeated `.tool(...)` calls for typed tools or `dynamic_tools(Vec<DynamicTool>)` for runtime-defined callbacks. Retrieval-backed `dynamic_tools(sample, index, toolset)` becomes `retrieved_tools`. On `ToolSetBuilder`, `static_tool` remains the typed-tool path, the former embedding-backed `dynamic_tool(ToolEmbedding)` becomes `retrieved_tool`, and runtime-defined callbacks use `dynamic_tool(DynamicTool)`.
  - Results and errors: replace `ToolError`, `ToolFailure`, `ToolFailureKind`, `ToolReturn`, `ToolReturnOutcome`, `ToolExecutionResult`, and `ToolOutcome` with `ToolExecutionError`, `ToolErrorKind`, and the read-only `ToolResult` observed by hooks.
  - Model presentation: serializable outputs convert once into canonical `ToolOutput` content blocks; strings remain literal text, explicit `serde_json::Value` values remain JSON, and multimodal tools use `ToolOutput::content` / `ToolOutput::one` or return typed `ToolResultContent` directly. Result hooks now rewrite `ToolOutput`, provider adapters preserve native JSON where supported or render it only at their terminal wire boundary, mixed user/tool-result blocks retain order, and Rig never reparses strings to infer rich content. Consumers can inspect `ToolResultContent` with `as_text` / `as_json` and explicitly decode either structured JSON or legacy JSON-bearing text with `deserialize_json`.
  - Error presentation: explicit `ToolExecutionError` constructors keep actionable diagnostics model-visible, while the generic `ToolExecutionError::from_error` path preserves operator diagnostics and the concrete source but defaults to safe kind-level model feedback. Use `with_model_feedback` for deliberate replacement text or `with_model_output` for JSON/multimodal feedback. MCP responses preserve ordered supported text/image content, retain unsupported and future blocks as typed JSON, and attach raw `CallToolResult`, `structuredContent`, and response metadata to `ToolContext`. MCP list installation and refresh are atomic and ownership-aware, so stale handlers cannot replace or remove newer registrations, while disconnected owners are retired during refresh, provider exposure, or direct dispatch.
  - Dispatch: replace `ToolSet::{call, call_with_extensions, call_structured}` with `ToolSet::execute`; replace `ToolServerHandle::{call_tool, call_tool_with_extensions, call_tool_structured}` with `ToolServerHandle::execute`.
  - Registration and definitions: `ToolSet` is the single ordered registry and records whether each tool is always advertised or retrieval-only. `ToolSet::{get_tool_definitions, documents}` are now synchronous and infallible, `ToolServerHandle` registration/removal methods no longer return an artificial `Result`, and the obsolete `ToolSetError` is removed.
  - Hooks: replace `AgentHook::on_event`, `StepEvent`, and `Flow` with the event-specific `AgentHook` methods and their corresponding action types (`CompletionCallAction`, `ToolCallAction`, `ToolResultAction`, `InvalidToolCallAction`, and `ObservationAction`). Result rewrites replace the effective model and result-content telemetry presentation while preserving the raw `ToolResult` and `ToolContext` for policy; result stops omit result-content telemetry. Invalid-tool hooks return `None` to defer; every explicit action, including `Fail`, is terminal for that hook stack.
  - Streaming execution observation: the atomically surfaced post-batch event is named `ToolExecutionCommitted`, reflecting that it is not a real-time start notification. Applications that need live host lifecycle events should observe `on_tool_call` / `on_tool_result`; typed result metadata remains available through `ToolResultEvent::tool_context` without entering model-facing messages.
- *(core)* [**breaking**] Mark `PromptError`, `StructuredOutputError`, and `VectorStoreError` as non-exhaustive, requiring downstream match expressions to include a wildcard arm. Conversation memory load failures now surface as the typed `PromptError::MemoryError` variant instead of `CompletionError::RequestError`.

### Fixed

- *(openai)* Treat empty `encrypted_content` in non-streaming Responses API
  reasoning items as absent, matching streaming behavior and avoiding empty
  encrypted reasoning blocks.

## [0.40.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.39.0...rig-core-v0.40.0) - 2026-07-10

### Added

- *(core)* expand rig::prelude to cover the everyday API surface ([#2057](https://github.com/0xPlaygrounds/rig/pull/2057)) (by @gold-silver-copper)
- *(tool)* [**breaking**] structured tool-execution results ([#2015](https://github.com/0xPlaygrounds/rig/pull/2015)) (by @gold-silver-copper)
- *(agent)* [**breaking**] hook system v2 — composable middleware ([#2012](https://github.com/0xPlaygrounds/rig/pull/2012)) (by @gold-silver-copper)
- *(ollama)* Extend `think` options with `max` ([#1982](https://github.com/0xPlaygrounds/rig/pull/1982)) (by @m-dreiling)
- *(examples)* human-in-the-loop tool-call approval — examples + tests ([#1967](https://github.com/0xPlaygrounds/rig/pull/1967)) (by @gold-silver-copper)
- *(rig-core)* steer the model request per turn from a hook via Flow::OverrideRequest ([#1966](https://github.com/0xPlaygrounds/rig/pull/1966)) (by @gold-silver-copper)
- *(rig-core)* rewrite tool results from a hook via Flow::RewriteResult ([#1965](https://github.com/0xPlaygrounds/rig/pull/1965)) (by @gold-silver-copper)
- *(rig-core)* rewrite tool-call arguments from a hook via Flow::RewriteArgs ([#1963](https://github.com/0xPlaygrounds/rig/pull/1963)) (by @gold-silver-copper)
- *(rig-core)* concurrent tool execution in the streaming driver (parity with blocking tool_concurrency) ([#1957](https://github.com/0xPlaygrounds/rig/pull/1957)) (by @gold-silver-copper)
- *(rig-core)* ToolCallExtensions — per-call tool context through the agent loop, MCP & sub-agents (supersedes #1537, #1953) ([#1954](https://github.com/0xPlaygrounds/rig/pull/1954)) (by @gold-silver-copper)
- *(openai)* preserve responses prompt cache parameters ([#1830](https://github.com/0xPlaygrounds/rig/pull/1830)) (by @Kade-Powell)
- *(streaming)* [**breaking**] surface unmodeled provider output items through the stream ([#1951](https://github.com/0xPlaygrounds/rig/pull/1951)) (by @gold-silver-copper)
- *(openai-responses)* [**breaking**] preserve unknown Output payloads ([#1950](https://github.com/0xPlaygrounds/rig/pull/1950)) (by @gold-silver-copper)
- *(rig-core)* [**breaking**] integrate hooks into AgentRun via a composable AgentRunner ([#1945](https://github.com/0xPlaygrounds/rig/pull/1945)) (by @gold-silver-copper)
- *(rig-core)* [**breaking**] broaden provider error-response inspection workspace-wide ([#1944](https://github.com/0xPlaygrounds/rig/pull/1944)) (by @gold-silver-copper)
- *(message)* add video helper constructors + OpenRouter audio/video conversion tests ([#1942](https://github.com/0xPlaygrounds/rig/pull/1942)) (by @gold-silver-copper)
- *(rig-core)* [**breaking**] expose provider error response inspection ([#1859](https://github.com/0xPlaygrounds/rig/pull/1859)) (by @Shaurya-Sethi)
- *(agent)* add OutputMode to compose structured output with tools ([#1928](https://github.com/0xPlaygrounds/rig/pull/1928)) ([#1929](https://github.com/0xPlaygrounds/rig/pull/1929)) (by @gold-silver-copper)

### Fixed

- *(telemetry)* keep GenAI message span fields empty ([#2066](https://github.com/0xPlaygrounds/rig/pull/2066)) (by @gold-silver-copper)
- *(chatgpt)* preserve non-success response errors ([#2053](https://github.com/0xPlaygrounds/rig/pull/2053)) (by @gold-silver-copper)
- *(chatgpt)* fallback on empty SSE output ([#2001](https://github.com/0xPlaygrounds/rig/pull/2001)) (by @gold-silver-copper)
- *(openai)* preserve reasoning text content ([#1999](https://github.com/0xPlaygrounds/rig/pull/1999)) (by @gold-silver-copper)
- preserve OpenAI Responses instructions ([#1995](https://github.com/0xPlaygrounds/rig/pull/1995)) (by @gold-silver-copper) - #1995
- *(openai)* accept null Responses metadata ([#1993](https://github.com/0xPlaygrounds/rig/pull/1993)) (by @gold-silver-copper)
- *(gemini)* tolerate omitted proto defaults ([#1984](https://github.com/0xPlaygrounds/rig/pull/1984)) (by @gold-silver-copper)
- *(openai)* make Responses API strict tools opt-in ([#1991](https://github.com/0xPlaygrounds/rig/pull/1991)) (by @gold-silver-copper)
- *(ollama)* omit `think` when unset so the model default applies ([#1990](https://github.com/0xPlaygrounds/rig/pull/1990)) (by @SarthakB11)
- *(agent)* stream concurrent tool results as they complete ([#1981](https://github.com/0xPlaygrounds/rig/pull/1981)) (by @gold-silver-copper)
- *(anthropic)* deserialize explicit null citations as empty vec ([#1972](https://github.com/0xPlaygrounds/rig/pull/1972)) (by @CharmingGroot)
- *(anthropic)* coerce tool_use.input to an object at the send boundary ([#1964](https://github.com/0xPlaygrounds/rig/pull/1964)) (by @wey-gu)
- *(openai-compat)* normalize evicted tool-call string arguments to an object ([#1958](https://github.com/0xPlaygrounds/rig/pull/1958)) (by @wey-gu)
- *(rig-core)* fix epub loader tests + prevent CWD-relative fixture-path regressions ([#1940](https://github.com/0xPlaygrounds/rig/pull/1940)) (by @gold-silver-copper)
- *(gemini)* default totalTokenCount to avoid deser crash on empty generations ([#1936](https://github.com/0xPlaygrounds/rig/pull/1936)) (by @gold-silver-copper)
- *(ollama)* preserve assistant reasoning from non-streaming responses ([#1926](https://github.com/0xPlaygrounds/rig/pull/1926)) ([#1927](https://github.com/0xPlaygrounds/rig/pull/1927)) (by @gold-silver-copper)

### Other

- Remove unused derive and core APIs ([#2087](https://github.com/0xPlaygrounds/rig/pull/2087)) (by @gold-silver-copper) - #2087
- Remove unused stream completion stdout helper ([#2085](https://github.com/0xPlaygrounds/rig/pull/2085)) (by @gold-silver-copper) - #2085
- Remove unused generation wrapper traits ([#2083](https://github.com/0xPlaygrounds/rig/pull/2083)) (by @gold-silver-copper) - #2083
- Remove unused Anthropic decoders ([#2082](https://github.com/0xPlaygrounds/rig/pull/2082)) (by @gold-silver-copper) - #2082
- *(agent)* [**breaking**] unify PromptResponse and FinalResponse into one type ([#2056](https://github.com/0xPlaygrounds/rig/pull/2056)) (by @gold-silver-copper)
- *(core)* [**breaking**] API paper cuts — duplicate names, hand-copied setters, dead types ([#2055](https://github.com/0xPlaygrounds/rig/pull/2055)) (by @gold-silver-copper)
- *(openrouter)* use generic completion model ([#2054](https://github.com/0xPlaygrounds/rig/pull/2054)) (by @gold-silver-copper)
- *(anthropic)* cover ANTHROPIC_BASE_URL from_env ([#2051](https://github.com/0xPlaygrounds/rig/pull/2051)) (by @gold-silver-copper)
- *(auth)* add non-interactive oauth cassette coverage ([#2050](https://github.com/0xPlaygrounds/rig/pull/2050)) (by @gold-silver-copper)
- *(providers)* [**breaking**] remove galadriel provider ([#2041](https://github.com/0xPlaygrounds/rig/pull/2041)) (by @gold-silver-copper)
- *(providers)* [**breaking**] collapse remaining providers onto GenericCompletionModel<Ext> (#2035 phases 2–4) ([#2040](https://github.com/0xPlaygrounds/rig/pull/2040)) (by @gold-silver-copper)
- *(providers)* [**breaking**] migrate llamafile onto GenericCompletionModel<Ext> (#2035 phase 1) ([#2038](https://github.com/0xPlaygrounds/rig/pull/2038)) (by @gold-silver-copper)
- *(core)* [**breaking**] delete unused evals module and experimental feature flag ([#2036](https://github.com/0xPlaygrounds/rig/pull/2036)) (by @gold-silver-copper)
- Flatten Tool metadata API ([#2029](https://github.com/0xPlaygrounds/rig/pull/2029)) (by @gold-silver-copper) - #2029
- Add Groq agent tool cassette regressions ([#2011](https://github.com/0xPlaygrounds/rig/pull/2011)) (by @gold-silver-copper) - #2011
- Add Mistral agent tool cassette regressions ([#2010](https://github.com/0xPlaygrounds/rig/pull/2010)) (by @gold-silver-copper) - #2010
- Add DeepSeek agent tool cassette regressions ([#2009](https://github.com/0xPlaygrounds/rig/pull/2009)) (by @gold-silver-copper) - #2009
- Add xAI agent tool cassette regressions ([#2008](https://github.com/0xPlaygrounds/rig/pull/2008)) (by @gold-silver-copper) - #2008
- *(openai)* production-grade Responses API cassette suite + tool_choice and replay-ID fixes ([#2002](https://github.com/0xPlaygrounds/rig/pull/2002)) (by @gold-silver-copper)
- *(providers)* add provider implementation checklist ([#1997](https://github.com/0xPlaygrounds/rig/pull/1997)) (by @gold-silver-copper)
- Fix provider ClientBuilder API key aliases ([#1996](https://github.com/0xPlaygrounds/rig/pull/1996)) (by @gold-silver-copper) - #1996
- Release-N cleanup: correctness fixes + dead-code removal ([#1987](https://github.com/0xPlaygrounds/rig/pull/1987)) (by @gold-silver-copper) - #1987
- *(agent)* unify streaming/non-streaming seams; fix Anthropic streaming output_schema drop ([#1986](https://github.com/0xPlaygrounds/rig/pull/1986)) (by @gold-silver-copper)
- *(agent)* unify the streaming and non-streaming drivers over one engine ([#1985](https://github.com/0xPlaygrounds/rig/pull/1985)) (by @gold-silver-copper)
- *(openai-compat)* genuinely exercise the #1958 tool-call eviction string-leak (+ live cassette) ([#1962](https://github.com/0xPlaygrounds/rig/pull/1962)) (by @gold-silver-copper)
- *(rig-core)* [**breaking**] remove the experimental pipeline module ([#1941](https://github.com/0xPlaygrounds/rig/pull/1941)) (by @gold-silver-copper)
- *(rig-core)* replace nanoid with fastrand for internal IDs ([#1938](https://github.com/0xPlaygrounds/rig/pull/1938)) (by @gold-silver-copper)

### Contributors

* @gold-silver-copper
* @SarthakB11
* @m-dreiling
* @CharmingGroot
* @wey-gu
* @Kade-Powell
* @Shaurya-Sethi

### Changed

- *(agent)* [**breaking**] `max_turns` and `default_max_turns` now bound the exact total number of model calls, including the initial call, tool continuations, and retries. A budget of `0` makes no model call, while `1` permits only the initial call. Unconfigured tool-then-answer flows now need an explicit total budget of `2`. To preserve the former maximum allowance of an explicit old budget `n`, account for the old effective `n + 2` calls; otherwise, set the intended literal total.
- *(agent)* [**breaking**] unify the blocking and streaming agent-run result types into a single [`PromptResponse`](https://docs.rs/rig-core/latest/rig_core/agent/struct.PromptResponse.html) ([#2046](https://github.com/0xPlaygrounds/rig/issues/2046)). The streaming surface previously returned a separate `FinalResponse` carrying the same run result under different names; it is removed and the terminal `MultiTurnStreamItem::FinalResponse` item now carries `PromptResponse`. One vocabulary works on both surfaces — `output`/`output()`, `usage`/`usage()`, `messages`/`messages()`, `completion_calls`/`completion_calls()`, and `content`/`content()` — and blocking callers gain the structured final-turn `content` for free. Migration: `FinalResponse` → `PromptResponse`, `.response()` → `.output()`, `.history()` → `.messages()`, `.aggregated_usage`/`assistant_content()` are gone (`.usage()`/`.content()`). The streamed `FinalResponse` stream item now serializes its fields as `snake_case` (matching the blocking type and the sibling `CompletionCall`) rather than `camelCase`.
- *(core)* [**breaking**] API paper-cut cleanups — collapse duplicate public names onto one canonical spelling per concept ([#2047](https://github.com/0xPlaygrounds/rig/issues/2047)):
  - `StreamingPromptRequest::multi_turn` is removed; use `max_turns`, matching the blocking `PromptRequest`/`TypedPromptRequest` builders.
  - The earlier low-level streaming stdout helper rename is superseded by the removal noted below; the high-level `agent::stream_to_stdout` helper (which drives a `StreamingResult`) remains unchanged.
  - The built-in `ThinkTool` moves from `rig::tools` to `rig::tool::builtin`; the one-line `tools` module is removed.
  - `StreamingPromptRequest`'s forwarding setters (`tool_extensions`, `history`, `conversation`, `without_memory`, `max_invalid_tool_call_retries`) are now generated by the shared `forward_prompt_setters!` macro instead of hand-copied, removing a drift risk (no API change).
  - The materially identical `AuthError` enums under `providers/chatgpt/auth` and `providers/copilot/auth` are unified into a single shared type (re-exported from each provider's `auth` module; no API change).
  - Dead code removed: the never-constructed Gemini embedding request structs (`EmbedContentRequest` and friends; the live response types are kept), commented-out lines in the OpenAI Responses API, and a placeholder tool-result error string replaced with a descriptive message (no API change).
- *(tool)* [**breaking**] flatten `Tool` and `ToolDyn`: tool implementations now provide `description()` and `parameters()` directly, while provider-facing `ToolDefinition`s are generated at registration/request boundaries. `Tool::definition(prompt)` and `ToolDyn::definition(prompt)` are removed, and the registered `Tool::NAME` / `Tool::name()` / `ToolDyn::name()` is the only advertised dispatch identity.

### Removed

- *(core)* [**breaking**] remove unused `Extractor::{get_inner, into_inner}` and the always-failing `TryFrom<String> for Nothing`; no direct replacements are provided.
- *(core)* [**breaking**] remove the unused public `streaming::stream_completion_to_stdout` helper; use the high-level `agent::stream_to_stdout` helper instead.
- *(core)* [**breaking**] remove the unused public `AudioGeneration<M>`, `ImageGeneration<M>`, and `Transcription<M>` wrapper traits; use the corresponding `AudioGenerationModel`, `ImageGenerationModel`, and `TranscriptionModel` APIs and request builders directly.
- *(anthropic)* [**breaking**] remove the unused public `providers::anthropic::decoders` module; Anthropic streaming uses the shared SSE machinery.

### Added

- *(doubleword)* new OpenAI-compatible provider for the [Doubleword](https://docs.doubleword.ai) inference API (`providers::doubleword`), covering realtime chat completions (with streaming) and embeddings (`Qwen/Qwen3-Embedding-8B`). Configure via `DOUBLEWORD_API_KEY` (and optional `DOUBLEWORD_BASE_URL`).
- *(telemetry)* [**breaking**] add opt-in sensitive-content telemetry for GenAI spans. `AgentBuilder::record_content_telemetry(bool)`, `PromptRequest::record_content_telemetry(bool)`, `TypedPromptRequest::record_content_telemetry(bool)`, `StreamingPromptRequest::record_content_telemetry(bool)`, and `AgentRunner::record_content_telemetry(bool)` consistently enable semantic-convention-shaped system instructions, prompts, model input/output messages, completions, and tool arguments/results. Content is disabled by default while structural metadata and token usage remain available. The low-level `CompletionRequestBuilder::record_content_telemetry(bool)` forwards the same local policy, but direct-model content fields remain provider- and surface-dependent; the policy is never serialized into provider request payloads.
- *(agent)* [**breaking**] hook system v2 — composable middleware for the agent run loop. Builds on the unified `AgentHook` below and makes hooks compose the way production middleware needs:
  - **Run-scoped `HookContext`.** `on_event` now takes `(&HookContext, StepEvent)` — the trait signature changed, so every `AgentHook` impl gains a context parameter. `HookContext` carries the run's identity and state: `run_id()`, `turn()`, `is_streaming()`, `agent_name()`, and a shared `Scratchpad` (an interior-mutable type-map: `insert`/`get`/`update`) so cooperating hooks share per-run state without rolling their own `Arc<Mutex<…>>`.
  - **Mergeable request patches.** `Flow::OverrideRequest`/`RequestOverride` become `Flow::PatchRequest`/`RequestPatch` (constructor `Flow::patch_request(..)`). On `StepEvent::CompletionCall`, patches from **every** hook now accumulate and merge in registration order instead of the first patch short-circuiting the rest — so a RAG hook, a tool-policy hook, and a provider-param hook all steer the same turn. Per-field merge rules are documented on `RequestPatch`: `extra_context` appends, `additional_params` shallow-merges (later wins), `active_tools` **intersects** (two narrowing guardrails compose), and scalars/`preamble`/`history` are last-writer-wins with a `tracing::warn!` on conflict. Patches remain per-turn and non-sticky.
  - **`RequestPatch::extra_context`.** Hooks can inject `Vec<Document>` context for a single model call (passive RAG), appended after the agent's static context in hook registration order. This includes documents produced by the hook-backed `dynamic_context` helper. Per-turn and non-sticky; works identically on `run()` and `stream()`.
  - **`RequestPatch::history`.** A per-turn replacement for the prior messages sent to the provider (context-window compaction / summarization). The persisted transcript is untouched and RAG query text still derives from the original history — only what is sent changes.
  - **Chained tool rewrites.** `Flow::RewriteArgs` / `Flow::RewriteResult` now compose across a `HookStack`: the rewritten value is threaded into the next hook's `ToolCall`/`ToolResult` event, so a redaction hook and a truncation hook stack (previously only the first rewrite took effect). The first result hook still observes the tool's actual output.
  - **`StepEvent::ModelTurnFinished { turn, content, usage }`.** A normalized per-turn event that fires exactly once per accepted model turn on **both** surfaces — including a streamed tool-only turn that fires no `StreamResponseFinish`. Observe-only; the medium-specific `CompletionResponse`/`StreamResponseFinish` events retain their lifecycle timing while exposing canonical Rig prompt, content, usage, and message-ID fields.
- *(agent)* [**breaking**] tool execution stream events split model-emitted from execution-lifecycle, and concurrent tool batches commit/surface **atomically**:
  - **Model tool call vs. execution start.** `MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall)` now reports the tool call the **model emitted** (surfaced when the model turn is committed, whether or not Rig executes it). A new `MultiTurnStreamItem::ToolExecutionStart { tool_call, internal_call_id }` marks that Rig has **started executing** a tool — emitted only after the tool passed its `ToolCall` hook checks and its body actually runs (never for a hook-skipped call, an invalid-recovery call, or a dropped sibling). Correlate the two, and the resulting `ToolResult`, via `internal_call_id`.
  - **Atomic per-batch commit/surface.** A turn's tool calls are collected, not streamed one-by-one: successful `ToolExecutionStart` + `ToolResult` items are surfaced (in call order) and committed to history **only after the whole batch settles successfully**. On the first hook termination / fail-closed error the batch fails fast — no new tool starts, not-yet-started concurrent siblings are dropped, already-started ones are drained, the deterministic lowest call-index error is surfaced, and **no** successful `ToolExecutionStart`/`ToolResult` is surfaced or committed (no orphan execution-start events, no partial history). `run()` and `stream()` return the same terminal reason. Previously the concurrent path streamed each result as its tool completed (in completion order) — results now surface in call order after the batch settles.
- *(agent)* [**breaking**] local validation of the effective tool set + `tool_choice` before the provider call. After per-turn request patches and `active_tools` filtering, `ToolChoice::Required` with no advertised tool (no executable tool and no synthetic output tool) and `ToolChoice::Specific` naming a tool not in the effective advertised set (executable tools + output tool) are **local request errors** with no provider round-trip. When a per-turn `active_tools` allow-list caused the incompatibility, the error says so and suggests setting a compatible `tool_choice` in the same `RequestPatch`. Structured-output Tool mode with no real tools still works when the synthetic output tool satisfies the choice.
- *(agent)* [**breaking**] unified, composable hook system integrated into the agent run loop. The hook trait (model-generic at the time) replaces the 8-method `PromptHook`. Hooks compose via `HookStack` and run in registration order (see the hook system v2 entry above for the per-event composition rules: `CompletionCall` patches accumulate, `ToolCall`/`ToolResult` rewrites chain, and observe-only/recovery events use first-non-`Continue`-wins). A new `AgentRunner<M>` driver — obtained with `agent.runner(prompt)` and run via `.run().await` (blocking) or `.stream()` (incremental) — pairs the sans-IO `AgentRun` state machine with hooks, model IO, tool execution and memory; both `agent.prompt(..)` and `agent.stream_prompt(..)` are now thin wrappers over it. Attach/compose hooks with `.add_hook(h)` (append-only; call it again to stack more) on the agent builder, the prompt request, or the runner. This closes the gap where hand-driving `AgentRun` had no hook support.
  - **Streaming ≡ non-streaming.** `run()` and `stream()` share one drive loop, run construction, and tool execution, so they produce the same message history, tool-result content, and recovery — only the inherently-streaming `TextDelta`/`ToolCallDelta`/`StreamResponseFinish` events differ. The **medium-independent** hook event sequence (model call, tool call/result, invalid-tool-call) is identical at the default tool concurrency of 1; the medium-specific `StreamResponseFinish` additionally fires only on turns that stream assistant text, so it is not a one-to-one match for the blocking `CompletionResponse`. See the behavioral note below for the observable changes when upgrading.
  - **Fail-closed `Flow`.** Each `StepEvent` honors a documented subset of `Flow` actions; an action an event cannot honor (e.g. `Flow::Fail` on a tool call) terminates the run rather than silently proceeding, so a blocking hook can never fail open.
  - **Event interest.** `AgentHook::observes(StepEventKind)` (default: all) lets a hook opt out of the high-frequency streaming delta events; the runner then skips building and dispatching them, so an empty stack — and hooks that only watch tool calls — pay nothing per delta.
- *(agent)* tool-call argument rewriting from a hook: a new `Flow::RewriteArgs { args }` action — constructor `Flow::rewrite_args(impl Into<serde_json::Value>)`, plus the typed `Flow::try_rewrite_args(&T)` — lets an `AgentHook` rewrite a `StepEvent::ToolCall`'s arguments before the tool runs, for guardrails that normalize, clamp, redirect, or inject scoped parameters. The rewritten arguments are what the tool executes against, what the following `StepEvent::ToolResult` reports, and what the `gen_ai.tool.call.arguments` span records; the model's assistant message is left unchanged, so this is an execution-args rewrite, not a transcript redactor. It is honored only for `ToolCall` (fail-closed on every other event) and wired through the shared `run_single_tool`, so it behaves identically on the blocking and streaming drivers. `Flow` is `#[non_exhaustive]`, so the addition is non-breaking. ([#1963](https://github.com/0xPlaygrounds/rig/pull/1963), closes [#1744](https://github.com/0xPlaygrounds/rig/issues/1744))
- *(agent)* tool-result rewriting from a hook: a new `Flow::RewriteResult { result }` action (constructor `Flow::rewrite_result(impl Into<String>)`) lets an `AgentHook` replace a tool's output on the `StepEvent::ToolResult` event before the model sees it — the post-execution counterpart of `Flow::RewriteArgs`, completing a symmetric `Rewrite{Args,Result}` family, for guardrails that redact, truncate, or normalize tool output. The replacement is what the model receives and what the `gen_ai.tool.call.result` span records; the `ToolResult` event still observes the tool's actual output (the rewrite is applied after it fires). Honored only for `ToolResult` (fail-closed on every other event) and wired through the shared `run_single_tool`, so it behaves identically on the blocking and streaming drivers. `Flow` is `#[non_exhaustive]`, so the addition is non-breaking. ([#1965](https://github.com/0xPlaygrounds/rig/pull/1965))
- *(agent)* per-turn model-request steering from a hook: a new `Flow::PatchRequest { patch: RequestPatch }` action (constructor `Flow::patch_request(..)`) lets an `AgentHook` patch the model request on the `StepEvent::CompletionCall` event before it is sent — adjusting the system prompt, sampling (`temperature`/`max_tokens`), `tool_choice`, the advertised tool set (a by-name `active_tools` allow-list), and provider `additional_params` from run state (e.g. force a tool on the first turn, lower the temperature on a critical step, or shrink the tool set after a phase). `RequestPatch` is a partial patch built with setters: a set field replaces the agent's configured value (`additional_params` is shallow-merged, the override winning), an unset field inherits it, and the patch applies to *that turn only* — it is non-sticky and never mutates the agent baseline. Honored only for `CompletionCall` (fail-closed on every other event) and applied in the shared request builder, so it behaves identically on the blocking and streaming drivers. The variant is additive (`Flow` is `#[non_exhaustive]`), but because the override carries an `f64` temperature, `Flow` is now `PartialEq` and no longer `Eq`. ([#1966](https://github.com/0xPlaygrounds/rig/pull/1966))
- *(providers)* broaden provider error-response inspection (`provider_response_body` / `provider_response_json` / `provider_response_status`) to all in-core providers (Anthropic, Gemini, Cohere, xAI, Hyperbolic, Ollama, Mira, VoyageAI, Mistral, Hugging Face, OpenRouter, OpenAI audio, …) and the gRPC/SDK companion crates (`rig-bedrock`, `rig-vertexai`, `rig-gemini-grpc`). Adds the shared `from_http_response(status, body)` and `from_provider_body(body)` constructors on every capability error so HTTP failures are no longer flattened into `ProviderError(String)` ([#1944](https://github.com/0xPlaygrounds/rig/pull/1944), closes [#1931](https://github.com/0xPlaygrounds/rig/issues/1931))
- *(tool)* per-call tool extensions: `ToolCallExtensions`, a type-erased, cloneable type-map (`TypeId` → value; a port of `http::Extensions` including the no-op `IdHasher`; zero-allocation when empty) that lets callers attach runtime values to a tool call — auth tokens, session IDs, A2A `context_id`/`task_id`, conversation state — without exposing them to the model. Tools opt in by overriding `Tool::call_with_extensions(args, &extensions)` (the default delegates to `call`, so existing `Tool`/`ToolDyn` impls are unchanged); read values with `extensions.get::<T>()` or `extensions.require::<T>()`. Attach per-run via `agent.prompt(..).tool_extensions(..)` / `agent.stream_prompt(..).tool_extensions(..)` (also on `TypedPromptRequest` and `AgentRunner`), threaded through the run loop into the single `run_single_tool` dispatch site for both the blocking and streaming drivers. The dispatch chain gains a parallel `call_with_extensions` on `ToolDyn` / `ToolType` / `ToolSet` / `ToolServerHandle`. Sub-agents (an `Agent` used as a tool) propagate the extensions into the inner run. MCP tools forward an `rmcp::model::Meta` placed in the extensions as the request `_meta` (SEP-1319) — per-call auth/session for MCP servers (`Meta` re-exported at `rig::tool::rmcp::Meta`). ([#1954](https://github.com/0xPlaygrounds/rig/pull/1954), closes [#1536](https://github.com/0xPlaygrounds/rig/issues/1536))
- *(tool)* [**breaking**] structured tool-execution results. A tool call now resolves to a `ToolExecutionResult { model_output, outcome, extensions }` that flows all the way to the `StepEvent::ToolResult` hook event, so hooks, tracing, telemetry, and policies can reason about *what happened* without parsing the model-visible string. The `outcome` is a `ToolOutcome` — `Success`, `Error(ToolFailure)`, `Skipped`, or `Denied` — where `ToolFailure { kind, message, retryable, code, http_status }` classifies the failure via a standard `ToolFailureKind` (`InvalidArgs`, `Timeout`, `Cancelled`, `NotFound`, `PermissionDenied`, `RateLimited`, `Provider`, `Network`, `Other`). Motivating use case: a hook can `outcome.is_error_kind(ToolFailureKind::Timeout)`, count timeouts in the run `Scratchpad`, and `Flow::terminate` after a threshold, while a `NotFound` falls through as recoverable model-visible feedback — all without string parsing.
  - **Authoring.** A tool classifies its own error type with `Tool::classify_error(&Self::Error) -> ToolFailure` (default: `Other`), and can return richer results from `Tool::call_structured(args, &extensions) -> Result<ToolReturn<Output>, Error>` (default: wraps `call` as `ToolReturn::success`). `ToolReturn<T>` attaches an outcome and/or `ToolResultExtensions` — type-erased, never-sent-to-the-model metadata (provider ids, raw headers, retry hints) — to its output; a plain `T: Serialize` output stays as ergonomic as before. A tool's declared outcome is a `ToolReturnOutcome` — `Success`, `Error(ToolFailure)`, or `Denied` — a strict subset of the observed `ToolOutcome` with **no `Skipped` variant**: `Skipped` is a framework-only outcome (a `ToolCall` hook returning `Flow::Skip`), so it is impossible by construction for a tool to return — or build a `ToolExecutionResult` claiming — a skipped outcome while having actually run. Tools express refusal with `denied`. (`ToolReturnOutcome` converts into `ToolOutcome` via `From`; `ToolExecutionResult`'s fields are read via `model_output()` / `outcome()` / `extensions()`.)
  - **Dispatch boundary.** `ToolDyn` gains `call_structured` (with a default that wraps `call`), threaded through `ToolType`, `ToolSet::call_structured`, and `ToolServerHandle::call_tool_structured`; the agent loop drives this structured path in the shared `run_single_tool`, so blocking and streaming observe identical outcomes. MCP tools classify a per-call timeout as `Timeout`, a transport error as `Provider`, and a tool-reported error as `Other` (`McpToolError` now carries a `ToolFailureKind`).
  - **Hooks.** `StepEvent::ToolResult` gains `outcome: &ToolOutcome` and `extensions: &ToolResultExtensions` alongside `result`. A `Flow::RewriteResult` still rewrites only the model-visible `result`; the raw `outcome`/`extensions` are unaffected, so a redaction hook cannot mask the true outcome from a later policy hook. **Behavioral changes:** `Flow::Skip` now fires `StepEvent::ToolResult` with a `Skipped` outcome (previously a skipped tool fired no result hook), so result hooks and denial-logging policies observe skips; and a tool error's model-visible text is now the tool's own error `Display` (formatted at the boundary) rather than the previous triple-wrapped `Toolset error: ToolCallError: ToolCallError: …` string. Because fields were added to the `#[non_exhaustive]` `ToolResult` variant, an exhaustive destructure must add `..` (or the new fields).
- *(agent)* concurrent tool execution on the **streaming** driver, gated behind the existing `tool_concurrency(n)` knob (default `1` = unchanged), bringing it to parity with the blocking path. `tool_concurrency(n)` is now exposed on `StreamingPromptRequest` (alongside `PromptRequest`/`AgentRunner`). A turn with several independent slow tools (HTTP/MCP) now finishes in ≈`max` rather than ≈`sum` under streaming. A turn's `ToolCall` items are emitted in call order, then — once the whole tool batch settles — the per-tool `ToolExecutionStart` + `ToolResult` items are surfaced in call order; persisted message history is deterministic in call order and matches the blocking driver (see the atomic per-batch commit/surface note above). ([#1955](https://github.com/0xPlaygrounds/rig/issues/1955), closes [#1872](https://github.com/0xPlaygrounds/rig/issues/1872))
- *(core)* `rig::prelude` now re-exports the everyday API surface — not just the provider-client traits — so a basic agent or RAG program compiles with `use rig::prelude::*` plus its provider module and nothing else. Added (all additive re-exports, non-breaking): `Agent`; the completion traits/types `Prompt`, `Chat`, `Completion`, `CompletionModel`, `Message`, `PromptError`, `CompletionError`; the streaming traits `StreamingPrompt`/`StreamingChat` and stream items `MultiTurnStreamItem`/`StreamingResult`; the embedding surface `Embed`, `EmbeddingModel`, `EmbeddingsBuilder`; the tool types `Tool`/`ToolSet`; the vector-store surface `VectorStoreIndex`, `InMemoryVectorStore`, `VectorSearchRequest`; and `OneOrMany`. Deliberately scoped to the common path — advanced surfaces (the hook system, run-loop stepping types, message content blocks, tool-authoring internals, extraction/loaders/memory) stay explicit imports from their modules. The `agent`, `vector_search`, and `agent_stream_chat` examples are updated to demonstrate it. ([#2044](https://github.com/0xPlaygrounds/rig/issues/2044))

### Changed

- *(openai)* [**breaking**] the Responses API conversion now sends Rig system instructions (the preamble and any leading system messages) through the official top-level `instructions` field instead of as `system` messages in `input`. Mid-conversation system messages keep their position in `input`; ChatGPT (whose backend rejects the `system` role entirely) lifts all of them. **Behavioral note for OpenAI-compatible endpoints:** a backend that ignores or rejects top-level `instructions` (some vLLM / mistral.rs / LM Studio setups) will silently lose the system prompt under the new default — call `with_system_instructions_as_messages()` on the `openai::Client` (applies to every model, agent, and extractor created from it) or on a `ResponsesCompletionModel` to restore the previous request shape. Placement is expressed by the new public `responses_api::SystemInstructionsPlacement` enum — selectable via `with_system_instructions_placement(..)` on the `openai::Client` and on `ResponsesCompletionModel` (including `AllInstructions` for backends that reject the `system` role), with `with_system_instructions_as_messages()` kept as a shorthand for the compatibility fallback — and a client-level default flows through the new `responses_api::ResponsesProviderExt` trait implemented by the client's `Ext` type (`system_instructions_placement` is a required method, so implementors must state their placement explicitly). A configured placement survives `completions_api()`/`responses_api()` round trips. Direct request conversion with a non-default placement uses the public `responses_api::ResponsesRequestParams` (a `TryFrom` source for the Responses `CompletionRequest`, mirroring the Chat Completions `OpenAIRequestParams` pattern). Also [**breaking**]: `OpenAIResponsesExt` and `OpenAICompletionsExt` are no longer unit structs — construct them with `::default()` instead of the struct literal. ([#1995](https://github.com/0xPlaygrounds/rig/pull/1995), closes [#1599](https://github.com/0xPlaygrounds/rig/issues/1599))
- *(anthropic)* the streaming completion path now honors the request's `tool_choice` instead of hardcoding `auto`, so a caller's tool choice (including one set per-turn via `Flow::PatchRequest`) takes effect under `stream_prompt`/streaming as it already did on the blocking path. A `tool_choice` Anthropic cannot represent (a multi-name `ToolChoice::Specific`) now surfaces as a request error on both the streaming and blocking paths instead of being silently downgraded to `auto`. ([#1966](https://github.com/0xPlaygrounds/rig/pull/1966))
- *(agent)* [**breaking**] `Flow` no longer implements `Eq` (it remains `PartialEq`), because `Flow::PatchRequest` carries a `RequestPatch` with an `f64` `temperature`. Code that relied on `Flow: Eq` (e.g. an `Eq` derive on a type embedding `Flow`, or an `Eq`/`Hash` bound) must drop that requirement. ([#1966](https://github.com/0xPlaygrounds/rig/pull/1966))
- *(agent)* [**breaking**] removed `PromptHook`, `HookAction` and `ToolCallHookAction`. Their capabilities are folded into `AgentHook` / `StepEvent` / `Flow`. `Agent`, `AgentBuilder`, `PromptRequest`, `TypedPromptRequest` and `StreamingPromptRequest` no longer carry a hook type parameter `P` (hooks moved to a runtime stack, which was model-parameterized at the time), and `AgentBuilder::hook(..)` becomes `add_hook(..)` (append-only; there is no stack-replacing setter). Per-request hooks likewise move from `PromptRequest::with_hook(..)` / `StreamingPromptRequest::with_hook(..)` — which *replaced* the agent's hook — to `add_hook(..)`, which *appends*; a mechanical `with_hook` → `add_hook` rename therefore now also runs any agent-default hooks the old override call would have dropped. `InvalidToolCallContext` and `InvalidToolCallHookAction` (used when hand-driving `AgentRun::resolve_invalid_tool_call`) are retained. The `agent::hook` module docs include a method-by-method `PromptHook` → `AgentHook` migration table (each old method → its `StepEvent` variant and the `Flow` to return).
- *(agent)* [**breaking**] aligned builder/runner method names for consistency: `AgentBuilder::conversation_id(..)` → `conversation(..)` (matching `PromptRequest`/`AgentRunner::conversation`), and the runner/request builders drop the lone `with_` prefix — `with_history(..)` → `history(..)` and `with_tool_concurrency(..)` → `tool_concurrency(..)`. The lower-level `AgentRun::with_history(..)` (part of its own `with_`-prefixed builder family) is unchanged.
- *(agent)* [**behavioral**] observable changes from the unified runner (the default `tool_concurrency` of 1 is unaffected by the ordering change):
  - Persisted tool results now land in **tool-call order** under `tool_concurrency(>1)`: the blocking path switched from `buffer_unordered` to `buffered`, so history/memory order is deterministic. The streaming path collects a turn's tool outcomes and surfaces the `ToolExecutionStart`/`ToolResult` stream items in tool-call order once the batch settles, persisting history in tool-call order too — matching the blocking driver (see the atomic per-batch commit/surface note above).
  - Synthetic tool results — hook/invalid-tool **skip** reasons and recovery feedback — are emitted **verbatim** as text in both drivers (and in streamed `StreamUserItem`s), no longer re-parsed through `from_tool_output`; a JSON-shaped reason is no longer reinterpreted as a structured/multimodal result.
  - Streaming `FinalResponse::history()` is now always `Some(..)` (parity with `run()`); it was `None` when no input history/memory was supplied, so a caller that branched on `None` to detect "no history" will observe a change.
  - Streaming `StreamResponseFinish` is now suppressed on invalid-tool-call **repaired** turns (parity with the blocking `CompletionResponse`); pre-PR it fired on the repaired turn.
- *(rerank)* [**breaking**] `RerankError` is now `#[non_exhaustive]` and gains a `ProviderResponse` variant, so rerank failures preserve the provider's raw status + body for inspection (parity with the other capability errors)
- *(streaming)* the OpenAI-compatible SSE stream now treats a present, non-empty `error` field (object or string) as a terminal provider error and ignores `{"error":null}` / empty values.
- *(streaming)* terminal streaming failures preserve the provider's error payload as `ProviderResponse` when present, otherwise surface `ProviderError` (so `provider_response_body()` may be `None`).
- *(providers)* [**behavioral**] migrated provider HTTP-error paths now yield `ProviderResponse` / `HttpError` instead of `ProviderError(String)` / `ResponseError(String)`. The error variant — and the `Display` / `to_string()` output for those failures — changed accordingly (e.g. `"ProviderError: …"` → `"HttpError: …"`). Exhaustive matches keep compiling (`#[non_exhaustive]`), but downstream code that matches specific variants or string-greps error messages will observe different runtime behavior.
- *(openai)* [**breaking**] OpenAI Responses `Output::Unknown` now carries the raw `serde_json::Value` of the unrecognized output item instead of being a fieldless unit variant, so provider-native hosted-tool items (`web_search_call`, `file_search_call`, `computer_call`, `code_interpreter_call`) survive the typed decode and are reachable on `CompletionResponse.output` instead of being discarded. Downstream exhaustive matches need the one-token `Output::Unknown(_)` update. ([#1950](https://github.com/0xPlaygrounds/rig/pull/1950), closes [#1861](https://github.com/0xPlaygrounds/rig/issues/1861))
- *(streaming)* [**breaking**] surface unmodeled provider output items through the stream: new `RawStreamingChoice::Unknown(serde_json::Value)` and public `StreamedAssistantContent::Unknown(serde_json::Value)` variants carry the raw item (e.g. OpenAI Responses hosted-tool results like `web_search_call`) to stream consumers instead of dropping it. The OpenAI Responses and Copilot streaming paths now emit it; it is forwarded to the consumer but not folded into the accumulated assistant message/history (there is no `AssistantContent::Unknown`). Adding the variants is breaking for exhaustive matches on these non-exhaustive enums.

## [0.39.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.38.2...rig-core-v0.39.0) - 2026-06-19

### Added

- *(providers)* add VoyageAI rerank support ([#1917](https://github.com/0xPlaygrounds/rig/pull/1917)) (by @sergiomeneses)
- *(agent)* [**breaking**] sans-IO AgentRun state machine; both agent loops become thin drivers ([#1899](https://github.com/0xPlaygrounds/rig/pull/1899)) (by @gold-silver-copper)

### Fixed

- *(rmcp)* bound MCP tool calls with a default, configurable, wasm-friendly timeout ([#1914](https://github.com/0xPlaygrounds/rig/pull/1914)) ([#1921](https://github.com/0xPlaygrounds/rig/pull/1921)) (by @gold-silver-copper)
- *(tool)* [**breaking**] deterministic, duplicate-safe tool registration + cassette tests ([#1913](https://github.com/0xPlaygrounds/rig/pull/1913)) (by @gold-silver-copper)

### Other

- Only append a slash to base_urls of api providers when they don't already end with a slash. ([#1903](https://github.com/0xPlaygrounds/rig/pull/1903)) (by @eriktews) - #1903
- *(tool)* back ToolSet with an IndexMap instead of HashMap + order Vec ([#1916](https://github.com/0xPlaygrounds/rig/pull/1916)) (by @gold-silver-copper)
- de-flake tracing span tests and deepseek permission_control race ([#1915](https://github.com/0xPlaygrounds/rig/pull/1915)) (by @gold-silver-copper) - #1915
- Fix streaming reasoning history order ([#1898](https://github.com/0xPlaygrounds/rig/pull/1898)) (by @gold-silver-copper) - #1898
- Fix context document ordering ([#1893](https://github.com/0xPlaygrounds/rig/pull/1893)) (by @gold-silver-copper) - #1893
- Add Gemini Nano Banana image generation ([#1889](https://github.com/0xPlaygrounds/rig/pull/1889)) (by @gold-silver-copper) - #1889

### Contributors

* @gold-silver-copper
* @eriktews
* @sergiomeneses
## [0.38.2](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.38.1...rig-core-v0.38.2) - 2026-06-09

### Fixed

- *(streaming)* record per-call token usage on chat generation spans ([#1880](https://github.com/0xPlaygrounds/rig/pull/1880)) (by @mateobelanger)
- support Anthropic mid-conversation system role ([#1862](https://github.com/0xPlaygrounds/rig/pull/1862)) (by @fangkangmi) - #1862
- *(openai)* make token usage details optional in responses API ([#1857](https://github.com/0xPlaygrounds/rig/pull/1857)) (by @sosal123tyu1)

### Other

- Add configurable Copilot intent ([#1883](https://github.com/0xPlaygrounds/rig/pull/1883)) (by @gold-silver-copper) - #1883
- [codex] support mistral.rs OpenAI-compatible reasoning ([#1864](https://github.com/0xPlaygrounds/rig/pull/1864)) (by @gold-silver-copper) - #1864
- [codex] cover Anthropic streaming tool result batching ([#1863](https://github.com/0xPlaygrounds/rig/pull/1863)) (by @gold-silver-copper) - #1863

### Contributors

* @gold-silver-copper
* @mateobelanger
* @fangkangmi
* @sosal123tyu1
## [0.38.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.38.0...rig-core-v0.38.1) - 2026-06-02

### Other

- unify workspace crate versions ([#1853](https://github.com/0xPlaygrounds/rig/pull/1853)) (by @gold-silver-copper) - #1853

### Contributors

* @gold-silver-copper
## [0.38.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.37.0...rig-core-v0.38.0) - 2026-06-02

### Added

- *(rig-derive)* replace hand-rolled schema with schemars in #[rig_tool] ([#1576](https://github.com/0xPlaygrounds/rig/pull/1576)) (by @tomasz-feliksik)
- *(embeddings)* expose token usage via embed_texts_with_usage ([#1791](https://github.com/0xPlaygrounds/rig/pull/1791)) (by @sergiomeneses)
- *(openrouter)* add prompt-caching support ([#1832](https://github.com/0xPlaygrounds/rig/pull/1832)) (by @gold-silver-copper)
- *(openrouter)* add with_app_identity and with_app_categories builders for app attribution ([#1806](https://github.com/0xPlaygrounds/rig/pull/1806)) (by @jimmiebfulton)
- *(openrouter)* surface cache token accounting in Usage ([#1808](https://github.com/0xPlaygrounds/rig/pull/1808)) (by @jimmiebfulton)
- *(gemini)* expose streaming response metadata ([#1790](https://github.com/0xPlaygrounds/rig/pull/1790)) (by @mateobelanger)
- *(anthropic)* support document citations ([#1778](https://github.com/0xPlaygrounds/rig/pull/1778)) (by @temrjan)
- *(gemini)* expose finish_reason and model_version on StreamingCompletionResponse ([#1776](https://github.com/0xPlaygrounds/rig/pull/1776)) (by @mateobelanger)

### Fixed

- *(openai)* tolerate object-form tool-call `arguments` in streaming ([#1822](https://github.com/0xPlaygrounds/rig/pull/1822)) (by @xavierforge)
- *(chatgpt)* Handle ChatGPT response.completed events without output field ([#1825](https://github.com/0xPlaygrounds/rig/pull/1825)) (by @geraschenko)
- *(rig-core)* Expose tools added via ToolServerHandle::append_toolset ([#1837](https://github.com/0xPlaygrounds/rig/pull/1837)) (by @mccormickt)
- avoid duplicate streaming reasoning history ([#1849](https://github.com/0xPlaygrounds/rig/pull/1849)) (by @gold-silver-copper) - #1849
- *(rig-gemini-grpc)* populate FunctionDeclaration.parameters from ToolDefinition ([#1763](https://github.com/0xPlaygrounds/rig/pull/1763)) (by @abhicris)
- *(openrouter)* avoid replaying generated images ([#1835](https://github.com/0xPlaygrounds/rig/pull/1835)) (by @gold-silver-copper)
- *(openrouter)* accept Gemini model role responses ([#1800](https://github.com/0xPlaygrounds/rig/pull/1800)) (by @puneetdixit200)
- *(tools)* safely normalize null tool call arguments ([#1814](https://github.com/0xPlaygrounds/rig/pull/1814)) (by @gold-silver-copper)
- *(ollama)* buffer NDJSON streaming across HTTP chunk boundaries bytes_stream may split a single NDJSON line across chunks, causing serde_json::from_slice to fail mid-stream with an EOF error on longer assistant messages ([#1759](https://github.com/0xPlaygrounds/rig/pull/1759)) (by @ChadBartley)
- *(gemini)* record tool use prompt token telemetry ([#1799](https://github.com/0xPlaygrounds/rig/pull/1799)) (by @gold-silver-copper)
- default OpenAI base64 image detail ([#1781](https://github.com/0xPlaygrounds/rig/pull/1781)) (by @fangkangmi) - #1781
- stream ToolCallDelta in prompt_request ([#1789](https://github.com/0xPlaygrounds/rig/pull/1789)) (by @notV4l) - #1789
- fix sqlite threshold and null tool call streaming ([#1786](https://github.com/0xPlaygrounds/rig/pull/1786)) (by @gold-silver-copper) - #1786
- *(anthropic)* serialize ToolResultContent::Image with source wrapper ([#1772](https://github.com/0xPlaygrounds/rig/pull/1772)) (by @Cyanistic)

### Other

- Fix parsing of streamed function-call argument deltas ([#1828](https://github.com/0xPlaygrounds/rig/pull/1828)) (by @geraschenko) - #1828
- Add invalid tool call recovery hooks ([#1840](https://github.com/0xPlaygrounds/rig/pull/1840)) (by @gold-silver-copper) - #1840
- [codex] Validate model tool calls ([#1823](https://github.com/0xPlaygrounds/rig/pull/1823)) (by @gold-silver-copper) - #1823
- Cap OpenRouter app categories header ([#1821](https://github.com/0xPlaygrounds/rig/pull/1821)) (by @gold-silver-copper) - #1821
- [codex] apply Anthropic cache control to tools ([#1815](https://github.com/0xPlaygrounds/rig/pull/1815)) (by @gold-silver-copper) - #1815
- Expose per-completion-call usage in agent responses ([#1787](https://github.com/0xPlaygrounds/rig/pull/1787)) (by @gold-silver-copper) - #1787
- Add replayable provider cassette tests ([#1769](https://github.com/0xPlaygrounds/rig/pull/1769)) (by @gold-silver-copper) - #1769

### Contributors

* @xavierforge
* @geraschenko
* @mccormickt
* @gold-silver-copper
* @tomasz-feliksik
* @sergiomeneses
* @abhicris
* @jimmiebfulton
* @puneetdixit200
* @ChadBartley
* @mateobelanger
* @temrjan
* @fangkangmi
* @notV4l
* @Cyanistic
## [0.37.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.36.0...rig-core-v0.37.0) - 2026-05-13

### Added

- *(openrouter)* add transcription (STT) and audio generation (TTS) support ([#1757](https://github.com/0xPlaygrounds/rig/pull/1757)) (by @fversaci)
- *(memory)* add Compactor trait, CompactingMemory adapter, and TemplateCompactor ([#1748](https://github.com/0xPlaygrounds/rig/pull/1748)) (by @ForeverAngry)
- *(ollama)* Enhance `think` parameter with string levels ([#1747](https://github.com/0xPlaygrounds/rig/pull/1747)) (by @cobaltburn)
- *(memory)* Rig-managed conversation memory + rig-memory companion crate ([#1702](https://github.com/0xPlaygrounds/rig/pull/1702)) (by @ForeverAngry)
- add copilot model listing ([#1700](https://github.com/0xPlaygrounds/rig/pull/1700)) (by @BigtoC) - #1700

### Fixed

- *(gemini)* Token usage correctness for posthog llm analytics ([#1761](https://github.com/0xPlaygrounds/rig/pull/1761)) (by @mateobelanger)
- *(openrouter)* skip serializing empty content in Assistant messages ([#1735](https://github.com/0xPlaygrounds/rig/pull/1735)) (by @pablof7z)
- *(openai)* send PDF Documents as file parts in chat completions ([#1732](https://github.com/0xPlaygrounds/rig/pull/1732)) (by @fangkangmi)
- *(core)* [**breaking**] make Chat append messages to caller history ([#1733](https://github.com/0xPlaygrounds/rig/pull/1733)) (by @gold-silver-copper)
- added a trailing newline after streamed agent response. The AgentImpl::request method streams tokens using print! macro with no trailing newline, so when the stream ends, the run loop prints the closing separator immediately which causes it to appear on the same line as the last response token - So added a println!() to the None arm of the streaming loop so a newline is always emitted after the final chunk which matches the ChatImpl path that uses println. ([#1712](https://github.com/0xPlaygrounds/rig/pull/1712)) (by @Shaurya-Sethi) - #1712
- *(mistral)* expose cached and audio token fields in Usage ([#1725](https://github.com/0xPlaygrounds/rig/pull/1725)) (by @byQuexo)

### Other

- Clean up root facade features and integration docs ([#1764](https://github.com/0xPlaygrounds/rig/pull/1764)) (by @gold-silver-copper) - #1764
- Improve GenAI token usage telemetry for Gemini and Responses API ([#1762](https://github.com/0xPlaygrounds/rig/pull/1762)) (by @gold-silver-copper) - #1762
- Memory adapter cancellation safety and trait-object forwarding ([#1756](https://github.com/0xPlaygrounds/rig/pull/1756)) (by @gold-silver-copper) - #1756
- Add demotion hooks for bounded conversation memory ([#1737](https://github.com/0xPlaygrounds/rig/pull/1737)) (by @ForeverAngry) - #1737
- Move reusable test doubles into rig_core::test_utils ([#1745](https://github.com/0xPlaygrounds/rig/pull/1745)) (by @gold-silver-copper) - #1745
- workspace and docs cleanup ([#1742](https://github.com/0xPlaygrounds/rig/pull/1742)) (by @gold-silver-copper) - #1742
- openrouter vars ([#1741](https://github.com/0xPlaygrounds/rig/pull/1741)) (by @gold-silver-copper) - #1741
- Add provider file ID support for document inputs ([#1740](https://github.com/0xPlaygrounds/rig/pull/1740)) (by @gold-silver-copper) - #1740
- bump dependencies ([#1728](https://github.com/0xPlaygrounds/rig/pull/1728)) (by @gold-silver-copper) - #1728
- Add a support of structured output for OpenRouter ([#1718](https://github.com/0xPlaygrounds/rig/pull/1718)) (by @Mnwa) - #1718
- set doctest to true, and update doc comments ([#1716](https://github.com/0xPlaygrounds/rig/pull/1716)) (by @gold-silver-copper) - #1716
- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- improve project organization and create rig crate ([#1699](https://github.com/0xPlaygrounds/rig/pull/1699)) (by @gold-silver-copper) - #1699

### Contributors

* @gold-silver-copper
* @fversaci
* @mateobelanger
* @ForeverAngry
* @cobaltburn
* @pablof7z
* @fangkangmi
* @BigtoC
* @Shaurya-Sethi
* @Mnwa
* @byQuexo

### Added

- `rig::memory::Compactor` trait. A `Compactor` produces a single
  `Message`-shaped `Artifact` from a slice of messages a memory policy
  has evicted, optionally combining it with the previous summary
  (`carry_over`) for rolling-summary semantics. Where `DemotionHook`
  is a one-way drain that observes evicted messages, `Compactor` is
  the inverse: it returns an artifact that the composing adapter
  splices back into the active history, so the loaded prompt shape is
  `[summary, ...recent_window]` instead of a verbatim suffix. The
  trait lives in `rig_core` so any memory backend can implement it
  without taking a `rig-memory` dependency; the composing
  `CompactingMemory` adapter and a zero-dependency `TemplateCompactor`
  reference implementation live in `rig-memory`. Implementations with
  durable side effects (LLM calls, vector-store writes) must
  deduplicate per the same idempotency contract as `DemotionHook`,
  since per-conversation watermarks are in-process only.

- Rig-managed conversation memory:
  - New `rig::memory` module with the `ConversationMemory` trait,
    `MemoryError` (with a `MemoryBackendError` source type),
    `MessageFilter` trait, and a default `InMemoryConversationMemory` backend
    with optional `with_filter` for shaping loaded history.
  - `AgentBuilder::memory(...)` and `AgentBuilder::conversation_id(...)` to
    attach a backend and an optional default conversation id to an agent.
  - `PromptRequest::conversation(id)` and `PromptRequest::without_memory()`
    (mirrored on the streaming builder) to control memory per-request.
  - `From<MemoryError> for PromptError` and `From<MemoryError> for
    StreamingError` so memory failures propagate via `?` through the existing
    `CompletionError::RequestError(Box<dyn Error>)` variant — no new
    top-level error variants, fully additive change.
  - `memory.append(...)` failures after a successful completion are
    best-effort: they emit `tracing::warn!` and the agent still returns the
    model response (parity for streaming `FinalResponse`). `memory.load(...)`
    failures remain fatal because the requested history is unavailable.
  - Examples: `agent_with_memory.rs` and `agent_with_memory_streaming.rs`.
  - Named history-shaping policies (sliding window, token budget) live in the
    new companion crate `rig-memory`.
- `rig::memory::DemotionHook` trait and `NoopDemotionHook` no-op default.
  Side-channel for messages that a memory policy or adapter removes from
  active history. Defined in `rig-core` so any memory backend can implement
  it without a `rig-memory` dependency; the composing
  `DemotingPolicyMemory<M, P, H>` adapter lives in `rig-memory`. Includes a
  forwarding `impl<H: DemotionHook + ?Sized> DemotionHook for Arc<H>` so
  hooks can be shared across multiple adapters.
- `MemoryError::Internal(String)` variant for in-process invariant
  violations (e.g. poisoned mutex guards), distinct from
  `MemoryError::Backend` which is reserved for failures of the underlying
  conversation store. `MemoryError` is now `#[non_exhaustive]` so future
  variants are not breaking changes.

  **Note for downstream crates:** `MemoryError` was previously a plain
  enum, so any existing `match` against it without a wildcard arm will
  now warn (and may need a wildcard arm if it was upgraded to a hard
  error elsewhere). Adding `_ => ...` is forward-compatible with future
  variants.

## [0.36.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.35.0...rig-core-v0.36.0) - 2026-04-28

### Added

- *(core)* add Xiaomi MiMo ([#1685](https://github.com/0xPlaygrounds/rig/pull/1685)) (by @BigtoC)
- rustls by default for everything ([#1682](https://github.com/0xPlaygrounds/rig/pull/1682)) (by @gold-silver-copper) - #1682
- gpt image 2, gpt 5.5, opus 4.7  ([#1679](https://github.com/0xPlaygrounds/rig/pull/1679)) (by @gold-silver-copper) - #1679
- *(core)* add DeepSeek model listing api ([#1672](https://github.com/0xPlaygrounds/rig/pull/1672)) (by @BigtoC)
- *(deepseek)* deprecate old model names ([#1664](https://github.com/0xPlaygrounds/rig/pull/1664)) (by @fu050409)
- *(ollama)* allow setting base_url and api_key programmatically ([#1511](https://github.com/0xPlaygrounds/rig/pull/1511)) (by @majiayu000)
- *(rig-core)* add ChatGPT Subscription, GitHub Copilot, and compatibility providers ([#1615](https://github.com/0xPlaygrounds/rig/pull/1615)) (by @wey-gu)
- *(providers)* add GitHub Copilot provider with relaxed response parsing ([#1451](https://github.com/0xPlaygrounds/rig/pull/1451)) (by @DAMEK86)
- *(rig-core)* Add model listing capability to OpenRouter client ([#1627](https://github.com/0xPlaygrounds/rig/pull/1627)) (by @nate-trojian)
- *(rig-derive)* support custom tool names ([#1619](https://github.com/0xPlaygrounds/rig/pull/1619)) ([#1620](https://github.com/0xPlaygrounds/rig/pull/1620)) (by @qaqland)

### Fixed

- pass generic parameter to gemini capability types ([#1687](https://github.com/0xPlaygrounds/rig/pull/1687)) (by @FayCarsons) - #1687
- *(openai)* carry reasoning_content on assistant tool-call messages ([#1649](https://github.com/0xPlaygrounds/rig/pull/1649)) (by @indrazm)
- preserve multimodal tool results in streaming chat history ([#1661](https://github.com/0xPlaygrounds/rig/pull/1661)) (by @gold-silver-copper) - #1661
- OpenAI text extraction  ([#1660](https://github.com/0xPlaygrounds/rig/pull/1660)) (by @gold-silver-copper) - #1660
- fixed n tests ([#1659](https://github.com/0xPlaygrounds/rig/pull/1659)) (by @gold-silver-copper) - #1659
- *(rig-1283)* handle llama.cpp reasoning_content as content ([#1657](https://github.com/0xPlaygrounds/rig/pull/1657)) (by @inqode-lars)
- *(responses_api)* add Unknown catch-all variant to Output enum ([#1552](https://github.com/0xPlaygrounds/rig/pull/1552)) (by @BillionClaw)

### Other

- Update permission_control.rs ([#1678](https://github.com/0xPlaygrounds/rig/pull/1678)) (by @gold-silver-copper) - #1678
- cleanup ([#1677](https://github.com/0xPlaygrounds/rig/pull/1677)) (by @gold-silver-copper) - #1677
- Add clippy no panic lints ([#1663](https://github.com/0xPlaygrounds/rig/pull/1663)) (by @gold-silver-copper) - #1663
- openai chat completions ([#1655](https://github.com/0xPlaygrounds/rig/pull/1655)) (by @gold-silver-copper) - #1655
- manual tool call example ([#1643](https://github.com/0xPlaygrounds/rig/pull/1643)) (by @gold-silver-copper) - #1643
- Remove `RwLock` from immutable state and execute futures concurrently ([#1641](https://github.com/0xPlaygrounds/rig/pull/1641)) (by @isSerge) - #1641
- Add Serialize/Deserialize derives to CompletionRequest, PromptResponse, TypedPromptResponse ([#1637](https://github.com/0xPlaygrounds/rig/pull/1637)) (by @geraschenko) - #1637
- wasm compat for model lister ([#1638](https://github.com/0xPlaygrounds/rig/pull/1638)) (by @gold-silver-copper) - #1638
- standardize required fields handling across builders ([#1611](https://github.com/0xPlaygrounds/rig/pull/1611)) (by @isSerge) - #1611
- remove deprecated code ([#1633](https://github.com/0xPlaygrounds/rig/pull/1633)) (by @gold-silver-copper) - #1633

### Contributors

* @BigtoC
* @FayCarsons
* @gold-silver-copper
* @fu050409
* @indrazm
* @inqode-lars
* @majiayu000
* @BillionClaw
* @isSerge
* @geraschenko
* @wey-gu
* @DAMEK86
* @nate-trojian
* @qaqland
## [0.35.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.34.0...rig-core-v0.35.0) - 2026-04-12

### Added

- *(rig-1197)* handle llama.cpp tool call ([#1408](https://github.com/0xPlaygrounds/rig/pull/1408)) ([#1409](https://github.com/0xPlaygrounds/rig/pull/1409)) (by @inqode-lars)

### Fixed

- *(#1604)* delay response_format on initial tool turns  (#1622) (by @gold-silver-copper)
- reduce `ToolServer` contention during tool lookup and execution ([#1607](https://github.com/0xPlaygrounds/rig/pull/1607)) (by @isSerge) - #1607
- *(streaming)* preserve tool call history, deduplicate prompt ([#1590](https://github.com/0xPlaygrounds/rig/pull/1590)) (by @gold-silver-copper)
- *(openai)* capture ResponseFailed errors in stream mode ([#1582](https://github.com/0xPlaygrounds/rig/pull/1582)) (by @gabrielrondon)

### Other

- (refactor): replace legacy Anthropic constants  ([#1616](https://github.com/0xPlaygrounds/rig/pull/1616)) (by @gold-silver-copper) - #1616
- Add ModelLister for Ollama, Anthropic, Mistral, OpenAI, Gemini ([#1587](https://github.com/0xPlaygrounds/rig/pull/1587)) (by @LHelge) - #1587
- gpt image 1.5 ([#1543](https://github.com/0xPlaygrounds/rig/pull/1543)) (by @kevinastock) - #1543
- *(rig-core)* [**breaking**] migrate examples to integration tests ([#1603](https://github.com/0xPlaygrounds/rig/pull/1603)) (by @gold-silver-copper)
- Do not stringify strings during tool output ([#1608](https://github.com/0xPlaygrounds/rig/pull/1608)) (by @gold-silver-copper) - #1608
- *(rig-core)* upgrade rmcp integration to 1.3, gate tests ([#1596](https://github.com/0xPlaygrounds/rig/pull/1596)) (by @gold-silver-copper)

### Contributors

* @gold-silver-copper
* @LHelge
* @kevinastock
* @isSerge
* @inqode-lars
* @gabrielrondon

## [0.34.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.33.0...rig-core-v0.34.0) - 2026-03-29

### Added

- *(rig-core)* respect custom Authorization headers set via http_headers() ([#1553](https://github.com/0xPlaygrounds/rig/pull/1553))
- make history generic and immutable ([#1563](https://github.com/0xPlaygrounds/rig/pull/1563))
- add grok xAI TTS ([#1530](https://github.com/0xPlaygrounds/rig/pull/1530))

### Fixed

- *(gemini)* infer string type for enum schemas in anyOf/oneOf ([#1547](https://github.com/0xPlaygrounds/rig/pull/1547))
- include assistant text in chat_history during multi-turn streaming ([#1560](https://github.com/0xPlaygrounds/rig/pull/1560))
- skip serializing encrypted_content when None ([#1534](https://github.com/0xPlaygrounds/rig/pull/1534))

### Other

- enable specifying native-tls instead of default rustls ([#1558](https://github.com/0xPlaygrounds/rig/pull/1558))
- Fix VoyageAI Usage deserialization failure on missing prompt_tokens ([#1568](https://github.com/0xPlaygrounds/rig/pull/1568))
- OTel GenAI semconv fix +  anthropic automatic prompt caching  ([#1572](https://github.com/0xPlaygrounds/rig/pull/1572))
- *(gemini)* Make `prompt_token_count` optional in gemini response ([#1548](https://github.com/0xPlaygrounds/rig/pull/1548))

## [0.33.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.32.0...rig-core-v0.33.0) - 2026-03-17

### Added

- *(rig-core)* add stateful WebSocket session for OpenAI Responses API ([#1500](https://github.com/0xPlaygrounds/rig/pull/1500))
- *(gemini)* add Gemini 3 model constants and thinking_level support ([#1520](https://github.com/0xPlaygrounds/rig/pull/1520))
- add llamafile provider ([#1519](https://github.com/0xPlaygrounds/rig/pull/1519))
- add grok imagine as image generation ([#1516](https://github.com/0xPlaygrounds/rig/pull/1516))
- *(rmcp)* `McpClientHandler` ([#1525](https://github.com/0xPlaygrounds/rig/pull/1525))
- *(telemetry)* emit gen_ai.usage.cached_tokens across all providers ([#1497](https://github.com/0xPlaygrounds/rig/pull/1497))
- add provider-native hosted tool support ([#1430](https://github.com/0xPlaygrounds/rig/pull/1430))

### Fixed

- *(openai)* make strict field optional in StructuredOutputsInput ([#1528](https://github.com/0xPlaygrounds/rig/pull/1528))
- *(llamafile)* apply embedding Number->f64 conversion for arbitrary_precision compat ([#1526](https://github.com/0xPlaygrounds/rig/pull/1526))
- embedding deserialization breaks with serde_json/arbitrary_precision ([#1518](https://github.com/0xPlaygrounds/rig/pull/1518))
- *(openai)* strengthen streaming tool call dedup to prevent false evictions ([#1510](https://github.com/0xPlaygrounds/rig/pull/1510))
- *(gemini)* [**breaking**] resolve embedding dimensions dynamically instead of hardcoding ([#1513](https://github.com/0xPlaygrounds/rig/pull/1513))
- *(gemini)* support URL-backed text documents ([#1507](https://github.com/0xPlaygrounds/rig/pull/1507))
- forward max_tokens in Chat Completions API requests ([#1495](https://github.com/0xPlaygrounds/rig/pull/1495))
- populate cached_input_tokens in Chat Completions streaming ([#1485](https://github.com/0xPlaygrounds/rig/pull/1485))
- *(gemini)* correct ProviderBuilder impl for GeminiInteractionsBuilder ([#1482](https://github.com/0xPlaygrounds/rig/pull/1482))
- *(rig-1218)* gemini MCP tool invalid tool argument ([#1462](https://github.com/0xPlaygrounds/rig/pull/1462))

### Other

- Change preamble to system message internally ([#1527](https://github.com/0xPlaygrounds/rig/pull/1527))
- fix link in rig-core README ([#1502](https://github.com/0xPlaygrounds/rig/pull/1502))
- Feat/gemini interactions api ([#1230](https://github.com/0xPlaygrounds/rig/pull/1230))


## [0.32.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.31.0...rig-core-v0.32.0) - 2026-03-05

### Added

- *(moonshot)* add Kimi K2 and K2.5 model constants ([#1457](https://github.com/0xPlaygrounds/rig/pull/1457)) (by @howardpen9)
- *(gemini)* add RAG extractor example and dynamic_context support ([#1456](https://github.com/0xPlaygrounds/rig/pull/1456)) (by @atellou)
- *(gemini)* Add support for RAG documents in dynamic context ([#1205](https://github.com/0xPlaygrounds/rig/pull/1205)) (by @snaumov)
- *(rig-core)* return conversation messages from non-streaming agent loop ([#1450](https://github.com/0xPlaygrounds/rig/pull/1450)) (by @illegalcall)
- *(extractor)* expose token usage via extract_with_usage methods ([#1447](https://github.com/0xPlaygrounds/rig/pull/1447)) (by @liamwh)
- add `.extended_details` to `TypedPromptRequest` via typestate ([#1446](https://github.com/0xPlaygrounds/rig/pull/1446)) (by @0xMochan) - #1446
- *(mistral)* implements audio transcription api ([#1424](https://github.com/0xPlaygrounds/rig/pull/1424)) (by @renanvieira)
- Reify SSE state machine ([#1428](https://github.com/0xPlaygrounds/rig/pull/1428)) (by @FayCarsons) - #1428
- feat(openrouter) Add support for openrouter embeddings ([#1418](https://github.com/0xPlaygrounds/rig/pull/1418)) ([#1419](https://github.com/0xPlaygrounds/rig/pull/1419)) (by @Lochlanna) - #1419
- *(azure-openai)* Add structured outputs support ([#1407](https://github.com/0xPlaygrounds/rig/pull/1407)) (by @austinsimpsond41)
- *(openrouter)* support audio and video ([#1413](https://github.com/0xPlaygrounds/rig/pull/1413)) (by @micllam)

### Fixed

- *(rig-1210)* deepseek content should not split into separate messages ([#1460](https://github.com/0xPlaygrounds/rig/pull/1460)) (by @joshua-mo-143)
- *(rig-1209)* reasoning content dropped from deepseektool messages ([#1459](https://github.com/0xPlaygrounds/rig/pull/1459)) (by @joshua-mo-143)
- *(openai)* add reasoning_content to StreamingDelta for OpenAI-compatible providers ([#1441](https://github.com/0xPlaygrounds/rig/pull/1441)) (by @Fromsko)
- properly support PDF doc URLs (anthropic) ([#1431](https://github.com/0xPlaygrounds/rig/pull/1431)) (by @joshua-mo-143) - #1431
- URL doc returns HTTP 400 (OpenAI) ([#1432](https://github.com/0xPlaygrounds/rig/pull/1432)) (by @joshua-mo-143) - #1432
- `total_usage` -> `usage` ([#1453](https://github.com/0xPlaygrounds/rig/pull/1453)) (by @0xMochan) - #1453
- *(deps)* enable reqwest system-proxy for proxy env var support ([#1442](https://github.com/0xPlaygrounds/rig/pull/1442)) (by @Phoenix500526)
- *(streaming)* disambiguate tool calls sharing the same index from API gateways ([#1443](https://github.com/0xPlaygrounds/rig/pull/1443)) (by @Phoenix500526)
- allow empty arguments openrouter ([#1438](https://github.com/0xPlaygrounds/rig/pull/1438)) (by @CremboC) - #1438

### Other

- *(rig-1220)* mark rig-eternalai deprecated ([#1472](https://github.com/0xPlaygrounds/rig/pull/1472)) (by @joshua-mo-143)
- *(rig-1200)* improve Client::builder DX ([#1436](https://github.com/0xPlaygrounds/rig/pull/1436)) (by @FayCarsons)
- *(deps)* update rmcp types for v0.16 API compatibility ([#1410](https://github.com/0xPlaygrounds/rig/pull/1410)) (by @adrianncovaci)

### Contributors

* @joshua-mo-143
* @Fromsko
* @howardpen9
* @atellou
* @snaumov
* @0xMochan
* @illegalcall
* @liamwh
* @Phoenix500526
* @FayCarsons
* @CremboC
* @renanvieira
* @Lochlanna
* @austinsimpsond41
* @adrianncovaci
* @micllam

## [0.31.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.30.0...rig-core-v0.31.0) - 2026-02-17

### Added

- *(rig-1192)* support pdf, image (openrouter) ([#1404](https://github.com/0xPlaygrounds/rig/pull/1404))
- cross-provider reasoning trace roundtrip ([#1396](https://github.com/0xPlaygrounds/rig/pull/1396))
- *(openrouter)* Add provider selection and prioritization support ([#1373](https://github.com/0xPlaygrounds/rig/pull/1373))
- *(rig-1189)* structured outputs ([#1382](https://github.com/0xPlaygrounds/rig/pull/1382))
- *(rig-core)* add optional model override to CompletionRequest ([#1374](https://github.com/0xPlaygrounds/rig/pull/1374))
- *(rig-1180)* support text docs (anthropic) ([#1377](https://github.com/0xPlaygrounds/rig/pull/1377))
- Add model listing capability ([#1243](https://github.com/0xPlaygrounds/rig/pull/1243))
- *(rig-1168)* add default prompt hook to agent (breaking) ([#1356](https://github.com/0xPlaygrounds/rig/pull/1356))
- [**breaking**] upgrade reqwest to 0.13 with rustls as default TLS backend ([#1218](https://github.com/0xPlaygrounds/rig/pull/1218))
- *(rig-1182)* single-text serialization to single string (openai) ([#1367](https://github.com/0xPlaygrounds/rig/pull/1367))
- add reqwest middleware example ([#1359](https://github.com/0xPlaygrounds/rig/pull/1359))

### Fixed

- *(rig-1195)* image urls don't work with anthropic ([#1403](https://github.com/0xPlaygrounds/rig/pull/1403))
- *(agents)* correct prompt hook docs, split modules, and fix install script ([#1384](https://github.com/0xPlaygrounds/rig/pull/1384))
- fix ollama dims miss ([#1199](https://github.com/0xPlaygrounds/rig/pull/1199))
- *(rig-1182)* assistantcontent serialization when empty (openai) ([#1369](https://github.com/0xPlaygrounds/rig/pull/1369))
- *(rig-1183)* invalid options provided (ollama) ([#1365](https://github.com/0xPlaygrounds/rig/pull/1365))

### Other

- add client builder test to all providers ([#1385](https://github.com/0xPlaygrounds/rig/pull/1385))
- add ironclaw ([#1400](https://github.com/0xPlaygrounds/rig/pull/1400))
- typed reasoning content model ([#1395](https://github.com/0xPlaygrounds/rig/pull/1395))
- *(streaming)* return updated history in FinalResponse ([#1210](https://github.com/0xPlaygrounds/rig/pull/1210))
- *(rig-1184)* remove AgentBuilderSimple ([#1368](https://github.com/0xPlaygrounds/rig/pull/1368))
- propagate current span to tool call ([#1361](https://github.com/0xPlaygrounds/rig/pull/1361))
- *(rig-1176)* unify prompt hook interfaces ([#1352](https://github.com/0xPlaygrounds/rig/pull/1352))

## [0.30.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.29.0...rig-core-v0.30.0) - 2026-02-03

### Added

- *(rig-1160)* encode control flow directly in type signature for hooks (breaking) ([#1305](https://github.com/0xPlaygrounds/rig/pull/1305))
- *(rig-1126)* tool image result support for gemini ([#1329](https://github.com/0xPlaygrounds/rig/pull/1329))
- support xhigh reasoning effort ([#1319](https://github.com/0xPlaygrounds/rig/pull/1319))
- *(agent)* allow on_tool_call hook to reject tool execution ([#1284](https://github.com/0xPlaygrounds/rig/pull/1284))

### Fixed

- avoid duplicate role in responses input ([#1314](https://github.com/0xPlaygrounds/rig/pull/1314))
- *(providers)* fixed azure openai embedding dimension ([#1303](https://github.com/0xPlaygrounds/rig/pull/1303))
- *(rig-1174)* openai responses requires reasoning in history ([#1335](https://github.com/0xPlaygrounds/rig/pull/1335))
- *(rig-1170)* concurrent tool execution ([#1326](https://github.com/0xPlaygrounds/rig/pull/1326))
- *(rig-1167)* fix deepseek-reasoner v3.2 invoke ([#1333](https://github.com/0xPlaygrounds/rig/pull/1333))
- *(rig-1156)* impl VectorStoreIndexDyn for mongodb and milvus ([#1300](https://github.com/0xPlaygrounds/rig/pull/1300))
- *(rig-1154)* gemini API tools mismatch ([#1291](https://github.com/0xPlaygrounds/rig/pull/1291))
- *(providers)* re-export gemini EmbeddingModel and constants at module root ([#1292](https://github.com/0xPlaygrounds/rig/pull/1292))

### Other

- *(rig-1164)* rename max_depth & related to max_turns (BREAKING) ([#1323](https://github.com/0xPlaygrounds/rig/pull/1323))
- remove unnecessary feature requirement for test ([#1341](https://github.com/0xPlaygrounds/rig/pull/1341))
- *(rig-1157)* Update xAI to Responses API ([#1316](https://github.com/0xPlaygrounds/rig/pull/1316))
- *(rig-1171)* update ollama docs ([#1327](https://github.com/0xPlaygrounds/rig/pull/1327))
- *(rig-1163)* ollama stream tool calls get ignored ([#1309](https://github.com/0xPlaygrounds/rig/pull/1309))
- Handle error for HTTP client response ([#1237](https://github.com/0xPlaygrounds/rig/pull/1237))
- Add default type parameter T = reqwest::Client to ollama's EmbeddingModel for consistency with other providers ([#1293](https://github.com/0xPlaygrounds/rig/pull/1293))

## [0.29.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.28.0...rig-core-v0.29.0) - 2026-01-20

### Added

- improve vector store documentation and filter ergonomics (breaking) ([#1258](https://github.com/0xPlaygrounds/rig/pull/1258))
- *(rig-1142)* include agent names in tracing ([#1270](https://github.com/0xPlaygrounds/rig/pull/1270))
- *(rig-1144)* deepseek reasoning content (non-streaming) ([#1269](https://github.com/0xPlaygrounds/rig/pull/1269))
- *(rig-1147)* re-export reqwest client ([#1277](https://github.com/0xPlaygrounds/rig/pull/1277))
- add custom vector store backend example ([#1252](https://github.com/0xPlaygrounds/rig/pull/1252))
- add default max depth to agents ([#1253](https://github.com/0xPlaygrounds/rig/pull/1253))
- Add the `user` parameter to openai-embedding. ([#1254](https://github.com/0xPlaygrounds/rig/pull/1254))
- *(rig-1135)* Agentic loop early termination reason ([#1248](https://github.com/0xPlaygrounds/rig/pull/1248))
- Add the ```encoding_format``` parameter to openai-embedding. ([#1203](https://github.com/0xPlaygrounds/rig/pull/1203))

### Fixed

- *(agent)* fix CancelSignal cancellation and reason sharing bugs ([#1282](https://github.com/0xPlaygrounds/rig/pull/1282))
- *(rig-1140)* do not prepend a forward slash to blank base URLs ([#1275](https://github.com/0xPlaygrounds/rig/pull/1275))

### Other

- bump dependencies ([#1257](https://github.com/0xPlaygrounds/rig/pull/1257))
- *(rig-1145)* update code snippet ([#1268](https://github.com/0xPlaygrounds/rig/pull/1268))
- fix gemini streaming ([#1262](https://github.com/0xPlaygrounds/rig/pull/1262))
- Add `AgentBuilder::tools` for adding static tools dynamically ([#1236](https://github.com/0xPlaygrounds/rig/pull/1236))
- *(rig-core)* Fix gemini doc example with wrong imports ([#1238](https://github.com/0xPlaygrounds/rig/pull/1238))

## [0.28.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.27.0...rig-core-v0.28.0) - 2026-01-06

### Added

- *(agent)* export StreamingError to public API ([#1200](https://github.com/0xPlaygrounds/rig/pull/1200))

### Fixed

- some completion providers send usage chunks with 0 completion choices causing 0 reported usage ([#1211](https://github.com/0xPlaygrounds/rig/pull/1211))
- *(rig-1109)* export agent StreamingResult type ([#1220](https://github.com/0xPlaygrounds/rig/pull/1220))
- docs typo ([#1219](https://github.com/0xPlaygrounds/rig/pull/1219))
- missing json header on send_streaming ([#1196](https://github.com/0xPlaygrounds/rig/pull/1196))
- *(rig-1113)* `calculate_max_tokens` assumes known model (anthropic) ([#1216](https://github.com/0xPlaygrounds/rig/pull/1216))
- add headers to get call ([#1178](https://github.com/0xPlaygrounds/rig/pull/1178))
- deepseek stream_prompt Invalid status code 415 ([#1170](https://github.com/0xPlaygrounds/rig/pull/1170))
- *(openrouter)* add default serde attr to reasoning_details for optional field deserialization ([#1173](https://github.com/0xPlaygrounds/rig/pull/1173))

### Other

- add tool name to tool call delta streaming events ([#1222](https://github.com/0xPlaygrounds/rig/pull/1222))
- *(deps)* update rmcp dependency to 0.12.0 ([#1182](https://github.com/0xPlaygrounds/rig/pull/1182))
- *(deps)* upgrade rmcp dependency to 0.11 ([#1165](https://github.com/0xPlaygrounds/rig/pull/1165))
- Respect custom http headers for outgoing client requests ([#1166](https://github.com/0xPlaygrounds/rig/pull/1166))

## [0.27.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.26.0...rig-core-v0.27.0) - 2025-12-15

### Added

- *(rig-1096)* pass tool call ID to prompt hook ([#1162](https://github.com/0xPlaygrounds/rig/pull/1162))
- *(rig-1059)* support `reqwest-middleware` client ([#1152](https://github.com/0xPlaygrounds/rig/pull/1152))

### Fixed

- *(groq)* rename StreamingOptions to StreamOptions ([#1159](https://github.com/0xPlaygrounds/rig/pull/1159))
- *(openai)* add None variant to ReasoningEffort enum ([#1158](https://github.com/0xPlaygrounds/rig/pull/1158))

### Other

- ToolCall Signature and additional parameters ([#1154](https://github.com/0xPlaygrounds/rig/pull/1154))
- fix incorrect variable name in AgentBuilder examples ([#1160](https://github.com/0xPlaygrounds/rig/pull/1160))
- *(rig-1085)* groq reasoning format ([#1151](https://github.com/0xPlaygrounds/rig/pull/1151))
- *(rig-1031)* remove worker crate dep ([#1149](https://github.com/0xPlaygrounds/rig/pull/1149))
- *(rig-1090)* crate re-org ([#1145](https://github.com/0xPlaygrounds/rig/pull/1145))

## [0.26.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.25.0...rig-core-v0.26.0) - 2025-12-04

### Added

- add Anthropic prompt caching support ([#1116](https://github.com/0xPlaygrounds/rig/pull/1116))
- *(rig-1076)* Providers should route all requests through `client::Client` ([#1115](https://github.com/0xPlaygrounds/rig/pull/1115))

### Fixed

- *(streaming)* use .instrument() instead of span.enter() to prevent span leak ([#1108](https://github.com/0xPlaygrounds/rig/pull/1108))

### Other

- *(rig-1077)* ensure log level enabled before logging messages ([#1114](https://github.com/0xPlaygrounds/rig/pull/1114))
- *(rig-1078)* remove messages from span telemetry ([#1112](https://github.com/0xPlaygrounds/rig/pull/1112))

## [0.25.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.24.0...rig-core-v0.25.0) - 2025-12-01

### Added

- Gemini Assistant Image Responses ([#1048](https://github.com/0xPlaygrounds/rig/pull/1048))
- *(gemini-request)* add response_json_schema to GenerationConfig ([#1077](https://github.com/0xPlaygrounds/rig/pull/1077))
- *(rig-985)* Consolidate provider clients ([#1050](https://github.com/0xPlaygrounds/rig/pull/1050))

### Fixed

- *(rig-1093)* gemini config error when no additional params used ([#1094](https://github.com/0xPlaygrounds/rig/pull/1094))
- OpenAI required props for structured output ([#1090](https://github.com/0xPlaygrounds/rig/pull/1090))
- *(rig-1055)* remove deprecated gemini-2.5-flash preview ([#1084](https://github.com/0xPlaygrounds/rig/pull/1084))
- rmcp derive clone ([#1080](https://github.com/0xPlaygrounds/rig/pull/1080))
- *(rig-1050)* Inconsistent model/agent initialisation methods ([#1069](https://github.com/0xPlaygrounds/rig/pull/1069))
- *(gemini-request)* add `#[serde(default)]` for missing `generation_config` field ([#1060](https://github.com/0xPlaygrounds/rig/pull/1060))
- update imported packages in the code example ([#1041](https://github.com/0xPlaygrounds/rig/pull/1041))

### Other

- add `Content-Type: application/json` to regular http requests ([#1106](https://github.com/0xPlaygrounds/rig/pull/1106))
- Deprecate `DynClientBuilder` ([#1105](https://github.com/0xPlaygrounds/rig/pull/1105))
- `client::Client` can leak api keys that have been inserted into its headers ([#1102](https://github.com/0xPlaygrounds/rig/pull/1102))
- *(rig-1071)* remove outdated models ([#1096](https://github.com/0xPlaygrounds/rig/pull/1096))
- *(rig-1068)* remove unused chatbot module ([#1092](https://github.com/0xPlaygrounds/rig/pull/1092))
- Simple JSON passthrough unwrapper ([#1086](https://github.com/0xPlaygrounds/rig/pull/1086))
- *(rig-777)* proper request modelling for every provider ([#1067](https://github.com/0xPlaygrounds/rig/pull/1067))
- *(deps)* upgrade `rmcp` ([#1079](https://github.com/0xPlaygrounds/rig/pull/1079))
- OpenAI parsing ([#1058](https://github.com/0xPlaygrounds/rig/pull/1058))
- *(rig-1046)* update list of who's using rig ([#1061](https://github.com/0xPlaygrounds/rig/pull/1061))
- clean up provider code ([#1052](https://github.com/0xPlaygrounds/rig/pull/1052))

## [0.24.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.23.1...rig-core-v0.24.0) - 2025-11-10

### Added

- *(rig-1024)* yield tool calls and results from multi-step stream prompt ([#1023](https://github.com/0xPlaygrounds/rig/pull/1023))
- *(providers)* Emit tool call deltas ([#1020](https://github.com/0xPlaygrounds/rig/pull/1020))
- export rig tool macro from main crate ([#1016](https://github.com/0xPlaygrounds/rig/pull/1016))

### Fixed

- *(rig-1035)* export StreamingPromptHook ([#1039](https://github.com/0xPlaygrounds/rig/pull/1039))
- Gemini responses lacking content ([#1030](https://github.com/0xPlaygrounds/rig/pull/1030))
- *(rig-1029)* Reasoning not handled properly for agent stream prompt ([#1024](https://github.com/0xPlaygrounds/rig/pull/1024))
- *(openai-responses)* add `#[serde(default)]` for missing `tools` field ([#1021](https://github.com/0xPlaygrounds/rig/pull/1021))
- *(rig-1027)* allow any error type to be used for rig tool macro ([#1017](https://github.com/0xPlaygrounds/rig/pull/1017))

### Other

- make CompletionModel  default type to reqwest::Client ([#1013](https://github.com/0xPlaygrounds/rig/pull/1013))
- *(deps)* upgrade rmcp dependency ([#1008](https://github.com/0xPlaygrounds/rig/pull/1008))

## [0.23.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.23.0...rig-core-v0.23.1) - 2025-10-28

### Fixed

- compliance with OpenAI API  stream error "message":"Model field is required." ([#1006](https://github.com/0xPlaygrounds/rig/pull/1006))

## [0.23.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.22.0...rig-core-v0.23.0) - 2025-10-27

### Added

- *(rig-1021)* allow language to be set to None for transcription ([#997](https://github.com/0xPlaygrounds/rig/pull/997))
- *(rig-1008)* add Send + Sync to ProviderClient ([#974](https://github.com/0xPlaygrounds/rig/pull/974))
- *(rig-976)* support filters for `VectorSearchRequest` ([#952](https://github.com/0xPlaygrounds/rig/pull/952))
- *(rig-1004)* expose tool call partials ([#960](https://github.com/0xPlaygrounds/rig/pull/960))
- convert video media mime type ([#959](https://github.com/0xPlaygrounds/rig/pull/959))
- *(rig-996)* generic streaming ([#955](https://github.com/0xPlaygrounds/rig/pull/955))
- *(gemini)* Support streaming thinking ([#947](https://github.com/0xPlaygrounds/rig/pull/947))
- *(ollama)* thinking ([#948](https://github.com/0xPlaygrounds/rig/pull/948))
- *(anthropic)* Expose the reasoning signature ([#945](https://github.com/0xPlaygrounds/rig/pull/945))

### Fixed

- CompletionError: ProviderError: {"error":{"code":null,"param":null,"message":"[] is too short - 'tools'","type":"invalid_request_error"}} ([#1003](https://github.com/0xPlaygrounds/rig/pull/1003))
- *(rig-1023)* reasoning/thinking stream sends redundant data ([#1002](https://github.com/0xPlaygrounds/rig/pull/1002))
- *(rig-1022)* GenericEventSource polling None should not error ([#999](https://github.com/0xPlaygrounds/rig/pull/999))
- *(huggingface)* align tool message serialization with OpenAI API spec ([#993](https://github.com/0xPlaygrounds/rig/pull/993))
- *(rig-1020)* add `futures-timer/wasm-bindgen` feature for wasm ([#995](https://github.com/0xPlaygrounds/rig/pull/995))
- *(rig-1019)* fix potentially incorrect provider URLs ([#991](https://github.com/0xPlaygrounds/rig/pull/991))
- *(rig-1016)* Huggingface completions API 404 ([#986](https://github.com/0xPlaygrounds/rig/pull/986))
- *(rig-1011)* docs mismatch ([#981](https://github.com/0xPlaygrounds/rig/pull/981))
- *(rig-1007)* tool servers broken in WASM ([#970](https://github.com/0xPlaygrounds/rig/pull/970))
- *(rig-1009)* Incorrect struct shape (OpenAI) ([#973](https://github.com/0xPlaygrounds/rig/pull/973))
- *(rig-997)* allow string documents for OpenAI Completions API ([#966](https://github.com/0xPlaygrounds/rig/pull/966))
- *(rig-1006)* text-embedding-ada-002 doesn't support custom dimensions ([#967](https://github.com/0xPlaygrounds/rig/pull/967))
- *(agent)* Apply tool_choice to completion request ([#958](https://github.com/0xPlaygrounds/rig/pull/958))
- *(rig-1005)* enable toggling "think" on ollama ([#962](https://github.com/0xPlaygrounds/rig/pull/962))
- *(openrouter)* use reqwest_post helper to construct full URL ([#943](https://github.com/0xPlaygrounds/rig/pull/943))
- *(rig-995)* include max tokens in Moonshot API request ([#935](https://github.com/0xPlaygrounds/rig/pull/935))

### Other

- InvalidCodeWithMessage error enum variant ([#963](https://github.com/0xPlaygrounds/rig/pull/963))
- *(rig-1003)* update list of production rig users ([#956](https://github.com/0xPlaygrounds/rig/pull/956))
- make streaming prompt module pub ([#944](https://github.com/0xPlaygrounds/rig/pull/944))
- *(rig-993)* re-import all items from embeddings module in rig::embeddings ([#936](https://github.com/0xPlaygrounds/rig/pull/936))

## [0.22.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.21.0...rig-core-v0.22.0) - 2025-10-14

### Added

- *(rig-937)* evals ([#905](https://github.com/0xPlaygrounds/rig/pull/905))
- *(rig-986)* tool servers ([#916](https://github.com/0xPlaygrounds/rig/pull/916))
- *(rig-988)* cancel streaming prompts from prompt hook ([#918](https://github.com/0xPlaygrounds/rig/pull/918))
- *(rig-990)* allow configuring optional lancedb features ([#923](https://github.com/0xPlaygrounds/rig/pull/923))
- return usage when streaming completions from a dynamic client ([#903](https://github.com/0xPlaygrounds/rig/pull/903))
- *(rig-979)* discord bot integration ([#900](https://github.com/0xPlaygrounds/rig/pull/900))
- *(rig-935)* support cancelling multi-turn prompt loop from hook ([#904](https://github.com/0xPlaygrounds/rig/pull/904))
- *(rig-951)* generic HTTP client ([#875](https://github.com/0xPlaygrounds/rig/pull/875))
- *(rig-977)* add description field to Agent, update tool impl ([#895](https://github.com/0xPlaygrounds/rig/pull/895))
- *(rig-848)* extract JSON with chat history ([#888](https://github.com/0xPlaygrounds/rig/pull/888))
- *(rig-955)* set up tool choice capability for Extractor ([#884](https://github.com/0xPlaygrounds/rig/pull/884))
- *(rig-964)* add tool choice to agent ([#883](https://github.com/0xPlaygrounds/rig/pull/883))
- *(rig-973)* DocumentSourceKind::String ([#882](https://github.com/0xPlaygrounds/rig/pull/882))

### Fixed

- *(rig-991)* nested struct conversion to Gemini OpenAPI type schema ([#926](https://github.com/0xPlaygrounds/rig/pull/926))
- *(rig-982)* embedding_model_with_ndims() doesn't pass dimensions parameter to OpenAI API
- *(rig-983)* http request fail due to no content type header set ([#909](https://github.com/0xPlaygrounds/rig/pull/909))
- Correct data structure for OpenAI responses images and PDFs ([#880](https://github.com/0xPlaygrounds/rig/pull/880))

### Other

- *(rig-975)* split streaming portion of PromptHook ([#889](https://github.com/0xPlaygrounds/rig/pull/889))
- *(rig-975)* split streaming portion of PromptHook
- *(rig-959)* Documents in Huggingface are not converted properly ([#874](https://github.com/0xPlaygrounds/rig/pull/874))

## [0.21.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.20.0...rig-core-v0.21.0) - 2025-09-29

### Added

- GenAI SemConv support (otel) ([#850](https://github.com/0xPlaygrounds/rig/pull/850))
- add streaming support to DynClientBuilder ([#824](https://github.com/0xPlaygrounds/rig/pull/824))
- *(rig-912)* rework `Chat` trait for multi-turn ([#846](https://github.com/0xPlaygrounds/rig/pull/846))
- *(rig-795)* support file URLs for audio, video, documents ([#823](https://github.com/0xPlaygrounds/rig/pull/823))
- *(rig-943)* support thinking for cohere ([#827](https://github.com/0xPlaygrounds/rig/pull/827))

### Fixed

- only youtube videos should accept null mime type (gemini) ([#873](https://github.com/0xPlaygrounds/rig/pull/873))
- *(rig-970)* file URLs should be able to accept empty media type (Gemini) ([#872](https://github.com/0xPlaygrounds/rig/pull/872))
- *(rig-970)* youtube video ingestion doesn't work (gemini)
- fix(rig-962)(deepseek): tool calls not recognised when put behind text content ([#862](https://github.com/0xPlaygrounds/rig/pull/862))
- fix-853 ([#854](https://github.com/0xPlaygrounds/rig/pull/854))
- *(rig-956)* DocumentSourceKind fails to serialize with common serializers ([#849](https://github.com/0xPlaygrounds/rig/pull/849))
- *(rig-957)* huggingface should convert image URLs ([#848](https://github.com/0xPlaygrounds/rig/pull/848))
- *(rig-950)* openai imagegen doesn't work with gpt-image-1 ([#837](https://github.com/0xPlaygrounds/rig/pull/837))
- ci lints ([#832](https://github.com/0xPlaygrounds/rig/pull/832))

### Other

- *(rig-969)* update features on README ([#870](https://github.com/0xPlaygrounds/rig/pull/870))
- *(rig-963)* fix feature regression in AWS bedrock ([#863](https://github.com/0xPlaygrounds/rig/pull/863))
- fix typo in comment ([#866](https://github.com/0xPlaygrounds/rig/pull/866))
- parse NDJSON correctly, fixes #825 ([#826](https://github.com/0xPlaygrounds/rig/pull/826))
- make Reasoning non-exhaustive ([#830](https://github.com/0xPlaygrounds/rig/pull/830))

## [0.20.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.19.0...rig-core-v0.20.0) - 2025-09-15

### Added

- think tool, vector store tool, better agent tool ([#424](https://github.com/0xPlaygrounds/rig/pull/424))
- *(rig-926)* make agent multi stream prompting more granular ([#796](https://github.com/0xPlaygrounds/rig/pull/796))
- *(rig-928)* allow openai chat completions to be used as an extractor ([#797](https://github.com/0xPlaygrounds/rig/pull/797))
- *(rig-831)* ensure all features are added to docs.rs ([#801](https://github.com/0xPlaygrounds/rig/pull/801))
- *(rig-931)* support file input for images on Gemini ([#790](https://github.com/0xPlaygrounds/rig/pull/790))

### Fixed

- *(rig-939)* incomplete byte sequence error when streaming from OpenAI Responses ([#812](https://github.com/0xPlaygrounds/rig/pull/812))
- *(rig-933)* openai responses api integration does not properly take images ([#799](https://github.com/0xPlaygrounds/rig/pull/799))

### Other

- *(cohere)* use `reqwest-eventsource`, some code cleanup ([#815](https://github.com/0xPlaygrounds/rig/pull/815))
- *(openAI, openrouter, deepseek, groq)* use `reqwest-eventsource` ([#814](https://github.com/0xPlaygrounds/rig/pull/814))
- remove unnecessary clone ([#808](https://github.com/0xPlaygrounds/rig/pull/808))
- *(rig-924)* update rmcp to 0.6 ([#785](https://github.com/0xPlaygrounds/rig/pull/785))
- optional candidates token count ([#793](https://github.com/0xPlaygrounds/rig/pull/793))
- allow prompt without preamble ([#791](https://github.com/0xPlaygrounds/rig/pull/791))

## [0.19.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.18.2...rig-core-v0.19.0) - 2025-09-02

### Added

- *(rig-core)* add fn cli_chatbot() back ([#769](https://github.com/0xPlaygrounds/rig/pull/769))
- *(rig-918)* expose more token usage metadata metrics for gemini ([#768](https://github.com/0xPlaygrounds/rig/pull/768))
- *(rig-911)* ConvertMessage trait ([#753](https://github.com/0xPlaygrounds/rig/pull/753))
- *(openai responses)* add `minimal` variant to ReasoningEffort ([#765](https://github.com/0xPlaygrounds/rig/pull/765))
- *(rig-904)* Rework CLI chatbot integration ([#756](https://github.com/0xPlaygrounds/rig/pull/756))
- Pauseable streams ([#733](https://github.com/0xPlaygrounds/rig/pull/733))
- *(rig-910)* function calls fail when using OpenAI Responses API with reasoning models ([#754](https://github.com/0xPlaygrounds/rig/pull/754))
- *(rig-901)* Make multi-turn stream return a `Send + 'static` stream ([#739](https://github.com/0xPlaygrounds/rig/pull/739))
- VerifyClient trait ([#724](https://github.com/0xPlaygrounds/rig/pull/724))
- *(rig-898)* make MultiTurnStreamItem pub ([#735](https://github.com/0xPlaygrounds/rig/pull/735))

### Fixed

- *(rig-core examples)* add `required` field to calculator example tool definitions ([#757](https://github.com/0xPlaygrounds/rig/pull/757))
- *(openai responses)* recursively add additionalProperties: false to nested schemas ([#755](https://github.com/0xPlaygrounds/rig/pull/755))
- empty type in Vec<T> schema conversion for Gemini API ([#721](https://github.com/0xPlaygrounds/rig/pull/721)) ([#748](https://github.com/0xPlaygrounds/rig/pull/748))

### Other
- 修改文档错误 ([#771](https://github.com/0xPlaygrounds/rig/pull/771))
- *(rig-907)* use where clause for trait bounds ([#749](https://github.com/0xPlaygrounds/rig/pull/749))
- *(rig-913)* add feature gated items to docs ([#764](https://github.com/0xPlaygrounds/rig/pull/764))
- Remove duplicate methods in perplexity ([#725](https://github.com/0xPlaygrounds/rig/pull/725))

## [0.18.2](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.18.1...rig-core-v0.18.2) - 2025-08-20

### Fixed

- docs are broken (...again) ([#722](https://github.com/0xPlaygrounds/rig/pull/722))

## [0.18.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.18.0...rig-core-v0.18.1) - 2025-08-19

### Fixed

- *(rig-890)* docs are broken ([#718](https://github.com/0xPlaygrounds/rig/pull/718))

## [0.18.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.17.1...rig-core-v0.18.0) - 2025-08-19

### Added

- *(rig-865)* multi turn streaming ([#712](https://github.com/0xPlaygrounds/rig/pull/712))
- implement Tool for Agent ([#704](https://github.com/0xPlaygrounds/rig/pull/704))
- Add capability to add custom logic while running prompts ([#632](https://github.com/0xPlaygrounds/rig/pull/632))
- *(rig-863)* add retries to extractor tool ([#685](https://github.com/0xPlaygrounds/rig/pull/685))
- *(gemini)* Accept plain-text tool result ([#686](https://github.com/0xPlaygrounds/rig/pull/686))
- video input for gemini ([#690](https://github.com/0xPlaygrounds/rig/pull/690))
- added get_tool_definitions ([#666](https://github.com/0xPlaygrounds/rig/pull/666))

### Fixed

- *(rig-886)* only GenerationConfig can be passed into additional_params ([#707](https://github.com/0xPlaygrounds/rig/pull/707))
- deepseek streaming endpoint ([#687](https://github.com/0xPlaygrounds/rig/pull/687))
- *(rig-864)* missing id from OpenAI Responses API for reasoning items ([#681](https://github.com/0xPlaygrounds/rig/pull/681))

### Other

- *(rig-883)* fully deprecate mcp feature flag ([#714](https://github.com/0xPlaygrounds/rig/pull/714))
- *(gemini)* Refactor parts to Vec instead of OneOrMany in Gemini ([#691](https://github.com/0xPlaygrounds/rig/pull/691))
- consistent visibility modifiers in openai ([#694](https://github.com/0xPlaygrounds/rig/pull/694))
- Update rmcp to version 0.5 ([#682](https://github.com/0xPlaygrounds/rig/pull/682))
- Fix SSE parsing in Gemini provider ([#683](https://github.com/0xPlaygrounds/rig/pull/683))
- *(rig-862)* remove sync bound from fn call() in tool trait ([#678](https://github.com/0xPlaygrounds/rig/pull/678))
- 删除gemini providers中重复的方法 ([#675](https://github.com/0xPlaygrounds/rig/pull/675))

## [0.17.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.17.0...rig-core-v0.17.1) - 2025-08-05

### Other

- remove unnecessary warning traces ([#672](https://github.com/0xPlaygrounds/rig/pull/672))
- *(rig-851)* update provider integrations list ([#651](https://github.com/0xPlaygrounds/rig/pull/651))

## [0.17.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.16.0...rig-core-v0.17.0) - 2025-08-05

### Added

- *(rig-845)* cosine similarity for vector search ([#664](https://github.com/0xPlaygrounds/rig/pull/664))
- add `delete_tool` method to `Toolset` ([#663](https://github.com/0xPlaygrounds/rig/pull/663))
- Read the OPENAI_BASE_URL env variable when constructing an OpenAI client from_env ([#659](https://github.com/0xPlaygrounds/rig/pull/659))
- add agent name ([#633](https://github.com/0xPlaygrounds/rig/pull/633))

### Fixed

- *(rig-853)* gemini streaming impl ignores reasoning chunks ([#654](https://github.com/0xPlaygrounds/rig/pull/654))
- Ollama provider handling of canonical URLs ([#656](https://github.com/0xPlaygrounds/rig/pull/656))
- *(rig-852)* dynamic context does not work correctly with ollama ([#660](https://github.com/0xPlaygrounds/rig/pull/660))

### Other

- *(rig-861)* make Agent<M> non-exhaustive ([#670](https://github.com/0xPlaygrounds/rig/pull/670))

## [0.16.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.15.1...rig-core-v0.16.0) - 2025-07-30

### Added

- *(rig-798)* `rig-wasm` ([#611](https://github.com/0xPlaygrounds/rig/pull/611))
- *(rig-819)* vector store index request struct ([#623](https://github.com/0xPlaygrounds/rig/pull/623))
- *(rig-830)* map documents to text for OpenAI Response API ([#622](https://github.com/0xPlaygrounds/rig/pull/622))
- Add GROK_4 model constant to xAI provider ([#614](https://github.com/0xPlaygrounds/rig/pull/614))
- *(rig-812)* yield final response with total usage metrics from streaming completion response in stream impl ([#584](https://github.com/0xPlaygrounds/rig/pull/584))
- *(rig-799)* add support for official rust sdk for mcp ([#553](https://github.com/0xPlaygrounds/rig/pull/553))
- *(rig-823)* impl size hint for OneOrMany types ([#606](https://github.com/0xPlaygrounds/rig/pull/606))
- *(rig-784)* thinking/reasoning ([#557](https://github.com/0xPlaygrounds/rig/pull/557))
- *(rig-821)* add tracing when submit tool is never called in extractor ([#603](https://github.com/0xPlaygrounds/rig/pull/603))
- make PromptResponse public ([#593](https://github.com/0xPlaygrounds/rig/pull/593))

### Fixed

- *(rig-824)* ToolResultContent should be serde-tagged ([#621](https://github.com/0xPlaygrounds/rig/pull/621))
- *(rig-828)* support done message on openai streaming completions api ([#619](https://github.com/0xPlaygrounds/rig/pull/619))
- *(rig-827)* openai responses streaming api placeholder panic ([#620](https://github.com/0xPlaygrounds/rig/pull/620))
- *(rig-834)* erroeneous tracing log level ([#626](https://github.com/0xPlaygrounds/rig/pull/626))
- *(rig-820)* ensure call ID is properly propagated ([#601](https://github.com/0xPlaygrounds/rig/pull/601))

### Other

- Add new claude models and default max tokens ([#634](https://github.com/0xPlaygrounds/rig/pull/634))
- *(rig-836)* deprecate mcp-core integration ([#631](https://github.com/0xPlaygrounds/rig/pull/631))
- Refactor clients with builder pattern ([#615](https://github.com/0xPlaygrounds/rig/pull/615))
- change log level to debug for input/output ([#627](https://github.com/0xPlaygrounds/rig/pull/627))
- fix spelling issue  ([#607](https://github.com/0xPlaygrounds/rig/pull/607))

### Migration
- If you are using `Client::from_url()`, you will now need to use `Client::builder()` and add it in from there. Otherwise if you don't care about changing your inner HTTP client or changing the base URL, you can still use `Client::new(<api_key_here>)` or `Client::from_env()` to achieve the same result as you normally would.
- `VectorStoreIndex` and `VectorStoreIndexDyn` now take a `rig::vector_search::VectorSearchRequest`, instead of a query and max result size. This has been done to enable much more ergonomic requesting in the future. Please see any of the `vector_search` examples for practical usage.
- The final response of a completion stream now yields the completion usage from the stream itself. You may wish to adjust your code to account for this.
- The `mcp-core` integration is now officially deprecated because the official Rust MCP SDK is now supported as it has feature parity. You will need to ensure you have moved to the `rmcp` integration (`rmcp` feature flag) by Rig 0.18.0 at the earliest.
- ToolResultContent is now `#[serde(tag = "type")]`. If you're storing the serialized Rig structs anywhere as JSON, you may need to account for this and write a script to backfill your stored JSON.

## [0.15.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.15.0...rig-core-v0.15.1) - 2025-07-16

### Fixed

- *(rig-815)* gemini completion fails when used with no tools ([#589](https://github.com/0xPlaygrounds/rig/pull/589))

## [0.15.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.14.0...rig-core-v0.15.0) - 2025-07-14

### Added

- *(rig-801)* DynClientBuilder::from_values ([#556](https://github.com/0xPlaygrounds/rig/pull/556))
- add `.extended_details` to `PromptRequest` ([#555](https://github.com/0xPlaygrounds/rig/pull/555))

### Fixed

- *(rig-811)* ollama fails to return results from multiple tools ([#581](https://github.com/0xPlaygrounds/rig/pull/581))
- *(rig-810)* prompting OpenAI reponses with message history fails ([#578](https://github.com/0xPlaygrounds/rig/pull/578))
- *(rig-809)* gemini function declarations should not be OneOrMany ([#576](https://github.com/0xPlaygrounds/rig/pull/576))

## [0.14.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.13.0...rig-core-v0.14.0) - 2025-07-07

### Added

- support inserting documents as a trait ([#563](https://github.com/0xPlaygrounds/rig/pull/563))
- Add max_tokens method to ExtractorBuilder ([#560](https://github.com/0xPlaygrounds/rig/pull/560))
- *(rig-780)* integrate openAI responses API ([#508](https://github.com/0xPlaygrounds/rig/pull/508))
- Stream cancellation using AbortHandle ([#525](https://github.com/0xPlaygrounds/rig/pull/525))
- *(rig-779)* allow extractor to be turned into inner agent ([#502](https://github.com/0xPlaygrounds/rig/pull/502))
- *(ollama)* add support for OLLAMA_API_BASE_URL environment var ([#541](https://github.com/0xPlaygrounds/rig/pull/541))
- *(rig-766)* add support for Voyage AI ([#493](https://github.com/0xPlaygrounds/rig/pull/493))
- *(rig-789)* add support for loading in pdfs/files as Vec<u8> ([#523](https://github.com/0xPlaygrounds/rig/pull/523))
- multi turn streaming example ([#413](https://github.com/0xPlaygrounds/rig/pull/413))
- *(rig-754)* support custom client configurations ([#511](https://github.com/0xPlaygrounds/rig/pull/511))

### Fixed

- Retain multi-turn tool call results in case of response error ([#526](https://github.com/0xPlaygrounds/rig/pull/526))
- *(rig-794)* parse openAI SSE response error ([#545](https://github.com/0xPlaygrounds/rig/pull/545))
- *(rig-796)* OpenRouter extractor fails ([#544](https://github.com/0xPlaygrounds/rig/pull/544))
- *(rig-792)* inconsistent implementations of with_custom_client ([#530](https://github.com/0xPlaygrounds/rig/pull/530))
- *(rig-783)* tool call example doesn't work with Gemini and OpenRouter ([#515](https://github.com/0xPlaygrounds/rig/pull/515))
- *(rig-773)* xAI embeddings endpoint is wrong ([#492](https://github.com/0xPlaygrounds/rig/pull/492))

### Other

- *(rig-803)* improve documentation for multi-turn ([#562](https://github.com/0xPlaygrounds/rig/pull/562))
- Migrate all crates to Rust 2024 ([#539](https://github.com/0xPlaygrounds/rig/pull/539))
- update deps ([#543](https://github.com/0xPlaygrounds/rig/pull/543))
- Declare shared dependencies in workspace ([#538](https://github.com/0xPlaygrounds/rig/pull/538))
- error fixes for clarity
- Make clippy happy on all targets ([#542](https://github.com/0xPlaygrounds/rig/pull/542))
- *(rig-791)* documents not consistently added to DeepSeek prompts ([#528](https://github.com/0xPlaygrounds/rig/pull/528))
- Fix `ToolResult` serialization in ollama provider ([#504](https://github.com/0xPlaygrounds/rig/pull/504))

## [0.13.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.12.0...rig-core-v0.13.0) - 2025-06-09

### Added

- add additional Gemini completion models ([#498](https://github.com/0xPlaygrounds/rig/pull/498))
- *(rig-758)* the extractor can pass additional params to be passed to the model ([#473](https://github.com/0xPlaygrounds/rig/pull/473))
- *(rig-744)* Add support for Milvus vector store ([#463](https://github.com/0xPlaygrounds/rig/pull/463))
- Improve Streaming API ([#388](https://github.com/0xPlaygrounds/rig/pull/388))

### Fixed

- OpenAI provider streaming tool call response for local LLM ([#442](https://github.com/0xPlaygrounds/rig/pull/442))
- *(rig-761)* ollama drops tool call results ([#478](https://github.com/0xPlaygrounds/rig/pull/478))
- Update of xAI model list ([#486](https://github.com/0xPlaygrounds/rig/pull/486))
- *(rig-757)* CI fails because of new clippy lints ([#470](https://github.com/0xPlaygrounds/rig/pull/470))
- *(extractor)* correct typo in extractor prompt ([#460](https://github.com/0xPlaygrounds/rig/pull/460))
- *(message)* correct ToolCall to Message conversion ([#461](https://github.com/0xPlaygrounds/rig/pull/461))
- Fix `dims` value for gemini's `EMBEDDING_004` ([#452](https://github.com/0xPlaygrounds/rig/pull/452)) ([#453](https://github.com/0xPlaygrounds/rig/pull/453))
- bump mcp-core to latest version and fixed breaking changes ([#443](https://github.com/0xPlaygrounds/rig/pull/443))

### Other

- Fix typo in AudioGenerationModel field name ([#487](https://github.com/0xPlaygrounds/rig/pull/487))
- Introduce Client Traits and Testing ([#440](https://github.com/0xPlaygrounds/rig/pull/440))
- Only PDF docs are supported by their API ([#465](https://github.com/0xPlaygrounds/rig/pull/465))
- Add mistral provider ([#437](https://github.com/0xPlaygrounds/rig/pull/437))
- `impl {Debug,Clone} for CompletionRequest` ([#457](https://github.com/0xPlaygrounds/rig/pull/457))
- fix some typos in comment ([#445](https://github.com/0xPlaygrounds/rig/pull/445))

## [0.12.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.11.1...rig-core-v0.12.0) - 2025-04-29

### Added

- add gpt-image-1 ([#418](https://github.com/0xPlaygrounds/rig/pull/418))
- multi-turn / reasoning loops + parallel tool calling ([#370](https://github.com/0xPlaygrounds/rig/pull/370))

### Fixed

- system and developer messages for openai ([#430](https://github.com/0xPlaygrounds/rig/pull/430))
- o-series models + constants ([#426](https://github.com/0xPlaygrounds/rig/pull/426))
- dynamically pull rag text from chat history ([#425](https://github.com/0xPlaygrounds/rig/pull/425))
- rig tool macro struct not public ([#409](https://github.com/0xPlaygrounds/rig/pull/409))
- function call conversion typo ([#415](https://github.com/0xPlaygrounds/rig/pull/415))
- deepseek function call conversion typo ([#414](https://github.com/0xPlaygrounds/rig/pull/414))

### Other

- Donot use async closure + Bump mcp-core ([#428](https://github.com/0xPlaygrounds/rig/pull/428))
- Remove broken xAI reference link in embedding.rs ([#427](https://github.com/0xPlaygrounds/rig/pull/427))
- Style/trace gemini embedding ([#411](https://github.com/0xPlaygrounds/rig/pull/411))
- Update agent_with_huggingface.rs ([#401](https://github.com/0xPlaygrounds/rig/pull/401))

## [0.11.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.11.0...rig-core-v0.11.1) - 2025-04-12

### Added

- trait for embedding images ([#396](https://github.com/0xPlaygrounds/rig/pull/396))
- Add `rig_tool` macro ([#353](https://github.com/0xPlaygrounds/rig/pull/353))
- impl From<mcp_core::types::Tool> for ToolDefinition ([#385](https://github.com/0xPlaygrounds/rig/pull/385))
- AWS Bedrock provider ([#318](https://github.com/0xPlaygrounds/rig/pull/318))

### Fixed

- gemini embeddings does not work for multiple documents ([#386](https://github.com/0xPlaygrounds/rig/pull/386))
- deserialization error due to serde rename of tool result ([#374](https://github.com/0xPlaygrounds/rig/pull/374))

### Other

- Updated broken link xaiAPI in `completion.rs` ([#384](https://github.com/0xPlaygrounds/rig/pull/384))
- Fix Clippy warnings for doc indentation and Error::other usage ([#364](https://github.com/0xPlaygrounds/rig/pull/364))

## [0.11.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.10.0...rig-core-v0.11.0) - 2025-03-31

### Added

- Add audio generation to all providers ([#359](https://github.com/0xPlaygrounds/rig/pull/359))
- Add image generation to all providers that support it ([#357](https://github.com/0xPlaygrounds/rig/pull/357))
- *(provider)* cohere-v2 ([#350](https://github.com/0xPlaygrounds/rig/pull/350))

### Fixed

- no params tools definition for Gemini ([#363](https://github.com/0xPlaygrounds/rig/pull/363))
- *(openai)* serde rename for image_url UserContent ([#355](https://github.com/0xPlaygrounds/rig/pull/355))

### Other

- New model provider: Anthropic Claude 3.7 Addition ([#341](https://github.com/0xPlaygrounds/rig/pull/341))
- added mcp_tool + Example ([#335](https://github.com/0xPlaygrounds/rig/pull/335))

## [0.10.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.9.1...rig-core-v0.10.0) - 2025-03-17

### Added

- Add streaming to all model providers ([#347](https://github.com/0xPlaygrounds/rig/pull/347))
- OpenRouter support ([#344](https://github.com/0xPlaygrounds/rig/pull/344))
- add reqwest/rustls-tls support ([#339](https://github.com/0xPlaygrounds/rig/pull/339))
- add transcription to all providers that support it ([#336](https://github.com/0xPlaygrounds/rig/pull/336))
- Azure OpenAI Token Authentication ([#329](https://github.com/0xPlaygrounds/rig/pull/329))
- SSE/JSONL decoders ported from Anthropic TS SDK ([#332](https://github.com/0xPlaygrounds/rig/pull/332))
- mira integration ([#282](https://github.com/0xPlaygrounds/rig/pull/282))
- Huggingface provider integration ([#321](https://github.com/0xPlaygrounds/rig/pull/321))

### Fixed

- unnecessary `unwrap`, skip serializing empty vec ([#343](https://github.com/0xPlaygrounds/rig/pull/343))
- fix error handling for Qwen's responses when using tools ([#351](https://github.com/0xPlaygrounds/rig/pull/351))
- reqwest can not use SOCKS proxy ([#311](https://github.com/0xPlaygrounds/rig/pull/311))
- fix wrong debug message ([#342](https://github.com/0xPlaygrounds/rig/pull/342))

### Other

- Update openai.rs ([#340](https://github.com/0xPlaygrounds/rig/pull/340))
- support svg ([#333](https://github.com/0xPlaygrounds/rig/pull/333))

## [0.9.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.9.0...rig-core-v0.9.1) - 2025-03-03

### Added

- Transcription Model support ([#322](https://github.com/0xPlaygrounds/rig/pull/322))
- Add EpubFileLoader for EPUB file processing ([#192](https://github.com/0xPlaygrounds/rig/pull/192))
- add ollama client ([#285](https://github.com/0xPlaygrounds/rig/pull/285))
- *(openai)* add updated OpenAI model constants ([#314](https://github.com/0xPlaygrounds/rig/pull/314))
- support together AI ([#230](https://github.com/0xPlaygrounds/rig/pull/230))

### Fixed

- *(openai)* skip serializing empty tool_calls vector ([#327](https://github.com/0xPlaygrounds/rig/pull/327))
- *(openai)* correct some fields for tools ([#286](https://github.com/0xPlaygrounds/rig/pull/286))
- *(loaders)* bump lodpf to allow more PDFs to parse correctly ([#307](https://github.com/0xPlaygrounds/rig/pull/307))

### Other

- rename DeepSeek_R1.pdf to deepseek_r1.pdf ([#316](https://github.com/0xPlaygrounds/rig/pull/316))

## [0.9.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.8.0...rig-core-v0.9.0) - 2025-02-17

### Added

- *(streaming)* add `Send` to `StreamingResult` inner Stream (#302)
- groq integration (#263)

### Fixed

- xai agent prompt provider error (#305) (#306)
- enhance tracing messages (#287)
- *(gemini)* fixed tool calling + tool extractor demo (#297)
- o3-mini doesn't support temperature (#266)

### Other

- EchoChambers Example Integration ([#244](https://github.com/0xPlaygrounds/rig/pull/244))
- deepseek message to remove dependencies with openai (#283)

## [0.8.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.7.0...rig-core-v0.8.0) - 2025-02-10

### Added

- fastembed integration (#268)
- *(core)* overhaul message API (#199)
- Add support for Azure OpenAI (#234)
- support moonshot language model (#223)
- galadriel api integration (redux) (#265)
- add Galadriel API integration (#188)
- support extractor for deepseek (#255)
- support tools for DeepSeek provider (#251)
- streaming API implementation for Anthropic provider (#232)

### Fixed

- deepseek client auth (#279)
- *(galadriel)* missed fixes from messages pr (#270)

### Other

- fix spelling errors in `Makefile` and `message.rs` (#284)
- Correct `tracing::debug` message. ([#275](https://github.com/0xPlaygrounds/rig/pull/275))
- agent recipes (#215)
- Revert "feat: add Galadriel API integration ([#188](https://github.com/0xPlaygrounds/rig/pull/188))" ([#264](https://github.com/0xPlaygrounds/rig/pull/264))
- *(example)* fix grammar mistake (#260)
- Fix typos  "substract" → "subtract" ([#256](https://github.com/0xPlaygrounds/rig/pull/256))
- fix typos (#242)
- add more provider notes (#237)

## [0.7.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.6.1...rig-core-v0.7.0) - 2025-01-27

### Added

- Add hyperbolic inference API integration (#238)
- *(rig-eternalai)* add support for EternalAI onchain toolset (#205)
- *(pipeline)* Add conditional op (#200)
- Add support for DeepSeek (#220)

### Fixed

- *(providers)* provider wasm support (#245)
- Use of deprecated `prelude` module (#241)
- anthropic tool use (#168)

### Other

- Fix typos (#233)
- *(README)* add SQLite as a supported vector store (#201)

## [0.6.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.6.0...rig-core-v0.6.1) - 2025-01-13

### Added

- Add `from_url` method to Gemini client (#194)
- Feature flag for CF worker compatibility (#176) (#175)
- *(eternal-ai)* Eternal-AI provider for rig (#171)
- Add gpt-4o-mini to openai model list (#187)

### Fixed

- *(example)* ollama example uses wrong url

### Other

- Add additional check for empty tool_calls ([#166](https://github.com/0xPlaygrounds/rig/pull/166))
- Mock provider API in vector store integration tests (#186)
- fix comment (#182)
- fix various typos

## [0.6.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.5.0...rig-core-v0.6.0) - 2024-12-19

### Added

- agent pipelines (#131)
- *(rig-anthropic)* Add default `max_tokens` for standard models (#151)

### Fixed

- *(openai)* Make integration more general (#156)

### Other

- *(ollama-example)* implement example showcasing ollama (#148)
- *(embeddings)* add embedding distance calculator module (#142)

## [0.5.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.4.1...rig-core-v0.5.0) - 2024-12-03

### Added

- Improve `InMemoryVectorStore` API ([#130](https://github.com/0xPlaygrounds/rig/pull/130))
- embeddings API overhaul ([#120](https://github.com/0xPlaygrounds/rig/pull/120))
- *(provider)* xAI (grok) integration ([#106](https://github.com/0xPlaygrounds/rig/pull/106))

### Fixed

- *(rig-lancedb)* rag embedding filtering ([#104](https://github.com/0xPlaygrounds/rig/pull/104))

## [0.4.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.4.0...rig-core-v0.4.1) - 2024-11-13

### Other

- Inefficient context documents serialization ([#100](https://github.com/0xPlaygrounds/rig/pull/100))

## [0.4.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.3.0...rig-core-v0.4.0) - 2024-11-07

### Added

- *(gemini)* move system prompt to correct request field
- *(provider-gemini)* add support for gemini specific completion parameters
- *(provider-gemini)* add agent support in client
- *(provider-gemini)* add gemini embedding support
- *(provider-gemini)* add gemini support for basic completion
- *(provider-gemini)* add gemini API client

### Fixed

- *(gemini)* issue when additionnal param is empty
- docs imports and refs
- *(gemini)* missing param to be marked as optional in completion res

### Other

- Cargo fmt
- Add module level docs for the `tool` module
- Fix loaders module docs references
- Add docstrings to loaders module
- Improve main lib docs
- Add `all` feature flag to rig-core
- *(gemini)* add utility config docstring
- *(gemini)* remove try_from and use serde deserialization
- Merge branch 'main' into feat/model-provider/16-add-gemini-completion-embedding-models
- *(gemini)* separate gemini api types module, fix pr comments
- add debug trait to embedding struct
- *(gemini)* add addtionnal types from the official documentation, add embeddings example
- *(provider-gemini)* test pre-commits
- *(provider-gemini)* Update readme entries, add gemini agent example

## [0.3.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.2.1...rig-core-v0.3.0) - 2024-10-24

### Added

- Generalize `EmbeddingModel::embed_documents` with `IntoIterator`
- Add `from_env` constructor to Cohere and Anthropic clients
- Small optimization to serde_json object merging
- Add better error handling for provider clients

### Fixed

- Bad Anthropic request/response handling
- *(vector-index)* In memory vector store index incorrect search

### Other

- Made internal `json_utils` module private
- Update lib docs
- Made CompletionRequest helper method private to crate
- lint + fmt
- Simplify `agent_with_tools` example
- Fix docstring links
- Add nextest test runner to CI
- Merge pull request [#42](https://github.com/0xPlaygrounds/rig/pull/42) from 0xPlaygrounds/refactor(vector-store)/update-vector-store-index-trait

## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.2.0...rig-core-v0.2.1) - 2024-10-01

### Fixed

- *(docs)* Docs still referring to old types

### Other

- Merge pull request [#45](https://github.com/0xPlaygrounds/rig/pull/45) from 0xPlaygrounds/fix/docs

## [0.2.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.1.0...rig-core-v0.2.0) - 2024-10-01

### Added

- anthropic models

### Fixed

- *(context)* displaying documents should be deterministic (sorted by alpha)
- *(context)* spin out helper method + add tests
- move context documents to user prompt message
- adjust version const naming
- implement review suggestions + renaming
- add `completion_request.documents` to `chat_history`
- adjust API to be cleaner + add docstrings

### Other

- Merge pull request [#43](https://github.com/0xPlaygrounds/rig/pull/43) from 0xPlaygrounds/fix/context-documents
- Merge pull request [#27](https://github.com/0xPlaygrounds/rig/pull/27) from 0xPlaygrounds/feat/anthropic
- Fix docstrings
- Deprecate RagAgent and Model in favor of versatile Agent
- Make RagAgent VectorStoreIndex dynamic trait objects

## [0.1.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.0.7...rig-core-v0.1.0) - 2024-09-16

### Added

- add o1-preview and o1-mini

### Fixed

- *(perplexity)* fix preamble and context in completion request
- clippy warnings

### Other

- Merge pull request [#18](https://github.com/0xPlaygrounds/rig/pull/18) from 0xPlaygrounds/feat/perplexity-support
- Add logging of http errors
- fmt code
