# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- *(tool)* [**breaking**] flatten `Tool` / `ToolDyn` metadata: tool authors now implement `description()` and `parameters()` directly, and `Tool::definition(prompt)` / `ToolDyn::definition(prompt)` are removed. `ToolDefinition` remains a provider/request artifact generated from registered tools, with `Tool::NAME` / `Tool::name()` / `ToolDyn::name()` as the single source of truth for advertised and dispatched tool names.

- *(providers)* [**breaking**] migrate `llamafile` onto the shared `GenericCompletionModel<Ext>` / `GenericEmbeddingModel<Ext>` path, deleting its hand-rolled completion model, request types, message flattening, and streaming profile. `llamafile::CompletionModel` / `llamafile::EmbeddingModel` are now type aliases for the generic models; the provider-specific `StreamingCompletionResponse` type is replaced by the shared OpenAI one. Requests now serialize messages in the shared OpenAI shape (single-text user content still flattens to a string; system/multi-part content is sent as a content-part array, which llama.cpp-family servers accept).
- *(openai)* [**breaking**] new `OpenAICompatibleProvider` trait (mirroring `AnthropicCompatibleProvider`) is now required by `GenericCompletionModel`'s `Ext` parameter; it carries the telemetry provider name (so minimax/zai/xiaomimimo spans stop reporting as "openai") and an `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` flag for llama.cpp-style streaming tool calls.
- *(providers)* [**breaking**] migrate the remaining OpenAI-chat-compatible providers onto `GenericCompletionModel<Ext>` — groq, deepseek, mistral, together, moonshot (OpenAI side), perplexity, hyperbolic, galadriel, mira, azure, and huggingface all lose their hand-rolled `CompletionModel` structs, request types, and `TryFrom<message::Message>` conversions; `CompletionModel` in each module is now a type alias for the generic model. Provider wire dialects live in `OpenAICompatibleProvider` hooks: an associated `Response` type, `completion_path` (Azure deployment URLs, `/v1`-prefixed routes), `prepare_request` (Groq native-tool folding, Moonshot `required` tool-choice coercion, HuggingFace Fireworks model ids, Perplexity/Mira tool stripping), `finalize_request_body` (DeepSeek string content + thinking-gated tool choice, Mistral `"any"` tool choice + `prefix` field + reasoning stripping, Mira raw-message flattening), and `SUPPORTS_RESPONSE_FORMAT` / `STREAM_INCLUDE_USAGE` consts. Provider-specific `StreamingCompletionResponse` types are replaced by the shared OpenAI one.
- *(openai)* [**breaking**] `ToolChoice` gains a `Function { name }` variant serializing OpenAI's `{"type":"function","function":{"name":...}}` form, so `message::ToolChoice::Specific` with one function is now supported instead of erroring; `CompletionRequest` fields are now public; `OpenAIRequestParams` gains a `supports_response_format` field.
- *(openai)* the shared `TryFrom<message::ToolResult>` conversion now prefers `call_id` over `id` for `tool_call_id` (matching provider-issued call ids); the shared streaming delta accepts `reasoning` as an alias for `reasoning_content` (Groq), and the deprecated `function_call` finish reason maps to tool-call handling.
- *(providers)* behavior notes from the migration: `max_tokens` is now forwarded by deepseek, together, hyperbolic, galadriel, and azure (previously silently dropped); together's streaming request uses standard `stream`/`stream_options` instead of `stream_tokens`, and `ToolChoice::Required` now serializes as `required` instead of erroring; perplexity's non-streaming endpoint drops its stray `/v1` prefix (matching its streaming path and the real API); galadriel tool-call ids in responses now use the provider id instead of the function name; mira's preamble is sent as a `system` message instead of `user`; `response_format` derived from `output_schema` is deferred while tools are pending a result (groq/mistral/azure previously applied it unconditionally); groq's streaming usage no longer falls back to the legacy `x_groq.usage` envelope.
- *(openrouter)* [**breaking**] de-fork OpenRouter's parallel message model (issue #2035 phase 4): `openrouter::{Message, UserContent, ImageUrl}` are now re-exports of the shared OpenAI types, and the fork's `FileContent`/`VideoUrlContent` are replaced by shared `FileData`/`VideoUrl`. To support this, the shared OpenAI types gain OpenRouter's optional extensions — `UserContent::Video`, `ImageUrl.detail` becomes `Option<ImageDetail>` (OpenAI still sends `"detail":"auto"`), and `Message::Assistant` gains skip-when-empty `reasoning_details`/`images` fields plus a deserialize-only `role: "model"` alias. `ReasoningDetails`/`ResponseImage` move into the openai module (re-exported from openrouter). OpenRouter-specific message conversion now goes through `openrouter::messages_from_rig_message`; `TryInto<Vec<openrouter::Message>>` resolves to the plain shared conversion. OpenRouter keeps its own request/response/streaming layer (provider preferences, cost accounting, reasoning-details grouping, generated-image extraction) as a documented exception.
- *(openai)* the `UserContent` audio part now serializes its tag as `input_audio` (matching OpenAI's actual API); `audio` is still accepted when deserializing.
- *(openai)* `StreamingCompletionResponse` is now generic over the provider's streaming usage payload (`StreamingCompletionResponse<U = Usage>`, selected via `OpenAICompatibleProvider::StreamingUsage`), so Mistral's cached-token fallbacks and DeepSeek's cache hit/miss counters survive streaming instead of being narrowed to OpenAI's usage shape.
- *(providers)* pre-migration request filtering is preserved where provider support is unverified: hyperbolic still drops `tools`/`tool_choice`/`output_schema` with warnings, galadriel and llamafile still drop `output_schema` with a warning, and perplexity flattens text-only message content back to plain strings (mixed multimodal content is passed through for sonar models).
- *(llamafile)* the chat cassettes are now recorded against an actual llama.cpp `llama-server`, confirming the shared OpenAI wire shape (content-part arrays, tool calls, tool results) against the real llamafile-family server rather than an OpenAI-compatible proxy.

### Removed

- *(core)* [**breaking**] remove the unused `evals` module (`Eval` trait, judge metrics, and builders) along with the `experimental` feature flag that gated it

## [0.39.0](https://github.com/0xPlaygrounds/rig/compare/v0.38.2...v0.39.0) - 2026-06-19

### Added

- *(providers)* add VoyageAI rerank support ([#1917](https://github.com/0xPlaygrounds/rig/pull/1917)) (by @sergiomeneses)
- *(agent)* [**breaking**] sans-IO AgentRun state machine; both agent loops become thin drivers ([#1899](https://github.com/0xPlaygrounds/rig/pull/1899)) (by @gold-silver-copper)

### Fixed

- correct possessive pronoun typo in CONTRIBUTING.md ([#1865](https://github.com/0xPlaygrounds/rig/pull/1865)) (by @abhicris) - #1865
- *(tool)* [**breaking**] deterministic, duplicate-safe tool registration + cassette tests ([#1913](https://github.com/0xPlaygrounds/rig/pull/1913)) (by @gold-silver-copper)

### Other

- *(deps)* bump uuid from 1.23.1 to 1.23.3 ([#1907](https://github.com/0xPlaygrounds/rig/pull/1907)) (by @dependabot[bot])
- *(deps)* bump lopdf from 0.40.0 to 0.41.0 ([#1877](https://github.com/0xPlaygrounds/rig/pull/1877)) (by @dependabot[bot])
- *(deps)* bump http from 1.4.0 to 1.4.2 ([#1909](https://github.com/0xPlaygrounds/rig/pull/1909)) (by @dependabot[bot])
- *(deps)* bump futures-timer from 3.0.3 to 3.0.4 ([#1908](https://github.com/0xPlaygrounds/rig/pull/1908)) (by @dependabot[bot])
- *(examples)* add Gemini mid-stream disruption token-counting example ([#1918](https://github.com/0xPlaygrounds/rig/pull/1918)) (by @gold-silver-copper)
- *(tool)* back ToolSet with an IndexMap instead of HashMap + order Vec ([#1916](https://github.com/0xPlaygrounds/rig/pull/1916)) (by @gold-silver-copper)
- de-flake tracing span tests and deepseek permission_control race ([#1915](https://github.com/0xPlaygrounds/rig/pull/1915)) (by @gold-silver-copper) - #1915
- *(agent)* cassette-backed AgentRun coverage against real Gemini turns ([#1901](https://github.com/0xPlaygrounds/rig/pull/1901)) (by @gold-silver-copper)
- Fix streaming reasoning history order ([#1898](https://github.com/0xPlaygrounds/rig/pull/1898)) (by @gold-silver-copper) - #1898
- Fix context document ordering ([#1893](https://github.com/0xPlaygrounds/rig/pull/1893)) (by @gold-silver-copper) - #1893
- Point ecosystem link to awesome-rig ([#1895](https://github.com/0xPlaygrounds/rig/pull/1895)) (by @gold-silver-copper) - #1895
- Add Gemini Nano Banana image generation ([#1889](https://github.com/0xPlaygrounds/rig/pull/1889)) (by @gold-silver-copper) - #1889

### Contributors

* @dependabot[bot]
* @abhicris
* @gold-silver-copper
* @sergiomeneses
## [0.38.2](https://github.com/0xPlaygrounds/rig/compare/v0.38.1...v0.38.2) - 2026-06-09

### Fixed

- support Anthropic mid-conversation system role ([#1862](https://github.com/0xPlaygrounds/rig/pull/1862)) (by @fangkangmi) - #1862

### Other

- *(deps)* bump tonic-prost-build from 0.14.5 to 0.14.6 ([#1874](https://github.com/0xPlaygrounds/rig/pull/1874)) (by @dependabot[bot])
- *(deps)* bump convert_case from 0.10.0 to 0.11.0 ([#1875](https://github.com/0xPlaygrounds/rig/pull/1875)) (by @dependabot[bot])
- *(deps)* bump reqwest from 0.13.3 to 0.13.4 ([#1873](https://github.com/0xPlaygrounds/rig/pull/1873)) (by @dependabot[bot])
- *(deps)* bump reqwest-middleware from 0.5.1 to 0.5.2 ([#1876](https://github.com/0xPlaygrounds/rig/pull/1876)) (by @dependabot[bot])
- Remove rig-redis integration ([#1887](https://github.com/0xPlaygrounds/rig/pull/1887)) (by @gold-silver-copper) - #1887
- migrate Copilot tests to cassette replay ([#1882](https://github.com/0xPlaygrounds/rig/pull/1882)) (by @gold-silver-copper) - #1882
- Redis vector store integration ([#1509](https://github.com/0xPlaygrounds/rig/pull/1509)) (by @daric93) - #1509
- add Ryzome to README nav links ([#1879](https://github.com/0xPlaygrounds/rig/pull/1879)) (by @mateobelanger) - #1879
- [codex] support mistral.rs OpenAI-compatible reasoning ([#1864](https://github.com/0xPlaygrounds/rig/pull/1864)) (by @gold-silver-copper) - #1864
- convert DeepSeek live tests to cassettes ([#1870](https://github.com/0xPlaygrounds/rig/pull/1870)) (by @gold-silver-copper) - #1870
- [codex] add OpenRouter cassette-backed provider coverage ([#1869](https://github.com/0xPlaygrounds/rig/pull/1869)) (by @gold-silver-copper) - #1869
- convert xAI live tests to cassettes ([#1868](https://github.com/0xPlaygrounds/rig/pull/1868)) (by @gold-silver-copper) - #1868
- [codex] cover Anthropic streaming tool result batching ([#1863](https://github.com/0xPlaygrounds/rig/pull/1863)) (by @gold-silver-copper) - #1863

### Contributors

* @dependabot[bot]
* @gold-silver-copper
* @daric93
* @mateobelanger
* @fangkangmi
## [0.38.1](https://github.com/0xPlaygrounds/rig/compare/v0.37.1...v0.38.1) - 2026-06-02

### Other

- unify workspace crate versions ([#1853](https://github.com/0xPlaygrounds/rig/pull/1853)) (by @gold-silver-copper) - #1853

### Contributors

* @gold-silver-copper
## [0.37.1](https://github.com/0xPlaygrounds/rig/compare/rig-v0.37.0...rig-v0.37.1) - 2026-06-02

### Added

- *(rig-derive)* replace hand-rolled schema with schemars in #[rig_tool] ([#1576](https://github.com/0xPlaygrounds/rig/pull/1576)) (by @tomasz-feliksik)
- *(gemini)* expose streaming response metadata ([#1790](https://github.com/0xPlaygrounds/rig/pull/1790)) (by @mateobelanger)
- *(anthropic)* support document citations ([#1778](https://github.com/0xPlaygrounds/rig/pull/1778)) (by @temrjan)

### Fixed

- *(chatgpt)* Handle ChatGPT response.completed events without output field ([#1825](https://github.com/0xPlaygrounds/rig/pull/1825)) (by @geraschenko)
- *(rig-gemini-grpc)* populate FunctionDeclaration.parameters from ToolDefinition ([#1763](https://github.com/0xPlaygrounds/rig/pull/1763)) (by @abhicris)
- fix sqlite threshold and null tool call streaming ([#1786](https://github.com/0xPlaygrounds/rig/pull/1786)) (by @gold-silver-copper) - #1786

### Other

- *(deps)* bump mongodb from 3.6.0 to 3.7.0 ([#1848](https://github.com/0xPlaygrounds/rig/pull/1848)) (by @dependabot[bot])
- *(deps)* bump zerocopy from 0.8.48 to 0.8.50 ([#1847](https://github.com/0xPlaygrounds/rig/pull/1847)) (by @dependabot[bot])
- *(deps)* bump google-cloud-aiplatform-v1 from 1.10.0 to 1.11.0 ([#1846](https://github.com/0xPlaygrounds/rig/pull/1846)) (by @dependabot[bot])
- *(deps)* bump serde_json from 1.0.149 to 1.0.150 ([#1845](https://github.com/0xPlaygrounds/rig/pull/1845)) (by @dependabot[bot])
- *(deps)* bump tonic from 0.14.5 to 0.14.6 ([#1844](https://github.com/0xPlaygrounds/rig/pull/1844)) (by @dependabot[bot])
- Fix parsing of streamed function-call argument deltas ([#1828](https://github.com/0xPlaygrounds/rig/pull/1828)) (by @geraschenko) - #1828
- *(deps)* port dependency bumps and Rust 1.91 ([#1842](https://github.com/0xPlaygrounds/rig/pull/1842)) (by @gold-silver-copper)
- *(deps)* bump quick-xml from 0.39.4 to 0.40.1 ([#1818](https://github.com/0xPlaygrounds/rig/pull/1818)) (by @dependabot[bot])
- *(deps)* bump google-cloud-auth from 1.9.0 to 1.10.0 ([#1817](https://github.com/0xPlaygrounds/rig/pull/1817)) (by @dependabot[bot])
- Stabilize MongoDB vector search test ([#1841](https://github.com/0xPlaygrounds/rig/pull/1841)) (by @gold-silver-copper) - #1841
- fix VT Code line grammar in README ([#1824](https://github.com/0xPlaygrounds/rig/pull/1824)) (by @Shaurya-Sethi) - #1824
- [codex] Validate model tool calls ([#1823](https://github.com/0xPlaygrounds/rig/pull/1823)) (by @gold-silver-copper) - #1823
- [codex] apply Anthropic cache control to tools ([#1815](https://github.com/0xPlaygrounds/rig/pull/1815)) (by @gold-silver-copper) - #1815
- *(deps)* bump tokio-tungstenite from 0.23.1 to 0.28.0 ([#1784](https://github.com/0xPlaygrounds/rig/pull/1784)) (by @dependabot[bot])
- *(deps)* bump rmcp from 1.6.0 to 1.7.0 ([#1783](https://github.com/0xPlaygrounds/rig/pull/1783)) (by @dependabot[bot])
- *(deps)* bump tokio from 1.52.1 to 1.52.3 ([#1782](https://github.com/0xPlaygrounds/rig/pull/1782)) (by @dependabot[bot])
- Expose per-completion-call usage in agent responses ([#1787](https://github.com/0xPlaygrounds/rig/pull/1787)) (by @gold-silver-copper) - #1787
- *(gemini)* add streaming metadata cassettes ([#1777](https://github.com/0xPlaygrounds/rig/pull/1777)) (by @gold-silver-copper)
- Add replayable provider cassette tests ([#1769](https://github.com/0xPlaygrounds/rig/pull/1769)) (by @gold-silver-copper) - #1769

### Contributors

* @dependabot[bot]
* @geraschenko
* @tomasz-feliksik
* @gold-silver-copper
* @abhicris
* @Shaurya-Sethi
* @mateobelanger
* @temrjan
## [0.37.0](https://github.com/0xPlaygrounds/rig/compare/rig-v0.36.0...rig-v0.37.0) - 2026-05-13

### Added

- *(openrouter)* add transcription (STT) and audio generation (TTS) support ([#1757](https://github.com/0xPlaygrounds/rig/pull/1757)) (by @fversaci)
- *(rig-bedrock)* add structured output support via Converse API ([#1667](https://github.com/0xPlaygrounds/rig/pull/1667)) (by @jdwil)
- *(memory)* Rig-managed conversation memory + rig-memory companion crate ([#1702](https://github.com/0xPlaygrounds/rig/pull/1702)) (by @ForeverAngry)
- add copilot model listing ([#1700](https://github.com/0xPlaygrounds/rig/pull/1700)) (by @BigtoC) - #1700

### Fixed

- *(gemini)* Token usage correctness for posthog llm analytics ([#1761](https://github.com/0xPlaygrounds/rig/pull/1761)) (by @mateobelanger)
- *(core)* [**breaking**] make Chat append messages to caller history ([#1733](https://github.com/0xPlaygrounds/rig/pull/1733)) (by @gold-silver-copper)

### Other

- Clean up root facade features and integration docs ([#1764](https://github.com/0xPlaygrounds/rig/pull/1764)) (by @gold-silver-copper) - #1764
- fix "a ancient" grammar in glarb-glarb sample text ([#1755](https://github.com/0xPlaygrounds/rig/pull/1755)) (by @abhicris) - #1755
- *(deps)* bump lopdf from 0.36.0 to 0.40.0 ([#1754](https://github.com/0xPlaygrounds/rig/pull/1754)) (by @dependabot[bot])
- *(deps)* bump quick-xml from 0.39.2 to 0.39.4 ([#1752](https://github.com/0xPlaygrounds/rig/pull/1752)) (by @dependabot[bot])
- *(deps)* bump tonic-build from 0.14.5 to 0.14.6 ([#1751](https://github.com/0xPlaygrounds/rig/pull/1751)) (by @dependabot[bot])
- Move reusable test doubles into rig_core::test_utils ([#1745](https://github.com/0xPlaygrounds/rig/pull/1745)) (by @gold-silver-copper) - #1745
- workspace and docs cleanup ([#1742](https://github.com/0xPlaygrounds/rig/pull/1742)) (by @gold-silver-copper) - #1742
- openrouter vars ([#1741](https://github.com/0xPlaygrounds/rig/pull/1741)) (by @gold-silver-copper) - #1741
- Add provider file ID support for document inputs ([#1740](https://github.com/0xPlaygrounds/rig/pull/1740)) (by @gold-silver-copper) - #1740
- add smoke test for completion across all Copilot models ([#1730](https://github.com/0xPlaygrounds/rig/pull/1730)) (by @BigtoC) - #1730
- bump dependencies ([#1728](https://github.com/0xPlaygrounds/rig/pull/1728)) (by @gold-silver-copper) - #1728
- remove needless files ([#1715](https://github.com/0xPlaygrounds/rig/pull/1715)) (by @gold-silver-copper) - #1715
- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- Add Bedrock integration tests ([#1707](https://github.com/0xPlaygrounds/rig/pull/1707)) (by @gold-silver-copper) - #1707

### Contributors

* @gold-silver-copper
* @fversaci
* @mateobelanger
* @abhicris
* @jdwil
* @dependabot[bot]
* @ForeverAngry
* @BigtoC
