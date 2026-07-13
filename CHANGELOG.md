# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- *(tool)* [**breaking**] Replace the parallel tool-execution APIs with one structured path. Typed tools now implement only `Tool::call(&mut ToolContext, Args) -> Result<Output, ToolExecutionError>`; `ToolContext` carries inbound values and host-only result metadata, `ToolResult` is the single runtime observation, and `ToolSet::execute` / `ToolServerHandle::execute` are the dispatch surfaces. Runtime erasure is internal, while event-specific hook action types make invalid event/action combinations unrepresentable.
  - Tool implementations: remove `type Error`, `classify_error`, `call_with_extensions`, and `call_structured`; return `ToolExecutionError` directly from `call`, return refusals with `ToolExecutionError::refused`, and attach host-only result metadata with `ToolContext::insert_result`.
  - Context: replace `ToolCallExtensions` and `ToolResultExtensions` with `ToolContext`; replace request/runner `.tool_extensions(...)` with `.tool_context(...)`. Each dispatch snapshots inbound context exactly once, isolates tool-local mutations, and publishes only result metadata back to the caller and hooks.
  - Dynamic tools: `ToolDyn` is removed from the public API; use `DynamicTool` for runtime-defined tools. Rig's erased dispatch trait is private. Typed tools use `Tool::NAME` as their sole identity; runtime-named agents convert explicitly with `Agent::into_tool()`.
  - Registration vocabulary: `AgentBuilder::tools(Vec<DynamicTool>)` becomes `dynamic_tools`, while retrieval-backed `dynamic_tools(sample, index, toolset)` becomes `retrieved_tools`. On `ToolSetBuilder`, `runtime_tool(DynamicTool)` becomes `dynamic_tool`, and the former embedding-backed `dynamic_tool(ToolEmbedding)` becomes `retrieved_tool`.
  - Results and errors: replace `ToolError`, `ToolFailure`, `ToolFailureKind`, `ToolReturn`, `ToolReturnOutcome`, `ToolExecutionResult`, and `ToolOutcome` with `ToolExecutionError`, `ToolErrorKind`, and the read-only `ToolResult` observed by hooks.
  - Model presentation: serializable outputs convert once into canonical `ToolOutput` content blocks; strings remain literal text, explicit `serde_json::Value` values remain JSON, and multimodal tools use `ToolOutput::content` / `ToolOutput::one` or return typed `ToolResultContent` directly. Result hooks now rewrite `ToolOutput`, provider adapters render JSON only at their terminal wire boundary, mixed user/tool-result blocks retain order, and Rig never reparses strings to infer rich content.
  - Error presentation: detailed execution-error messages remain model-visible by default so argument, validation, and custom tool errors are actionable. Use `redact_model_feedback` when diagnostics contain secrets, `with_model_feedback` for replacement text, or `with_model_output` for JSON/multimodal feedback. MCP responses preserve ordered supported text/image content, retain unsupported and future blocks as typed JSON, and attach raw `CallToolResult`, `structuredContent`, and response metadata to `ToolContext`. MCP list installation and refresh are atomic and ownership-aware, so stale handlers cannot replace or remove newer registrations.
  - Dispatch: replace `ToolSet::{call, call_with_extensions, call_structured}` with `ToolSet::execute`; replace `ToolServerHandle::{call_tool, call_tool_with_extensions, call_tool_structured}` with `ToolServerHandle::execute`.
  - Registration and definitions: `ToolSet` is the single ordered registry and records whether each tool is always advertised or retrieval-only. `ToolSet::{get_tool_definitions, documents}` are now synchronous and infallible, `ToolServerHandle` registration/removal methods no longer return an artificial `Result`, and the obsolete `ToolSetError` is removed.
  - Hooks: replace `AgentHook::on_event`, `StepEvent`, and `Flow` with the event-specific `AgentHook` methods and their corresponding action types (`CompletionCallAction`, `ToolCallAction`, `ToolResultAction`, `InvalidToolCallAction`, and `ObservationAction`). Invalid-tool hooks return `None` to defer; every explicit action, including `Fail`, is terminal for that hook stack.
- *(core)* [**breaking**] Mark `PromptError`, `StructuredOutputError`, and `VectorStoreError` as non-exhaustive, requiring downstream match expressions to include a wildcard arm. Conversation memory load failures now surface as the typed `PromptError::MemoryError` variant instead of `CompletionError::RequestError`.

## [0.40.0](https://github.com/0xPlaygrounds/rig/compare/v0.39.0...v0.40.0) - 2026-07-10

### Added

- *(tool)* [**breaking**] structured tool-execution results ([#2015](https://github.com/0xPlaygrounds/rig/pull/2015)) (by @gold-silver-copper)
- *(agent)* [**breaking**] hook system v2 — composable middleware ([#2012](https://github.com/0xPlaygrounds/rig/pull/2012)) (by @gold-silver-copper)
- *(examples)* human-in-the-loop tool-call approval — examples + tests ([#1967](https://github.com/0xPlaygrounds/rig/pull/1967)) (by @gold-silver-copper)
- *(rig-core)* steer the model request per turn from a hook via Flow::OverrideRequest ([#1966](https://github.com/0xPlaygrounds/rig/pull/1966)) (by @gold-silver-copper)
- *(rig-core)* rewrite tool results from a hook via Flow::RewriteResult ([#1965](https://github.com/0xPlaygrounds/rig/pull/1965)) (by @gold-silver-copper)
- *(rig-core)* rewrite tool-call arguments from a hook via Flow::RewriteArgs ([#1963](https://github.com/0xPlaygrounds/rig/pull/1963)) (by @gold-silver-copper)
- *(openai)* preserve responses prompt cache parameters ([#1830](https://github.com/0xPlaygrounds/rig/pull/1830)) (by @Kade-Powell)
- *(streaming)* [**breaking**] surface unmodeled provider output items through the stream ([#1951](https://github.com/0xPlaygrounds/rig/pull/1951)) (by @gold-silver-copper)
- *(rig-core)* [**breaking**] integrate hooks into AgentRun via a composable AgentRunner ([#1945](https://github.com/0xPlaygrounds/rig/pull/1945)) (by @gold-silver-copper)
- *(message)* add video helper constructors + OpenRouter audio/video conversion tests ([#1942](https://github.com/0xPlaygrounds/rig/pull/1942)) (by @gold-silver-copper)
- *(agent)* add OutputMode to compose structured output with tools ([#1928](https://github.com/0xPlaygrounds/rig/pull/1928)) ([#1929](https://github.com/0xPlaygrounds/rig/pull/1929)) (by @gold-silver-copper)

### Fixed

- *(telemetry)* keep GenAI message span fields empty ([#2066](https://github.com/0xPlaygrounds/rig/pull/2066)) (by @gold-silver-copper)
- *(chatgpt)* preserve non-success response errors ([#2053](https://github.com/0xPlaygrounds/rig/pull/2053)) (by @gold-silver-copper)
- *(vertexai)* preserve signed thought text parts ([#2052](https://github.com/0xPlaygrounds/rig/pull/2052)) (by @gold-silver-copper)
- *(chatgpt)* fallback on empty SSE output ([#2001](https://github.com/0xPlaygrounds/rig/pull/2001)) (by @gold-silver-copper)
- *(openai)* preserve reasoning text content ([#1999](https://github.com/0xPlaygrounds/rig/pull/1999)) (by @gold-silver-copper)
- preserve OpenAI Responses instructions ([#1995](https://github.com/0xPlaygrounds/rig/pull/1995)) (by @gold-silver-copper) - #1995
- *(openai)* accept null Responses metadata ([#1993](https://github.com/0xPlaygrounds/rig/pull/1993)) (by @gold-silver-copper)
- *(postgres)* update sqlx and pgvector ([#1992](https://github.com/0xPlaygrounds/rig/pull/1992)) (by @gold-silver-copper)
- *(openai)* make Responses API strict tools opt-in ([#1991](https://github.com/0xPlaygrounds/rig/pull/1991)) (by @gold-silver-copper)
- *(agent)* stream concurrent tool results as they complete ([#1981](https://github.com/0xPlaygrounds/rig/pull/1981)) (by @gold-silver-copper)
- *(rig-core)* fix epub loader tests + prevent CWD-relative fixture-path regressions ([#1940](https://github.com/0xPlaygrounds/rig/pull/1940)) (by @gold-silver-copper)
- *(ollama)* preserve assistant reasoning from non-streaming responses ([#1926](https://github.com/0xPlaygrounds/rig/pull/1926)) ([#1927](https://github.com/0xPlaygrounds/rig/pull/1927)) (by @gold-silver-copper)

### Other

- Remove unused derive and core APIs ([#2087](https://github.com/0xPlaygrounds/rig/pull/2087)) (by @gold-silver-copper) - #2087
- add Bedrock cassette coverage ([#2084](https://github.com/0xPlaygrounds/rig/pull/2084)) (by @gold-silver-copper) - #2084
- Remove unused stream completion stdout helper ([#2085](https://github.com/0xPlaygrounds/rig/pull/2085)) (by @gold-silver-copper) - #2085
- Remove unused generation wrapper traits ([#2083](https://github.com/0xPlaygrounds/rig/pull/2083)) (by @gold-silver-copper) - #2083
- Remove unused Anthropic decoders ([#2082](https://github.com/0xPlaygrounds/rig/pull/2082)) (by @gold-silver-copper) - #2082
- *(agent)* [**breaking**] unify PromptResponse and FinalResponse into one type ([#2056](https://github.com/0xPlaygrounds/rig/pull/2056)) (by @gold-silver-copper)
- *(core)* [**breaking**] API paper cuts — duplicate names, hand-copied setters, dead types ([#2055](https://github.com/0xPlaygrounds/rig/pull/2055)) (by @gold-silver-copper)
- *(examples)* add force_tool_first_turn hook example ([#2014](https://github.com/0xPlaygrounds/rig/pull/2014)) (by @gold-silver-copper)
- *(auth)* add non-interactive oauth cassette coverage ([#2050](https://github.com/0xPlaygrounds/rig/pull/2050)) (by @gold-silver-copper)
- *(perplexity)* add cassette coverage ([#2049](https://github.com/0xPlaygrounds/rig/pull/2049)) (by @gold-silver-copper)
- *(providers)* [**breaking**] remove galadriel provider ([#2041](https://github.com/0xPlaygrounds/rig/pull/2041)) (by @gold-silver-copper)
- *(providers)* [**breaking**] collapse remaining providers onto GenericCompletionModel<Ext> (#2035 phases 2–4) ([#2040](https://github.com/0xPlaygrounds/rig/pull/2040)) (by @gold-silver-copper)
- *(providers)* [**breaking**] migrate llamafile onto GenericCompletionModel<Ext> (#2035 phase 1) ([#2038](https://github.com/0xPlaygrounds/rig/pull/2038)) (by @gold-silver-copper)
- *(core)* [**breaking**] delete unused evals module and experimental feature flag ([#2036](https://github.com/0xPlaygrounds/rig/pull/2036)) (by @gold-silver-copper)
- Flatten Tool metadata API ([#2029](https://github.com/0xPlaygrounds/rig/pull/2029)) (by @gold-silver-copper) - #2029
- *(gemini)* live cassette hook-system stress suite ([#2013](https://github.com/0xPlaygrounds/rig/pull/2013)) (by @gold-silver-copper)
- Add Groq agent tool cassette regressions ([#2011](https://github.com/0xPlaygrounds/rig/pull/2011)) (by @gold-silver-copper) - #2011
- Add Mistral agent tool cassette regressions ([#2010](https://github.com/0xPlaygrounds/rig/pull/2010)) (by @gold-silver-copper) - #2010
- Add DeepSeek agent tool cassette regressions ([#2009](https://github.com/0xPlaygrounds/rig/pull/2009)) (by @gold-silver-copper) - #2009
- Add xAI agent tool cassette regressions ([#2008](https://github.com/0xPlaygrounds/rig/pull/2008)) (by @gold-silver-copper) - #2008
- Gate Gemini image cassette tests on image feature ([#2007](https://github.com/0xPlaygrounds/rig/pull/2007)) (by @gold-silver-copper) - #2007
- Add OpenRouter agent tool cassette regressions ([#2006](https://github.com/0xPlaygrounds/rig/pull/2006)) (by @gold-silver-copper) - #2006
- Add ChatGPT Codex cassette regression suite ([#2005](https://github.com/0xPlaygrounds/rig/pull/2005)) (by @gold-silver-copper) - #2005
- *(gemini)* production-grade generateContent cassette suite ([#2004](https://github.com/0xPlaygrounds/rig/pull/2004)) (by @gold-silver-copper)
- *(anthropic)* production-grade Messages API cassette suite ([#2003](https://github.com/0xPlaygrounds/rig/pull/2003)) (by @gold-silver-copper)
- *(openai)* production-grade Responses API cassette suite + tool_choice and replay-ID fixes ([#2002](https://github.com/0xPlaygrounds/rig/pull/2002)) (by @gold-silver-copper)
- *(providers)* add provider implementation checklist ([#1997](https://github.com/0xPlaygrounds/rig/pull/1997)) (by @gold-silver-copper)
- *(deps)* bump assert_fs from 1.1.3 to 1.1.4 ([#1933](https://github.com/0xPlaygrounds/rig/pull/1933)) (by @dependabot[bot])
- *(deps)* bump trybuild from 1.0.116 to 1.0.117 ([#1935](https://github.com/0xPlaygrounds/rig/pull/1935)) (by @dependabot[bot])
- *(deps)* bump chrono from 0.4.44 to 0.4.45 ([#1934](https://github.com/0xPlaygrounds/rig/pull/1934)) (by @dependabot[bot])
- *(deps)* bump uuid from 1.23.3 to 1.23.4 ([#1975](https://github.com/0xPlaygrounds/rig/pull/1975)) (by @dependabot[bot])
- *(deps)* bump scylla from 1.6.0 to 1.7.0 ([#1932](https://github.com/0xPlaygrounds/rig/pull/1932)) (by @dependabot[bot])
- *(anthropic)* add null citation streaming cassette ([#1978](https://github.com/0xPlaygrounds/rig/pull/1978)) (by @gold-silver-copper)
- update agent and contribution guidance ([#1974](https://github.com/0xPlaygrounds/rig/pull/1974)) (by @gold-silver-copper) - #1974
- *(openai-compat)* genuinely exercise the #1958 tool-call eviction string-leak (+ live cassette) ([#1962](https://github.com/0xPlaygrounds/rig/pull/1962)) (by @gold-silver-copper)
- *(rig-core)* [**breaking**] remove the experimental pipeline module ([#1941](https://github.com/0xPlaygrounds/rig/pull/1941)) (by @gold-silver-copper)
- run doctests and stop rig-sqlite opting out of them ([#1939](https://github.com/0xPlaygrounds/rig/pull/1939)) (by @gold-silver-copper) - #1939
- *(rig-core)* replace nanoid with fastrand for internal IDs ([#1938](https://github.com/0xPlaygrounds/rig/pull/1938)) (by @gold-silver-copper)
- *(examples)* migrate to a package-per-example layout ([#1937](https://github.com/0xPlaygrounds/rig/pull/1937)) (by @gold-silver-copper)
- add Archestra to "Who is using Rig?" section ([#1925](https://github.com/0xPlaygrounds/rig/pull/1925)) (by @arsenyinfo) - #1925

### Contributors

* @gold-silver-copper
* @dependabot[bot]
* @Kade-Powell
* @arsenyinfo

### Changed

- *(agent)* [**breaking**] `max_turns` and `default_max_turns` now bound the exact total number of model calls, including the initial call, tool continuations, and retries. A budget of `0` makes no model call, while `1` permits only the initial call. Unconfigured tool-then-answer flows now need an explicit total budget of `2`. To preserve the former maximum allowance of an explicit old budget `n`, account for the old effective `n + 2` calls; otherwise, set the intended literal total.

- *(tool)* [**breaking**] flatten `Tool` / `ToolDyn` metadata: tool authors now implement `description()` and `parameters()` directly, and `Tool::definition(prompt)` / `ToolDyn::definition(prompt)` are removed. `ToolDefinition` remains a provider/request artifact generated from registered tools, with `Tool::NAME` / `Tool::name()` / `ToolDyn::name()` as the single source of truth for advertised and dispatched tool names.

- *(providers)* [**breaking**] migrate `llamafile` onto the shared `GenericCompletionModel<Ext>` / `GenericEmbeddingModel<Ext>` path, deleting its hand-rolled completion model, request types, message flattening, and streaming profile. `llamafile::CompletionModel` / `llamafile::EmbeddingModel` are now type aliases for the generic models; the provider-specific `StreamingCompletionResponse` type is replaced by the shared OpenAI one. Requests now serialize messages in the shared OpenAI shape (single-text user content still flattens to a string; system/multi-part content is sent as a content-part array, which llama.cpp-family servers accept).
- *(openai)* [**breaking**] new `OpenAICompatibleProvider` trait (mirroring `AnthropicCompatibleProvider`) is now required by `GenericCompletionModel`'s `Ext` parameter; it carries the telemetry provider name (so minimax/zai/xiaomimimo spans stop reporting as "openai") and an `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` flag for llama.cpp-style streaming tool calls.
- *(providers)* [**breaking**] migrate the remaining OpenAI-chat-compatible providers onto `GenericCompletionModel<Ext>` — groq, deepseek, mistral, together, moonshot (OpenAI side), perplexity, hyperbolic, mira, azure, and huggingface all lose their hand-rolled `CompletionModel` structs, request types, and `TryFrom<message::Message>` conversions; `CompletionModel` in each module is now a type alias for the generic model. Provider wire dialects live in `OpenAICompatibleProvider` hooks: an associated `Response` type, `completion_path` (Azure deployment URLs, `/v1`-prefixed routes), `prepare_request` (Groq native-tool folding, Moonshot `required` tool-choice coercion, HuggingFace Fireworks model ids, Perplexity/Mira tool stripping), `finalize_request_body` (DeepSeek string content + thinking-gated tool choice, Mistral `"any"` tool choice + `prefix` field + reasoning stripping, Mira raw-message flattening), and `SUPPORTS_RESPONSE_FORMAT` / `STREAM_INCLUDE_USAGE` consts. Provider-specific `StreamingCompletionResponse` types are replaced by the shared OpenAI one.
- *(openai)* [**breaking**] `ToolChoice` gains a `Function { name }` variant serializing OpenAI's `{"type":"function","function":{"name":...}}` form, so `message::ToolChoice::Specific` with one function is now supported instead of erroring; `CompletionRequest` fields are now public; `OpenAIRequestParams` gains a `supports_response_format` field.
- *(openai)* the shared `TryFrom<message::ToolResult>` conversion now prefers `call_id` over `id` for `tool_call_id` (matching provider-issued call ids); the shared streaming delta accepts `reasoning` as an alias for `reasoning_content` (Groq), and the deprecated `function_call` finish reason maps to tool-call handling.
- *(providers)* behavior notes from the migration: `max_tokens` is now forwarded by deepseek, together, hyperbolic, and azure (previously silently dropped); together's streaming request uses standard `stream`/`stream_options` instead of `stream_tokens`, and a rig-level `ToolChoice::Required` now serializes as `required` instead of erroring; perplexity's non-streaming endpoint drops its stray `/v1` prefix (matching its streaming path and the real API); mira's preamble is sent as a `system` message instead of `user`; `response_format` derived from `output_schema` is deferred while tools are pending a result (groq/mistral/azure previously applied it unconditionally); groq's streaming usage no longer falls back to the legacy `x_groq.usage` envelope.
- *(openrouter)* [**breaking**] de-fork OpenRouter's parallel message model (issue #2035 phase 4): `openrouter::{Message, UserContent, ImageUrl}` are now re-exports of the shared OpenAI types, and the fork's `FileContent`/`VideoUrlContent` are replaced by shared `FileData`/`VideoUrl`. To support this, the shared OpenAI types gain OpenRouter's optional extensions — `UserContent::Video`, `ImageUrl.detail` becomes `Option<ImageDetail>` (OpenAI still sends `"detail":"auto"`), and `Message::Assistant` gains a skip-when-empty `reasoning_details` field, an inbound-only `images` field (never serialized back into requests), plus a deserialize-only `role: "model"` alias. `ReasoningDetails`/`ResponseImage` move into the openai module (re-exported from openrouter). OpenRouter-specific message conversion now goes through `openrouter::messages_from_rig_message`; `TryInto<Vec<openrouter::Message>>` resolves to the plain shared conversion. OpenRouter keeps its own request/response/streaming layer (provider preferences, cost accounting, reasoning-details grouping, generated-image extraction) as a documented exception.
- *(openai)* the `UserContent` audio part now serializes its tag as `input_audio` (matching OpenAI's actual API); `audio` is still accepted when deserializing.
- *(openai)* `StreamingCompletionResponse` is now generic over the provider's streaming usage payload (`StreamingCompletionResponse<U = Usage>`, selected via `OpenAICompatibleProvider::StreamingUsage`), so Mistral's cached-token fallbacks and DeepSeek's cache hit/miss counters survive streaming instead of being narrowed to OpenAI's usage shape.
- *(providers)* pre-migration request filtering is preserved where provider support is unverified: hyperbolic still drops `tools`/`tool_choice`/`output_schema` with warnings, and perplexity flattens text-only message content back to plain strings (mixed multimodal content is passed through for sonar models). llamafile keeps the current mapping of `output_schema` to a `json_schema` response format (as on the shared path since the llamafile migration; modern llama.cpp servers support it).
- *(llamafile)* the chat cassettes are now recorded against an actual llama.cpp `llama-server`, confirming the shared OpenAI wire shape (content-part arrays, tool calls, tool results) against the real llamafile-family server rather than an OpenAI-compatible proxy.
- *(openai)* the assistant tool-call echo now serializes `call_id` (falling back to `id`) so it stays consistent with the tool-result side when history recorded via the Responses API is replayed through chat completions; streaming delta `content` tolerates content-part arrays (Mistral reasoning models) instead of dropping the chunk.
- *(providers)* review fixes: mira and perplexity no longer send `stream_options` (their APIs never received it pre-migration); moonshot rejects a specific forced tool client-side again; openrouter serializes plain assistant reasoning under its documented `reasoning` key; azure telemetry spans report `azure.openai` again; mira usage math saturates instead of overflowing; perplexity strips tool-exchange remnants from shared histories.
- *(providers)* second review round: openrouter tool-result messages prefer the provider-issued `call_id` (matching the assistant echo side); Azure's deployment URL stays pinned to the model the handle was created with (a per-request `model` override only changes the body, as pre-migration); shared streaming spans record `gen_ai.system_instructions` again; providers without tool support (perplexity, mira, and now hyperbolic) sanitize tool-exchange remnants from shared histories via one shared helper that also preserves strict role alternation (tool-call-only assistant turns are dropped and consecutive assistant turns merged); openrouter's dead pre-migration `ToolChoice` type is removed, and `ToolChoice::Specific` with multiple function names now errors client-side for openrouter (the old fork serialized a non-standard array).
- *(moonshot)* [**breaking**] reasoning-only assistant history turns are no longer preserved: the shared conversion drops assistant messages with neither text nor tool calls. Reasoning attached to text or tool-call turns still round-trips via `reasoning_content`.
- *(providers)* [**breaking**] responses with empty assistant content and no tool calls now surface the shared path's "empty response" error for hyperbolic, perplexity, and huggingface (previously they returned an empty text completion).
- *(providers)* [**breaking**] additional removed public items: the raw response types of perplexity, hyperbolic, and huggingface (each module keeps a `CompletionResponse` alias to the shared OpenAI payload; `Message`/`Choice`/`Usage`/`Delta`/`Role` companions are gone), `together::ToolChoice`/`ToolChoiceFunctionKind`, `moonshot::ToolChoice`, `groq::send_compatible_streaming_request` and `deepseek::send_compatible_streaming_request` (use `openai::send_compatible_streaming_request`), and openrouter's `UserContent` builder helpers (`image_url`, `file_base64`, `video_url`, ...) — construct the shared `openai` content variants directly.
- *(openai)* [**breaking**] sending rig `Video` user content to providers on the shared conversion now serializes a `video_url` content part (an OpenRouter/gateway extension) instead of returning a client-side conversion error; providers without video support will reject it server-side.
- *(providers)* [**breaking**] telemetry: migrated providers' streaming spans are now named `chat` with `gen_ai.operation.name = "chat"` (previously `chat_streaming`). GenAI message-content span fields (`gen_ai.input.messages` / `gen_ai.output.messages`) are intentionally left empty instead of recording serialized request/response messages, preserving the privacy/cardinality behavior from #2065; the public `SpanCombinator::record_model_output` helper is removed. `gen_ai.request.model` reports the per-request model override when one applies.
- *(providers)* third review round: history sanitization treats `refusal` parts as text when flattening and, for alternation-strict perplexity, merges consecutive same-role turns (dropping a tool exchange could previously leave `user/user` adjacency its API rejects); streaming no longer overwrites caller-supplied `stream_options`; openrouter's encrypted reasoning details now correlate with the wire tool-call id and its non-streaming usage uses the reported `completion_tokens` (no underflow); base64 videos with unrecognized MIME types round-trip as data-URI URLs instead of failing conversion.
- *(providers)* fourth review round — agent structured output: `GenericCompletionModel` no longer claims native structured output composes with tools for every provider; it now follows `SUPPORTS_RESPONSE_FORMAT`. Agents with tools plus an output schema on deepseek/together/moonshot/huggingface/hyperbolic/perplexity/mira fall back to tool-mode schema enforcement as their pre-migration models did (the migration had silently dropped the schema entirely); groq/mistral/azure now compose natively like openai.
- *(openai)* new `OpenAICompatibleProvider::SUPPORTS_TOOLS` const (default true): perplexity, hyperbolic, and mira set it false and `tools`/`tool_choice` are dropped with a warning during request conversion — before tool-choice validation, so a multi-name `ToolChoice::Specific` no longer errors client-side on providers that ignored it pre-migration.
- *(openai)* streaming robustness: `include_usage` is inserted into caller-supplied `stream_options` instead of being skipped (or clobbering the caller's keys, the pre-migration behavior); a delta carrying both `reasoning_content` and `reasoning` no longer fails as a serde duplicate-field error that dropped the whole chunk; streaming tool-call `index` defaults to 0 when omitted (Mistral marks it optional); `CompletionResponse.object`/`created` are defaulted on deserialization for gateways that omit them (HuggingFace router sub-providers).
- *(openrouter)* non-streaming usage falls back to `total - prompt` (saturating) when the gateway omits `completion_tokens`; streaming spans follow the shared telemetry behavior of leaving GenAI message-content fields empty.
- *(openai)* [**breaking**] `ToolChoice` is now `#[non_exhaustive]`; `GenericCompletionModel`'s `strict_tools`/`tool_result_array_content` fields are private (use the `with_*` builder methods) and the redundant `with_model` constructor is removed (use `new`).

### Removed

- *(derive)* [**breaking**] remove the unused public `rig_derive::ProviderClient` derive macro and its `deluxe` dependency; `Embed` and `rig_tool` are unchanged, and no replacement is provided.
- *(core)* [**breaking**] remove unused `Extractor::{get_inner, into_inner}` and the always-failing `TryFrom<String> for Nothing`; no direct replacements are provided.
- *(core)* [**breaking**] remove the unused public `streaming::stream_completion_to_stdout` helper; use the high-level `agent::stream_to_stdout` helper instead.
- *(core)* [**breaking**] remove the unused public `AudioGeneration<M>`, `ImageGeneration<M>`, and `Transcription<M>` wrapper traits; use the corresponding `AudioGenerationModel`, `ImageGenerationModel`, and `TranscriptionModel` APIs and request builders directly.
- *(core)* [**breaking**] remove the unused `evals` module (`Eval` trait, judge metrics, and builders) along with the `experimental` feature flag that gated it
- *(anthropic)* [**breaking**] remove the unused public `providers::anthropic::decoders` module; Anthropic streaming uses the shared SSE machinery.
- *(providers)* [**breaking**] remove the Galadriel provider integration (`providers::galadriel`), including its client, model constants, environment-variable support, and ignored live tests.

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
