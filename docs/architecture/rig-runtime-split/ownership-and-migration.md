# Ownership matrix, public APIs, conformance, and migration plan

This document maps the source revision
`87f3f5b77a3caeffa10d60225c41e386753bf05e`. “Proposed owner” describes the
ideal runtime boundary. Built-in provider movement is explicitly staged outside
the initial runtime extraction.

Portability classes used below are:

- **contract**: stable provider/backend authoring interface;
- **canonical value**: runtime-neutral data exchanged across a boundary;
- **classic behavior**: progression or extension behavior owned by `rig-agent`;
- **Bevy behavior**: ECS-native progression or extension behavior;
- **facade**: construction or re-export convenience;
- **integration**: provider/store/UI implementation;
- **testing**: fixtures or validators, not production behavior;
- **mixed**: a current module that must split internally.

## Complete public `rig-core` module ownership

The authoritative current module declarations are
[`crates/rig-core/src/lib.rs:152-198`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/lib.rs#L152).
Feature gates shown in the table are current facts.

| Current public module | Current responsibility | Class | Proposed owner/action | Allowed dependencies | Rationale and migration hazard | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `agent` | `Agent`, builder, runner, sans-I/O run, prompt requests, hooks, output mode, agent-as-tool | classic behavior | Move whole module to `rig-agent`; preserve internal layout first | `rig-core`; no Bevy | It is the classic runtime. Moving pieces independently risks parity and retry rollback. | [`agent/mod.rs:148-178`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/mod.rs#L148) |
| `audio_generation` (`audio`) | audio model/request/response contracts | contract/value | Remain in `rig-core` | portable HTTP/value crates | No agent progression dependency. Keep WASM-compatible bounds. | [`audio_generation.rs:22-82`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/audio_generation.rs#L22) |
| `client` | provider construction/capabilities plus model and agent/extractor conveniences | mixed | Keep low-level client/provider traits in core; move `agent()`/`extractor()` to `rig-agent` extension trait | core client must not depend on runtimes | Direct imports of `AgentBuilder`/`ExtractorBuilder` create the reverse dependency. | [`client/completion.rs:1-60`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/client/completion.rs#L1), [`client/mod.rs:111-300`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/client/mod.rs#L111) |
| `completion` | canonical messages/request/response/model plus classic prompt facade/errors | mixed | Keep messages, definitions, `CompletionModel`, requests, `CompletionError`, response, usage in core; move `Prompt`, `Chat`, `TypedPrompt`, `PromptError`, `StructuredOutputError` to `rig-agent` | core contracts only | Prompt traits promise tool execution/history mutation. Current revision has no older `Completion` facade trait. | [`completion/request.rs:87-631`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L87) |
| `embeddings` | embedding model/value/builders and vector-store tool integration | mixed | Keep models/values/builders in core; split automatic agent tool conveniences into runtime adapters | core plus vector contracts; no runtime in portable pieces | Embedding is provider-neutral, but turning retrieval into an executable tool is runtime behavior. | [`embeddings/mod.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/embeddings/mod.rs), [`embeddings/tool.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/embeddings/tool.rs) |
| `extractor` | classic agent-based structured extraction/retry | classic behavior | Move to `rig-agent` | core + classic runtime | It stores `Agent`, builds output-tool mode, attaches hooks, and owns retries. A Bevy extractor can be separate later. | [`extractor.rs:37-79`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/extractor.rs#L37), [`extractor.rs:199-335`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/extractor.rs#L199) |
| `http_client` | portable request/response/SSE/multipart/retry abstraction | contract/integration | Remain in core | portable HTTP dependencies; target-gated transports | Provider integrations need it without a runtime. Preserve target-specific stream bounds. | [`http_client/mod.rs:15-125`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/http_client/mod.rs#L15) |
| `id` | provider message/content identifier helpers | canonical value | Remain in core | none/runtime-neutral | IDs participate in provider round trips, not runtime ownership. | [`id.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/id.rs) |
| `image_generation` (`image`) | image model/request/response contracts | contract/value | Remain in core | portable HTTP/value crates | Independent provider capability. Its raw `Send + Sync` usage should be reviewed separately for WASM consistency. | [`image_generation.rs:14-79`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/image_generation.rs#L14) |
| `integrations` | CLI chatbot and Discord adapters built around classic `Agent` | integration/classic | Move agent integrations to `rig-agent::integrations`; future non-runtime integrations may remain core | `rig-agent` + UI deps | Both current implementations import `Agent`; leaving them core prevents extraction. | [`integrations/mod.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/integrations/mod.rs), [`cli_chatbot.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/integrations/cli_chatbot.rs), [`discord_bot.rs:3`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/integrations/discord_bot.rs#L3) |
| `loaders` | file/PDF/EPUB document loading | integration/value | Remain in core for now; optionally a utility crate later | core document values + gated parsers | Runtime-neutral and unrelated to the split. Do not broaden migration. | [`loaders/mod.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/loaders/mod.rs) |
| `markers` | builder type-state markers | canonical utility | Remain in core | none | Used by low-level request builders. Runtime crates may define their own markers. | [`markers.rs:7-13`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/markers.rs#L7) |
| `memory` | backend contract, errors, filters, demotion/compaction contracts, in-memory backend; docs describe agent timing | mixed | Keep backend/policy contracts and in-memory backend in core; move load-before-run/append-after-commit orchestration and agent-facing docs to each runtime | core messages + WASM compatibility | Backends are portable; commit timing is runtime behavior. | [`memory.rs:85-117`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/memory.rs#L85), [`memory.rs:188-348`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/memory.rs#L188) |
| `model` | model listing values | contract/value | Remain in core | none/runtime-neutral | Provider capability contract. | [`model/mod.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/model/mod.rs) |
| `one_or_many` | non-empty canonical collection | canonical value | Remain in core | serde/schemars only | Used throughout provider message contracts. | [`one_or_many.rs:15`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/one_or_many.rs#L15) |
| `prelude` | client, classic agent, prompt/streaming traits, tools, vectors | facade/mixed | Core gets a contracts prelude; `rig-agent` and `rig-ecs` get runtime preludes; root composes namespaced preludes | each prelude only exports its owner/dependencies | Current glob path hides ownership and would cause runtime collisions. | [`prelude.rs:14-57`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/prelude.rs#L14) |
| `providers` | 25 built-in public provider modules | provider integration | Temporarily remain while runtime extraction lands; ideal follow-up moves them to provider integration crates | core contracts/HTTP only | Provider-neutral ideal conflicts with current packaging, but moving 25 providers with runtimes destroys reviewability. | [`providers/mod.rs:96-121`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/providers/mod.rs#L96) |
| `rerank` | rerank model/request/response contracts | contract/value | Remain in core | WASM-compatible portable dependencies | No runtime progression dependency. | [`rerank.rs:20-75`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/rerank.rs#L20) |
| `streaming` | provider raw stream accumulation plus classic prompt/chat streaming traits | mixed | Keep raw choices, deltas, pause/control, final provider response accumulator in core; move runtime traits/items/requests to `rig-agent` | core side only completion/HTTP | `StreamingPrompt`/`StreamingChat` return classic request types; raw provider streams are shared. | [`streaming.rs:28-261`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/streaming.rs#L28), [`streaming.rs:565-626`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/streaming.rs#L565) |
| `test_utils` (`test-utils`) | mock models/HTTP/streaming/embeddings plus model and runtime conformance | testing/mixed | Keep core provider/model mocks in core test support; move classic driver tests to `rig-agent`; add test-only cross-runtime conformance package | dev-only dependencies | Existing model conformance mixes provider and agent behavior and must split by test level. | [`test_utils/mod.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/test_utils/mod.rs), [`model_conformance.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/test_utils/model_conformance.rs) |
| `tool` | authoring trait/canonical outputs plus context, registry, server, dispatch | mixed | Keep portable authoring/canonical values in core; move current context/registry/server/dispatch to `rig-agent`; add ECS adapters/grants/effects in `rig-ecs` | core contract must not depend on runtimes | Highest-risk internal split; derive macro and vector-tool blanket impls depend on it. | [`tool/mod.rs:111-180`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/mod.rs#L111), [`tool/mod.rs:511-680`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/mod.rs#L511) |
| `transcription` | transcription model/request/response contracts | contract/value | Remain in core | WASM-compatible portable dependencies | Independent provider capability. | [`transcription.rs:18-149`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/transcription.rs#L18) |
| `vector_store` | vector index/search/filter contracts, in-memory store, automatic Tool impl | mixed | Keep store contracts/filters/in-memory store in core; move runtime tool registration/convenience to adapters | core embeddings + tool authoring contract only | Search is portable; tool execution/registration is runtime-specific. | [`vector_store/mod.rs:34-182`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/vector_store/mod.rs#L34) |
| `wasm_compat` | target-dependent Send/Sync/future/stream and timeout | contract/target | Remain in core; runtime crates reuse it where portable and may add stricter native adapters | no Bevy | Prevents ECS/native requirements from weakening WASM provider contracts. | [`wasm_compat.rs`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/wasm_compat.rs) |
| `telemetry` | provider GenAI spans/response metadata plus awareness of classic agent span target | mixed | Keep provider semantic conventions and `ProviderResponseExt` in core; move agent-run span creation/recording to `rig-agent`; Bevy owns ECS run/effect telemetry | provider helpers depend on canonical values; runtime telemetry depends on core | Current `CompletionSpanBuilder` conditionally reuses `rig::agent_chat`, a hidden runtime coupling. | [`telemetry/mod.rs:66-156`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/telemetry/mod.rs#L66), [`telemetry/mod.rs:426-461`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/telemetry/mod.rs#L426) |

Private `json_utils` remains with whichever canonical merge/serialization helpers
need it. Private `provider_response` and the public root re-export
`ProviderResponseError` remain core provider contracts
([`lib.rs:166-191`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/lib.rs#L166)).

## Symbol and responsibility ownership matrix

| Item/symbols | Current owner | Proposed owner | Action | Allowed dependencies | Rationale/hazards/evidence |
| --- | --- | --- | --- | --- | --- |
| `CompletionModel`, `CompletionRequest`, `CompletionRequestBuilder`, `CompletionResponse<T>`, `CompletionError` | `rig-core::completion` | `rig-core` | Remain | HTTP, canonical content, WASM compatibility | Provider-facing boundary; generic `raw_response: T` preserves provider types ([`request.rs:451-458`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L451), [`request.rs:576-631`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L576)). |
| `CompletionModel::composes_native_output_with_tools` | `rig-core::completion`; its docs and only caller reach into classic `OutputMode` policy | retain as a narrowly named provider capability in `rig-core`; move `OutputMode` and resolution to `rig-agent` | Redesign boundary | core boolean capability only; no runtime types or links | The capability reports provider behavior needed by either runtime. Remove the core-to-agent rustdoc link and let each runtime decide policy ([`request.rs:615-629`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L615), [`agent/completion.rs:330-350`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/completion.rs#L330)). |
| `Message`, `UserContent`, `AssistantContent`, tool calls/results, documents/media | `rig-core::completion::message` and root `message` re-export | `rig-core` | Remain | canonical value dependencies only | Both runtimes and every provider map these values. |
| `Usage`, `GetTokenUsage`, completion-call accounting input | `rig-core::completion` | `Usage` remains core; per-run `CompletionCall` records live in each runtime | Split | core values only | Provider usage is canonical; call indexing and commit accounting are progression. |
| `RawStreamingChoice`, `RawStreamingToolCall`, `StreamingCompletionResponse<R>`, `StreamedAssistantContent<R>` | `rig-core::streaming` | `rig-core` | Remain/trim runtime references | completion values + futures | Provider mappings and direct streaming consumers need them. Runtime-specific multi-turn events move. |
| `Prompt`, `Chat`, `TypedPrompt`, `StreamingPrompt`, `StreamingChat` | core completion/streaming | `rig-agent` for current traits | Move | core + classic request types | Their contracts include tool execution and history mutation. Bevy gets handle methods, not duplicate global traits. |
| Older `Completion` facade trait | absent at source revision; present in PR #6 base | none unless a concrete use returns | Do not recreate | n/a | Current main uses `CompletionModel::completion_request`; research prompt's named item is historically relevant but not current public API. |
| `PromptError`, `StructuredOutputError` | `rig-core::completion` | `rig-agent` | Move/split provider forwarding | core errors + memory + classic diagnostics | Variants are max-turn/cancel/tool/runtime states ([`request.rs:140-279`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L140)). |
| `CompletionClient::completion_model` | `rig-core::client` | `rig-core` | Remain | core completion | Low-level capability construction. |
| `CompletionClient::agent`, `CompletionClient::extractor` | `rig-core::client` | `rig-agent::client::CompletionClient` | Move | core client + classic builders | Removes core-to-agent imports while preserving facade ergonomics. |
| OpenAI `GenericCompletionModel::into_agent_builder` and other provider/model agent conveniences | OpenAI defines an inherent method returning `crate::agent::AgentBuilder`; client conveniences also come from `CompletionClient` | blanket `rig-agent::model::AgentModelExt` for `CompletionModel` plus `CompletionClient` for clients | Remove the core inherent method and redesign as runtime extensions | provider code depends on core only; extension implementation depends on core + classic builder | The exact OpenAI method is a hidden provider-to-runtime edge ([`openai/completion/mod.rs:1898-1901`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/providers/openai/completion/mod.rs#L1898)). A blanket model extension preserves `model.into_agent_builder()` without editing each provider or making providers depend on a runtime. |
| `Agent`, `AgentBuilder`, `AgentRunner` | `rig-core::agent` | `rig-agent` | Move without semantic changes | core contracts + tool/memory runtime infrastructure | Primary classic API ([`builder.rs:97`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/builder.rs#L97), [`completion.rs:555`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/completion.rs#L555), [`runner.rs:205`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/runner.rs#L205)). |
| `AgentRun`, `AgentRunStep`, `ModelTurn`, `ModelTurnOutcome`, pending tool calls | `rig-core::agent::run` | `rig-agent` | Move intact | core canonical values | Classic sans-I/O state machine; not a shared runtime engine ([`run/mod.rs:114-280`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L114)). |
| `PromptRequest`, typed/streaming request and response types | `rig-core::agent::prompt_request` | `rig-agent` | Move | classic runner + core values | Runtime builders and final transcript/accounting surface. |
| `AgentHook`, `HookStack`, events/actions, `HookContext`, `RunId`, `Scratchpad` | `rig-core::agent::hook` | `rig-agent` | Move intact | classic runtime + core values | Exact lifecycle and composition semantics are classic behavior. |
| `RequestPatch` | hook module | `rig-agent` | Move intact | core request values | Per-turn patch merge is HookStack behavior, not provider contract ([`hook.rs:579-724`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L579)). Bevy defines policy components/effects instead. |
| response retry (`RetryRequest`, `ModelTurnAction`) | hook/run/runner | `rig-agent` | Move and preserve | classic state machine | Merged PR #2182; retry must rerun request preparation, consume total budget, and rollback rejected content. |
| `OutputMode` and structured-output recovery | `rig-core::agent::run` | `rig-agent` | Move | core schemas/tool definitions | Native format is core request data; choosing mode, synthetic output tool, re-prompting, and fallback are runtime behavior. |
| `Tool`, `ToolEmbedding`, `DynamicTool` metadata, `ToolDefinition` | `rig-core::tool` | narrow authoring contracts in `rig-core` | Split/redesign | WASM-compatible portable bounds | Must be usable by both runtimes. Do not accept `World` or ECS IDs. Context API is unresolved. |
| `ToolOutput`, `IntoToolOutput`, `ToolExecutionError`, `ToolErrorKind`, raw/model presentation | `rig-core::tool` | `rig-core` | Remain | canonical content + error sources | Both runtimes require typed canonical output and failure classification. |
| `ToolContext`, `TypeMap`, result metadata | `rig-core::tool::extensions` | `rig-agent::tool` | Move | classic dispatch only | Mutable type-map is current classic inbound/result context; ECS should use owned effect input and components. |
| `ToolSet`, erased dispatch, `ToolSetBuilder` | `rig-core::tool` | `rig-agent::tool` | Move | core authoring contract | Registry order, execution, parsing, and snapshots are runtime behavior. |
| `ToolServer`, `ToolServerHandle`, live registration/snapshots | `rig-core::tool::server` | `rig-agent::tool::server` | Move | classic tool runtime | Bevy models capability/grant entities and retirement instead. |
| ECS tool entities, grants, policies, effect calls, generations | absent/current PR #6 experiment | `rig-ecs` | New vertical slices | core tool contracts + Bevy | ECS-specific; never core. |
| `VectorStoreIndex`, dyn adapter, filters, top-N values | `rig-core::vector_store` | `rig-core` | Remain | embeddings/canonical values/WASM | Portable backend contract. |
| blanket vector-store `Tool` and dynamic-context automatic integrations | vector/agent modules | runtime adapters | Split | runtime + core index | Advertising and invoking retrieval is runtime behavior. |
| `ConversationMemory`, `MemoryError`, demotion/compaction contracts, in-memory backend | `rig-core::memory` | `rig-core` | Remain | core messages/WASM | Portable storage and policy contracts. |
| memory load/append, conversation IDs on requests, commit timing | agent builder/driver | each runtime | Duplicate behavior under conformance | runtime + core memory | Must append only committed canonical messages and handle errors consistently. |
| `Extractor`, `ExtractorBuilder`, extraction retry loop | `rig-core::extractor` | `rig-agent::extractor` | Move | classic runtime | Current implementation is explicitly agent-based and hook-aware. |
| future Bevy extraction | absent | `rig-ecs` if demanded | Defer | Bevy + core schemas | Do not force a shared extractor abstraction before its semantics exist. |
| `CompletionSpanBuilder`, `ProviderResponseExt`, `SpanCombinator` | `rig-core::telemetry` | provider helpers remain core | Split | core values/tracing | Provider semantic conventions are portable. Remove awareness of classic span target from core. |
| agent run/tool spans and content telemetry | runner + telemetry helper | `rig-agent` | Move | classic events | Runtime-specific lifecycle and accepted/rejected content semantics. |
| ECS schedule/effect/run telemetry | absent/PR #6 experiment | `rig-ecs` | New | ECS state + core provider metadata | Must observe committed authoritative state and stale/late outcomes. |
| provider modules and mapping implementations | `rig-core::providers` | provider integration crates ideally; temporary core | Defer separate decomposition | core only | Runtime split must not rewrite provider behavior or cassettes. |
| HTTP/provider response error helpers | core private module + public error re-export | `rig-core` | Remain | HTTP/serde | Required for provider failures and runtime error forwarding. |
| `rig-derive::Embed` | `rig-derive`, emits core embedding paths | `rig-derive` targeting `rig-core` | Remain/update resolver tests | proc macro only | Portable derive ([`rig-derive/src/embed.rs:50`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-derive/src/embed.rs#L50)). |
| `rig_tool` macro | `rig-derive`, emits `rig_core::tool::Tool` and `ToolContext` | split portable tool expansion and explicit classic contextual mode | Redesign | generated path may target core or agent | Current path resolver tries `rig-core`, then `rig` ([`rig-derive/src/lib.rs:22-34`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-derive/src/lib.rs#L22)); context emission is at lines 691-722. |
| core test models/HTTP/provider validators | `rig-core::test_utils` | `rig-core` test support | Remain | core only | Used by provider mappings and both runtime harnesses. |
| classic run/hook conformance | core tests today | `rig-agent` tests + shared scenario inputs | Move | rig-agent + test-only conformance | Keeps exact current semantics. |
| Bevy runtime conformance | PR #6 experiment only | `rig-ecs` tests + shared scenario inputs | New | rig-ecs + test-only conformance | Tests the same observations through different state. |
| examples | 70 external example files directly import or fully qualify a runtime-bearing `rig`/`rig_core` surface | facade/runtime-specific example directories | Split/update | selected runtime | Examples are public migration evidence; avoid hiding Bevy imports in classic prelude. Complete paths are in the [runtime-import inventory](runtime-import-inventory.md#complete-example-paths). |
| external runtime-importing tests | 256 files: 234 provider tests, 12 other root tests/fixtures, and 10 `rig-derive` tests | core/provider, classic-runtime, facade, and derive targets according to the contract under test | Split/update | the narrow target under test | The exhaustive groups and paths in the [runtime-import inventory](runtime-import-inventory.md#complete-test-import-groups) prevent the crate move from leaving implicit facade/runtime dependencies behind. |
| provider cassettes | root `tests/cassettes` | core provider tests | Remain once per provider mapping | core/provider integration only | Do not rerecord for runtime refactors. Runtime acceptance should reuse scripted or selected cassettes without changing fixtures. |
| root facade and prelude | `rig` glob re-exports core and declares 41 forwarding/composition features | `rig` composes core + classic; `rig::ecs` opt-in | Redesign | may depend on both runtimes/integrations | Facade is the only allowed convergence point. The [complete feature map](runtime-import-inventory.md#complete-root-facade-feature-map) identifies runtime-free and runtime-bearing forwarding. |

## Current companion dependencies

All 18 current companion libraries have a normal dependency on `rig-core`:

| Category | Crates | Target dependency after split |
| --- | --- | --- |
| Provider/model integrations | `rig-bedrock`, `rig-candle`, `rig-fastembed`, `rig-gemini-grpc`, `rig-vertexai` | `rig-core`; runtime-specific examples may separately dev-depend on `rig-agent` or `rig` |
| Vector stores | `rig-helixdb`, `rig-lancedb`, `rig-milvus`, `rig-mongodb`, `rig-neo4j`, `rig-postgres`, `rig-qdrant`, `rig-s3vectors`, `rig-scylladb`, `rig-sqlite`, `rig-surrealdb`, `rig-vectorize` | `rig-core` only for index/value contracts |
| Memory | `rig-memory` | `rig-core` only for memory/message contracts |
| Derive | `rig-derive` | no production Rust dependency; generated paths split as above; dev tests may depend on core/agent |

`rig-bedrock` currently also depends on `rig-derive`. The root `rig` facade is
the only package that should have normal dependencies on companion crates and
both runtimes.

## Proposed public namespaces

The following sketches resolve ownership and collision problems; exact type
names remain subject to implementation review.

### `rig-core`

```rust,ignore
pub mod client;       // ProviderClient, CompletionClient::completion_model
pub mod completion;   // model/request/response/errors/usage
pub mod message;      // canonical message and content values
pub mod streaming;    // raw provider stream values/accumulator
pub mod tool;         // portable Tool contract and canonical output/errors
pub mod embeddings;
pub mod vector_store;
pub mod memory;       // backend contracts, not run orchestration
pub mod telemetry;    // provider semantic helpers
pub mod wasm_compat;
```

There is no `agent`, `hook`, `runtime`, `ecs`, or agent-based `extractor`
module in the target core. If built-in providers remain temporarily, their
module is explicitly documented as transitional integration code.

### `rig-agent`

```rust,ignore
pub mod agent;        // Agent, AgentBuilder, AgentRunner
pub mod run;          // AgentRun and stepping types
pub mod hook;         // AgentHook, HookStack, actions/events/RequestPatch
pub mod prompt;       // Prompt, Chat, TypedPrompt and request/response types
pub mod streaming;    // multi-turn runtime events and facade traits
pub mod tool;         // ToolContext, ToolSet, ToolServer, dispatch
pub mod extractor;
pub mod integrations;
pub mod telemetry;
pub mod prelude;

pub mod client {
    pub trait CompletionClient: rig_core::client::CompletionClient {
        fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel>;
        fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<Self::CompletionModel, T>;
    }
}

pub mod model {
    pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
        fn into_agent_builder(self) -> AgentBuilder<Self>;
    }
}
```

The extension traits are blanket-implemented for core completion clients and
models. The default root prelude imports them, retaining `client.agent(...)`
and OpenAI's `model.into_agent_builder()` spelling without making providers
depend on `rig-agent`.

### `rig-ecs`

```rust,ignore
pub mod components;
pub mod topology;
pub mod schedule;
pub mod effects;
pub mod policy;
pub mod adapters;
pub mod persistence;
pub mod debug;
pub mod runtime;      // local/hosted handles
pub mod prelude;

pub trait EcsCompletionClientExt: rig_core::client::CompletionClient {
    fn ecs_agent(&self, model: impl Into<String>) -> AgentSpec<Self::CompletionModel>;
}
```

Bevy construction should create explicit model/agent/tool/store entities or
specifications consumed by a runtime/world. A handle-based prompt API is normal;
it need not implement classic `Prompt`/`Chat` if doing so hides subscriptions,
entity identity, or scheduling semantics.

### Root `rig`

```rust,ignore
pub use rig_core::{completion, embeddings, memory, tool, vector_store, /* values */};
pub use rig_agent as agent;

#[cfg(feature = "ecs")]
pub mod ecs {
    pub use rig_ecs::*;
}

pub mod prelude {
    pub use rig_core::prelude::*;
    // Re-export only non-colliding classic conveniences deliberately.
    pub use rig_agent::prelude::{Agent, CompletionClient, Prompt, StreamingPrompt};
}
```

The actual facade should prefer deliberate re-exports over the current
`pub use rig_core::*` glob once modules split. `rig::prelude` keeps portable
contract identities and adds classic ergonomics. Contextual classic tools are
explicit at `rig::agent::tool`; `rig::tool` remains portable.
`rig::ecs::prelude` selects Bevy ergonomics. Applications that import both
traits see `agent()` and `ecs_agent()`, not two competing `agent()` methods.

## Raw provider response policy

`CompletionResponse<T>` and `StreamingCompletionResponse<R>` retain typed raw
provider finals at direct core boundaries. A runtime consumes canonical choice,
usage, message ID, and finish metadata for progression. Runtime-specific APIs
may expose the typed final without placing it in serializable state:

- classic streaming already emits a provider-typed final item;
- classic blocking may expose an opt-in typed completion callback/collector;
- Bevy may publish a non-persisted typed subscription/event at the effect
  boundary or return it from a typed local handle;
- hosted/erased paths may expose a documented operator-diagnostics envelope,
  but must not claim it is the provider type.

Do not force arbitrary provider responses through `serde_json::Value`, store
them in ECS domain snapshots, or make canonical progression depend on them.

## Shared behavioral conformance

### Test layers

| Layer | Runs where | Purpose | Examples |
| --- | --- | --- | --- |
| Core value/unit | `rig-core` | canonical serialization, request validation, usage math, tool output/error values, WASM bounds | message round trips, tool definition ordering rules, raw stream accumulation |
| Provider mapping/cassette | once per provider integration against core | provider request conversion and raw response conversion only | existing `tests/providers/*/cassette` and `tests/cassettes/*` |
| Shared runtime conformance | in both `rig-agent` and `rig-ecs` via dev-only harness | observable orchestration invariants | scenario list below |
| Provider-backed runtime acceptance | small provider/runtime matrix | catches integration assumptions not represented by scripts | OpenAI Responses, Anthropic Messages, Gemini Generate/Interactions; blocking and streaming |
| Runtime-specific | owner crate only | extension/scheduler behavior that should differ | HookStack merge tests; Bevy schedule/relationship/stale-generation tests |

The conformance package contains no production runtime implementation and does
not define a public runtime trait. It provides a scripted `CompletionModel`,
scripted tools/stores, controllable futures/effects, scenario inputs, canonical
expected transcripts/events, and a private dev-only harness interface.

### Required reusable scenarios

| Scenario | Shared observations/invariants | Current classic evidence | Bevy-specific proof needed |
| --- | --- | --- | --- |
| Model-call budgets | zero rejects initial call; N permits exactly N total calls including retries/continuations | [`run/mod.rs:1820-1879`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L1820) | operation entities cannot dispatch after budget; late result cannot bypass budget |
| Canonical transcript validity | role order valid; no empty synthetic turn; rejected response retained only where diagnostic contract requires | run and runner history tests; PR #2182 empty retry hardening | committed transcript query and snapshot validate roles/order |
| Tool-call/result pairing | every committed tool call has exactly one matching result; parallel results commit in call order | [`runner.rs:4096-4479`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/runner.rs#L4096) | call entities correlate by stable ID/generation; arrival order irrelevant |
| Usage and call accounting | one `CompletionCall` per billed completed model operation; aggregate usage includes rejected/retried calls where reported | [`run/mod.rs:887`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L887), streaming record tests | effect completion commit is idempotent; duplicate/late completions do not double count |
| Invalid-tool recovery | Fail/Retry/Repair/Skip/Stop; suppressed calls never execute | [`run/mod.rs:1884-2037`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L1884), streaming parity tests | policy decisions bind to immutable advertised tool snapshot |
| Response retry and rollback | only tool-free turns retry; corrective feedback/history correct; request preparation/hooks rerun; budget consumed | [`hook.rs:940-949`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L940), [`run/mod.rs:1673-1739`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L1673) | policy/effect vertical slice rolls back provisional state and creates a new model operation |
| Stop/cancellation | stop terminal; no later dispatch/commit; diagnostics preserved; cancellation distinct from cleanup | runner stop tests and `PromptCancelled` | cancel components prevent dispatch, late results classified, terminal state externally observable before cleanup |
| Structured output/output mode | Native/Tool/Prompted/Auto behavior, required fields, bounded recovery, best-effort exhaustion | [`output_mode.rs:12-54`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/output_mode.rs#L12), [`run/mod.rs:2404-2618`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L2404) | explicit output state/policy; provider constraints and synthetic terminal tool modeled without hidden classic runner |
| Memory load/append | load before first request; append only committed new messages; failure mapping; no append on stop/error | memory contract plus `drive_agent` | store effects correlated; persistence completion and cancellation semantics explicit |
| Blocking/streaming terminal parity | same committed history, usage, final content, errors, and stop behavior | shared `drive_agent`; [`runner.rs:2422`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/runner.rs#L2422) and parity suite | same schedules/state; blocking ignores deltas while streaming subscribes |
| Provider-final exposure | typed provider final observable when completed; provider error after a final suppresses false success | [`runner.rs:2102-2145`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/runner.rs#L2102) | typed local/effect side channel; non-persisted hosted policy documented |
| Provisional streaming output | deltas can be observed before terminal; rejected/stopped provisional output is never committed as accepted final | stream finish ordering/stop tests at [`runner.rs:2008-2044`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/runner.rs#L2008) | subscription events distinguish provisional from committed; rollback/cancel terminal emitted |
| Tool execution suppression | hook/policy skip, invalid peer, output-tool finalization, or cancellation prevents body execution | [`run/mod.rs:2037`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L2037), runner skip tests | no effect request entity or request marked suppressed before executor sees it |
| Concurrency | bounded tool concurrency; transcript and events deterministic; terminal action drains/cancels defined siblings | [`runner.rs:4280-5084`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/runner.rs#L4280) | multiple operation entities, deterministic batch commit, explicit in-flight drain policy |
| Stale-result handling | unavailable in classic process-local driver beyond cancellation; outcome must not mutate newer run state | classic cancellation/serialized state tests | generation/correlation checks reject duplicate, stale, superseded, and foreign-world completions |

### Provider acceptance matrix

At minimum, run these scenarios for both runtimes before `rig-ecs` is supported:

| Provider surface | Blocking | Streaming | Tools | Structured output | Raw final |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI Responses | yes | yes | yes | native + tool | yes |
| Anthropic Messages | yes | yes | yes | tool/prompted | yes |
| Gemini Generate Content or Interactions | yes | yes | yes | native + recovery | yes |

Provider cassette fixtures should not be rerecorded solely because a runtime
changes. Low-level requests must stay byte/semantically equivalent unless a
separate provider behavior change is intended.

## Migration PR DAG

Every PR below leaves an acyclic graph. “Compatibility” is optional and must not
violate dependency direction. Because breaking changes are allowed, a clean
break is preferable to a shim that makes `rig-core` depend on `rig-agent`.

### PR 1 — architecture package and conformance fixture skeleton

- **Prerequisites:** architecture decision accepted.
- **Scope:** land this package; create an unpublished test-only conformance
  package with scripted core model/value fixtures and no runtime adapter.
- **Source/API changes:** documentation and test support only.
- **Tests:** JSON parse, Mermaid validation, graph acyclicity, source-anchor
  validator, fixture unit tests.
- **Compatibility:** none needed.
- **Rollback:** remove docs/test-only package.
- **Risks:** freezing incorrect observations. Mitigate by deriving each scenario
  from current tests and maintainer review.
- **Completion:** ownership accepted; scenario ledger names current oracle tests;
  no production dependencies on the conformance package.

### PR 2 — split portable tool contracts from classic execution infrastructure

- **Prerequisites:** PR 1; maintainer decision on context-free versus contextual
  tool authoring and macro behavior.
- **Scope:** isolate `Tool` authoring metadata/call boundary, `ToolDefinition`,
  `ToolOutput`, and canonical errors from `ToolContext`, `ToolSet`, server,
  snapshots, dispatch, and concurrency.
- **Moves/API:** create internal/classic-owned modules without yet moving crates;
  make dependencies point from classic infrastructure to portable contracts.
- **Tests:** typed output/error tests, argument parsing, result metadata,
  registry ordering, live replacement/snapshot tests, WASM compile tests,
  derive trybuild tests.
- **Compatibility:** temporary re-exports may preserve paths inside the same
  crate; no new runtime-neutral type map.
- **Rollback:** revert module split while fixtures remain.
- **Risks:** breaking derive output, dynamic tools, RMCP, vector tool blanket
  impls, agent-as-tool. Address all in this PR or explicitly keep a same-crate
  bridge.
- **Completion:** portable module imports no registry/server/context execution;
  classic tests are behavior-identical.

### PR 3 — remove core-to-classic dependencies by dependency inversion

- **Prerequisites:** PR 1; PR 2 where tool types intersect.
- **Scope:** split low-level completion/streaming/client/telemetry/memory from
  classic prompt/extractor/orchestration types while still in `rig-core` source
  tree if useful.
- **Moves/API:** introduce classic extension trait for `agent()`/`extractor()`;
  relocate `PromptError`, prompt/chat traits, runtime stream items, extractor,
  agent integrations, agent telemetry, and OpenAI's inherent
  `into_agent_builder()` behavior behind an internal classic boundary.
- **Tests:** compile-fail dependency checks, all current agent unit tests,
  provider direct completion tests, WASM builds, rustdoc.
- **Compatibility:** same-crate re-exports can preserve paths temporarily.
- **Rollback:** revert re-export/module changes; no crate movement yet.
- **Risks:** cyclic imports hidden by current crate privacy; raw provider final or
  error forwarding lost during split.
- **Completion:** a dependency scan shows the planned core subset has no imports
  of classic modules; `CompletionClient` only constructs models and no provider
  module returns a classic builder.

### PR 4 — create `rig-agent` and move the classic runtime without semantics changes

- **Prerequisites:** PRs 2-3.
- **Scope:** move `agent`, classic prompt/streaming, tool infrastructure,
  extractor, integrations, and runtime telemetry into `rig-agent`.
- **Moves/API:** source moves plus import rewrites only; retain builder/run/hook
  behavior and serialization formats during this PR.
- **Tests:** move and run all classic unit tests; run shared conformance adapter;
  compare serialized `AgentRun` golden fixtures; run blocking/streaming parity,
  response retry, hook stack, concurrency, memory, and output-mode suites.
- **Compatibility:** root `rig` re-exports classic paths. `rig-core` must not
  re-export `rig-agent` because that creates a cycle.
- **Rollback:** revert crate/source move as one commit/PR.
- **Risks:** accidental semantic cleanup during movement; missing test-only
  visibility. Prohibit refactors beyond imports/build plumbing.
- **Completion:** `rig-agent -> rig-core`, never reverse; complete classic test
  suite is green with no changed behavioral expectations.

### PR 5 — finalize agent construction and extractor conveniences

- **Prerequisites:** PR 4.
- **Scope:** publish `CompletionClient`; move/document extractor and agent-as-tool
  conveniences; publish blanket `AgentModelExt`; resolve provider-specific
  examples.
- **Moves/API:** core `CompletionClient` loses runtime methods; classic prelude
  exports the client and model extension traits; OpenAI loses its inherent
  `into_agent_builder()` and receives the same spelling from `AgentModelExt`;
  direct `rig-agent` usage is documented.
- **Tests:** trait method resolution with core-only, agent-only, and combined
  dependencies; explicit OpenAI model construction; extractor retry/usage/hooks;
  provider client/model type matrix.
- **Compatibility:** root facade keeps `client.agent()` via prelude. Explicit
  imports change from `rig_core::client::CompletionClient` to the agent trait.
- **Rollback:** temporarily restore same method only if it can be implemented
  without core depending on agent; otherwise breaking rollback is impossible.
- **Risks:** method ambiguity and provider crates accidentally depending on
  runtime.
- **Completion:** every provider client and completion model gets classic
  construction by blanket extensions; core-only builds and provider source
  cannot name `AgentBuilder`.

### PR 6 — update root facade, features, and preludes

- **Prerequisites:** PR 5.
- **Scope:** root dependencies/features/namespaces; deliberate re-exports.
- **Moves/API:** `rig::agent`, default `rig::prelude`, placeholder feature wiring
  for future `rig::ecs`; stop relying on an undifferentiated core glob where it
  causes collisions.
- **Tests:** feature powerset for default/core-only/agent, doc examples, facade
  path compile tests, no duplicate method candidates.
- **Compatibility:** classic remains default. No Bevy types in default prelude.
- **Rollback:** restore previous facade exports while keeping crate direction.
- **Risks:** feature unification and docs.rs all-features collisions.
- **Completion:** facade graph matches the decision and remains acyclic under all
  feature combinations.

### PR 7 — update derive macros, companion crates, examples, and tests

- **Prerequisites:** PR 6 and final tool authoring decision.
- **Scope:** generated paths, trybuild cases, companion examples, root examples,
  tests, READMEs, crate docs.
- **Moves/API:** `Embed` continues targeting core; tool macro targets portable
  tool contract or explicit classic contextual contract; imports name runtime.
- **Tests:** all trybuild cases including renamed `rig-core`/`rig` dependencies;
  example checks; companion crate tests; docs; WASM example build.
- **Compatibility:** macro diagnostics explain migrations. Do not silently infer
  unrelated user types named `ToolContext`.
- **Rollback:** revert macro and call-site changes together.
- **Risks:** proc-macro path resolution and downstream contextual tools.
- **Completion:** no generated path points at moved core symbols; all workspace
  examples select an explicit runtime; cassettes unchanged.

### PR 8 — create minimal `rig-ecs` boundary

- **Prerequisites:** PR 7; Bevy MSRV/target policy accepted.
- **Scope:** new crate with dependency on core and Bevy, feature-gated facade
  namespace, empty/minimal installer and documented experimental status.
- **Moves/API:** no agent behavior yet; define crate boundaries and compile-time
  prohibited dependency checks.
- **Tests:** native compile, target policy checks, facade feature compile, graph
  acyclicity.
- **Compatibility:** opt-in only.
- **Rollback:** remove feature/crate without affecting classic/core.
- **Risks:** premature public API. Keep surface minimal and concrete.
- **Completion:** `rig-core` dependency tree has no Bevy; neither runtime depends
  on the other.

### PR 9a — ECS topology, identity, and deterministic schedule skeleton

- **Prerequisites:** PR 8.
- **Scope:** agent/model/run/call/capability components and relationships, stable
  IDs, schedule labels/sets, progress/quiescence, local world driver.
- **Moves/API:** spawn/query handles with no real provider effects.
- **Tests:** deterministic ordering, foreign/stale handle rejection, tenant/grant
  isolation skeleton, explicit `ApplyDeferred` boundaries, quiescence/livelock.
- **Compatibility:** experimental Bevy API may break.
- **Rollback:** remove vertical slice without core/classic effect.
- **Risks:** central mega-components or query-order dependence.
- **Completion:** authoritative state exists only in ECS; schedule progression is
  deterministic under shuffled insertion order.

### PR 9b — owned model effects, streaming, and terminal commit

- **Prerequisites:** PR 9a.
- **Scope:** adapt `CompletionModel`; owned request effects; completion ingress;
  usage/call accounting; stream subscriptions; terminal state.
- **Moves/API:** local/hosted run handles; typed raw-final side channel decision.
- **Tests:** out-of-order/duplicate/late/stale effects, provider errors,
  cancellation, model budgets, provisional streaming, blocking/streaming parity.
- **Compatibility:** experimental.
- **Rollback:** retain topology, remove effect adapter/systems.
- **Risks:** async tasks borrowing world, unbounded queues, false terminal
  exposure, provider response erasure.
- **Completion:** no ECS borrow enters a future; each completion validates stable
  identity/generation and commits once.

### PR 9c — tools, grants, policy, invalid calls, and concurrency

- **Prerequisites:** PR 9b and portable tool contract.
- **Scope:** tool capability/grant entities, immutable turn snapshots, tool
  effects, policy components/systems, structured output recovery, parallel
  batches.
- **Moves/API:** ECS-native policy/install extension surface, not HookStack.
- **Tests:** tool pairing/order, replacement/retirement, grant/tenant isolation,
  invalid recovery, response retry, suppression, concurrency, simultaneous
  stop/cancel, output modes.
- **Compatibility:** experimental; no hook adapter advertised as native policy.
- **Rollback:** remove tool/policy slice; model-only runs remain.
- **Risks:** recreating registries/callback stacks in components, executing a
  different tool than advertised, nondeterministic terminal winner.
- **Completion:** conformance tool/retry scenarios pass; policies are ECS data and
  systems with explicit order.

### PR 9d — stores, memory, persistence, debugging, and cleanup

- **Prerequisites:** PR 9b; can run parallel with later parts of 9c where safe.
- **Scope:** vector/memory/store adapters, load/append behavior, explicit domain
  snapshots, restoration, retirement, explanation/debug views, cleanup.
- **Moves/API:** store capability entities; stable serialized records never raw
  `World` serialization.
- **Tests:** memory conformance, crash/restart fixtures, missing implementation
  rebinding, snapshot determinism, result retention before cleanup, cancellation
  and late effects, redacted secrets.
- **Compatibility:** versioned experimental snapshot schema.
- **Rollback:** disable persistence/store features independently.
- **Risks:** persisting runtime-only handles/tasks/provider raw responses;
  cleanup racing observers.
- **Completion:** restored domain reproduces canonical state after explicit
  adapter rebinding; no secrets or raw `Entity` values in snapshots.

### PR 10 — full shared conformance and provider acceptance

- **Prerequisites:** 9b-9d; all required features implemented.
- **Scope:** run every shared scenario against both runtimes and selected
  provider-backed acceptance matrix.
- **Moves/API:** none unless a confirmed contract gap requires a separately
  reviewed correction.
- **Tests:** full scenario and matrix tables above; runtime-specific suites;
  WASM/native policies; docs examples.
- **Compatibility:** divergences must be named and approved, not hidden by
  weakening assertions.
- **Rollback:** keep `rig-ecs` experimental if gates fail.
- **Risks:** overfitting harness to classic API or accepting indirect evidence.
- **Completion:** every shared scenario passes or has an explicit, documented,
  maintainer-approved runtime-specific classification.

### PR 11 — support and default-readiness decision

- **Prerequisites:** PR 10 and at least one release cycle of experimental use.
- **Scope:** governance/status docs and feature stability, not automatic default
  switch.
- **Tests/evidence:** conformance history, provider CI reliability, fuzz/stress
  results for stale/concurrent effects, snapshot compatibility, issue backlog,
  security/tenant audit, compile/MSRV metrics.
- **Compatibility:** classic remains default unless a separate decision changes
  it.
- **Rollback:** retain experimental status.
- **Risks:** declaring support from API shape rather than operational evidence.
- **Completion:** maintainers explicitly label `rig-ecs` experimental or
  supported and publish the evidence; default eligibility is a later ADR.

## Per-step acyclicity invariant

At every migration step:

```text
provider/store/memory integrations -> rig-core
rig-agent -> rig-core
rig-ecs -> rig-core
rig -> rig-core + selected runtimes + integrations
```

Temporary compatibility may live in the root facade or within the crate that
owns the implementation. It may not be implemented by adding
`rig-core -> rig-agent`, `rig-agent -> rig-ecs`, or `rig-ecs -> rig-agent`.

## Completion audit for the future migration

The split is not complete merely when crates compile. Completion requires:

1. every module row above has the proposed owner or an explicit approved defer;
2. `cargo tree` proves the prohibited edges absent across all features;
3. core and classic WASM guarantees are unchanged unless separately approved;
4. classic tests and serialized run fixtures show semantic preservation;
5. all shared conformance scenarios pass both runtimes;
6. provider cassettes show no unrelated rerecording/churn;
7. facade/macro/docs/examples resolve to intended namespaces;
8. Bevy stale/duplicate/late/cancel/concurrency/persistence tests pass;
9. independent full-diff review finds no hidden shared orchestration engine;
10. runtime support status and unresolved divergences are documented.
