# Single-architecture plan: retiring the classic trait runtime

**Status: PLAN — no implementation.** Branch `data-oriented-migration` (PR #2228),
validated against the tree at `d1e7e49`+. This document catalogs every remaining
generic trait bound, behavioral trait, associated type, trait object, `Any`
usage, boxed future/stream, typestate marker, and callback-serving lifetime in
rig-core and rig-agent, classifies each, and lays out a phased removal plan
whose endgame is **one** execution architecture: `ProviderConfig` /
`EmbedderConfig` + `Runtime`, `AgentSession` / `AgentStream` over the sans-IO
`AgentRun` + `prepare_request`, owned events, and host-supplied decisions.

Method: four parallel audits (client/provider layer; rig-core data/streaming/
tools/wasm; rig-agent; consumer inventory), each validated definition-by-
definition against source with file:line anchors, followed by cross-checks of
the full `dyn`/`Any`/`Box<dyn Fn` census (`Box<dyn|Arc<dyn`: 142 sites;
`std::any`: 3 files; function-pointer tables: none).

---

## 1. The target layer is already clean

`crates/rig-agent/src/{provider,session,stream,extract}.rs`, `agent/run/`,
`agent/prepare.rs`, `agent/config.rs` contain **zero** trait objects, zero
`Any`, zero boxed futures, and no behavior-bounding generics. The only
residual generics are R3-style leaf functions over caller-owned `T`
(`extract<T>`, `top_n_as<T>`) — sanctioned. These modules survive untouched as
THE architecture. Everything below is about deleting the parallel classic
layer and closing the feature gaps that keep it alive.

## 2. Catalog verdicts

### 2.1 REMOVE — runtime polymorphism / erasure

| Construct | Definition | Classification | Replacement |
|---|---|---|---|
| `AgentHook` + `DynAgentHook` + `HookStack` (`Vec<Arc<dyn DynAgentHook>>`) + `HookContext`/`Scratchpad` (TypeId map) + `StepEventKind` + rewrite frames | rig-agent `agent/hook.rs:920,1043,1172,303,157` | runtime polymorphism + Any-erasure over an already-data decision vocabulary | host `match` on `SessionEvent`/`AgentStreamItem` + `SessionPolicy`; **keep** the decision enums, `RequestPatch`, and the closure-free `fold_*`/`ToolCallResolution` helpers (promote to `agent::mod` re-exports) |
| `ErasedTool`, `ErasedEmbeddingTool`, `DynamicTool`/`DynamicCallback`, `ToolSet` dispatch half, `ToolServer`/`ToolServerHandle` (`Arc<RwLock<…>>` registry), `Agent::into_tool` | rig-agent `tool/mod.rs:293,502,373; tool/server.rs:128,215; agent/tool.rs:26` | runtime polymorphism | host-executed tools over `ToolCatalog` + `PendingToolCall`; `router_support::{execute_typed,dispatch_pending}` is the promoted seed; sub-agent-as-tool = host composes an inner `AgentSession` |
| `ToolContext` / `TypeMap` / `AnyMap` | rig-agent `tool/extensions.rs:10-294` (`HashMap<TypeId, Box<dyn AnyClone>>`) | **Any-erasure** | delete; MCP's typed payloads are concrete data in rig-mcp (`McpCallOutcome{result, raw}`); host closes over what it needs |
| `tool/rmcp.rs` (1 979 lines: McpTool, handler, generation tokens) | rig-agent | runtime polymorphism + registry machinery | **already replaced**: `crates/rig-mcp` `McpToolset` (push→poll `refresh()` is a deliberate behavior change, see §6) |
| `IntoToolOutput` blanket impl with 3-way `&dyn Any` sieve | rig-core `tool/output.rs:157-235` | Any-as-specialization | `#[rig_tool]` knows the concrete output type at expansion: emit `ToolOutput::from(…)`/`serialize_to_tool_output(value)` free fn; delete the blanket + `Any` |
| `PortableTool` trait + `PortableToolEmbedding` + `ToolSchema`/`embeddings/tool.rs` | rig-core `tool/portable.rs:19,48` | compile-time capability trait | collapse onto `PortableDynamicTool` records; `#[rig_tool]` generates a constructor fn (generated inherent methods); tool-embedding discovery dies per the dynamic-tools→`active_tools` decision |
| classic `Tool` trait (context param delta only) | rig-agent `tool/mod.rs:163` | typed polymorphism | rename/collapse into the portable contract (104 `impl Tool` sites drop `_context`) |
| `ConversationMemory` (`Arc<dyn>` on Agent) + `MessageFilter` + `DemotionHook` + `Compactor` | rig-core `memory.rs:103,188,230,306`; rig-agent `builder.rs:132` | runtime polymorphism; the `WasmBoxedFuture<'a>` method lifetimes are **callback symptoms** | owned events / host decision: load into `with_history`, append `PromptResponse.messages` after `Done`; rig-memory becomes concrete policy structs the host calls (§6 decision) |
| `Prompt`/`Chat`/`TypedPrompt` (GAT)/`StreamingPrompt`/`StreamingChat` | rig-agent `completion.rs:139-158`, `streaming.rs:9-15` | compile-time dispatch (pure sugar) | inherent methods on the (thin, post-plan) `Agent` over `AgentSession`; `extract<T>` for typed |
| `PromptRequest<S>` typestate (`PromptType`/`Standard`/`Extended` + `PhantomData`), `TypedPromptRequest<T,S>`, 4× `WasmBoxedFuture` IntoFuture impls | rig-agent `agent/prompt_request/mod.rs:195-219,694` | typestate + boxed-future portability | delete; always return `PromptResponse`; typed = `extract<T>` |
| `StreamingPromptRequest`, `StreamingResult` (`Pin<Box<dyn Stream<MultiTurnStreamItem>>>`), `MultiTurnStreamItem`, `stream_to_stdout`, `DriveStream<'a>` | rig-agent `agent/prompt_request/streaming.rs:44,286` | boxed-stream portability | `AgentStream::next_item` (unboxed, structural backpressure); `AgentStreamItem` gains serde (gap 7) |
| `AgentRunner` (27 fields; memory + HookStack + ToolServerHandle slots), `drive_agent`, `TurnSource` enum fusion, `drive_tool_calls<F>` | rig-agent `agent/runner.rs:197`, `prompt_request/streaming.rs:388,511,722` | the classic driver (~20k lines, ~88 % tests) | `AgentSession`/`AgentStream` are already the two concrete drivers; the unary/streaming fusion is obsolete |
| `AgentBuilder<ToolState>` typestate (`NoToolConfig`/`WithToolServerHandle`/`WithBuilderTools`, three duplicated `build()`s) | rig-agent `agent/builder.rs:44-61` | typestate whose sole reason is the ToolServer/ToolSet split | plain construction from `AgentConfig` + `ProviderConfig` + `ToolCatalog` |
| `Extractor<T>`/`ExtractorBuilder<T>` | rig-agent `extractor.rs:76,250` | leaf-typed wrapper over the classic runner | `extract::extract<T>` free function (close gaps first, §4) |
| cli_chatbot `CliChat` trait + `Missing/Provided` typestate; discord `DiscordExt` | rig-agent `integrations/` | sugar | concrete struct over `AgentStream`; inherent method |
| model_conformance `Fn(AgentBuilder)->AgentBuilder` closures (10×) | rig-agent `test_utils/model_conformance.rs` | behavior-as-value | plain-data scenario-override struct (`AgentConfig` + `RequestPatch`) |

### 2.2 REMOVE — classic client/provider trait stack (rig-core)

| Construct | Definition | Replacement |
|---|---|---|
| `Client<Ext,H>` + `ClientBuilder<Ext,ApiKey,H>` typestate + `Provider`/`ProviderBuilder` (53 impls) + `ApiKey`/`BearerAuth`/`Nothing` + `DebugExt` (`&dyn Debug` iter) + `VerifyClient` | `client/mod.rs:173,579,231,134,181` | `functions::Config` + `HttpRuntime` + `ApiKeyLocation`; verify = per-provider free fn or descriptor field |
| `Capabilities<H>`/`Capability`/`Capable<M>(PhantomData)` + the 7 blanket capability-client traits (`CompletionClient`, `EmbeddingsClient`, …, `ModelListingClient`, RPITIT) | `client/mod.rs:276,262,259`, `client/*.rs` | `ProviderDescriptor` booleans (as-data capability sheet, already in tree) + `ProviderConfig`/free functions |
| `ProviderClient` (`type Input/Error`, 31 impls) | `client/mod.rs:113` | `Config::new` defaults `ApiKeyLocation::Env`; optional generated `from_env` inherent fns |
| `CompletionModel` (10 impls), `EmbeddingModel` (+`MAX_DOCUMENTS`), `ImageEmbeddingModel`, `TranscriptionModel`/`ImageGenerationModel`/`AudioGenerationModel` (assoc `Response` payload leak), `RerankModel` | `completion/request.rs:378`, `embeddings/embedding.rs:97,180`, modality files | `provider::complete/open_stream`, `functions::embed/transcribe/generate_image/generate_audio/rerank` (all exist); capability method → descriptor field |
| `GenericCompletionModel<Ext,H>` ×2, `GenericResponsesCompletionModel`, `GenericEmbeddingModel` | `openai/completion/mod.rs:1558`, `anthropic/completion.rs:1559`, `responses_api`, `openai/embedding.rs:217` | the functions modules; openai/functions.rs doc already promises "the generic path is retired later" |
| `OpenAICompatibleProvider` (5 assoc consts, 2 heavily-bounded assoc types, five-stage hook pipeline = vtable-as-trait; 17 impls), `AnthropicCompatibleProvider`, `ResponsesProviderExt`, `CompatibleStreamProfile` (+`OpenAICompatibleProfile` PhantomData) | `openai/completion/mod.rs:1424`, `anthropic/completion.rs:40`, `responses_api/mod.rs:1143`, `internal/openai_chat_completions_compatible.rs:158` | consts → descriptor; hooks → straight-line per-provider `build_request_body` code; streaming profile → sans-IO chunk-normalizer state machine. **Load-bearing inversion**: functions modules currently delegate *through* these traits (`openai/functions.rs:91-115`) — must invert first |
| `HttpClientExt` (RPITIT ×3, generic body in/out) + `LazyBytes`/`LazyBody` boxed futures + `ModelLister<H>` | `http_client/mod.rs:124,78`, `client/model_listing.rs:118` | `http_runtime::Transport` exhaustive enum (self-described as "the replacement for the `H` type parameter"); `list_models(cfg, rt)` free fns (**the one missing functions-path capability**) |
| `EmbeddingsBuilder<M,D>` | `embeddings/builder.rs:53` | free `embed_documents` over `EmbedderConfig`/functions + the existing `batching.rs` machinery; buffer_unordered concurrency folds into it |
| `Embed` trait + `TextEmbedder` visitor | `embeddings/embed.rs:65` | `#[derive(Embed)]` emits an inherent `embed_texts() -> Vec<String>`; builder bound becomes a plain argument |
| `SearchFilter` tagless-final trait + `Filter<V>` genericity + `VectorSearchRequestBuilder<F,Q,S>` typestate + modality request-builder typestates + `markers.rs` | `vector_store/request.rs:123,259`, `markers.rs`, `{audio,image,transcription}` builders | concretize `Filter<serde_json::Value>`; per-backend `from_filter(Filter) -> BackendFilter` free fns; `VectorSearchRequest::new(query, samples)` + `with_*` (P7 `CompletionRequestBuilder` precedent); modality requests get plain constructors |
| loader iterator erasure (`Box<dyn Iterator>` ×6) | `loaders/{file,pdf,epub}` | concrete iterator types or eager `Vec` (leaf, low priority) |
| `ToProviderConfig` + `AgentClientExt` bridge (~30 impls) | rig-agent `client.rs:38,348` | transitional by design; dies automatically with `Client<Ext,H>` |

### 2.3 KEEP — with justification

| Construct | Why harmless |
|---|---|
| `OneOrMany<T>` | ordinary container genericity carrying a load-bearing non-empty invariant; pervasive plain data |
| `Box<dyn Error>` error-source variants (8 sites: memory, embed, embedding, vector_store, completion, http_client, modalities, epub) + `ToolExecutionError.source: Option<Arc<dyn Error>>` with downcast | std error-chain idiom, mirrors `Error::source`; not behavior dispatch |
| serde `'de` lifetimes, `PhantomData<fn() -> T>` visitor structs (`json_utils`), `empty_params_as_none` | plain borrowed data in deserializers, already free functions |
| `RankingItem<'a>`, `embed_image<'a>`, `DebugExt`'s `&dyn Debug`, HRTB `for<'a> …From<&'a T>` | plain borrows / formatting; **no lifetime in either crate serves a callback except `ConversationMemory`'s `WasmBoxedFuture<'a>` methods (removed above)** |
| `StreamingResult` box + transport-edge `BoxedStream` (sse.rs:30) | one box at the IO boundary carrying **owned events** (`RawStreamingChoice` is a concrete enum) — the endgame shape; the sans-IO parser removes the generic plumbing above it, not this box |
| `PortableDynamicTool` + `Arc<dyn PortableDynamicCallback>` | **the single sanctioned callback seam**: tool bodies are arbitrary host code — irreducible runtime polymorphism by definition; rig-agent's duplicate (`DynamicCallback`) is deleted in its favor |
| `ProviderConfig`/`EmbedderConfig` x-macro row tables | declarative data tables, not function-pointer tables; deliberate exhaustiveness is the feature |
| `ProviderDescriptor`, `ApiKeyLocation`, `functions::Config`s, `HttpRuntime`/`Transport`, `MockScript`/`MockEmbedder` | target-state data |
| R3 leaf typed helpers: `extract<T>`, `top_n_as<T>`/`insert_as<T>`/`payload_as<T>`, `deserialize_json<T>`, `embed_chunked<B,P>` closure params, `from_documents_with_id_f` | generic *functions* over caller-owned `T`/pure closures; nothing stores the parameter |
| `SearchHit.payload`/`StoreRecord.payload`/`Unknown(Value)`/`additional_params` | `serde_json::Value` as *transport for genuinely schemaless provider/store metadata*, each with a typed leaf escape (`payload_as<T>`), never the universal erased payload |
| `wasm_compat` markers + `timeout` + macros | portability machinery that **evaporates mechanically** as the callback traits above are removed (already absent from the entire session layer); deleting it first is order-of-operations backwards. Endgame residue: `timeout`, `if_wasm!`, transport boxing |
| `extern crate self as rig` (rig-core lib.rs:146) | macro path resolution; removable only after derives emit inherent methods |

## 3. `Any` / vtable census after the plan completes

- `std::any` remaining: **one** — `ToolExecutionError` source downcasting (error idiom).
- `dyn` remaining: transport-edge stream boxes, error sources, `PortableDynamicCallback` (the sanctioned tool-body seam).
- Typestate remaining: none. Function-pointer tables: none (there are none today either).

## 4. Gap list — what the session layer must gain before deletion

1. **Automatic tool execution** (largest): an optional layer over
   session/stream reproducing `drive_tool_calls` semantics — bounded
   `tool_concurrency`, atomic batch commit with lowest-call-index error
   selection, skip→`ToolResult::skipped`, per-tool spans. Seed:
   `router_support::dispatch_pending`. `AgentSession::run()` today refuses
   executable catalogs (session.rs:507).
2. **Event coverage parity**: surface tool-call/tool-result decision points
   (approval/permission-control pattern), completion-response observation,
   delta-observation gating; re-export the `fold_*` decision vocabulary from
   `agent::mod`.
3. **Memory recipe**: load-before/append-after with warn-and-proceed and the
   explicit-history-bypass rule as a documented host recipe or thin optional
   wrapper (exact classic semantics at runner.rs:606-621, 1123-1133).
4. **Telemetry**: `invoke_agent`/`execute_tool` spans, span adoption,
   `record_content_telemetry` gating in session/stream (classic parity).
5. **Per-request overrides**: `RequestPatch` covers fewer knobs than the ~20
   runner setters; add the missing per-run fields or bless config-scoping.
6. **Typed output**: `extract<T>` lacks cross-attempt usage accounting and a
   Native-mode `prompt_typed` equivalent.
7. **Serde**: `AgentStreamItem` needs Serialize/Deserialize (its predecessor
   `MultiTurnStreamItem` has it); `StreamedTurnAssembler` is not Serialize →
   no durable mid-turn stream suspension.
8. **One-liner ergonomics**: `Agent` survives as a thin concrete struct
   (`AgentConfig` + `ProviderConfig` + `Arc<Runtime>` + `ToolCatalog`) with
   inherent `prompt`/`chat`/`stream`/`extract` methods over the session layer
   — this keeps the 224 `.prompt(` test files and ~40 examples mechanical.

## 5. Phased plan

Each phase = one commit, workspace green, cassettes byte-identical
(`cargo test -p rig --features agent,test-utils,derive,mcp --no-fail-fast -- --test-threads=1`),
clippy/fmt/doc clean (`cargo clippy --workspace --all-features --all-targets`,
`cargo test --workspace --all-features --doc`). Cassette YAMLs are never
edited.

**R1 — Close the gaps (additive).**
Ship gap items 1–8 (§4): the tool-execution layer (`session::ToolExecutor` or
inherent `AgentSession::run_with_tools`), event-coverage parity, memory
recipe, telemetry, patched overrides, serde on `AgentStreamItem`, and the thin
inherent-method `Agent`. Port `test_utils/model_conformance.rs` (the
highest-coupling internal consumer, 25 scenarios) onto the session layer with
plain-data scenario overrides. Add `list_models` free functions (the one
missing functions capability). Give rig-vertexai a `functions::Config` face
(precedent: every other companion).
*Deps: none. Risk: telemetry parity is fiddly; verify with the existing span
tests. This phase unblocks everything.*

**R2 — Tool system collapse.**
Delete `tool/rmcp.rs` (→ rig-mcp), `ToolServer`/`Handle`/`Snapshot`
(→ `ToolCatalog`), `Agent::into_tool` (→ host-composed sessions),
`DynamicTool`/`DynamicCallback` (→ `PortableDynamicTool`), `ToolEmbedding` +
`ErasedEmbeddingTool` + `ToolSchema`/`embeddings/tool.rs`, `ToolSet`,
`ErasedTool` (the rmcp/wasm `compile_error!` vanishes with it), `ToolContext`/
`extensions.rs` (the Any map). Retarget `#[rig_tool]` to emit a
`PortableDynamicTool` constructor; collapse classic `Tool` into the portable
contract; delete the `IntoToolOutput` blanket + `&dyn Any` sieve (macro emits
concrete conversion). Migrate the 104 `impl Tool` sites (mostly mechanical:
drop `_context`).
*Deps: R1 (tool execution layer). Risk: rmcp push→poll behavior change (§6).*

**R3 — Hooks and memory inversion.**
Delete `AgentHook`/`DynAgentHook`/`HookStack`/`HookContext`/`Scratchpad`/
`StepEventKind`/rewrite-frames; keep and re-export the decision vocabulary +
fold helpers. Migrate the 44 test `impl AgentHook`s and 17 examples onto
`SessionPolicy` + `SessionEvent` matching (semantic work: the gemini
hook_stress cassettes encode ordering — replay must stay byte-identical).
Invert memory: drop `Arc<dyn ConversationMemory>` from Agent; rig-memory
re-ships its policies as concrete structs the host calls around the run
(design decision, §6); delete `MessageFilter`/`DemotionHook`/`Compactor`
callback traits in favor of owned events.
*Deps: R1 (event parity). Risk: hook_stress ordering; rig-memory rewrite.*

**R4 — Prompting surface.**
Delete `PromptRequest<S>`/`TypedPromptRequest` (+ typestate + boxed
IntoFutures), `Prompt`/`Chat`/`TypedPrompt`/`StreamingPrompt`/`StreamingChat`,
`Extractor<T>`/`ExtractorBuilder<T>` (→ `extract<T>` with R1's usage
accounting), `stream_to_stdout`, integrations' trait/typestate sugar
(cli_chatbot/discord rebuilt on `AgentStream`/inherent methods). ~230 files of
`.agent(…).prompt(…)` call sites move to the inherent methods — mechanical
because R1's thin `Agent` preserves the shape. Migration example:

```rust
// before
let answer = client.agent(GPT_5_2).preamble("…").build()
    .prompt("hello").await?;
// after (R1 thin Agent, inherent method over AgentSession)
let agent = Agent::new(AgentConfig::new().with_preamble("…"),
                       client.provider_config(GPT_5_2));
let answer = agent.prompt("hello").await?;   // inherent, not a trait
```

**R5 — Classic driver deletion.**
Delete `AgentRunner`, `agent/prompt_request/` (`drive_agent`, `TurnSource`,
`DriveStream`, `MultiTurnStreamItem`, `StreamingResult`), collapse
`AgentBuilder<ToolState>` to plain construction. ~20k lines (~88 % tests —
their intents were ported in R1/R3/R4 sweeps).
*Deps: R2–R4. This is the point of no return for the classic runtime.*

**R6 — Provider trait-stack inversion (rig-core).**
Invert the functions→trait delegation: move pure body-building out of
`OpenAICompatibleProvider`/`AnthropicCompatibleProvider` into per-provider
straight-line code parameterized by descriptor + plain options; land the
sans-IO stream chunk-normalizer (deleting `CompatibleStreamProfile`,
`OpenAICompatibleProfile`, `send_compatible_streaming_request`, and the
`HttpRuntime::transport()` leak); then delete the `Generic*Model<Ext,H>`
types, the modality traits (B1–B7), the modality/vector typestate builders,
and `markers.rs`.
*Deps: none on R2–R5 (parallel track possible after R1), but sequencing after
R5 avoids double-migrating tests. Risk: byte-fidelity of the compat dialects
(mistral tool-choice, deepseek flattening) — the cassette suites are the
proof.*

**R7 — Client layer deletion (rig-core).**
Delete `Client<Ext,H>`, `ClientBuilder` typestate, `Provider`/
`ProviderBuilder` (53 impls), `ProviderClient` (31), `ApiKey`/`BearerAuth`,
`Capabilities`/`Capable`/`Nothing`, the 7 capability client traits,
`ModelLister` (→ R1's free fns), `DebugExt`, `VerifyClient`, `HttpClientExt` +
`LazyBody`, `EmbeddingsBuilder` (→ free fn), and `CompletionModel`/
`EmbeddingModel`/modality model traits. rig-agent's `ToProviderConfig`/
`AgentClientExt` bridge dies with it; construction becomes
`Config::new`/`from_env` + `ProviderConfig`. Companions: bedrock/gemini-grpc
already have functions faces + enum arms; candle/vertexai drive `AgentRun` +
their own inherent completion fns (vertexai face from R1).
*Deps: R6. Largest doctest/README sweep.*

**R8 — rig-core leaf cleanups.**
`SearchFilter`/`Filter<V>` concretization + per-backend `from_filter` (13
store crates ripple), `Embed` → derive-emitted inherent method, loader
iterator boxes, `extern crate self as rig` removal, wasm_compat shrink to
`timeout` + macros + transport boxing.

Expected net deletions: ~35–45k lines (classic drivers ~20k, hook machinery
~2.5k, tool server/rmcp ~3k, client layer + 31-provider trait stacks ~8k,
typestate/builders/misc ~3k), against ~3–4k added in R1.

## 6. Design decisions the maintainer must make

1. **Third-party provider extensibility.** With `CompletionModel` gone and
   `ProviderConfig` deliberately closed, out-of-tree providers have no trait
   contract. The in-tree answer (vertexai/candle precedent) is: *drive the
   public `AgentRun` + `prepare_request` protocol directly with your own
   inherent completion functions*. If first-class `client.agent()`-style
   integration for out-of-tree providers is required, the only alternatives
   are "PR a variant into the enum" or re-introducing one erased seam — the
   plan recommends the protocol answer and documents it.
2. **rig-memory's contract.** Deleting `ConversationMemory` orphans the crate
   unless it re-ships as concrete policy structs (`PolicyMemory` etc.) with
   inherent `load/append` the host calls. Recommended; the alternative (keep
   the trait host-side only) preserves one callback trait indefinitely.
3. **MCP push→poll.** `McpToolset::refresh()` replaces rmcp's
   `on_tool_list_changed` live-registry push. Observable behavior change;
   accepted in the rig-mcp design, restated here for sign-off.
4. **Classic `Agent` name.** The plan keeps `Agent` as a thin concrete struct
   with inherent methods (gap 8). Alternative: delete it and bless
   `AgentSession` as the only entry point — cheaper, but turns ~270 files of
   mechanical migration into rewrites.

## 7. What would make complete removal impossible

Nothing found is strictly impossible. The two asymptotes:
- **Zero traits + closed enum + third-party providers** is logically
  unsatisfiable as a trio; §6.1 resolves it by making the sans-IO protocol
  (data in, data out) the extension contract — no trait needed, but
  integration is at the protocol level, not the convenience level.
- **Arbitrary host tool bodies** are code by definition; the single
  `PortableDynamicCallback` seam is the irreducible minimum and is retained
  deliberately.

## 8. Verification (every phase)

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-features --all-targets   # zero warnings
cargo check --workspace --all-targets
cargo test -p rig-core --all-features
cargo test -p rig-agent --all-features
cargo test -p rig --features agent,test-utils,derive,mcp --no-fail-fast -- --test-threads=1
cargo test -p rig --features bedrock,agent,test-utils,derive --test bedrock -- --test-threads=1
cargo test --workspace --all-features --doc
git diff --stat -- tests/cassettes   # must be empty, always
```
