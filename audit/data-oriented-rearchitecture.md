# rig without abstraction machinery: the full data-oriented rearchitecture

Status: design, executable by a maintainer. Supersedes
`audit/generic-bounds-rearchitecture.md` (which only removed the model
parameter) and corrects `audit/generic-bounds.md` where verification found its
numbers wrong (§2). Every anchor below was checked against the working tree on
branch `docs/migrating-0.30-onward` (rig 0.41.0).

> **Revision 2.2 — maintainer direction (2026-07-29).** The `rig` facade
> crate must remain a *pure facade*: re-exports only, no implementation.
> Everything this document places "in the `rig` facade" — the
> `ProviderConfig` enum, `Runtime`, the free `complete`/`open_stream`
> dispatch functions, `AgentSession`, `AgentStream`, `extract<T>` — is
> implemented in `rig-agent` (`src/provider.rs`, `src/session.rs`,
> `src/stream.rs`, `src/extract.rs`) and re-exported unchanged at the
> familiar `rig::provider`/`rig::session`/`rig::stream`/`rig::extract`
> paths. This works without a dependency cycle because the companion
> provider crates (`rig-bedrock`, `rig-gemini-grpc`) depend on `rig-agent`
> only as dev-dependencies; `rig-agent` takes them as optional normal
> dependencies behind `bedrock`/`gemini-grpc` features, which the facade's
> features forward. The migrated classic `Agent` likewise stays in
> `rig-agent`. Read every "facade" placement below through this lens.

> **Revision 2 — maintainer direction (2026-07-28).** Three scope changes
> supersede the original endgame wherever this document says otherwise:
> 1. **The classic runtime is never deleted.** The original P7 ("the
>    deletion") is retired. Instead the classic runtime is *migrated onto*
>    the new data-oriented substrate: `Agent<M>` loses its model generic
>    (holding `ProviderConfig` + `Arc<Runtime>`), its internals are rewired
>    onto `prepare_request` + the facade provider functions, and its
>    ergonomic surface — prompt API, `AgentHook` callbacks, memory backends,
>    tool server — is **preserved** as the classic runtime's convenience
>    layer. The mandate's full rigor applies to `rig-core` and the protocol
>    layer; the classic runtime is *a* consumer of them, kept for users who
>    want batteries.
> 2. **A simple `bevy_ecs` runtime (`rig-bevy`) is a deliverable**, not just
>    a validation gate: a new crate with `AgentConfig`/`ProviderConfig`/
>    `ToolCatalog`/`AgentRun` as components and systems driving the
>    `CallModel`/`CallTools` effects. This is the second runtime the whole
>    rearchitecture exists to enable.
> 3. **Vector-store crates are kept**, but lose all generic trait machinery
>    per §10.3 (pre-embedded queries, concrete inherent methods, no shared
>    trait) — de-genericized, not removed.
>
> Decision rule for everything left open in this document: choose whatever
> yields idiomatic, safe, beautiful Rust, a bug-free future, and the easiest
> `bevy_ecs` integration for `rig-core`. Concretely resolved by that rule:
> the `Tool` trait stays (better diagnostics = fewer bugs); the
> `ToolExecutionError` public downcast API is dropped at P7 unless the test
> port proves a need; §14's phase table is rewritten below to match this
> revision.
>
> **Revision 2.1.** A reverted P1 reconnaissance pass validated the §5.4 and
> §9.1 signatures against the compiler; three details are amended in place:
> `CompletionResponse.provider` is `String` (a `&'static str` field cannot
> satisfy `Deserialize`); `StreamFinal` additionally carries `provider` and
> `model` so the stream→response conversion can fill the normalized fields;
> and the `GetTokenUsage` deletion's full dependent set plus the
> shared-conversion provider-stamping pattern are recorded on the §3.2 row
> and the §14 P1 row.

**Mandate.** `rig-core` and `rig-agent` become data-oriented libraries with no
polymorphism in their public or internal architecture: no `dyn`, no hand-rolled
fn-pointer vtables, no `Any`, no `Value`-as-uniform-payload, no behavior-bounding
generics, no behavior-varying associated types, no RPITIT-for-polymorphism, no
typestate markers, no adapter blanket impls. Variation is enums matched
exhaustively; behavior lives in free functions over data or outside the library.

**Verdict up front.** The mandate is achievable for everything, including
streaming (§9 — the single most important finding: the streaming data protocol
already exists in-tree as `StreamedTurnAssembler` and needs only the removal of
one generic parameter). The exceptions ledger (§16) has five entries; the
widest — the `Tool` authoring contract — retains two forbidden-list
constructs (associated types, RPITIT) and is argued as a codegen-only
contract with a fully-compliant inherent-method fallback, not waved through
as merely "capability-shaped".
The four behavior slots in `Agent` — model, hooks, memory, tools — all leave
the core, each with a concrete replacement below. Two *additional* banned
constructs the prompt did not list were found during verification and are also
resolved here: `ToolContext`'s `Any`-based type map (§7.4) and
`IntoToolOutput`'s runtime `Any` downcasts (§7.5).

---

## 1. Rules adopted

The mandate leaves three boundary questions open. The rules below are the
answers, applied consistently through the rest of the document.

**R1 — Container generics over caller-owned concrete payloads: KEEP.**
A type parameter is compatible with the mandate iff all three hold:
(a) it carries no bounds beyond capability traits (`Clone`, `Debug`, `Serialize`,
`Deserialize`, `PartialEq`); (b) the values are supplied and consumed by the
*same* party (the caller), so the library never varies behavior on it; (c) the
library never stores it behind an abstraction that outlives the call.
`OneOrMany<T>` (`crates/rig-core/src/one_or_many.rs:15`) passes: it has 4
load-bearing public instantiations (`AssistantContent`, `UserContent`,
`ToolResultContent`, `Embedding`, plus marginal `SystemContent`/`Message`/`String`),
its only bound is `T: Clone` on convenience methods, and it is exactly `Vec<T>`
with a non-empty invariant. Replacing it with four concrete copies would
quadruple a data type to remove zero dispatch. Same verdict for the loader
iterators and `TypedPromptResponse<T>`/`ExtractionResponse<T>`.
`CompletionResponse<T>` **fails** the rule — `T` is chosen by provider identity
(`CompletionModel::Response`), which is exactly the abstraction being removed —
so it is deleted (§5.4), not grandfathered.

**R2 — `impl Into<X>` / `impl IntoIterator<Item = X>` argument sugar: KEEP,
argument-position only.** These monomorphize away, cannot be stored, and are
the call-site face of `From`/`TryFrom`, which the mandate's allowed-list keeps.
`AgentRun::new(impl Into<Message>)` (`run/mod.rs:325`) already relies on this
in the zero-generic core the mandate holds up as the target. Removing them
would force `.into()` at several hundred call sites (61 `impl Into<Message>`
/ `impl Into<String>` occurrences in `rig-agent`, ~230 in `rig-core`) for zero architectural
gain. The rule is scoped: argument position, converting **into a concrete rig
type**, never bounding stored state, never appearing in a data-protocol type.

**R3 — Capability-trait generics on free functions: KEEP.** A free function
generic over `serde::Deserialize` / `schemars::JsonSchema` (e.g. structured
extraction, §10.6) is behavior supplied by *derive-generated data schemas*, not
by runtime dispatch; the mandate explicitly allows serde as capability. No such
parameter may be stored in a struct that outlives the call.

Everything else in the forbidden list is applied without exception. In
particular: the run-scoped `Scratchpad` (`agent/hook.rs:157`, a mutex-guarded
`TypeMap`) and `ToolContext` (`tool/extensions.rs:157`, a dual `TypeMap` over
`HashMap<TypeId, Box<dyn AnyClone>>`) are `Any`-based type erasure and are
**deleted**, not ported (§6.5, §7.4).

---

## 2. Corrections to the prior inventory

Verification (grep + read, this session) against `audit/generic-bounds.md`
(GB) and `audit/generic-bounds-rearchitecture.md` (GBR):

| Prior claim | Verified reality |
|---|---|
| GB §1: "17 `M: CompletionModel` sites in rig-core (8 in providers)" | **3** literal bound sites (`completion/request.rs:594`, `:611`, `client/mod.rs:760`); ≤10 under the broadest grep. The rig-agent count of exactly **80** is correct. GBR's "~97 sites" is therefore ~83–90. |
| GBR §2/§6: "~32 providers", "~35 providers" | **25** provider modules under `crates/rig-core/src/providers/` (+1 `pub(crate) mod internal`). 25 `pub type Client<H=reqwest::Client>` aliases (GB said ~60). 17 `OpenAICompatibleProvider` impls; 10 in-core `CompletionModel` impls (the OpenAI-compatible 17 share `GenericCompletionModel`). |
| GBR Move 3: `OpenAICompatibleProvider` "is five `const`s plus two small functions" | **Wrong in a load-bearing way.** The trait (`providers/openai/completion/mod.rs:1409-1527`) has five consts, **two associated types** (`type StreamingUsage` :1440, `type Response` :1451 with a `TryInto<CompletionResponse<Self::Response>>` bound) and **six** methods (`completion_path` :1464, `build_completion_request` :1473, `prepare_request` :1492, `finalize_request_body` :1501, `finalize_request_body_with_options` :1509, `decorate_streaming_tool_call` :1520). "Consts are struct fields in a costume" survives; "the rest is two functions" does not. §12 gives the corrected split. |
| GB §6 Option A: "~6 files outside providers consume the typed raw value" | Undercount: `StreamedAssistantContent::Final` appears in **20 files** outside `crates/rig-core/src` (38 occurrences); `raw_response` in **34** non-core files (mostly cassette tests, plus `rig-candle`, `rig-vertexai`, `rig-bedrock`, `rig-gemini-grpc`). Still zero references in `crates/rig-agent/src` — that claim holds. |
| GB §3: `markers::{Missing, Provided<T>, Nothing}` in `markers.rs` | `Nothing` lives at `client/mod.rs:163`, not `markers.rs`. |
| GB file sizes imply migration scale | Misleading: `runner.rs` production code is lines 1–1181 of 10,191 (**88% tests**); `prompt_request/streaming.rs` production ends at :1631 of 6,740 (**76% tests**); `run/streamed.rs` production ends at :674 of 1,256. The implementation to migrate is ~4,500 lines, not ~18,000 — but the test suites *are* real migration cost (§15). |
| GB §7: `GetTokenUsage` "obtained at `streaming.rs:236`, `agent/completion.rs:697`" | Those are bound declarations; the actual call is `self.response.token_usage()` at `rig-core/src/streaming.rs:310` (and `:417`), plus `drain_stream_usage` (`agent/prompt_request/streaming.rs:193`) and `StreamedTurnAssembler::ingest`'s `Final` arm (`run/streamed.rs:504`). |
| Anchors drifted | `Agent.memory` is `agent/completion.rs:594` (not :597); `StreamingTurnSource` impl at `streaming.rs:1026` (not :1027); `AgentModelExt::into_agent_builder` at `client.rs:49` (not :52); `TurnSource` spans :405–~:470 (not –:450). All other ~90 checked anchors were correct. |

Two facts the prior docs missed entirely, found by this verification:

1. **`ToolServerHandle` is not a server.** It is `Arc<RwLock<ToolServerState>>`
   (`tool/server.rs:238`) — no task, no channel, no message protocol. "Delete
   the tool server" (§7) removes a lock-wrapped map, not an actor system.
2. **The main agent loop is not in `runner.rs`.** Both surfaces share
   `drive_agent` (`agent/prompt_request/streaming.rs:486-673`); blocking
   `AgentRunner::run` (`runner.rs:1118`) merely folds that stream and discards
   items. There is *one* loop to replace, not two.

---

## 3. Verified inventory and disposition

Every generic parameter, trait bound, trait object, and associated type in
`rig-core` and `rig-agent`, with its fate. Verdicts: **DEL** (removed with its
host construct), **ENUM** (replaced by an exhaustively-matched enum), **DATA**
(replaced by plain data + free functions), **KEEP** (kept, with the rule that
justifies it).

### 3.1 Trait objects (the four slots plus everything else)

| Construct | Anchor | Fate |
|---|---|---|
| `Agent<M>.model: Arc<M>` | `agent/completion.rs:560` | **ENUM** — `ProviderConfig` in the `rig` facade (§5) |
| `Agent.hooks: HookStack` = `Vec<Arc<dyn DynAgentHook>>` | `agent/completion.rs:586`, `hook.rs:1169-1171` | **DATA** — event/decision protocol (§6); trait, stack, and erased twin deleted |
| `Agent.memory: Option<Arc<dyn ConversationMemory>>` | `agent/completion.rs:594` | **DEL from core** — trait moves to `rig-memory`; history crosses the boundary as `Vec<Message>` (§8) |
| `Agent.tool_server_handle: ToolServerHandle` (owns `Arc<dyn ErasedTool>`) | `agent/completion.rs:578`, `tool/server.rs:238`, `tool/mod.rs:292` | **DATA** — core holds `ToolDefinition`s; execution is the host's, with a `ToolRouter` derive for ergonomics (§7) |
| `Arc<dyn DynamicCallback>` in `DynamicTool` | `tool/mod.rs:348-378` | **DEL** — host-side execution makes runtime-closure tools a host pattern, not a library type |
| `Arc<dyn VectorStoreIndexDyn>` in builder/tool server | `agent/builder.rs:119`, `tool/server.rs:70` | **DEL** — dynamic tool retrieval becomes host-computed data fed into the per-turn `ToolCatalog` (§7.7); interacts with the separately-planned vector-store removal |
| `Pin<Box<dyn Stream>>` (`StreamingResult`) | `rig-core/src/streaming.rs:223,228` | **ENUM** — concrete `ModelStream` with inherent `async fn next_item` (§9.2) |
| `WasmBoxedFuture` (= `Pin<Box<dyn Future>>`) | `wasm_compat.rs:67,71` | **DEL** — exists only to make traits dyn-compatible; no traits, no boxed futures. `wasm_compat` shrinks to cfg helpers |
| `Box<dyn AnyClone>` in `TypeMap` (`Scratchpad`, `ToolContext`) | `tool/extensions.rs:10-68` | **DEL** — §6.5, §7.4 |
| `Any` downcasts in `IntoToolOutput` | `rig-core/src/tool/output.rs:211-218` | **DATA** — explicit `Json<T>` wrapper + concrete `From` impls (§7.5) |
| `ToolExecutionError` boxed source + downcast API | `rig-core/src/tool/result.rs:194,:207,:262-269,:315-327` | **KEEP-with-justification** — std's `Error::source` signature forces `dyn` at the std boundary; note rig code *does* branch on a downcast (the error-flattening `from_error` constructor, cfg branches at `:194`/`:207`) and exposes a public downcast API — ledgered in full in §16.2, not glossed |

### 3.2 Behavior-varying traits and their associated types

| Trait | Anchor | Fate |
|---|---|---|
| `CompletionModel` (`type Response`, `type StreamingResponse`, `type Client`, `fn make`, RPITIT ×2, `Clone` supertrait) | `rig-core/src/completion/request.rs:338-388` | **DEL** — providers become config structs + free `complete`/`open_stream` functions (§5.2); dispatch is the facade enum (§5.1) |
| `OpenAICompatibleProvider` (5 consts, 2 assoc types, 6 methods) | `providers/openai/completion/mod.rs:1409-1527` | **DATA** — descriptor const + per-provider free functions calling shared helpers (§12) |
| `EmbeddingModel` (+`const MAX_DOCUMENTS`, `type Client`, `fn make`, RPITIT) | `embeddings/embedding.rs:98-170` | **DATA/ENUM** — §10.1; widest blast radius (13 companion crates) |
| `TranscriptionModel` / `ImageGenerationModel` / `AudioGenerationModel` | `transcription.rs:62`, `image_generation.rs:49`, `audio_generation.rs:55` | **DATA/ENUM** — same treatment as completion, §10.2 |
| `RerankModel` (1 implementor) | `rerank.rs:44` | **DATA** — free function on the single provider; facade enum arm when a second appears (§10.2) |
| `VectorStoreIndex` (`type Filter`, generic `top_n<T>`) + `VectorStoreIndexDyn` | `vector_store/mod.rs:84-155` | **DATA** — query-by-embedding data API (§10.3); supersedes the dyn companion |
| `CompletionClient` (`type CompletionModel<Client = Self>`) + siblings | `client/completion.rs:5-27` | **DEL** — die with `CompletionModel`; construction is struct literals / `Config::from_env()` |
| `Capabilities<H>` / `Capability` / `Capable<M>` (7 assoc types, `const CAPABLE`) | `client/mod.rs:259-293` | **DATA** — compile-time booleans become `ProviderDescriptor` fields checked at request-build (§13.2) |
| `Provider` / `ProviderBuilder` (GAT `type Extension<H>`) | `client/mod.rs:231-325` | **DEL** — plumbing for `Client<Ext, H>`; replaced by config structs + the shared executor |
| `ModelLister<H>` (`type Client`, `fn new`, RPITIT; 11 impls) | `client/model_listing.rs:108-127` | **DATA/ENUM** — free `list_models(&ProviderConfig)` in the facade |
| `Prompt` / `Chat` (RPITIT + `impl Into` params) | `rig-agent/src/completion.rs:139-155` | **DEL** — replaced by concrete methods on `AgentSession` (§5.5); no type stores a `dyn Prompt` today, so nothing but call-site sugar is lost |
| `TypedPrompt` (GAT `type TypedRequest<T>`) | `rig-agent/src/completion.rs:157-168` | **DEL** — replaced by `AgentSession::extract::<T>` under R3 (§10.6) |
| `AgentHook` (9 RPITIT methods) + private `DynAgentHook` | `hook.rs:917-1032`, `:1040` | **DEL** — §6. Action/event *types* survive as the protocol vocabulary |
| `Tool` (`const NAME`, `type Args/Output/Error`, RPITIT) | `tool/mod.rs:162-204` | **KEEP-with-justification** — codegen contract consumed only by derives; its associated types and RPITIT are themselves forbidden-list items, so this is a real exception (ledger §16.1) with a compliant inherent-method fallback (§7.2) |
| `ErasedTool` / `ErasedEmbeddingTool` / blanket impls | `tool/mod.rs:292-346, 501-517` | **DEL** |
| `ConversationMemory` (dyn-safe, boxed futures) | `rig-core/src/memory.rs:93-117` | **MOVE** to `rig-memory` unchanged in shape; core loses the trait and the `Arc<dyn>` field (§8) |
| `MessageFilter` / `DemotionHook` / `Compactor` | `memory.rs:178, 220, 296` | **MOVE** with their module to `rig-memory` |
| `TurnSource<M>` (`type Raw`; 2 impls) | `agent/prompt_request/streaming.rs:405-470` | **DEL** — the blocking/streaming seam becomes two concrete session types sharing free functions (§9) |
| `PortableTool` / `PortableToolEmbedding` + blanket bridges | `rig-core/src/tool/portable.rs:19-46`, `tool/mod.rs:206-272` | **DEL** — one `Tool` contract remains, in rig-core, context-free (§7.2) |
| `GetTokenUsage` | `rig-core/src/completion/request.rs:229` (bound at `streaming.rs:236`) | **DEL** — usage becomes a field on the concrete stream-final record (§9.1). Full dependent set (Revision 2.1, compiler-verified): ~31 provider impl files; `telemetry/mod.rs:776-795` (`record_token_usage<U: GetTokenUsage>` → takes `&Usage`); the `OpenAICompatibleProvider` bounds (`openai/completion/mod.rs:1440-1456`, call sites `:1208`/`:1363`/`:2000`) and `internal/openai_chat_completions_compatible.rs:159-161` — wire usage types retarget to `Into<Usage>` bounds (allowed `From`/`Into` capability machinery); `test_utils` mocks (`streaming.rs:38`, `completion.rs`) |
| `InsertDocuments` (generic method over `Embed`) | `vector_store/mod.rs:75-81` | **DATA** — insertion takes `Vec<(serde_json::Value, OneOrMany<Embedding>)>`-shaped concrete records; embedding happens before the store boundary (§10.3) |
| `Embed` (user-implemented text extraction) / `EmbeddingsBuilder<M, D>` | `embeddings/` | **DATA + plain contract** — builder becomes the typed `embed_batches` free fn (§10.1); `#[derive(Embed)]` targets a single sync method with no associated types and no RPITIT — fully compliant, no exception needed (§10.1, §16.1) |

### 3.3 Runtime-generic types (Tier 1/2) — all deleted with their parameter

`Agent<M>` (`completion.rs:551`), `AgentRunner<M>` (`runner.rs:198` — 25 fields,
exactly one generic), `AgentBuilder<M, ToolState>` (`builder.rs:148`),
`PromptRequest<S,M>` / `TypedPromptRequest<T,S,M>` (`prompt_request/mod.rs:209,711`),
`StreamingPromptRequest<M>` (`streaming.rs:291`), `PreparedCompletionRequest<M>`
(`completion.rs:22`), `Extractor<M,T>` / `ExtractorBuilder<M,T>`
(`extractor.rs:76,252`), `StreamingPrompt<M,R>` / `StreamingChat<M,R>`
(`streaming.rs:12,26`), `CompletionRequestBuilder<M>` (`request.rs:594`),
integration wrappers (`cli_chatbot.rs:15-31`, `discord_bot.rs:25,40`);
`CompletionResponse<T>` (`request.rs:213`), `RawStreamingChoice<R>`
(`streaming.rs:70`), `StreamingCompletionResponse<R>` (`streaming.rs:234`),
`StreamedAssistantContent<R>` (`streaming.rs:959`), `MultiTurnStreamItem<R>`
(`prompt_request/streaming.rs:54`), `DriveStream<'a,R>` / `DriveItem<R>`
(`:369,:387`), `StreamedTurnAssembler::ingest<R>` (`run/streamed.rs:388` — the
*only* generic in the run module, used solely to call `token_usage()` in the
`Final` arm at `:504`).

Replacements: `AgentConfig` (§5.3), `AgentSession`/`AgentStream` (§5.5, §9.2),
concrete `CompletionResponse` (§5.4), concrete `StreamedAssistantContent` with
a `StreamFinal` record (§9.1), `extract::<T>` (§10.6).

### 3.4 Construction-time generics (Tier 4/5) — die with their hosts

`Client<Ext = Nothing, H = reqwest::Client>` (`client/mod.rs:173`),
`ClientBuilder<Ext, ApiKey, H>` (`:579`), 25 provider `Client<H>` aliases,
`GenericCompletionModel<Ext, H>` / `GenericEmbeddingModel<Ext, H>`, typestate
markers `Missing`/`Provided<T>` (`markers.rs:7,13`), `Nothing`
(`client/mod.rs:163`), `PromptRequest`'s `Standard`/`Extended` phantom
(`prompt_request/mod.rs:195-197`), `AgentBuilder`'s `NoToolConfig`/
`WithToolServerHandle`/`WithBuilderTools` (`builder.rs:98-125`),
`VectorSearchRequestBuilder<F, Q, S>` (`vector_store/request.rs:253`),
`ChatBotBuilder<T>` (`cli_chatbot.rs:29`). All **DEL**: the replacement
constructors are plain config structs validated at build time with `Result`,
which the mandate prefers over type-level proof. `Standard`-vs-`Extended`
deserves its own note: the typestate controls only the *await return type*
(`String` vs `PromptResponse`, `prompt_request/mod.rs:282-304`); the session
replaces it with two concrete methods (`run()` → `PromptResponse`; callers
wanting a `String` take `.output`).

### 3.5 Kept (rule R1/R2/R3)

`OneOrMany<T>` + iterators; `FileLoader`/`PdfFileLoader`/`EpubFileLoader`;
`ApiResponse<T>` (provider-internal wire envelope, one per provider module,
never crosses the provider boundary); argument-position `impl Into<Message>`/
`impl Into<String>`; `extract::<T>`'s `T: JsonSchema + DeserializeOwned`;
`Json<T: Serialize>` tool-output wrapper (ledgered, §16.5); derive/std
capability traits throughout.

### 3.6 Verdicts the first draft missed (added after adversarial review)

| Construct | Anchor | Fate |
|---|---|---|
| `WasmCompatSend`/`WasmCompatSync` marker traits + blanket impls (~363 bound sites) | `wasm_compat.rs:8-57` | **DEL** — they exist to abstract `Send`/`Sync` across targets for trait objects and RPITIT bounds; with no traits left, plain `async fn` on concrete types compiles on both targets and the markers delete mechanically with their bounds |
| `IntoFuture` impls + `Prompt`'s `-> impl IntoFuture` | `prompt_request/mod.rs:282,294,847,860`, `streaming.rs:1573`, `completion.rs:144` | **DEL** with their host types; session methods are plain `async fn` |
| `PauseControl` (public API) | `rig-core/src/streaming.rs:22` | **DEL** — a pull stream pauses by not being polled (§9.3); its busy-spin defect (`:433-436`) dies with it; public-API removal, listed under §15.1 |
| `AgentClientExt` / `AgentModelExt` | `rig-agent/src/client.rs:26,:49` | **DEL** with the builder chain (§10.4) |
| `InMemoryVectorIndex<M, D>` | `vector_store/in_memory_store.rs:469` | **DATA** — a concrete in-memory store over pre-embedded records per §10.3 |
| `test_utils` (public; `MockCompletionModel` implements the deleted trait) | `rig-core/src/test_utils/`, `rig-core/src/lib.rs:175` | **DATA/ENUM** — scripted mocks at the data boundary: `ProviderConfig::Mock(MockScript)` + `ModelStream::Mock`, feature-gated `test-utils` (§14 P5). Hand-driven `AgentRun` tests need no mock at all — they feed `ModelTurn` values directly, as the run tests already do |

---

## 4. Target architecture

Four layers, dependency order strictly downward. The design's central move is
unchanged from GBR — promote the sans-IO protocol, delete the shells — but the
three unsolved slots and streaming are now designed rather than deferred, and
one hypothesis is amended (§5.3: `CallModel` does *not* grow a full
`CompletionRequest`; a pure request-preparation function is provided instead,
which keeps `AgentRun` independently usable — the composability requirement).

```
rig-core        data + pure functions. Messages, CompletionRequest,
                concrete CompletionResponse, concrete streaming items,
                ToolDefinition, ToolOutput/ToolResult, Usage, errors.
                Per-provider: Config structs, ProviderDescriptor consts,
                free fns build_request/parse_response/SseParser + async
                complete()/open_stream() over a concrete HttpRuntime.
                The context-free Tool authoring contract + #[rig_tool].
                NO traits with associated types, NO dyn, NO Client<Ext,H>.

rig-agent       the sans-IO protocol crate — after this design it performs
                ZERO IO (today's tokio/http deps drop). AgentRun (unchanged
                heart), StreamedTurnAssembler (de-genericized), AgentConfig,
                prepare_request(), ToolCatalog, the hook event/decision
                vocabulary + pure composition helpers, PromptResponse.
                Everything Serialize + Deserialize.

rig (facade)    the batteries. ProviderConfig enum (25 in-core arms +
                feature-gated companion arms) + exhaustive complete()/
                open_stream()/list_models(); ModelStream enum;
                AgentSession (blocking driver) + AgentStream (streaming
                driver); telemetry; integrations. The ONLY crate with a
                closed provider set, and the only place `match provider`
                appears.

companions      rig-memory (ConversationMemory moves here), rig-rmcp/rmcp
                (McpToolset, typed), rig-bedrock / rig-gemini-grpc /
                rig-candle (Config + free fns; fulfilment contract is
                CompletionRequest in / CompletionResponse out, so non-HTTP
                transports are ordinary facade arms), vector stores
                (§10.3), rig-derive (#[rig_tool] + #[derive(ToolRouter)]).
```

Composability check (§13.4 verifies each): `AgentRun` runs without
`AgentConfig` (it already does); `ToolCatalog` is usable without any runner
(it is a `Vec<ToolDefinition>` plus name sets); `rig_core::providers::openai::complete()`
is callable with no agent at all; the facade enum is skippable by any host
that fulfils `CallModel` itself.

---

## 5. The model slot

### 5.1 Dispatch: enum over provider *config* in the facade

Confirmed from GBR with corrected counts, restated here as normative:

```rust
// rig (facade) — src/provider.rs
// Deliberately NOT #[non_exhaustive]: the mandate's exhaustiveness
// requirement extends to hosts outside the facade (an ECS host matching
// provider configs itself, §13.1). Adding a provider is a compile-breaking
// event for every exhaustive matcher BY DESIGN — that is the feature the
// mandate asks for, and #[non_exhaustive] would silently cancel it with a
// forced wildcard arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderConfig {
    Anthropic(rig_core::providers::anthropic::Config),
    Azure(rig_core::providers::azure::Config),
    // ... one arm per in-core provider (25 today) ...
    OpenAi(rig_core::providers::openai::Config),
    Xai(rig_core::providers::xai::Config),
    #[cfg(feature = "bedrock")]     Bedrock(rig_bedrock::Config),
    #[cfg(feature = "gemini-grpc")] GeminiGrpc(rig_gemini_grpc::Config),
    #[cfg(feature = "candle")]      Candle(rig_candle::Config),
    /// Scripted responses for tests — the successor to MockCompletionModel
    /// (which implements the deleted trait; see §3.6 and §14 P5). Plain data
    /// plus an interior-mutable, serde-skipped turn cursor (today's mock
    /// already shares state via Arc, test_utils/completion.rs:171-172);
    /// stated rule: clone SHARES the cursor, deserialize resets it.
    #[cfg(feature = "test-utils")]  Mock(MockScript),
}

pub async fn complete(
    provider: &ProviderConfig,
    rt: &Runtime,
    request: CompletionRequest,
) -> Result<CompletionResponse, CompletionError> {
    match provider {
        ProviderConfig::OpenAi(cfg) => rig_core::providers::openai::complete(cfg, &rt.http, request).await,
        ProviderConfig::Anthropic(cfg) => rig_core::providers::anthropic::complete(cfg, &rt.http, request).await,
        // ... every arm, exhaustively; adding a provider fails to compile
        //     here and in open_stream/list_models until handled ...
        #[cfg(feature = "candle")]
        ProviderConfig::Candle(cfg) => rig_candle::complete(rt.candle_model(cfg).await?, request).await,
    }
}
```

Rationale unchanged from GBR §2 and re-verified: (1) the fulfilment contract is
`CompletionRequest` in / `CompletionResponse` out, which the three non-HTTP
providers satisfy as ordinary arms (`rig-bedrock` completion.rs:212 over
`aws-sdk-bedrockruntime`; `rig-gemini-grpc` completion.rs:44 over tonic;
`rig-candle` model.rs:421 in-process); (2) config arms are serde-able
end-to-end, extending `AgentRun`'s suspend/resume to the full agent definition;
(3) in-core providers already all compile unconditionally
(`crates/rig-core/Cargo.toml:99-129` has no provider features), so the enum
adds no binary size; (4) exhaustiveness makes a missing fulfilment a compile
error. The one real cost stands: an out-of-tree provider cannot add an arm and
instead drives the public `AgentRun`/`prepare_request` protocol directly — it
loses the facade conveniences, nothing else. **No `Custom` arm ships.** GBR
offered a `struct ProviderOps { build: fn(..), parse: fn(..) }` escape hatch;
under this mandate that is a hand-rolled vtable and is rejected. If a real
out-of-tree provider needs first-class facade support, the answer is a PR
adding an arm.

Credentials: config structs hold an `ApiKeyLocation` enum
(`Env(String)` | `Inline(String)`), so a serialized `ProviderConfig` can
reference the environment instead of embedding secrets; `Inline` supports the
current explicit-key path. Serializing an `Inline` key is the caller's choice
and documented as such.

**Live handles: the `Runtime` struct.** "Serde-able end-to-end" applies to
*configuration only*; live transports are deliberately not serializable and
are reconstructed on resume — the same rule the ECS reference analysis states
for runtime-only clients. Verified constraints this must satisfy:
`rig-bedrock` wraps an `aws_sdk_bedrockruntime::Client` built from an
async-loaded credential chain (`rig-bedrock/src/client.rs:29-33, :84-89`),
`rig-gemini-grpc` holds a connected `tonic::transport::Channel`
(`client.rs:66`), and `rig-candle` holds loaded model weights — none is
serde, and none can be rebuilt per call (credential-chain resolution, TLS
connect, multi-second weight loads). They live in one concrete facade struct:

```rust
pub struct Runtime {
    pub http: HttpRuntime,   // rig-core's HTTP executor (covers all 25 in-core arms)
    #[cfg(feature = "bedrock")]     bedrock: BedrockCache,
    #[cfg(feature = "gemini-grpc")] gemini:  GeminiChannelCache,
    #[cfg(feature = "candle")]      candle:  rig_candle::ModelCache,
}
// Three MONOMORPHIC cache structs (config fingerprint → once-initialized
// handle), NOT a shared ClientCache<K, V>: a generic cache would fail rule
// R1's own test (the facade, not the caller, builds and consumes the cached
// values, and they outlive the call), and parameterizing its build step
// would smuggle a stored closure — a one-entry vtable. Each cache instead
// has ONE concrete accessor (rt.bedrock_client(cfg).await,
// rt.gemini_channel(cfg).await, rt.candle_model(cfg).await) whose
// check-then-build-then-insert body names the construction code directly.
// Runtime::new() starts every cache empty; handles are rebuilt on resume.
```

One resume caveat stated plainly: feature-gated arms mean a `ProviderConfig`
serialized by a build with `bedrock` enabled fails to deserialize in a build
without it — an inherent property of feature-gated enum arms, surfacing as an
ordinary serde error rather than silent misbehavior.

### 5.2 Providers: config + descriptor + free functions

Per provider module in `rig-core` (the 17 OpenAI-compatible ones share almost
everything via §12's helper layer):

```rust
// rig-core — providers/openai/mod.rs (shape; every provider mirrors it)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub base_url: String,          // defaulted constructor provided
    pub api_key: ApiKeyLocation,
    pub model: String,
    pub extra_headers: Vec<(String, String)>,
}

pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "openai",
    supports_tools: true,
    supports_response_format: true,
    stream_include_usage: true,
    emits_complete_single_chunk_tool_calls: false,
    composes_native_output_with_tools: true,   // absorbs request.rs:385
    max_embedding_documents: Some(2048),       // absorbs EmbeddingModel::MAX_DOCUMENTS
};

// Pure: no IO, unit-testable against cassette request bodies byte-for-byte.
pub fn build_request(cfg: &Config, req: &CompletionRequest) -> Result<http::Request<Vec<u8>>, CompletionError>;
pub fn parse_response(status: http::StatusCode, body: &[u8]) -> Result<CompletionResponse, CompletionError>;

// Sans-IO streaming: a push-parser struct, same pattern as StreamedTurnAssembler.
pub struct SseParser { /* concrete accumulation state */ }
impl SseParser {
    pub fn ingest(&mut self, bytes: &[u8]) -> Result<Vec<StreamedAssistantContent>, CompletionError>;
    pub fn finish(self) -> Result<Vec<StreamedAssistantContent>, CompletionError>;
}

// IO wrappers over the shared executor (async fn on concrete types — allowed).
pub async fn complete(cfg: &Config, rt: &HttpRuntime, req: CompletionRequest)
    -> Result<CompletionResponse, CompletionError>;
pub async fn open_stream(cfg: &Config, rt: &HttpRuntime, req: CompletionRequest)
    -> Result<HttpModelStream, CompletionError>;
```

`HttpRuntime` is one concrete struct owning the HTTP client. The
`H = reqwest::Client` parameter threaded through 25 aliases exists to swap
backends per target; that becomes cfg-selection *inside* `HttpRuntime`
(`reqwest` native, fetch-based on wasm), deleting the parameter everywhere.
The evidence this decomposition is faithful is unchanged
(`GenericCompletionModel::completion`, `providers/openai/completion/mod.rs:1941-2022`:
pure build :1952, pure path :1981, transport send :1983-1990, pure parse
:1996-2009) — the trait was scaffolding around one pure pair plus a shared call.

### 5.3 `AgentConfig`, and the amendment to "complete the effect"

```rust
// rig-agent — plain data, all Serialize + Deserialize
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct AgentConfig {
    pub name: Option<String>,
    pub description: Option<String>,
    pub preamble: Option<String>,
    pub static_context: Vec<Document>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<serde_json::Value>,
    pub tool_choice: Option<ToolChoice>,
    pub max_turns: Option<usize>,
    pub max_invalid_tool_call_retries: usize,
    pub output_schema: Option<schemars::Schema>,
    pub output_mode: OutputMode,
    pub output_tool_description: Option<String>,
    pub augment_output_preamble: bool,
    pub tool_concurrency: usize,
    pub record_telemetry_content: bool,
}
```

Plain `with_*` field helpers accompany it (ordinary methods on a plain struct
— no builder type, no typestate), covering the fluent style in the examples
below. This is `Agent<M>`'s 17 fields (`agent/completion.rs:551-597`) minus
the four behavior slots. Note what is *not* here: no model (that is the paired
`ProviderConfig` value), no hooks, no memory, no tool handle. In an ECS world
these are naturally *two* components (`AgentConfig` + `ProviderConfig`) plus a
per-entity `ToolCatalog` — small components, per the archetype consideration
in §13.1.

**Amendment to the working hypothesis.** The prompt's §5.1 proposed moving all
request fields into the run so `CallModel` emits a complete
`CompletionRequest`. Verification argues against literally that:

- `RequestPatch.history` (`hook.rs:596`) replaces the *outgoing* history for
  one turn without touching the run's canonical history. If the machine
  emitted final requests, patch application would have to become a
  machine-ingestion step and the machine would need to distinguish canonical
  from effective history — state it deliberately does not carry.
- Tool definitions are per-turn data (`snapshot_tool_defs`,
  `tool/server.rs:474`) whose refresh cadence belongs to the driver; embedding
  them in serialized run state would duplicate the catalog every turn.
- `AgentRun` must stay drivable *without* `AgentConfig` (composability).

So `AgentRunStep::CallModel { prompt, history, turn }` is **unchanged**, and
the completion of the effect happens in a pure free function that any driver —
the facade session, an ECS system, hand-rolled code — calls between
`next_step` and the provider:

```rust
// rig-agent — pure; the de-generified successor of
// build_prepared_completion_request (agent/completion.rs:218-522)
pub fn prepare_request(
    config: &AgentConfig,
    tools: &ToolCatalog,               // plain data, §7.1
    prompt: &Message,
    history: &[Message],
    committed_output_tool: Option<&str>, // read back from AgentRun, run/mod.rs:464
    patch: &RequestPatch,               // merged hook decisions, §6
) -> Result<PreparedRequest, PromptError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedRequest {
    pub request: CompletionRequest,               // concrete, rig-core
    pub executable_tool_names: BTreeSet<String>,  // feeds ModelTurn
    pub allowed_tool_names: BTreeSet<String>,
    pub output_tool_name: Option<String>,         // feeds AgentRun::set_output_tool_name
}
```

`prepare_request` absorbs today's patch-vs-baseline resolution
(`completion.rs:242-265`), tool-choice validation (`completion.rs:500-513`),
output-mode interception, and preamble assembly (`completion.rs:443-449`) —
all already pure computation. The "emit a complete request" property the
hypothesis wanted is delivered one level up: `AgentSession::advance` *does*
build and send the full request; the protocol crate just does not force that
coupling on every driver.

### 5.4 The concrete response

```rust
// rig-core — replaces CompletionResponse<T> (request.rs:213)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionResponse {
    pub choice: OneOrMany<AssistantContent>,
    pub usage: Usage,
    pub message_id: Option<String>,
    pub finish_reason: Option<FinishReason>,   // closes #2090 / #1886
    pub provider: String,                      // descriptor name (String, not
                                               // &'static str: the field must
                                               // Deserialize — Revision 2.1)
    pub model: Option<String>,                 // provider-reported model id
}
```

**Normalization, not a `RawResponse` enum.** The prompt asked for an explicit
evaluation of `enum RawResponse { OpenAi(..), Anthropic(..), … }` versus plain
normalization. Verdict: normalization wins.

- The enum's only benefit is transporting the provider-typed payload through
  the runtime — and the runtime demonstrably never reads it (`raw_response`:
  zero references in `crates/rig-agent/src`, re-verified).
- The enum has real costs: 25 arms of large response types inflate every
  `CompletionResponse` by the maximal variant (these flow through hot code and
  history handling); the three companion providers *cannot appear in it* if it
  lives in rig-core (dependency direction), so it would be partial exactly
  where GBR's HTTP-signature plan failed; and it couples the core's most
  central type to every provider's wire schema.
- The typed payload remains fully reachable without any generic: call the
  provider's own `parse_response` (or `complete`) directly — it is a public
  free function returning provider-typed data on the provider's side of the
  boundary. Of the 34 non-core files that touch `raw_response` today (§2),
  most are cassette tests, which migrate to exactly that (or to the new typed
  fields — most only read usage/finish data anyway); seven are
  companion-crate source files (`rig-bedrock`, `rig-candle`,
  `rig-gemini-grpc`, `rig-vertexai`), which migrate as part of their P4
  provider split.

`FinishReason` is new in P1 (the #2090 ask), defined here so the signature is
complete: `enum FinishReason { Stop, Length, ToolCalls, ContentFilter,
Other(String) }`, normalized per provider inside `parse_response`.

What dies with `T`: the `GetTokenUsage` trait (usage is a field), the
`Serialize + DeserializeOwned` bounds on `type Response`, and the entire Tier-2
cascade (§3.3). The streaming counterpart is §9.1.

### 5.5 `AgentSession` — the facade driver, and what user code looks like

```rust
// rig (facade). Fully concrete; no callbacks anywhere.
pub struct AgentSession {
    pub config: AgentConfig,
    pub provider: ProviderConfig,
    pub tools: ToolCatalog,
    pub policy: SessionPolicy,     // which decision points surface (plain bools)
    run: AgentRun,
    rt: Arc<Runtime>,              // shared live handles, §5.1 — the one non-serde field
    next_patch: RequestPatch,      // consumed by the next turn's prepare_request
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionPolicy {
    pub surface_model_turns: bool,      // emit TurnFinished instead of auto-accepting
    pub surface_completion_calls: bool, // emit BeforeModelCall instead of auto-sending
}

/// Deliberately exhaustive, like AgentRunStep.
#[derive(Debug)]
pub enum SessionEvent {
    /// policy.surface_completion_calls: the turn about to be prepared —
    /// surfaced PRE-BUILD (prompt/history/turn, like today's CompletionCall
    /// event, hook.rs:389-396), because a patch's active_tools/history must
    /// flow through prepare_request to keep the advertised and allowed tool
    /// sets in sync; patching an already-built request would desync them.
    /// Respond with reply_before_call(CompletionCallAction); the session
    /// then builds and sends the patched request.
    BeforeModelCall { prompt: Message, history: Vec<Message>, turn: usize },
    /// policy.surface_model_turns: an accepted, tool-free-or-not model turn.
    /// Respond with reply_turn(ModelTurnAction); Retry is valid only for
    /// tool-free turns and is rejected for tool-bearing ones, exactly as
    /// retry_model_turn does today (run/mod.rs:522). The full provider
    /// response is available via last_response().
    TurnFinished { turn: usize, content: OneOrMany<AssistantContent>, usage: Usage },
    /// Always surfaced: the model called an unknown/disallowed tool.
    /// Respond with resolve_invalid(InvalidToolCallAction).
    InvalidToolCall(InvalidToolCallContext),
    /// Always surfaced: execute these and respond with provide_tool_results(..).
    ToolCallsReady(Vec<PendingToolCall>),
    Done(PromptResponse),
}

impl AgentSession {
    pub fn new(config: AgentConfig, provider: ProviderConfig, rt: Arc<Runtime>,
               prompt: impl Into<Message>) -> Self;
    pub fn with_history(self, history: Vec<Message>) -> Self;
    pub fn with_tools(self, catalog: ToolCatalog) -> Self;

    /// Drive until the next event the policy surfaces. Performs model IO.
    pub async fn advance(&mut self) -> Result<SessionEvent, PromptError>;

    // Decision inboxes — thin wrappers over AgentRun's existing entry points.
    pub fn resolve_invalid(&mut self, action: InvalidToolCallAction) -> Result<(), PromptError>;
    pub fn provide_tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError>;
    pub fn reply_turn(&mut self, action: ModelTurnAction) -> Result<(), PromptError>;
    pub fn reply_before_call(&mut self, action: CompletionCallAction) -> Result<(), PromptError>;
    pub fn patch_next_turn(&mut self, patch: RequestPatch);   // sugar over reply_before_call

    /// Observation parity with today's on_completion_response hook: the last
    /// provider response in full (message_id, finish_reason, provider fields
    /// — §5.4). None before the first turn; cleared when invalid-call
    /// recovery suppresses the response event (response_hook_suppressed,
    /// run/mod.rs:212), preserving today's suppression semantics.
    pub fn last_response(&self) -> Option<&CompletionResponse>;

    /// Convenience: default policy → drives to Done. Errors up front
    /// (PromptError::PromptCancelled) if the catalog advertises executable
    /// tools — a run() caller has nowhere to answer ToolCallsReady, and
    /// silently hanging or panicking is not an option under workspace lints.
    pub async fn run(mut self) -> Result<PromptResponse, PromptError>;
}
```

Before/after for the three canonical users:

```rust
let rt = Arc::new(Runtime::new());

// 1. Tool-less prompt (today: agent.prompt("...").await?)
let out = AgentSession::new(cfg, provider, rt.clone(), "Entertain me!").run().await?.output;

// 2. Tools (today: builder.tool(Add).build(); agent.prompt(..).multi_turn(3).await?)
#[derive(ToolRouter)]                 // §7.3
struct MyTools { add: Add, search: Search }
let tools = MyTools { add: Add, search: Search::new(db) };
let mut s = AgentSession::new(cfg.with_max_turns(3), provider, rt.clone(), prompt)
    .with_tools(tools.catalog());
let response = loop {
    match s.advance().await? {
        SessionEvent::ToolCallsReady(calls) => {
            let results = tools.dispatch_all(&calls, s.config.tool_concurrency).await?;
            s.provide_tool_results(results)?;
        }
        SessionEvent::InvalidToolCall(_) => s.resolve_invalid(InvalidToolCallAction::fail())?,
        SessionEvent::Done(r) => break r,
        _ => continue, // BeforeModelCall/TurnFinished are not surfaced under the default policy
    }
};

// 3. A "hook" (today: impl AgentHook + add_hook; here: a match arm)
//    policy.surface_model_turns = true, then:
SessionEvent::TurnFinished { content, .. } if needs_retry(&content) =>
    s.reply_turn(ModelTurnAction::retry_with_feedback("Return a complete answer."))?,
```

The loop-with-match replaces callback registration everywhere. That is the
mandate's intended shape: control flow is visible in application code, every
decision is a value, and the whole session (minus `rt`) is serializable
between events.

---

## 6. Hooks — the event/decision protocol

This was flagged as the hardest problem. Verification shows it is instead the
most *finished* one, because the machine was already built this way for its
hardest case and the remaining cases follow the same groove.

### 6.1 The decisive facts

1. **Every hook action is already plain data.** `CompletionCallAction`
   (`hook.rs:727`), `RequestPatch` (`:580`), `ModelTurnAction` (`:447`),
   `InvalidToolCallAction` (`:835`), `ToolCallAction` (`:755`),
   `ToolResultAction` (`:797`), `ObservationAction` (`:897`). None contains a
   callback. Missing serde derives are an additive change.
2. **`AgentRun` already ingests hook decisions as data.** The invalid-tool-call
   flow is a literal inbox/outbox: `ModelTurnOutcome::NeedsResolution(InvalidToolCallContext)`
   out (`run/mod.rs:220`), `resolve_invalid_tool_call(InvalidToolCallAction)`
   in (`:934`), with full recovery semantics (fail/retry/repair/skip/stop,
   budget accounting, `ToolChoice::None` rejection) inside the machine.
   Likewise `retry_model_turn(RetryRequest)` (`:511`) *is*
   `ModelTurnAction::Retry` as a data entry point, `cancel_error` (`:575`) is
   `Stop`, and `ModelTurnOutcome::Continue { response_hook_suppressed }`
   (`:212`) already tells a driver when the response event must not fire.
3. **Every remaining hook power is an operation on data the driver already
   holds.** With the driver owning the request (§5.3), tool execution (§7),
   and the delta stream (§9), each hook method maps to a plain transformation
   at a point where the value is in the driver's hands:

| Hook (today) | Power | Where it lands in the protocol |
|---|---|---|
| `on_completion_call` → `Patch`/`Stop` (`hook.rs:922`, fired `runner.rs:594`) | mutate the outgoing request | driver passes a `RequestPatch` into `prepare_request` (§5.3); `Stop` → `run.cancel_error` |
| `on_completion_response` (observe) (`runner.rs:984`) | observe / stop | driver looks at the `CompletionResponse` it just received; on the session surface, `AgentSession::last_response()` (§5.5); suppression rule comes from `response_hook_suppressed` |
| `on_model_turn_finished` → `Retry`/`Stop` (`runner.rs:1007`) | reject a tool-free turn | `run.retry_model_turn(RetryRequest)` — already exists |
| `on_invalid_tool_call` (`runner.rs:952`) | recover invalid calls | `run.resolve_invalid_tool_call` / `resolve_streamed_invalid_tool_call` — already exist |
| `on_tool_call` → `Rewrite`/`Skip`/`Stop` (`runner.rs:695`) | rewrite args, veto execution | the driver executes tools: rewriting = passing different args; skip = answering with synthetic content (the `PendingToolCall::preresolved_result` shape, `run/mod.rs:153`, is the existing precedent) |
| `on_tool_result` → `Rewrite`/`Stop` (`runner.rs:782`) | rewrite model-visible presentation | driver edits the `UserContent` before `run.tool_results(..)`; the raw `ToolResult` stays whatever the driver kept |
| `on_text_delta` / `on_tool_call_delta` / `on_stream_response_finish` (observe-only, `streaming.rs:1144/:1176/:1328`) | observe / stop | the driver is polling the stream; verification confirmed these **cannot** mutate or suppress deltas today (`ObservationAction` is `Continue`/`Stop` only), so pure observation loses nothing |

So the core needs **zero new ingestion points**. The redesign is: keep the
event and action types as the protocol vocabulary (adding serde derives and
owned-variant forms where the current structs borrow), delete the callback
machinery (`AgentHook`, `DynAgentHook`, `HookStack`, `HookContext`,
`Scratchpad`, `ToolCallRewriteFrames`), and encode the *composition* semantics
as pure functions.

### 6.2 Composition as pure functions

Today's `HookStack` semantics, each with its replacement:

```rust
// rig-agent::hook — plain data and pure state machines; no futures, no callbacks.

/// on_completion_call: patches accumulate in order, Stop short-circuits.
/// A true fold — verified the event handed to each hook is NOT mutated
/// mid-chain (hook.rs:1259-1266), so pre-collected decisions are faithful.
/// Merge rules preserved verbatim from RequestPatch::merge (hook.rs:696-722):
/// extra_context appends; JSON-object additional_params shallow-merge
/// (later keys win); active_tools intersect; scalars & history last-wins.
pub fn fold_completion_actions(actions: Vec<CompletionCallAction>) -> CompletionCallAction;

/// on_tool_call and on_tool_result are NOT foldable over pre-collected
/// decisions: in the real semantics each later decider's INPUT carries the
/// earlier rewrites (the event is rebuilt with effective args per hook,
/// hook.rs:1217-1223; with effective presentation, hook.rs:1331-1335), so
/// the decisions themselves depend on prior outcomes. The replacement is a
/// plain accumulator the host drives, computing each next decision against
/// the current effective value:
pub struct ToolCallResolution {
    effective: serde_json::Value,
    terminal: Option<ToolCallAction>,
}
impl ToolCallResolution {
    pub fn new(original_args: serde_json::Value) -> Self;
    /// Input for computing the NEXT decision (original or rewritten args).
    pub fn args(&self) -> &serde_json::Value;
    /// Apply one decision; returns false after a terminal Skip/Stop, which
    /// short-circuits while KEEPING the accumulated rewrite — the exact
    /// salvage semantics of HookStack::resolve_tool_call (hook.rs:1211-1237).
    /// The ToolCallRewriteFrames machinery (hook.rs:219-299, ~80 lines of
    /// mutex-guarded frame stacks) existed only to smuggle that salvage
    /// across the erased-hook boundary; with no erased boundary it deletes.
    pub fn apply(&mut self, action: ToolCallAction) -> bool;
    pub fn finish(self) -> (ToolCallAction, serde_json::Value);
}

/// Same shape for on_tool_result: later deciders see the rewritten
/// presentation, Stop short-circuits (hook.rs:1325-1343).
pub struct ToolResultResolution { /* ToolOutput accumulator, same protocol */ }

/// Observations: first non-Continue wins (hook.rs:1240-1250). Foldable —
/// observation events carry no chained state.
pub fn first_stop(actions: Vec<ObservationAction>) -> ObservationAction;

/// on_invalid_tool_call: first Some wins (hook.rs:1303-1314). Foldable.
pub fn first_resolution(actions: Vec<Option<InvalidToolCallAction>>) -> Option<InvalidToolCallAction>;
```

A host with one policy calls none of these. A host composing several policy
sources (the "stack of hooks" use case) folds the *independent* events —
completion-call patches, observations, invalid-call resolutions, whose
per-hook input is verifiably unchanged mid-chain — and drives the
accumulators for the two *chained* events, computing each source's decision
against `resolution.args()` before applying it. Ordered composition, patch
merging, and short-circuit-with-salvage are thereby *specified once, in
testable pure types*, instead of being emergent behavior of a vec of vtables.
The existing `HookStack` unit tests (`hook.rs:1384-1597` and the migrated
suite) port directly onto the folds and accumulators.

### 6.3 What replaces `HookContext` and `Scratchpad`

`HookContext` (`hook.rs:303-361`) carries `run_id`, `turn`, `is_streaming`,
`agent_name`, and the `Scratchpad`. The first four are data the host already
has (`turn` is in every relevant event; the session exposes a `RunId`). The
`Scratchpad` — a mutex-guarded `Any` map for cross-hook state (`hook.rs:157-209`)
— is banned erasure, and its replacement is nothing: host decision code is a
match in a loop, so "run-scoped hook state" is a local variable. The doc
example at `hook.rs:66-115` (retry counter with an atomic ID allocator and a
`HashMap` in the scratchpad) becomes `let mut attempts = 0;` above the loop.
This is the strongest single ergonomics *win* in the whole redesign.

### 6.4 Semantics parity checklist

Verified against the real behaviors the prompt listed:

- **Ordered composition**: fold/apply order = registration order. ✓
- **Patch merging**: `RequestPatch::merge` survives verbatim as the fold's
  combine step; the "two hooks set the same field" warning
  (`hook.rs:599-610`) stays in the fold. ✓
- **Short-circuiting**: folds stop at the first terminal action for the
  independent events; the accumulators encode terminate-with-salvage for the
  chained ones (§6.2 — a fold alone cannot, because later decisions consume
  earlier rewrites as input). ✓
- **Response-hook suppression for recovered turns**: already a machine output
  (`ModelTurnOutcome::Continue { response_hook_suppressed }`). ✓
- **Streaming delta events**: observe-only today (verified), so host-side
  observation is semantics-preserving; a `Stop` maps to ceasing to poll +
  `run.cancel_error`, which is what `observe_action` does now
  (`runner.rs:108`, `streaming.rs:1154-1158`). ✓
- **`observes()` interest gating** (`hook.rs:1029`, precomputed at
  `streaming.rs:1019-1020` so no-hook runs never build delta events): the host
  simply doesn't write observation code it doesn't want; the zero-cost default
  is structural. ✓

### 6.5 What is lost, honestly

Reusable *drop-in* hooks. Today a crate can ship
`impl AgentHook for RateLimiter` and any agent adds it with one line. In the
new world a reusable policy ships as a *decision function* over event data —
`fn rate_limit_decision(&mut RateLimiterState, &CompletionRequest) -> CompletionCallAction`
— and the host must call it in the right arm of its loop. Composition is
explicit rather than automatic. That is a real regression in plug-and-play
convenience, accepted deliberately: the callback registry was precisely the
behavior-holding slot the mandate removes. (Cost quantified in §15.)

---

## 7. Tools

### 7.1 The core holds definitions; the host executes

```rust
// rig-agent — plain data; the de-generified ToolRegistrySnapshot
// (tool/server.rs:29-64) minus the IndexMap<String, RegisteredTool> of
// Arc<dyn ErasedTool> handles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCatalog {
    pub definitions: Vec<ToolDefinition>,       // order = advertisement order
    pub executable: BTreeSet<String>,           // names the host will actually run
}

impl ToolCatalog {
    pub fn retain_names(&mut self, allow: &BTreeSet<String>);  // per-turn active_tools
}
```

`prepare_request` (§5.3) consumes the catalog; `AgentRun` keeps validating
model-emitted calls against `executable`/allowed names exactly as today
(`ModelTurn.executable_tool_names`, `run/mod.rs:173-175`). Execution is the
host's, via the `CallTools` effect — the machine already never executes a tool
(`RunState::ExecutingTools` parks the calls; `tool_results` validates the
answers as a multiset by call ID, `run/mod.rs:1108-1146`). MCP was the
existence proof that this boundary works: `McpTool::execute` is already a
remote call whose "implementation" is a wire message (`tool/rmcp.rs:481-512`).
The redesign makes every tool "remote" from the core's perspective.

### 7.2 Authoring is unchanged; the contract moves down

The typed `Tool` trait survives **as a codegen contract only**, merged into
`rig-core` in the context-free shape of today's `PortableTool`
(`rig-core/src/tool/portable.rs:19-46`): `const NAME`, `type Args: Deserialize`,
`type Output: Into<ToolOutput>` (§7.5), `type Error`, `async fn call(&self, args)`.
`#[rig_tool]` (`rig-derive/src/lib.rs:145`, expansion `tool/expand.rs:121-402`)
keeps working with one change: it always targets this one trait (the
contextual/portable fork at `expand.rs:332-336` disappears with `ToolContext`,
§7.4).

Justification for keeping a trait at all (§16 entry): no rig type ever stores
`dyn Tool` or takes `T: Tool` — the only consumer is the `ToolRouter` derive,
which needs a *named method shape* to generate calls against. The alternative
(duck-typed inherent methods) provides strictly worse diagnostics for the same
architecture. The `Tool: Sized` bound and the deleted `ErasedTool` guarantee
it can never become a vtable again.

### 7.3 Ergonomics: `#[derive(ToolRouter)]`

The user's tool set is a struct of concrete tools in the *user's* crate —
which is downstream of everything and can therefore hold the closed set the
core cannot:

```rust
#[derive(ToolRouter)]
struct MyTools {
    add: Add,                       // #[rig_tool] output
    search: Search,                 // hand-impl'd Tool
    #[tool_router(dynamic)]
    github: McpToolset,             // runtime-defined set, §7.6
}

// Generated (exhaustive over the fields — adding a field updates everything):
impl MyTools {
    pub fn catalog(&self) -> ToolCatalog;                    // definitions in field order
    pub async fn dispatch(&self, call: &ToolCall) -> ToolResult;  // match on NAME
    pub async fn dispatch_all(&self, calls: &[PendingToolCall], concurrency: usize)
        -> Result<Vec<UserContent>, ToolBatchError>;
        // honors preresolved_result; buffer_unordered internally; Err is the
        // fail-fast path (lowest-call-index error wins, nothing committed) —
        // an infallible signature could not express the batch semantics below
}
```

`dispatch` is a name-`match` over the fields — string comparison on wire data,
not dispatch machinery. `dispatch_all` ports the batch semantics that matter
from `drive_tool_calls` (`streaming.rs:698-980`): call-order result assembly,
fail-fast with lowest-index error winning, atomic commit (all results or
none), `preresolved_result` passthrough without execution, no `tokio::spawn`
so tools need not be `'static`. Those semantics move from the library's driver
into generated code the user owns — same behavior, visible in `cargo expand`.

What this deletes: `ToolSet` (`tool/mod.rs:604`), `ToolSetBuilder`,
`ToolServer`/`ToolServerHandle`/`ToolServerState` (`tool/server.rs`),
`RegisteredTool`/`ToolRegistration`, `ErasedTool`/`ErasedEmbeddingTool`/
`DynamicTool`/`DynamicCallback`, the `String`-typed args erasure boundary and
its `Value → String → Args` round-trip (`runner.rs:680` → `tool/mod.rs:278`),
and the builder typestates. Runtime add/remove of tools (`add_tool`/
`remove_tool` on the handle) becomes the host mutating its own struct/marker
state between turns and re-snapshotting the catalog — which is all the handle
did, minus the `RwLock`.

### 7.4 `ToolContext` is deleted, and typed replacements cover its three uses

`ToolContext` (`tool/extensions.rs:157-272`) is a clone-on-dispatch dual `Any`
map — banned. Its actual uses, each with the typed replacement:

1. **MCP metadata injection** (`rmcp.rs:486` pulls `rmcp::model::Meta` out of
   the map): `McpToolset::call` takes `Option<&rmcp::model::Meta>` as a real
   parameter (§7.6).
2. **Dispatch-result metadata** (`preserve_mcp_result`, `rmcp.rs:450-458`,
   stuffs the raw `CallToolResult` in for `on_tool_result` hooks):
   `McpToolset::call` returns a typed `McpCallOutcome { result: ToolResult,
   raw: rmcp::model::CallToolResult }` — the host holds it directly; no map
   needed to smuggle it past an erased boundary that no longer exists.
3. **Sub-agent context inheritance** (`agent/tool.rs:48`,
   `context.inbound_only()`): sub-agents-as-tools become a field holding an
   `AgentSession` factory config; whatever state the parent wants to share is
   a field on the user's tool struct. User state never needed a type map —
   the type map existed because the *library* stood between the user's tool
   and the user's driver. Both sides are the user now.

### 7.5 `IntoToolOutput` without `Any`

Today `impl<T: Serialize> IntoToolOutput for T` sniffs concrete types at
runtime via `Any` downcasts to rescue `ToolResultContent` /
`OneOrMany<ToolResultContent>` from generic JSON serialization
(`rig-core/src/tool/output.rs:199-229`). Replacement: delete the blanket impl;
`type Output: Into<ToolOutput>` with concrete `From` impls for `String`,
`&str`, `serde_json::Value`, `ToolResultContent`, `OneOrMany<ToolResultContent>`,
`ToolOutput` itself, plus a `Json<T: Serialize>(pub T)` newtype (ledgered as
§16.5 — R1/R2/R3 as written do not license it) for
the serialize-me case. Tools returning plain custom structs write
`Json(my_struct)` — one explicit wrapper replaces a runtime type sniff.
`#[rig_tool]` inserts the wrapper automatically for non-special return types,
so macro users see no change.

### 7.6 MCP

`rig-agent`'s `tool/rmcp.rs` moves to a companion (`rig-rmcp`), where foreign
`dyn` internals (rmcp's own machinery) are out of the mandate's scope. Shape:

```rust
// rig-rmcp — concrete; no rig trait implemented.
pub struct McpToolset { client: rmcp::service::ServerSink, tools: Vec<rmcp::model::Tool>,
                        timeout: Option<Duration> }
impl McpToolset {
    pub async fn connect(transport: ...) -> Result<Self, McpError>;
    pub async fn refresh(&mut self) -> Result<(), McpError>;      // replaces list-changed reconciliation
    pub fn definitions(&self) -> Vec<ToolDefinition>;
    pub fn is_live(&self) -> bool;
    pub async fn call(&self, name: &str, args: &serde_json::Value, meta: Option<&rmcp::model::Meta>)
        -> Result<McpCallOutcome, McpError>;
}
```

The `#[tool_router(dynamic)]` attribute makes the derive route
not-statically-known names to the field's `definitions()`/`call()` — the
catalog is rebuilt per turn, which is what `snapshot_tool_defs` does today.
What this deletes wholesale: the generation-token reconciliation
(`ManagedToolToken` `Arc`-pointer-identity, `server.rs:105-122`,
`reconcile_managed_erased_tools` `server.rs:312-393`, the write-lock-on-read
retirement at `server.rs:447,523`) — ~300 lines of concurrency-sensitive code
that exist only because MCP pushed mutations into a shared registry behind the
agent's back. With the host owning the toolset, refresh is a host-initiated
`&mut` call; there is no shared registry to reconcile.

### 7.7 Dynamic tool retrieval (RAG tools)

Today: `retrieved_tools()` stores `Arc<dyn VectorStoreIndexDyn>` in the tool
server; `snapshot_tool_defs(prompt)` fans searches out per turn
(`server.rs:474-534`). Under the mandate — and converging with the separately
planned core vector-store removal — retrieval is host-side data flow: the host
queries whatever store it likes (§10.3) and appends the resulting
`ToolDefinition`s to the turn's `ToolCatalog` before `advance()`. The
`RequestPatch.extra_context`/`active_tools` fields already model per-turn
injection; retrieved tools are the same pattern on the catalog.

---

## 8. Memory

Confirmed easiest. Verified touchpoints: load at `runner.rs:1136-1139` and
`streaming.rs:1523-1527` (only when no explicit history was given); append at
the single choke point `append_run_messages` (`runner.rs:610-624`, invoked
from the `Done` arm, `streaming.rs:656`); `clear` never called by the runtime.
History is already `Vec<Message>` on both edges.

Design: **memory leaves the library entirely; it does not become an effect.**
`LoadHistory`/`SaveHistory` steps would put IO policy (swallow-append-errors,
`tracing::warn` at `runner.rs:621`, bypass-when-explicit-history) into the
machine for zero gain — the data already crosses the protocol boundary as
`with_history(Vec<Message>)` in and `PromptResponse::messages` out.

- `rig_core::memory` (trait + `InMemoryConversationMemory` + filters) moves to
  `rig-memory`, which already exists and composes memories
  (`PolicyMemory<M,P>` etc. — generic over *memory*, a companion-crate
  concern outside this mandate's scope).
- Host pattern, replacing `conversation("id")`:

```rust
let history = store.load(&conversation_id).await?;               // rig-memory
let done = AgentSession::new(cfg, provider, rt.clone(), prompt)
    .with_history(history).run().await?;
store.append(&conversation_id, done.messages.clone().unwrap_or_default()).await.ok();
```

The facade may ship exactly this as a documented five-line recipe rather than
a parameter. Cost: one convenience (`.conversation("id")`) becomes three
visible lines; the swallow-errors policy becomes the host's explicit `.ok()`.

---

## 9. Streaming — the data protocol already expresses it

The prompt marked this the potential design-killer. Finding: **the sans-IO
streaming protocol already exists and is shipping**, as
`StreamedTurnAssembler` (`run/streamed.rs:313-672`) plus `AgentRun`'s streamed
entry points (`record_streamed_completion_call` `run/mod.rs:1240`,
`resolve_streamed_invalid_tool_call` `:1289`, `streamed_turn` `:1407`). The
assembler is a push-parser — `ingest(&item) -> Vec<StreamedTurnEvent>` — that
handles delta accumulation, name-validation buffering/replay, mid-stream
invalid-call surfacing, reasoning-signature quarantine, and canonical turn
assembly, with zero IO. The module doc (`run/streamed.rs:1-34`) states the
complete driving protocol. What remains outside it in `streaming.rs` is: the
provider stream lifecycle, seven locals of per-turn scratch state — four
booleans plus `last_usage`, `pending_final`, and the assembler
(`streaming.rs:1066-1079`) — hook dispatch, telemetry, and item ordering —
driver code by nature. There is **no generic and no trait object required by
streaming semantics**; today's `R` parameter and `Pin<Box<dyn Stream>>` are
inherited from `CompletionModel`, not demanded by streaming.

### 9.1 De-genericized stream items

```rust
// rig-core — replaces StreamedAssistantContent<R> (streaming.rs:959)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamedAssistantContent {
    Text(Text),
    ToolCall { tool_call: ToolCall, internal_call_id: String },
    ToolCallDelta { id: String, internal_call_id: String, content: ToolCallDeltaContent },
    Reasoning(Reasoning),
    ReasoningDelta { id: Option<String>, reasoning: String },
    Final(StreamFinal),
    Unknown(serde_json::Value),      // stays last; see ordering note below
}

/// The provider's terminal record, normalized. Kills the GetTokenUsage bound
/// (usage was only ever read via token_usage(): rig-core/streaming.rs:310,417,
/// agent drain :193, assembler Final arm run/streamed.rs:504).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamFinal {
    pub kind: StreamFinalKind,       // required discriminant — see below
    pub usage: Usage,
    pub finish_reason: Option<FinishReason>,
    pub message_id: Option<String>,
    pub provider: String,            // Revision 2.1: needed so the
    pub model: Option<String>,       // stream→CompletionResponse conversion
                                     // can fill the normalized fields
}
```

`kind` (a unit-ish enum serialized as a required string field) is the
discriminating field GB §7's caveat asked for: under `#[serde(untagged)]`,
`Final` now demands a field no `Unknown` payload will accidentally satisfy,
so `Final`-before-`Unknown` ordering round-trips. The reverse hazard is
accepted and stated: an `Unknown` provider payload that structurally supplies
`kind` + `usage` would deserialize as `Final` — the same class of untagged
risk the current enum already carries (`rig-core/streaming.rs:957-996`),
now confined to one improbable shape instead of "any `Value`". With `R` gone,
`StreamedTurnAssembler::ingest` loses its only type parameter and the **entire
run module is generic-free**.

### 9.2 The concrete stream types

```rust
// rig-core: one concrete HTTP stream (SSE/chunked), shared by all in-core providers.
pub struct HttpModelStream { body: HttpBody, parser: StreamParser }
// HttpBody: the cfg-selected concrete transport inside HttpRuntime's module
// (reqwest::Response native, a fetch-body reader on wasm) — the same cfg
// story as §5.2, never a Box<dyn>. StreamParser: a rig-core enum over the
// 25 per-provider parser structs (openai::SseParser, anthropic::SseParser,
// …), maintained by a core-local x-macro (§12).
impl HttpModelStream {
    /// Pull-based; backpressure is "don't call next_item".
    pub async fn next_item(&mut self) -> Option<Result<StreamedAssistantContent, CompletionError>>;
}

// rig (facade): transport variation as an enum, replacing Pin<Box<dyn Stream>>.
pub enum ModelStream {
    Http(HttpModelStream),
    #[cfg(feature = "bedrock")]     Bedrock(rig_bedrock::BedrockStream),
    #[cfg(feature = "gemini-grpc")] GeminiGrpc(rig_gemini_grpc::GrpcStream),
    #[cfg(feature = "candle")]      Candle(rig_candle::LocalStream),
    #[cfg(feature = "test-utils")]  Mock(MockStream),      // scripted items, §3.6/§14
}
impl ModelStream {
    pub async fn next_item(&mut self) -> Option<Result<StreamedAssistantContent, CompletionError>>;
    pub fn close(self);   // explicit early termination (replaces cancel()/Abortable)
}

// rig (facade): the user-facing multi-turn stream — an explicit state machine,
// NOT async_stream (whose generators are unnameable and force Box<dyn Stream>).
pub struct AgentStream {
    session_state: /* AgentConfig + ProviderConfig + ToolCatalog + AgentRun + Arc<Runtime> */,
    policy: SessionPolicy,                                // same decision gating as AgentSession
    turn: Option<(ModelStream, StreamedTurnAssembler)>,   // present while a turn streams
    pending_tools: Option<Vec<PendingToolCall>>,
    buffered: VecDeque<AgentStreamItem>,                  // assembler fan-out only
}
impl AgentStream {
    pub async fn next_item(&mut self) -> Option<Result<AgentStreamItem, PromptError>>;
    // Decision inboxes — full parity with AgentSession's (§5.5); request
    // patching and turn retry fire through the shared drive_agent on today's
    // streaming surface, so the replacement surface must accept them too.
    pub fn provide_tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError>;
    pub fn resolve_invalid(&mut self, action: InvalidToolCallAction) -> Result<(), PromptError>;
    pub fn reply_before_call(&mut self, action: CompletionCallAction) -> Result<(), PromptError>;
    pub fn reply_turn(&mut self, action: ModelTurnAction) -> Result<(), PromptError>;
    pub fn patch_next_turn(&mut self, patch: RequestPatch);
    // Accessors §9.3 relies on:
    pub fn usage(&self) -> Usage;          // live total, valid after failure too
    pub fn close_turn(&mut self);          // early provider-socket termination
    pub fn last_response(&self) -> Option<&StreamFinal>;   // observation parity
}

/// MultiTurnStreamItem<R> minus R, plus the decision points that used to be
/// hook callbacks. Deliberately exhaustive, like SessionEvent and
/// AgentRunStep: a new decision-bearing variant must fail to compile in
/// every streaming host rather than fall through a wildcard arm — the same
/// doctrine §5.1 applies to ProviderConfig.
#[derive(Debug, Clone)]
pub enum AgentStreamItem {
    Assistant(StreamedAssistantContent),                   // deltas, incl. Final(StreamFinal)
    CompletionCall(CompletionCall),
    /// policy.surface_completion_calls — answer via reply_before_call (same
    /// pre-build contract as SessionEvent::BeforeModelCall, §5.5).
    BeforeModelCall { prompt: Message, history: Vec<Message>, turn: usize },
    /// policy.surface_model_turns — answer via reply_turn; a Retry surfaces
    /// ModelTurnRetried so consumers discard the turn's provisional deltas.
    TurnFinished { turn: usize, content: OneOrMany<AssistantContent>, usage: Usage },
    ModelTurnRetried { turn: usize },
    InvalidToolCall(InvalidToolCallContext),               // answer via resolve_invalid
    ToolCallsReady(Vec<PendingToolCall>),                  // answer via provide_tool_results
    ToolExecutionCommitted { tool_call: ToolCall, internal_call_id: String },
    User(StreamedUserContent),
    Final(PromptResponse),
}
```

Ordering parity for tool calls: complete `ToolCall` items surface to the
consumer as `Assistant(StreamedAssistantContent::ToolCall { .. })` in call
order immediately before `ToolCallsReady`, preserving today's
announce-before-execute contract (`streaming.rs:752-775`); the assembler
still emits nothing for them mid-turn (`run/streamed.rs:446-448`), and
`ToolExecutionCommitted`/`User` items remain post-commit, in call order.

`AgentStream::next_item`'s body is a small explicit state machine because
`AgentRun` *is* the state machine — the driver-side state is exactly what
`streaming.rs` keeps in locals today (`streaming.rs:1066-1079`), reified into
named fields. This is the one place the implementation gets structurally
harder: `drive_agent`'s `async_stream::stream!` elegance (`streaming.rs:499`)
cannot be stored in a struct without `Box<dyn Stream>`, so the loop becomes a
resumable `match` over those fields. Estimated at roughly the same line count
as today's driver (~600 lines), minus the hook plumbing, plus state
bookkeeping.

### 9.3 Semantics audit against the current implementation

- **Backpressure**: verified all-pull today — every layer is a generator with
  no channels or buffers (`drive_agent` :499, poll loop :1109); a slow
  consumer stops polling the provider socket. `async fn next_item` chains
  preserve this exactly. ✓
- **Cancellation**: today's three mechanisms are (1) `cancel()` on the
  provider stream — **not reachable from the agent surface** (verified: no
  plumbing in `rig-agent`); (2) dropping the stream — loses all run state,
  memory never appended; (3) hook `Stop`. The redesign strictly improves this:
  the host *owns* `AgentStream`, so stopping = stop polling (state intact,
  serializable, resumable), `close()` gives explicit provider-socket
  termination with the run still holding partial usage, and the
  incremental-usage discipline the kill example demands
  (`examples/gemini_stream_kill_token_count/src/main.rs:206-248`) is the
  natural shape since `record_streamed_completion_call` is host-visible. The
  `PauseControl` busy-spin defect (`rig-core/streaming.rs:433-436`,
  `wake_by_ref` in a loop) dies with `PauseControl`: pausing a pull stream is
  not polling it. ✓ (improvement)
- **Per-delta cost**: today every delta crosses `Pin<Box<dyn Stream>>`
  vtable polls at two layers plus an optional `Box::pin` per observed hook
  event; the redesign's enum-match `next_item` is monomorphic and the
  `DriveItem`-inline-vs-box tuning (`streaming.rs:382-386`) carries over as
  "return by value". Strictly fewer allocations per delta. ✓
- **Mid-stream invalid tool calls**: unchanged — assembler surfaces
  `StreamedTurnEvent::InvalidToolCall`, host answers through
  `resolve_streamed_invalid_tool_call`, `StreamedResolution::Repaired`
  replays buffered deltas (`run/streamed.rs:523-569`). Already data. ✓
- **Turn atomicity for tools**: `dispatch_all` + buffered
  `ToolExecutionCommitted`/`User` items preserve the collect-then-commit
  discipline (`streaming.rs:782-784, :971-978`). ✓
- **Streaming-error usage loss** (today `error_usage` only works on the
  blocking surface — `runner.rs:228,1102`; streaming drops it): fixed
  structurally, the host holds the run and can read `run.usage()` after any
  failure. The `Arc<Mutex<Usage>>` side-channel wart deletes. ✓ (improvement)

**Conclusion for §6.4 of the task:** streaming does not merely survive the
mandate — the mandated shape (pull-based, host-owned, machine-mediated) fixes
four latent defects of the callback/trait-object shape (unreachable cancel,
state-losing drop, busy-spin pause, lost error usage). What is genuinely lost
is `futures::Stream` interop (§16.3).

---

## 10. The rest of the trait surface — verdict per trait

### 10.1 `EmbeddingModel` — first, because 13 companion crates bound on it

Verified shape (`embeddings/embedding.rs:98-170`): `const MAX_DOCUMENTS`,
`type Client`, `fn make(client, model, dims)`, RPITIT `embed_texts`; 14
implementors; consumed by `EmbeddingsBuilder<M,D>`, `InMemoryVectorIndex<M,D>`,
and every vector-store crate (lancedb, qdrant, mongodb, postgres, neo4j,
milvus, sqlite, scylladb, s3vectors, surrealdb, helixdb, vectorize, fastembed —
plus bedrock/gemini-grpc as implementors).

Same treatment as completion, one level simpler (no streaming):

```rust
// rig-core, per provider with embeddings:
pub async fn embed(cfg: &Config, rt: &HttpRuntime, texts: Vec<String>)
    -> Result<EmbeddingResponse, EmbeddingError>;      // batching honors DESCRIPTOR.max_embedding_documents

// rig facade:
pub enum EmbedderConfig { OpenAi(openai::Config), Cohere(cohere::Config), /* … */
    #[cfg(feature="fastembed")] FastEmbed(rig_fastembed::Config), /* local; weights via rt cache */ }
pub async fn embed(cfg: &EmbedderConfig, rt: &Runtime, texts: Vec<String>) -> …;
```

`MAX_DOCUMENTS` (the hardest dyn-blocker in the old plan) is trivial here: a
descriptor field consulted by the free function's chunking loop. `fn make`'s
`dims` parameter becomes a `Config` field. **The vector-store crates stop
holding an embedder at all** — see next.

`EmbeddingsBuilder<M, D>` and the `Embed` trait get explicit verdicts of
their own (deliverable item 1 — the builder's model side dies with the free
functions; the document side is two parts). Text *extraction* — today the
user-implemented `Embed` trait — becomes `#[derive(Embed)]` generating
`fn embed_texts(&self) -> Vec<String>`: one sync method, no associated
types, no RPITIT, no async. Unlike `Tool`, this contract needs nothing from
the forbidden list and requires **no ledger entry**. Batch *embedding* stays
fully typed — the caller's document type is nameable, so pre-erasing it to
`Value` for pipeline uniformity would be exactly the banned pattern:
`embed_batches(cfg, rt, texts: Vec<Vec<String>>) ->
Result<(Vec<OneOrMany<Embedding>>, Usage), EmbeddingError>` returns
embeddings **aligned to input order**, chunked per
`DESCRIPTOR.max_embedding_documents`, with the `Usage` total riding
alongside so batch callers keep the accounting `EmbeddingResponse` (the
existing type, `embedding.rs:174-179`) carries on the single-call path;
the caller zips the embeddings against its own documents. Conversion to a store record
— `StoreRecord { id: String, payload: serde_json::Value, embeddings:
OneOrMany<Embedding> }` — happens only at the store-insertion boundary,
where `payload` carries the same transport-only defense as `SearchHit`
(§10.3): the *store* is genuinely schemaless about documents; the embedding
pipeline never was.

### 10.2 `TranscriptionModel`, `ImageGenerationModel`, `AudioGenerationModel`, `RerankModel`

All CompletionModel-shaped (verified: `type Response` + `type Client` +
`fn make` + RPITIT; Rerank is EmbeddingModel-shaped with `MAX_DOCUMENTS`).
Same recipe: per-provider `transcribe`/`generate_image`/`generate_audio`/
`rerank` free functions over concrete request/response structs (the request
builders `TranscriptionRequestBuilder<M,D>` etc. become plain request structs;
their `Missing` typestates die per §3.4), plus facade enums gated behind the
existing `audio`/`image` features. Implementor counts justify sequencing them
last: 7/7/5/1. `RerankModel` with its single implementor (`voyageai.rs:350`)
gets no enum until a second provider exists — a free function is the whole
API.

### 10.3 `VectorStoreIndex` / `VectorStoreIndexDyn` / `InsertDocuments`

Two forces converge: the mandate (the trait has `type Filter` and a generic
`top_n<T>`; the dyn twin erases documents to `Value` — disguised-erasure
adjacent) and the separately-planned removal of `vector_store` +
`dynamic_context` from rig-core. Design consistent with both:

- rig-core keeps only the *data* vocabulary: `Embedding`,
  `VectorSearchRequest` (query as `OneOrMany<Embedding>` — **pre-embedded**),
  and a concrete `SearchHit { id: String, score: f64, payload: serde_json::Value }`.
  `payload` is `Value` as *transport only* — a store's raw record is
  genuinely schemaless from the library's perspective — and the typed path is
  normative, not a footnote: every store also provides
  `async fn top_n_as<T: serde::de::DeserializeOwned>(&self, req: VectorSearchRequest)
  -> Result<Vec<(f64, String, T)>, VectorStoreError>` (rule R3: a generic
  function on a concrete type), so retrieval keeps the named type today's
  generic `top_n<T>` gives and `Value` never becomes the terminal API. The
  write side gets the symmetric R3 sugar: `insert_as<T: Serialize>` builds
  the `StoreRecord`s from the caller's typed documents.
- Each store crate exposes concrete types and inherent async methods:
  `LanceDbStore::top_n(&self, req: VectorSearchRequest) -> Vec<SearchHit>`,
  `insert(&self, Vec<StoreRecord>)` (§10.1). **No shared trait.** The trait
  existed so `rig-agent` could store `Arc<dyn VectorStoreIndexDyn>` for
  dynamic context/tools; with retrieval host-side (§7.7), nothing in rig
  needs to abstract over stores — the host names its store concretely.
- The embedder bound disappears from all 13 crates because queries arrive
  pre-embedded: `store.top_n(req.with_query(embed(&embedder, rt, vec![q]).await?…))`.
  Filters: each store keeps its own concrete filter type (they were never
  actually interchangeable — `type Filter` admitted it).

### 10.4 `CompletionClient`, `ProviderClient`, `Capabilities`/`Capability`/`Capable`, `Provider`/`ProviderBuilder`, `ModelLister`

All die with `Client<Ext, H>`. Construction: `openai::Config::from_env()` /
struct literals (replaces `ProviderClient::from_env`). Capability gating
(`const CAPABLE`, 7 associated types at `client/mod.rs:276-293`) becomes
runtime-checkable descriptor booleans — see §13.2 for what that trades away.
`ModelLister` (11 impls) becomes per-provider `list_models(cfg, rt)` functions
and one facade match.

### 10.5 `Prompt` / `Chat` / `TypedPrompt`

Deleted, not dyn-ified. Verified: no type in the workspace stores a
`dyn Prompt`; the traits exist so `Agent<M>`, `&Agent<M>`, and extractors share
call-site sugar (`agent/completion.rs:644-793`). The session's concrete
methods replace them; `Chat`'s history-splicing convenience
(`completion.rs:676-692`) becomes the memory recipe in §8. The `TypedPrompt`
GAT (`completion.rs:158-168`) — the single hardest-to-erase item in the old
plan — simply has no successor to need it: `extract` is a generic *function*
(R3), not an associated-type family.

### 10.6 Extraction

```rust
// rig facade — replaces Extractor<M,T> / ExtractorBuilder<M,T> / TypedPromptRequest<T,S,M>
pub async fn extract<T>(config: AgentConfig, provider: ProviderConfig, rt: Arc<Runtime>,
                        prompt: impl Into<Message>, retries: u64)
    -> Result<ExtractionResponse<T>, StructuredOutputError>
where T: schemars::JsonSchema + serde::de::DeserializeOwned;
```

Internally: sets `output_schema = schema_for!(T)`, `OutputMode::Tool` with the
`submit` tool (`extractor.rs:49,270-280` semantics preserved, including
first-submit-wins `:240` and the retry loop), drives an `AgentSession`,
deserializes. `T` is caller-owned output under R3; nothing stores it.

### 10.7 Integrations (`cli_chatbot`, `discord_bot`)

Rewritten over `AgentSession` (their `M`/`T` parameters existed only to hold
`Agent<M>`). These are leaf conveniences; the Discord one is feature-gated
already.

---

## 11. Ergonomic bounds (`impl Into<…>`) — decision

Rule R2: **retained in argument position, banned in stored types** — and the
protocol types already comply (every `AgentRunStep`/`ModelTurn`/decision-enum
field is concrete). Quantified cost of the alternative (full removal): 61
`impl Into<Message>`/`impl Into<String>` sites in rig-agent and ~230 in
rig-core (grep, this session),
every one forcing `.into()`/`.to_string()` noise onto user code; zero
architectural benefit since none can capture behavior or survive into state.
The one place the sugar is *removed* anyway is data-protocol constructors that
serde must mirror (`ModelTurn::new` keeps concrete parameters, as today).
`impl IntoIterator<Item = …>` on `embed_texts` becomes `Vec<String>` because
the function crosses the new provider-function boundary where requests should
be plain data; builder-side `IntoIterator` sugar stays.

---

## 12. Codegen and the 25-arm enums — is the maintenance load real?

Sites that must match every provider, exhaustively: facade `complete`,
`open_stream`, `list_models`, `embed` (+ 3 modality functions when their turn
comes), `ProviderId` naming, and the in-core `StreamParser` enum (§9.2).
Roughly 8 matches × 25+ arms.

**Mechanism: one x-macro in the facade** (macro_rules, no proc-macro cost):

```rust
macro_rules! for_each_builtin_provider {
    ($apply:ident) => { $apply! {
        (OpenAi,    openai,    "openai"),
        (Anthropic, anthropic, "anthropic"),
        /* … 25 rows; one row added per new provider … */
    } };
}
// Generates: the ProviderConfig enum, the uniform fulfilment matches,
// ProviderId, Display/FromStr, and the conformance-test matrix. Adding a
// provider is: write the provider module (build/parse/descriptor/config),
// add ONE row, and the compiler walks you through anything a macro can't
// reach.
```

Two macros, one per crate: this facade x-macro, and a core-local x-macro
generating the `StreamParser` enum over the per-provider parser structs — a
facade macro cannot generate a rig-core type. Non-uniform arms (candle's
`rt.candle_model(..)` shape, the cfg-gated companion arms, the `Mock` arm)
are written by hand outside the macro rows; the macro covers the uniform
HTTP majority.

**The OpenAI-compatible 17 after the `Ext` correction (§2).** The trait's six
methods and two associated types decompose as:

- 5 consts + `STREAM_INCLUDE_USAGE`-style knobs → `ProviderDescriptor` fields
  (values, so the 17 share one runtime-parameterized helper:
  `openai_compat::build_request(&GROQ_DESCRIPTOR, cfg, req)`).
- `completion_path`, `prepare_request`, `finalize_request_body(_with_options)`,
  `decorate_streaming_tool_call` → default free functions in
  `providers::openai_compat`; a provider that deviates writes its own free
  function that calls the shared ones. Static function composition — no
  parameterization. Deviation count (adversarially re-verified against the
  impl blocks, not just `fn`-name grep): **11 of the 17 override at least one
  method beyond the consts** (azure, mira, hyperbolic, together, huggingface,
  mistral, moonshot, groq, perplexity, deepseek, openrouter). OpenAI's own
  compat impl (`completion/mod.rs:1529-1534`) overrides nothing, and the
  responses-websocket `prepare_request` (`websocket.rs:462`) is a private
  inherent method with a different signature — a name collision two earlier
  counts of this number both tripped over. Still the common case, not the
  exception: budget ~10–40 lines of non-shared free-function code per
  deviating provider; the sharing win is real but smaller than the prior doc
  implied.
- `type Response: TryInto<CompletionResponse<…>>` → each provider's
  `parse_response` names its wire struct as a local and converts — the
  associated type existed only to tell the generic core which local to use;
  free functions don't need to be told.

**Compile-time impact**: bounded and mostly *favorable*. All 25 providers
already compile unconditionally today; the enum adds one match-generating
macro expansion. Deleted in exchange: `GenericCompletionModel<Ext, H>` × 17
monomorphizations, the `Client<Ext, H>`/`ClientBuilder` GAT machinery, ~80
RPITIT futures, and the double-generic builder chain — all of which are
per-instantiation codegen today. The facade's `complete` future is one enum
state machine sized by its largest arm (one per in-flight call — irrelevant
next to an HTTP round-trip). Risk to watch: a single giant `match` fn is one
codegen unit; if it ever bottlenecks, split per-capability matches into
separate modules (the macro already does this naturally).

Verdict: codegen is load-bearing, and the per-provider override functions —
not the macro — are the bulk of P4's mechanical work (two x-macros, one
descriptor struct, 11 providers' deviation functions). No proc-macro, no
build step.

---

## 13. Challenges from the task, verified

### 13.1 The ECS premise

Verified against the real experiment: `gold-silver-copper/rig-ecs` PR #6
(analysis at `rig1/rig/docs/architecture/rig-runtime-split/ecs-reference-analysis.md`;
674 files, bevy_ecs 0.19 made a mandatory rig-core dependency). What it proves
the ECS actually needs: plain-data components; owned effect requests/
completions with stable IDs and generation checks; immutable per-turn
snapshots; explicit domain snapshots rather than raw `World` serialization;
futures that never borrow the world. Every one of those is what this design
produces natively (`AgentConfig`/`ProviderConfig`/`ToolCatalog`/`AgentRun` as
components; `CallModel`/`CallTools`/decision enums as effect data;
`ToolCatalog` per-turn pinning as the snapshot). What PR #6 had to *build*
inside rig-core — the effects layer, snapshot records, capability erasure at
adapters — exists here as the library's ordinary surface, which is precisely
the "no adaptation layer" goal. The archetype consideration holds: config
splits into several small components rather than one aggregate (§5.3), and
per-provider marker-component dispatch remains available to an ECS host that
prefers open dispatch over the facade enum (the host matches; the core doesn't
care). The premise survives scrutiny, with one honest caveat: PR #6 also
demonstrates that a *full* ECS runtime wants ~30k lines of scheduling/policy/
identity machinery that no core redesign obviates — this mandate makes
`rig-ecs` thin, not free.

### 13.2 What replaces compile-time safety

Lost: `client.agent(GPT_5_2)` can't send an OpenAI model to Anthropic — the
provider/model pairing was type-checked. Gained back at request-build time:
`prepare_request`/facade `complete` validate against `ProviderDescriptor`
(tool support, response-format support, modality presence) and fail before
any network call — same stage where tool-choice-vs-toolset conflicts already
fail today (`completion.rs:500-513`). Honest residue: a typo'd *model name*
was already a runtime error (model ids are `&str` consts today), so the net
new runtime-error surface is "config names a provider whose descriptor can't
satisfy the request", caught at build-request, plus "wrong credentials", which
was always runtime. Exhaustive matching restores a *different* compile-time
guarantee the trait system never gave: a new provider cannot ship half-wired.

### 13.3 Performance

- Unary path: one enum branch per model call, replacing `Arc<M>` deref +
  static call; unmeasurable against a network round-trip. Removes one boxed
  future per call that Option-C-style dyn plans would have added.
- Per-delta path (the tuned one, `streaming.rs:382-386`): today each delta
  crosses two `Pin<Box<dyn Stream>>` vtable polls (provider stream + engine
  stream) and clones per the `item_slot` discipline. Redesign: monomorphic
  `async fn` chains + one enum match per item; the `DriveItem` no-box rule
  becomes "return by value". Fewer indirections, no new allocations. The only
  hot-loop `match provider` is inside `HttpModelStream::next_item`'s parser
  enum — a jump table over 25 arms, constant per item.
- Tool batches: `dispatch_all` keeps `buffer_unordered` without `tokio::spawn`
  (tools stay non-`'static`), unchanged from `streaming.rs:885`.

### 13.4 Composability check

- `AgentRun` without `AgentConfig`: unchanged from today (`run/mod.rs:36-61`
  doc example compiles against nothing else). ✓
- Tool definitions without the runner: `ToolCatalog` is two plain fields;
  `#[derive(ToolRouter)]` output has no rig-agent dependency beyond data
  types. ✓
- Provider functions without any agent: `openai::complete(cfg, rt, req)` with
  a hand-built `CompletionRequest`. This is *new* — today the equivalent
  requires constructing a `Client<Ext,H>`, a model handle, and a
  `CompletionRequestBuilder<M>`. ✓
- Hooks/memory/tools each usable alone: fold helpers are free functions;
  memory is a separate crate; the assembler ingests any provider item
  stream. ✓

---

## 14. Migration plan

Each phase compiles, passes tests, and ships. "Reversible" = deletable without
touching other phases' work. Cassette discipline throughout: provider request
bytes must not change except in P1's normalization fields (recorded once,
reviewed as the proof of behavioral identity — per the PR #6 lesson that
wholesale cassette churn destroys reviewability).

| # | Phase | Contents | Breaking? | Reversible? |
|---|---|---|---|---|
| P1 | Normalize the payloads | Concrete `CompletionResponse` (+`finish_reason`, closes #2090/#1886); concrete `StreamedAssistantContent`/`StreamFinal`; de-genericize `StreamedTurnAssembler::ingest`; delete `GetTokenUsage` (dependent set on the §3.2 row). `CompletionModel` temporarily keeps erased-free signatures by returning the concrete types. Transitional pattern while `GenericCompletionModel` survives: shared compat wire conversions fill a placeholder and the generic model stamps `response.provider = Ext::PROVIDER_NAME` post-`try_into` (in-crate field mutation under `#[non_exhaustive]`). Working order: core vocabulary first (`request.rs`, `streaming.rs`), then the provider fleet in dependency-clean waves, with the `openai`/`internal` shared core done attentively before the 17 thin compat files. | Yes — `raw_response`/`Final(R)` consumers (34 + 20 files, §2; mostly in-repo tests) | No (data-model change) — do first, with the issue-closing payoff |
| P2 | The pure protocol layer | `AgentConfig`, `ToolCatalog`, `prepare_request` in rig-agent; hook fold + resolution-accumulator helpers (§6.2) + serde on action/context types. Existing runner internally rewires `build_prepared_completion_request` onto `prepare_request`. | No (additive) | Yes |
| P3 | Provider pilot | `openai`: `Config`/`DESCRIPTOR`/`build_request`/`parse_response`/`SseParser`/`complete`/`open_stream` + `HttpRuntime`; existing trait impl becomes a thin delegate. Byte-identical cassettes prove the split total. | No | Yes |
| P4 | Provider fleet | Remaining 24 in-core providers via `openai_compat` helpers + x-macro; companion crates (`bedrock`/`gemini-grpc`/`candle`) gain `Config` + free fns alongside their trait impls. | No | Yes (mechanical) |
| P5 | The facade runtime | `ProviderConfig`/`Runtime` (live-handle caches)/`ModelStream`/facade `complete`/`open_stream`; `AgentSession`; `AgentStream`; `#[derive(ToolRouter)]`; `McpToolset` in rig-rmcp; extraction fn; `ProviderConfig::Mock`+`ModelStream::Mock` scripted test doubles (the successor to `MockCompletionModel`, §3.6); telemetry ported into the session drivers. Ships **alongside** the classic runtime. | No | Yes |
| P6 | Classic runtime re-plumb (Revision 2) | `Agent<M>` → `Agent` (holds `ProviderConfig` + `Arc<Runtime>`); `AgentRunner`/`PromptRequest`/`StreamingPromptRequest`/`Extractor` lose `M`; internals rewired onto `prepare_request` + facade `complete`/`open_stream`; implementation stays in `rig-agent` (Revision 2.2), rewired onto the in-crate `provider` module; **hooks, memory, and the tool server are kept** as the classic runtime's convenience layer. Examples/integrations/docs updated; `MIGRATING.md` chapter written. | Yes (type-param removal; API shape preserved) | No (the migration itself) |
| P7 | Cleanup of orphaned plumbing (Revision 2 — was "the deletion") | Delete only what the re-plumb orphaned: `GenericCompletionModel<Ext,H>`, `Client<Ext,H>`/`ClientBuilder`/`Provider(Builder)`/`Capabilities` internals (per-provider `Client::from_env()` survives as thin sugar over `Config` + `Runtime`), `CompletionRequestBuilder<M>`, `TurnSource`, `CompletionModel` retired after nothing in-tree consumes it (out-of-tree implementors migrate to `Config` + free fns or drive `AgentRun` directly), `wasm_compat` shrinks. The classic runtime surface itself is untouched. | Yes (trait retirement) | No |
| P8 | Modalities & stores (Revision 2) | `EmbeddingModel` → embed functions + `EmbedderConfig`; transcription/image/audio/rerank per §10.2; **all 13 store crates kept and de-genericized** per §10.3 (drop the `EmbeddingModel` bound, pre-embedded queries, `StoreRecord`/`SearchHit`/`top_n_as`, no shared trait). | Yes (per-crate majors) | Per-crate |
| P9 | `rig-bevy` (Revision 2 — new) | A simple `bevy_ecs` runtime crate: components = `AgentConfig`, `ProviderConfig`, `ToolCatalog`, `AgentRun`; systems fulfil `CallModel`/`CallTools` as owned effects (spawned entities + write-back), per the PR #6 invariants §13.1 lists; a mock-provider example proving a world of heterogeneous agents driven by one system set. Deliberately minimal — scheduling/policy sophistication is future work. | No (new crate) | Yes |

P1–P5 are individually valuable on their own (P1 closes two open issues;
P3/P4 make providers unit-testable as pure functions; P5 is the ECS-ready
runtime). Under Revision 2 the sequence runs to completion: the classic
runtime survives migrated, and P9's `rig-bevy` is the proof that the
"additional runtime with no adaptation layer" goal is real.

---

## 15. Cost accounting

Stated against the whole redesign, not per-phase:

1. **User-facing breakage (Revision 2 — much smaller than originally
   scoped).** With the classic runtime migrated rather than deleted, user
   code keeps `Agent`/`prompt`/`stream_prompt`/hooks/tool registration; the
   breakage is the `M` parameter's removal (construction sites and any code
   naming `Agent<SomeModel>` in types) plus the P7 trait retirements. The
   ~23k-line classic test suites are *updated in place*, not ported to a new
   surface. Sessions and routers are the new, additional surface — adopted
   by choice, mandatory only for the bevy runtime.
2. **The hook ecosystem survives (Revision 2).** `AgentHook`/`HookStack`
   remain the classic runtime's convenience layer. The decision-protocol
   pattern (§6) is how the *protocol layer and rig-bevy* express the same
   powers; §6.5's plug-in-to-pattern regression now applies only to hosts
   that leave the classic runtime.
3. **Out-of-tree providers lose the trait plug-in point at P7.** No
   `CompletionModel` to implement. Options: PR an arm into the facade, or
   drive `AgentRun`+`prepare_request` directly (fully supported, but they
   re-own the driver loop). Private/proprietary providers are the loss case;
   the three in-repo companions are unaffected (they get arms).
4. **Dynamic tool sets change idiom.** Runtime add/remove via a shared handle
   becomes host-owned catalog rebuilds. Long-lived multi-tenant servers that
   mutated one agent's tools concurrently must now own that state machine
   themselves (most such users were fighting the `RwLock` semantics anyway —
   §7.6's reconciliation deletion is the same trade seen from the other side).
5. **`futures::Stream`/`StreamExt` interop is lost** at the agent surface
   (§16.3). `while let Some(item) = s.next_item().await` replaces combinators;
   `tokio::select!` still works (it selects on futures).
6. **Typed builder ergonomics.** `openai.agent(GPT_5_2).preamble(…).build()`
   becomes config structs + session constructors; per-provider convenience
   constructors can restore most of the feel but not the type-level pairing
   proof (§13.2).
7. **Migration engineering size.** Production code: ~4.5k lines rewritten
   (driver + hooks + tools + streaming shells), 25 provider modules split
   (mostly mechanical, shared through `openai_compat`), 13 store crates
   re-cut. Tests are the majority of the work by volume (point 1). Two major
   releases minimum (P1, P7), realistically spanning several release trains.
8. **What is *not* lost, despite appearances**: wasm support (concrete async
   fns compile on wasm more easily than the marker-trait lattice they
   replace); provider-typed raw responses (reachable via provider parse
   functions); suspend/resume (extended, now covering config); tool authoring
   (`#[rig_tool]` unchanged); observability (telemetry lives in the facade
   drivers, and `record_telemetry_content` stays a config field).

---

## 16. Exceptions ledger — constructs that survive with justification

Per the task's instruction to argue exceptions explicitly rather than smuggle
them:

1. **The `Tool` authoring contract** (§7.2). Stated precisely: this trait
   retains `type Args`/`type Output`/`type Error` and an RPITIT `call` — both
   items on the forbidden list — so this is a genuine exception, not
   capability machinery by another name (`From`/`Iterator` are std traits;
   this is not). The argument for granting it: the trait is consumed
   exclusively by derives in the *user's* crate; no rig type stores it,
   bounds a stored field on it, or erases it (the deleted `ErasedTool` was
   what manufactured the vtable, and nothing reintroduces one); and the
   associated types vary the *user's own payloads*, never provider,
   transport, or library behavior. The fully-compliant fallback is specified
   and cheap: the derive targets inherent methods + `const NAME` by
   convention, eliminating both constructs at the cost of worse diagnostics
   when a tool's shape is wrong. If the mandate is read strictly, take the
   fallback — the architecture is identical either way; this entry exists so
   the choice is made consciously rather than smuggled. (`Embed`, §10.1,
   deliberately does **not** share this entry: its contract is one sync
   method with no associated types and no RPITIT — an exception ledger that
   over-grants is itself a compliance defect, so `Embed` gets none.)
2. **`std::error::Error` source chains and the `ToolExecutionError` downcast
   API** (`rig-core/src/tool/result.rs:194,:207,:262-269,:315-327`). The
   sound argument: std's `Error::source` signature returns
   `&(dyn Error + 'static)`, so source-chaining forces `dyn` at the std
   boundary, and the mandate keeps `Error` as a capability trait. The first
   draft additionally claimed "no control flow branches on the downcast in
   rig code" — **that was false**: the error-flattening constructor
   (`from_error`, whose native/wasm cfg branches sit at `:194`/`:207`)
   `match`es `source.downcast::<Self>()` to collapse nested
   `ToolExecutionError`s, `with_source<E>` stores a caller-supplied `E`
   behind `Arc` erasure, and `downcast_ref::<E>()`/`is::<E>()` is a public
   `Any`-shaped API. Ledgered in full: the `dyn` source and the flattening
   branch stay (std idiom; the branch inspects error *identity* for
   diagnostics shape, never provider or tool behavior); the public
   `downcast_ref`/`is` pair is reviewed at P7 — kept only if the test port
   surfaces real error-inspection use cases, otherwise deleted along with
   the erased tool boundary that motivated it.
3. **No `futures::Stream` implementation on `ModelStream`/`AgentStream`.** Not
   a kept construct but a declared *incapability*: implementing `poll_next` by
   hand would require storing the in-flight chunk future, whose type is
   unnameable without `Box<dyn Future>` — banned. Hence inherent
   `async fn next_item` only. Hosts wanting a `Stream` can wrap in their own
   crate (`async_stream` around a value they own); rig will not ship the
   wrapper because rig would have to box it.
4. **Foreign-crate internals** (`reqwest`, `tokio`, `rmcp`, `aws-sdk`,
   `tonic`, `tracing`). The mandate governs rig's architecture; these are
   dependencies' implementation details behind concrete rig types. `rmcp` is
   additionally quarantined into the `rig-rmcp` companion so `rig-core`/
   `rig-agent` stay clean even transitively at the API level.
5. **`Json<T: Serialize>` and `impl<T: Serialize> From<Json<T>> for ToolOutput`**
   (§7.5). A generic wrapper appearing as `Tool::Output` type state, converted
   by a blanket-shaped `From` impl — licensed by none of R1/R2/R3 as written
   (R3 covers free-function bounds only), so it is ledgered rather than
   rule-cited: the wrapper is opt-in and nominal (the author writes
   `Json(value)`; nothing is silently adapted, unlike the deleted
   `impl<T: Serialize> IntoToolOutput for T`), `From` is on the mandate's
   allowed list, `T` is the author's own payload consumed at the conversion
   boundary rather than stored by rig, and the construct exists precisely to
   delete the `Any`-sniffing at `output.rs:199-229`. Compliant fallback:
   authors call a `to_tool_output(&impl Serialize)`-shaped free function (R3)
   explicitly, which `#[rig_tool]` can inline — strict readers lose only
   hand-impl ergonomics.

Nothing else needs an exception. In particular — verified, not assumed — the
following all fit inside the constraints with designs given above: streaming
(§9), mid-stream recovery (§9.3), hook composition semantics (§6.2), parallel
tool batches (§7.3), MCP (§7.6), suspend/resume (§5.1, §9.3), wasm (§15.8).

---

*Prepared against rig 0.41.0, branch `docs/migrating-0.30-onward`. Companion
verification data: §2 of this document; structural maps of runner/streaming/
tools/trait-surface captured during this session's audit. Prior documents
`generic-bounds.md` (inventory; use with §2's corrections) and
`generic-bounds-rearchitecture.md` (model-slot-only design; superseded).*
