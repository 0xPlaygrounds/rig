# Generic trait bounds in rig — inventory and de-generification options

Goal: reduce/eliminate generic trait bounds so rig's runtime types become plain,
concrete, `'static + Send + Sync` values that can be used as ECS components.

The core ECS problem: `Agent<M>` is not one type — it is *N* types, one per
provider model. A `bevy`-style world cannot query "all agents"; it can only query
`Agent<openai::CompletionModel>`, `Agent<anthropic::CompletionModel>`, … Every
generic parameter that survives into a runtime value multiplies component types
and blocks uniform systems. Generics that only exist at *construction* time
(builders, typestate markers, provider `Ext`/`H` parameters) never become
components and are therefore harmless.

---

## 0. TL;DR

There is essentially **one root cause** and it cascades:

```
CompletionModel { type Response; type StreamingResponse; type Client; fn make(); -> impl Future }
        │
        ├── forces `R` onto every response/stream type  (CompletionResponse<T>, StreamingCompletionResponse<R>,
        │   RawStreamingChoice<R>, StreamedAssistantContent<R>, MultiTurnStreamItem<R>, DriveStream<'a,R>)
        │
        └── forces `M` onto every runtime type          (Agent<M>, AgentRunner<M>, PromptRequest<S,M>,
            StreamingPromptRequest<M>, Extractor<M,T>, AgentBuilder<M,_>, TurnSource<M>)
```

Two findings make removal much cheaper than it looks:

1. **The raw response type is already required to be `Serialize + DeserializeOwned`**
   (`completion/request.rs:340-348`). Erasing it to `serde_json::Value` is
   lossless on the wire, and `Value` satisfies the remaining `Clone + Unpin`
   bounds for free.
2. **Nothing in the runtime reads it.** `raw_response` has zero consumers in
   `rig-agent`, and only a handful outside providers (candle tests, two Gemini
   examples, a few cassette tests) — all of which read concrete fields and could
   `serde_json::from_value` back.

Recommended path: **erase `R` first (Phase 1), then make `CompletionModel`
dyn-compatible (Phase 2)**. That collapses `Agent<M>` → `Agent` and leaves the
classic runtime with no runtime-level generics at all. The codebase already
contains three precedents for exactly this move (§4).

---

## 1. Inventory — Tier 1: runtime types generic over the model (`M: CompletionModel`)

These are the ECS blockers. All in `crates/rig-agent/src` unless noted.

| Type | Anchor | Why `M` is there |
|---|---|---|
| `pub struct Agent<M> where M: CompletionModel` | `agent/completion.rs:551` | Holds `model: Arc<M>`. Every other field is already concrete or `dyn`. |
| `pub struct AgentRunner<M> where M: CompletionModel` | `agent/runner.rs:198` | Holds `model: Arc<M>`; rest is concrete. |
| `pub struct AgentBuilder<M, ToolState = NoToolConfig>` | `agent/builder.rs:148` | Holds `model: M` until `build()`. `ToolState` is a typestate marker (Tier 5). |
| `pub struct PromptRequest<S, M>` | `agent/prompt_request/mod.rs:209` | Wraps `AgentRunner<M>`; `S` is `PhantomData` typestate (Tier 5). |
| `pub struct TypedPromptRequest<T, S, M>` | `agent/prompt_request/mod.rs:711` | Same, plus output type `T`. |
| `pub struct StreamingPromptRequest<M>` | `agent/prompt_request/streaming.rs:291` | Extra bound `M::StreamingResponse: WasmCompatSend + GetTokenUsage`. |
| `pub(crate) trait TurnSource<M>` | `agent/prompt_request/streaming.rs:405` | Internal blocking/streaming seam; carries `type Raw`. |
| `pub(crate) struct PreparedCompletionRequest<M: CompletionModel>` | `agent/completion.rs:22` | Wraps `CompletionRequestBuilder<M>`. |
| `pub struct Extractor<M, T>` | `extractor.rs:76` | Wraps `Agent<M>`; `T` is the extracted type. |
| `pub struct ExtractorBuilder<M, T>` | `extractor.rs:252` | Same. |
| `pub trait StreamingPrompt<M, R>` | `streaming.rs:12` | Trait is generic over *both* model and raw response. |
| `pub trait StreamingChat<M, R>` | `streaming.rs:26` | Same. |
| `pub struct AgentImpl<M>` / `ChatImpl<T>` / `ChatBot<T>` | `integrations/cli_chatbot.rs:15-31` | Integration wrappers. |
| `struct BotState<M: CompletionModel>` / `Handler<M>` | `integrations/discord_bot.rs:25,40` | Integration wrappers. |
| `pub struct CompletionRequestBuilder<M: CompletionModel>` | `rig-core/src/completion/request.rs:594` | Stores the model so `.send()`/`.stream()` can call it. |

**Bound-site count** (`M: CompletionModel` occurrences):

```
15  test_utils/model_conformance.rs      7  agent/runner.rs        4  extractor.rs
14  agent/prompt_request/mod.rs          6  integrations/discord_bot.rs   2  streaming.rs
11  agent/completion.rs                  6  agent/builder.rs       2  agent/tool.rs
 9  agent/prompt_request/streaming.rs    4  integrations/cli_chatbot.rs
```
≈ 80 sites in `rig-agent`, 17 in `rig-core` (of which 8 are in providers).

---

## 2. Inventory — Tier 2: the raw-response payload generic (`R` / `T`)

Driven entirely by `CompletionModel::Response` and `::StreamingResponse`.

| Type | Anchor | Where `R` appears |
|---|---|---|
| `pub struct CompletionResponse<T>` | `rig-core/src/completion/request.rs:213` | `pub raw_response: T` — the only use. |
| `pub enum RawStreamingChoice<R> where R: Clone` | `rig-core/src/streaming.rs:70` | Single variant `FinalResponse(R)`. |
| `pub type StreamingResult<R>` | `rig-core/src/streaming.rs:223,228` | `Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, _>>>>` — **already boxed/dyn**. |
| `pub struct StreamingCompletionResponse<R>` | `rig-core/src/streaming.rs:234` | `inner: Abortable<StreamingResult<R>>`, `response: Option<R>`. |
| `pub enum StreamedAssistantContent<R>` | `rig-core/src/streaming.rs:959` | Single variant `Final(R)`. Note it already has an `Unknown(serde_json::Value)` variant. |
| `pub enum MultiTurnStreamItem<R>` | `agent/prompt_request/streaming.rs:54` | Only via `StreamAssistantItem(StreamedAssistantContent<R>)`. |
| `pub type StreamingResult<R>` (agent) | `agent/prompt_request/streaming.rs:44,48` | Stream of `MultiTurnStreamItem<R>`. |
| `pub(crate) type DriveStream<'a, R>` | `agent/prompt_request/streaming.rs:369,373` | Internal driver stream. |
| `pub(crate) enum DriveItem<R>` | `agent/prompt_request/streaming.rs:387` | Internal driver item. |
| `TurnSource::Raw` | `agent/prompt_request/streaming.rs:410` | `= M::Response` (blocking) / `= M::StreamingResponse` (streaming). |

**Consumers of the typed raw value outside providers** (the entire migration cost
of erasing it):

- `crates/rig-candle/tests/real_model.rs`, `crates/rig-candle/src/model/tests.rs` — read `.text`, `.generated_tokens`, `.finish_reason`, …
- `examples/gemini_stream_kill_token_count/src/main.rs:237`, `examples/candle_local/src/main.rs:42` — `StreamedAssistantContent::Final(resp)`
- `tests/providers/gemini/cassette/{agent_run_streamed,interactions_api}.rs` — same

`raw_response` is referenced **zero** times in `crates/rig-agent/src`.

---

## 3. Inventory — Tiers 3–6

### Tier 3 — sibling modality traits with the identical shape

Every one of these repeats the `type Response` + `type Client` + `fn make` +
`-> impl Future` pattern, so each needs the same treatment if you want its
handles to be ECS components:

| Trait | Anchor | Blockers |
|---|---|---|
| `EmbeddingModel` | `rig-core/src/embeddings/embedding.rs:98` | `const MAX_DOCUMENTS`, `type Client`, `fn make(impl Into<String>)`, `embed_texts(impl IntoIterator)`, RPITIT |
| `TranscriptionModel` | `rig-core/src/transcription.rs:62` | `type Response`, `type Client`, `fn make`, RPITIT, `Clone` supertrait |
| `ImageGenerationModel` | `rig-core/src/image_generation.rs:49` | same |
| `AudioGenerationModel` | `rig-core/src/audio_generation.rs:55` | same |
| `RerankModel` | `rig-core/src/rerank.rs:44` | `const MAX_DOCUMENTS`, `type Client`, `fn make`, RPITIT |
| `VectorStoreIndex` | `rig-core/src/vector_store/mod.rs:84` | `type Filter`, generic method `top_n<T>` — **already has a `VectorStoreIndexDyn` companion at `:106`** |

Payload/builder types riding on them: `EmbeddingsBuilder<M, T>`
(`embeddings/builder.rs:53`), `TranscriptionRequestBuilder<M, D>`
(`transcription.rs:149`), `ImageGenerationRequestBuilder<M, P>`
(`image_generation.rs:79`), `AudioGenerationRequestBuilder<M, T, V>`
(`audio_generation.rs:82`), `InMemoryVectorIndex<M: EmbeddingModel, D: Serialize>`
(`vector_store/in_memory_store.rs:469`).

### Tier 4 — provider construction generics (`Ext`, `H`) — **leave alone**

`Client<Ext = Nothing, H = reqwest::Client>` (`rig-core/src/client/mod.rs:173`),
`Capabilities<H>` (`:276`), `Capable<M>` (`:259`), `ModelLister<H>`
(`client/model_listing.rs:111`), plus ~60 provider aliases of the form
`pub type Client<H = reqwest::Client> = client::Client<XExt, H>` and
`GenericCompletionModel<Ext, H>`.

These are compile-time provider-plumbing parameters. The *model handle* is what
flows into the runtime, and it is already a concrete per-provider type. Erasing
`M` (Phase 2) means these never reach an ECS component. **No action needed.**

`CompletionClient::CompletionModel` (`client/completion.rs:5`) is an associated
type, not a struct generic — it stays as the provider-author-facing factory.

### Tier 5 — typestate / marker generics (`PhantomData`-only) — **low priority**

| Type | Anchor |
|---|---|
| `PromptRequest<S, M>` — `S ∈ {Standard, Extended}` | `agent/prompt_request/mod.rs:195-209` |
| `AgentBuilder<M, ToolState = NoToolConfig>` | `agent/builder.rs:148` |
| `ClientBuilder<Ext, ApiKey = Missing, H = Missing>` | `rig-core/src/client/mod.rs:579` |
| `VectorSearchRequestBuilder<F, Q = Missing, S = Missing>` | `rig-core/src/vector_store/request.rs:253` |
| `ChatBotBuilder<T = Missing>` | `integrations/cli_chatbot.rs:29` |
| `markers::{Missing, Provided<T>, Nothing}` | `rig-core/src/markers.rs` |

These are erased at `.build()` and never become components. Removing them buys
ECS nothing and costs compile-time safety. Keep unless you want the API
uniformly non-generic for other reasons.

### Tier 6 — benign payload generics — **keep**

`OneOrMany<T>`, `Iter<'a,T>`/`IntoIter<T>` (`one_or_many.rs`),
`FileLoader<'a,T>`/`PdfFileLoader<'a,T>`/`EpubFileLoader<'a,T,P>` (`loaders/`),
`ApiResponse<T>` (per-provider), `WasmBoxedFuture<'a,T>`, `Filter<V>`,
`TypedPromptResponse<T>`, `ExtractionResponse<T>`, `Extractor`'s `T`.
These are ordinary container/output generics, not dispatch bounds.

### Companion crates

`rig-memory/src/lib.rs:514,623,1015` (`PolicyMemory<M,P>`,
`DemotingPolicyMemory<M,P,H>`, `CompactingMemory<M,P,C>` — `M` here is the inner
*memory*, not a model), `rig-vectorize/src/lib.rs:52`,
`rig-mongodb/src/lib.rs:115` (`MongoDbVectorIndex<C, M: EmbeddingModel>`), and
4–7 `EmbeddingModel` bound sites each in the sqlite/lancedb/surrealdb/scylladb/
neo4j/qdrant/postgres/milvus/s3vectors/helixdb stores. All follow from Tier 3.

---

## 4. Prior art already in this codebase

You have already done this move three times. Reuse the patterns rather than
inventing a fourth.

| Precedent | Anchor | Pattern |
|---|---|---|
| **Tools** | `rig-agent/src/tool/mod.rs:162` (`Tool: Sized` w/ `type Args`/`type Output`), `:292` (`pub(crate) trait ErasedTool`), `:311` (blanket `impl<T: Tool> ErasedTool for T`), `tool/server.rs:238` (`ToolServerHandle(Arc<RwLock<…>>)`) | Typed author-facing trait + crate-private erased trait + blanket impl + a **non-generic handle** stored in `Agent`. This is the exact target shape. |
| **Memory** | `rig-core/src/memory.rs:93` — `ConversationMemory` returns `WasmBoxedFuture<'a, …>`, stored as `Option<Arc<dyn ConversationMemory>>` in `Agent` (`agent/completion.rs:597`) | A dyn-safe async trait already living inside `Agent` today. Proof the pattern works under the wasm gating. |
| **Vector stores** | `rig-core/src/vector_store/mod.rs:106` — `VectorStoreIndexDyn` + blanket `impl<I: VectorStoreIndex<Filter=F>, F> VectorStoreIndexDyn for I`, erasing documents to `serde_json::Value` | Public dyn companion for a trait with associated types + generic methods, erasing payloads to JSON. Directly analogous to what `CompletionModel` needs. |

And the destination already exists:

> **`pub struct AgentRun` (`rig-agent/src/agent/run/mod.rs:283`) is completely
> generic-free and `serde`-derived.** The sans-IO state machine — turn budget,
> pending tool calls, output mode, invalid-call retries — is already a perfect
> ECS component today. Everything blocking you is in the *IO shell* around it.

---

## 5. Why the bounds can't be `dyn` today — blocker-by-blocker

For `CompletionModel` (`rig-core/src/completion/request.rs:338-388`):

| Blocker | Line | Fix |
|---|---|---|
| `type Response: … Serialize + DeserializeOwned` | `:340` | Erase to `serde_json::Value` (already serializable — lossless). |
| `type StreamingResponse: Clone + Unpin + … + GetTokenUsage` | `:342-348` | Erase to `serde_json::Value`; carry `Usage` **alongside** the value, computed at the provider boundary (see §7 caveat). |
| `type Client` + `fn make(client: &Self::Client, …)` | `:351-354` | Not needed at runtime. Move to a separate `CompletionModelFactory` trait, or leave on `CompletionClient` (`client/completion.rs:5`) which already owns construction. |
| `fn completion(..) -> impl Future` (RPITIT) | `:357-362` | `-> WasmBoxedFuture<'a, Result<…>>` — exactly what `ConversationMemory` does (`memory.rs:97-100`). One `Box` per model call, negligible against an HTTP round-trip. |
| `fn stream(..) -> impl Future<Output = StreamingCompletionResponse<..>>` | `:364-369` | Same. The inner stream is **already** `Pin<Box<dyn Stream>>` (`streaming.rs:223`), so dynamic dispatch there costs nothing new. |
| `Clone` supertrait | `:338` | Drop it. `Agent`/`AgentRunner` already store `Arc<M>`; only `CompletionRequestBuilder::send` clones (`request.rs:861`), which becomes `Arc::clone`. |
| `fn completion_request(&self) -> CompletionRequestBuilder<Self>` | `:372` | Uses `Self` by value → move to an extension trait with a `Sized` bound, or return a non-generic builder once `M` is erased. |
| `fn composes_native_output_with_tools(&self) -> bool` | `:385` | Already dyn-compatible. No change. |

For `Tool` and friends: `Sized` + `const NAME` + `type Args`/`type Output`
(`tool/mod.rs:162-174`) — already solved by `ErasedTool`.

For `Prompt` / `Chat` / `TypedPrompt` (`rig-agent/src/completion.rs:139-168`):
the *traits* are already non-generic, but the *methods* are not dyn-compatible
(`impl Into<Message>` params, RPITIT, `TypedPrompt::TypedRequest<T>` GAT). If you
want `Arc<dyn Prompt>` in the world, add a dyn companion taking `Message` by
value and returning `WasmBoxedFuture`.

Note `WasmCompatSend`/`WasmCompatSync` (`rig-core/src/wasm_compat.rs:7-58`) are
**not auto traits** — they are blanket-implemented marker traits. Any erased
trait object must spell out the marker supertraits explicitly and gate the
`Send` in `WasmBoxedFuture` the same way (`wasm_compat.rs:60-71` documents this
trap).

---

## 6. Options, ranked

### Option A — **Erase the raw response type** (`R` → `serde_json::Value`)

Change `type Response` / `type StreamingResponse` to a concrete
`serde_json::Value` (or a small concrete `RawResponse { value: Value, usage: Usage }`).

- **Removes:** every type in Tier 2 (10 types), plus the `R` parameter from
  `StreamingPrompt<M,R>`/`StreamingChat<M,R>` and `TurnSource::Raw`.
- **Cost:** ~6 files outside providers deserialize back; per-response
  `serde_json::to_value` at the provider boundary.
- **Breaking?** Yes for anyone touching `raw_response` / `Final(R)`.
- **Prerequisite for Option B** (associated types must be gone before dyn).
- **Verdict:** do this first. Highest ratio of generics removed to code touched,
  and the `Serialize + DeserializeOwned` bounds mean the data model does not change.

### Option B — **Make `CompletionModel` dyn-compatible; delete `M` outright**

After A: box the futures, move `make`/`type Client` off the trait, drop `Clone`.
`Agent<M>` → `Agent { model: Arc<dyn CompletionModel>, … }`.

- **Removes:** all of Tier 1 (~80 bound sites in `rig-agent`, 17 in `rig-core`).
- **Result:** `Agent`, `AgentRunner`, `PromptRequest<S>`, `StreamingPromptRequest`,
  `Extractor<T>` all become single concrete types. Combined with the already
  generic-free `AgentRun`, `ToolServerHandle`, and `Arc<dyn ConversationMemory>`,
  the classic runtime has **no runtime-level generics left**.
- **Cost:** one boxed future per model call; provider authors must adapt `make`.
- **Verdict:** this is the destination. Do it as the second phase.

### Option C — **Dyn companion trait + blanket impl** (non-breaking variant of B)

Keep `CompletionModel` verbatim for provider authors; add
`DynCompletionModel` with `impl<M: CompletionModel> DynCompletionModel for M`
returning `WasmBoxedFuture` and erased responses. `Agent` stores
`Arc<dyn DynCompletionModel>`. This is precisely `VectorStoreIndexDyn`
(`vector_store/mod.rs:106-155`).

- **Pro:** zero provider churn; both APIs coexist; ships incrementally.
- **Con:** two traits to keep in sync forever; the blanket impl still needs
  Option A's erasure inside it, so it does not avoid the response-type work.
- **Verdict:** the right *migration mechanism* for Option B — land C, migrate
  `Agent` onto it, then collapse the two traits in a later major release.

### Option D — **Enum dispatch** (`enum AnyModel { OpenAI(..), Anthropic(..), … }`)

- **Verdict: reject *as stated here*.** An enum over model *handles* closes the
  provider set while still requiring Option A to unify the response types — all
  cost, none of the openness.
- **But see `generic-bounds-rearchitecture.md` §2.** An enum over provider
  *config* under the rearchitecture is a different and much better proposition:
  the variants are plain serde data rather than handles, so the closed-set
  objection is confined to out-of-tree providers and the response-erasure
  prerequisite disappears. That is the recommended dispatch mechanism there.

### Option E — **Keep generics, add a non-generic ECS façade**

Leave `rig-agent` alone; build `rig-ecs` components that wrap
`Arc<dyn SomeErasedRunner>` and hold the generic agent behind it.

- **Pro:** zero breakage to `rig-agent`.
- **Con:** you write the erasure anyway (Options A + C in a new crate), and the
  duplication drifts. Only worth it if `rig-agent`'s API must be frozen.

### Option F — **Remove typestate markers** (Tier 5)

Replace `PhantomData` typestates with runtime-validated builders returning
`Result`.

- **Verdict: not for ECS.** Builders never become components. Do this only if
  API uniformity is a goal in itself.

---

## 7. Recommended sequencing

**Phase 0 — freeze the seam.** The IO boundary is already isolated at
`TurnSource<M>` (`agent/prompt_request/streaming.rs:405-450`) and
`PreparedCompletionRequest<M>` (`agent/completion.rs:22`). Two impls only:
`UnaryTurnSource` (`runner.rs:899`) and `StreamingTurnSource`
(`streaming.rs:1027`). Everything else already talks to the generic-free
`AgentRun`. Confirm nothing else calls a model directly — currently only
`CompletionRequestBuilder::{send,stream}` (`rig-core/src/completion/request.rs:860-875`) does.

**Phase 1 — erase `R`.** Change `CompletionResponse<T>` → `CompletionResponse`,
`RawStreamingChoice<R>` → `RawStreamingChoice`, and the six dependents. Update
the ~6 non-provider consumers to `serde_json::from_value`. `TurnSource::Raw`
disappears; `DriveStream<'a, R>` → `DriveStream<'a>`.

*Caveat to design around:* `StreamedAssistantContent` is `#[serde(untagged)]`
(`rig-core/src/streaming.rs:957-996`) and already has an
`Unknown(serde_json::Value)` variant deliberately ordered last because "a raw
`Value` matches anything." Erasing `Final(R)` to `Final(Value)` creates two
`Value`-shaped untagged variants that cannot be distinguished on deserialize.
Fix by giving `Final` a concrete wrapper struct with a discriminating field
(e.g. `Final(FinalResponse { usage, raw })`) rather than a bare `Value`.

*Second caveat:* `GetTokenUsage` is currently obtained by calling a method on the
typed `StreamingResponse` (`streaming.rs:236`, `agent/completion.rs:697`). After
erasure, usage must be computed at the provider boundary and carried as a
concrete `Usage` field. That wrapper struct handles both caveats at once.

**Phase 2 — dyn-ify `CompletionModel`.** Add `DynCompletionModel` (Option C),
blanket-impl it, switch `Agent`/`AgentRunner`/`PromptRequest`/
`StreamingPromptRequest`/`Extractor` to `Arc<dyn DynCompletionModel>`, delete the
`M` parameters. `Agent` and `AgentRunner` are both `#[non_exhaustive]`, so field
churn is not itself a breaking change. `AgentClientExt::agent` (`client.rs:26`)
and `AgentModelExt::into_agent_builder` (`:52`) keep working — they just return
the non-generic builder.

**Phase 3 — replicate for Tier 3** (`EmbeddingModel` first: it has the widest
companion-crate blast radius — 10 vector-store crates). `VectorStoreIndexDyn`
already shows the shape.

**Phase 4 — optional.** Tier 5 typestate removal; dyn companions for
`Prompt`/`Chat`/`TypedPrompt` if you want `Arc<dyn Prompt>` in the world.

## 8. What stays generic, deliberately

- Provider `Client<Ext, H>` plumbing (Tier 4) — never reaches a component.
- `OneOrMany<T>` and loader iterators (Tier 6) — ordinary containers.
- `Extractor`'s `T` and `TypedPromptRequest`'s `T` — the caller's output type;
  intrinsic to the API, and an ECS system would hold an erased runner plus a
  schema anyway.
- The typed author-facing `Tool` / `CompletionModel` traits themselves —
  provider and tool authors keep the ergonomic typed surface; only the *stored
  handles* get erased. That is the whole point of the `ErasedTool` pattern.
