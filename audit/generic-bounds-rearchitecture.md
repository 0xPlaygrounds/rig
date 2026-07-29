# Rearchitecting rig to eliminate model trait bounds (no type erasure)

> **Status note: superseded.** This document handles only the *model*
> parameter and is kept for its dispatch-mechanism rationale (§2). The full
> design — covering hooks, tools, memory, streaming, the whole trait
> surface, and the migration plan — is `audit/data-oriented-rearchitecture.md`,
> which also corrects several factual claims made here (its §2: provider
> count is 25, not ~32/~35; `OpenAICompatibleProvider` has 2 associated
> types + 6 methods, not "five consts + two small functions"; the "~97
> bound sites" figure is ~83–90). Two of this document's positions were
> later overturned: the `Custom` fn-pointer arm offered in §2 is a
> hand-rolled vtable and was rejected outright, and §5's final step
> ("delete the generic shell") is retired by the full design's Revision 2 —
> the classic runtime is migrated onto the new substrate, never deleted.

---

## 1. Thesis: the target architecture already exists in this repo

`crates/rig-agent/src/agent/run/mod.rs` already contains exactly the
architecture you are asking for:

```rust
pub struct AgentRun { /* ...20 plain-data fields... */ }   // :283 — zero generics, Serialize + Deserialize

pub enum AgentRunStep {                                     // :122 — effects out, as data
    CallModel { prompt: Message, history: Vec<Message>, turn: usize },
    CallTools { calls: Vec<PendingToolCall> },
    Done(PromptResponse),
}

impl AgentRun {
    pub fn next_step(&mut self) -> Result<AgentRunStep, PromptError>;      // :619
    pub fn model_response(&mut self, turn: ModelTurn) -> Result<ModelTurnOutcome, _>;  // :849
    pub fn tool_results(&mut self, results: Vec<UserContent>) -> Result<(), _>;        // :1108
}
```

Its module doc (`run/mod.rs:25-27`) already states the principle:

> `AgentRun` deliberately contains no model, tool registry, memory backend, or
> hook stack.

It owns every *decision* — turn budget, tool-call validation, invalid-call
recovery, history threading, usage aggregation, final response construction —
and performs no IO. It is `Serialize + Deserialize`. It is a valid ECS component
**today, unmodified**.

The problem is one of *hierarchy*, not of missing design. `AgentRun` is
positioned as a low-level escape hatch — the same doc says "constructing an
`AgentRun` directly is not an alternate way to execute an `Agent`" — while the
generic `Agent<M>` / `AgentRunner<M>` shell is the product. Everything you want
to delete lives in that shell.

**The rearchitecture is to invert this.** Promote the sans-IO protocol to *the*
architecture; demote the generic shell to a thin, optional convenience façade
built on top of it (or delete it). No erasure is involved because the model never
enters the core in the first place — not as a type parameter, not as a trait
object.

---

## 2. Where dispatch goes instead (the honest part)

Generics, `dyn`, and enums are three answers to the same question: *the core
holds behavior supplied by the caller, so how does it call it?* Erasure answers
"vtable." The rearchitecture rejects the question: **the core never holds
behavior at all.** It emits a `CompletionRequest` (already a concrete,
`Serialize + Deserialize` struct at `rig-core/src/completion/request.rs:392`) and
consumes a `CompletionResponse`. Something else — outside the core, outside its
type system — turns one into the other.

That "something else" still has to pick OpenAI code over Anthropic code.

### Use an enum

```rust
#[derive(Clone, Serialize, Deserialize)]      // plain data, all the way down
pub enum ProviderConfig {
    OpenAi(OpenAiConfig),
    Anthropic(AnthropicConfig),
    /* ...one arm per bundled provider... */
    #[cfg(feature = "bedrock")]  Bedrock(BedrockConfig),
    #[cfg(feature = "candle")]   Candle(CandleConfig),
}
```

Fulfilment is a `match`. This is the recommended mechanism, for four reasons:

1. **It is the only shape that covers every provider.** See §3 Move 3 — three of
   rig's providers are not HTTP/JSON at all (`rig-bedrock` uses the AWS SDK,
   `rig-gemini-grpc` uses tonic/protobuf, `rig-candle` runs in-process with no
   network). A uniform "build an `http::Request`, parse bytes" function signature
   cannot express those. An enum arm can hold arbitrary fulfilment logic.
2. **An enum over config is serde-able end to end.** This is the decisive
   advantage over both trait objects and function tables, neither of which can be
   serialized. `AgentRun` is already `Serialize + Deserialize` and supports
   suspend/resume across processes; a serde `ProviderConfig` extends that to the
   whole agent, so ECS save/load, rollback, and replay cover configuration too.
   Note the variants must hold *config* (endpoint, model name, credential
   reference), not live handles — an `Arc<reqwest::Client>` in an arm forfeits
   this.
3. **It costs nothing in binary size.** rig-core's ~32 providers are not
   feature-gated today (`crates/rig-core/Cargo.toml:99-129`), so they already
   compile into every build. Companion crates are already facade-feature-gated,
   and the `cfg` arms above match that existing scheme exactly.
4. **The match is exhaustive.** Adding a provider fails to compile until every
   fulfilment site handles it — a genuine advantage over a registry, where a
   missing entry is a runtime error.

### Where the enum has to live

It must be defined downstream of every variant it holds. `rig-core` **cannot**
own it: `rig-bedrock`, `rig-candle`, and `rig-gemini-grpc` all depend on
`rig-core`, so a `Bedrock(BedrockConfig)` arm there would invert the dependency.

It belongs in the **`rig` facade** (`src/lib.rs`), which already re-exports the
core and feature-gates companion integrations. That gives clean layering: core
defines the protocol and per-provider functions and stays open; the facade offers
the closed, convenient, exhaustively-matched enum over the bundled set.

### The one real cost, and the alternatives

A private out-of-tree provider cannot add an arm. Three responses, in order of
preference:

- **Accept it.** `AgentRun` and its protocol are public. An out-of-tree provider
  drives the machine itself and never touches the facade enum. It loses the
  bundled convenience layer, nothing else.
- **In ECS, skip the enum entirely.** Marker components give open dispatch with
  no dispatch construct at all: each provider registers
  `fn openai_completion_system(q: Query<(&PendingCompletion, &OpenAiConfig)>)`
  and selection becomes query filtering done by the scheduler. Third parties
  register their own systems. The enum still works fine in ECS as a component
  field matched by one system — simpler, marginally less parallelizable — so
  choose per situation rather than globally.
- **Add a single `Custom` arm** holding a `struct ProviderOps { build: fn(..), parse: fn(..) }`.
  Honestly: that arm is a vtable you built by hand. It is `Copy`, `'static`,
  plain-data, and not serializable — so it is one contained exception rather
  than the architecture, unlike `Arc<dyn CompletionModel>`. Add it only if a real
  out-of-tree provider demands it.

---

## 3. The four moves

### Move 1 — Make the emitted effect a complete request

Today `AgentRunStep::CallModel` emits `{ prompt, history, turn }`
(`run/mod.rs:125-134`). The other twelve request fields — preamble, documents,
tools, temperature, max_tokens, tool_choice, additional_params, output_schema —
live in `AgentRunner<M>` (`agent/runner.rs:198-231`) and are assembled by
`PreparedCompletionRequest<M>` (`agent/completion.rs:22`). That split is the only
reason a driver needs the generic shell at all.

Move those fields into the run (or a sibling `AgentConfig` component) and emit
the finished request:

```rust
pub enum AgentRunStep {
    CallModel { request: CompletionRequest, turn: usize },
    CallTools { calls: Vec<PendingToolCall> },
    Done(PromptResponse),
}
```

Every one of those fields is already plain data. The only field needing care is
`tools`: building it currently requires an `async` read of `ToolServerHandle`.
The seam already exists — `ToolRegistrySnapshot` (`tool/server.rs:29`) holds
`definitions: Vec<ToolDefinition>`, pure data. A system refreshes the snapshot
and stores it; `next_step` merges definitions in synchronously. The state machine
stays sans-IO.

After this move, a driver needs nothing but `AgentRun` + data to run an agent.

### Move 2 — Agent configuration becomes a plain-data component

`Agent<M>` (`agent/completion.rs:551`) has 17 fields. Sixteen are already
concrete data or non-generic handles. Exactly one — `model: Arc<M>` — carries the
generic. Replace it with a data reference:

```rust
pub struct AgentConfig {          // no generics, Serialize + Deserialize
    pub name: Option<String>,
    pub model: ModelRef,          // { provider: ProviderId, model: String, params: ... }
    pub preamble: Option<String>,
    pub static_context: Vec<Document>,
    pub temperature: Option<f64>,
    /* ... the remaining existing fields, unchanged ... */
}
```

`ModelRef` is a name, not a handle. Resolving it to credentials and an endpoint
is the fulfilling system's job. `Agent<M>` → `AgentConfig` deletes the `M`
parameter from `Agent`, `AgentRunner`, `PromptRequest<S, M>`,
`TypedPromptRequest<T, S, M>`, `StreamingPromptRequest<M>`, `Extractor<M, T>`,
`ExtractorBuilder<M, T>`, `AgentBuilder<M, _>`, `TurnSource<M>`, and
`PreparedCompletionRequest<M>` — Tier 1 in its entirety, ~97 bound sites.

### Move 3 — Providers become descriptor data + pure functions

This is the move that makes 2 possible without a trait object, and the evidence
that it is achievable is in the provider code itself. `GenericCompletionModel::completion`
(`providers/openai/completion/mod.rs:1941-2022`) does exactly four things:

| Step | Line | Nature |
|---|---|---|
| `ext().build_completion_request(model, req, options)` → value | `:1952` | **pure transform** |
| `ext().completion_path(&model)` → path | `:1981` | **pure** |
| `client.post(path).body(bytes)`; `client.send(req).await` | `:1983-1990` | **IO, provider-independent** |
| `from_str::<ApiResponse<_>>(text)` → `try_into()` → `CompletionResponse` | `:1996-2009` | **pure transform** |

The whole `CompletionModel` trait, the `Ext` parameter, the `H` parameter,
`type Client`, `fn make`, the RPITIT futures, and the `Clone` supertrait are
scaffolding wrapped around **one pure function pair plus a shared HTTP call**.

Better still, the provider-variation trait is already mostly *data*.
`OpenAICompatibleProvider` (`providers/openai/completion/mod.rs:1405-1470`) is
five `const`s — `PROVIDER_NAME`, `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS`,
`SUPPORTS_TOOLS`, `SUPPORTS_RESPONSE_FORMAT`, `STREAM_INCLUDE_USAGE` — plus two
small functions. Consts are struct fields wearing a costume:

```rust
pub struct ProviderDescriptor {          // plain data — one const per provider
    pub name: &'static str,
    pub supports_tools: bool,
    pub supports_response_format: bool,
    pub stream_include_usage: bool,
    pub emits_complete_single_chunk_tool_calls: bool,
}

// Free functions. No Self, no associated types, no async, no bounds.
pub fn build_request(d: &ProviderDescriptor, cfg: &ProviderConfig, req: &CompletionRequest)
    -> Result<http::Request<Vec<u8>>, CompletionError>;

pub fn parse_response(bytes: &[u8], status: http::StatusCode)
    -> Result<CompletionResponse, CompletionError>;

pub fn parse_chunk(state: &mut StreamParser, bytes: &[u8])
    -> Result<Vec<RawStreamingChoice>, CompletionError>;
```

These are `fn` items with no type parameters. There is no trait to bound on and
nothing to erase. One shared, non-generic executor performs the HTTP call, which
also deletes the `H = reqwest::Client` parameter threaded through ~60 provider
type aliases (Tier 4): the backend is a cfg-selected concrete type in the
executor, not a parameter of every provider type.

#### This shape is not universal — three providers do not fit

The signatures above cover the ~32 HTTP/JSON providers in `rig-core`. They
**cannot** express these three, which implement `CompletionModel` outside
`rig-core` over entirely different transports:

| Crate | Transport | Why `build_request -> http::Request` fails |
|---|---|---|
| `rig-bedrock` | `aws-sdk-bedrockruntime` | Request signing/retry owned by the AWS SDK; never yields a plain `http::Request` for rig to send. |
| `rig-gemini-grpc` | `tonic` + protobuf | gRPC framing over HTTP/2, not a JSON body. |
| `rig-candle` | `candle-core`/`-nn`/`-transformers` | In-process local inference. No network at all. |

So the fulfilment boundary must be defined in terms of **`CompletionRequest` in,
`CompletionResponse` out** — not bytes in, bytes out. `build_request`/`parse_response`
are then an *implementation convenience shared by the HTTP/JSON majority*, not
the contract. This is precisely why the enum in §2 is the right dispatch
mechanism: each arm fulfils the request however its transport requires, and the
three outliers are ordinary arms rather than exceptions to a universal signature.

This move deletes `CompletionModel` outright, along with `type Client`,
`fn make`, `CompletionRequestBuilder<M>`, and `CompletionClient::CompletionModel`.

### Move 4 — The raw response leaves the core

`R` exists solely because `CompletionModel::Response` is an associated type. With
the trait gone, the question becomes: what does `parse_response` return?

Not `serde_json::Value` — that is the erasure you rejected. Instead, **normalize**:

```rust
pub struct CompletionResponse {          // concrete. no <T>.
    pub choice: OneOrMany<AssistantContent>,
    pub usage: Usage,
    pub message_id: Option<String>,
    pub finish_reason: Option<FinishReason>,   // normalized — closes #2090
    pub provider_model_id: Option<String>,
}
```

The provider's typed response struct (`openai::CompletionResponse`,
`anthropic::CompletionResponse`, …) still exists and is still strongly typed — it
is the local variable inside `parse_response`, on the provider's own side of the
boundary. A caller who genuinely needs it calls that provider's parse function
directly and gets the real type back, with no generic anywhere. In ECS, a
provider system that wants to keep it writes it to its own component.

This is justified by the inventory: `raw_response` has **zero** references in
`crates/rig-agent/src`. The core has been carrying a type parameter through ten
types (Tier 2) to transport a value it never reads. The rearchitecture stops
modelling it rather than erasing it.

Streaming follows identically: `RawStreamingChoice::FinalResponse(R)` becomes a
concrete terminal record carrying `usage` and `finish_reason`, which also removes
the `GetTokenUsage` bound — usage becomes a field, computed by `parse_chunk`,
instead of a trait method on a caller-supplied type.

---

## 4. What this deletes, against the inventory

| Inventory tier | Fate |
|---|---|
| **Tier 1** — `Agent<M>`, `AgentRunner<M>`, `PromptRequest<S,M>`, `StreamingPromptRequest<M>`, `Extractor<M,T>`, `AgentBuilder<M,_>`, `TurnSource<M>`, `CompletionRequestBuilder<M>` (~97 sites) | **Deleted** by Moves 2+3. Model never enters the core. |
| **Tier 2** — `CompletionResponse<T>`, `RawStreamingChoice<R>`, `StreamingCompletionResponse<R>`, `StreamedAssistantContent<R>`, `MultiTurnStreamItem<R>`, `DriveStream<'a,R>`, `DriveItem<R>`, `StreamingResult<R>`, `TurnSource::Raw` | **Deleted** by Move 4. Also removes the untagged-enum and `GetTokenUsage` hazards flagged in the erasure plan. |
| **Tier 3** — `EmbeddingModel`, `TranscriptionModel`, `ImageGenerationModel`, `AudioGenerationModel`, `RerankModel` | Same treatment, same shape: descriptor + `build_request`/`parse_response`. Do `EmbeddingModel` first — 10 vector-store crates depend on it. |
| **Tier 4** — `Client<Ext, H>`, `Capabilities<H>`, `ModelLister<H>`, ~60 provider aliases | **Deleted** by Move 3's shared executor. Capabilities become descriptor fields, checked as data. |
| **Tier 5** — typestate markers (`PromptRequest<S,_>`, `AgentBuilder<_,ToolState>`, `ClientBuilder<Ext,ApiKey,H>`) | Fall out with their hosts; anything left is builder-only and never a component. |
| **Tier 6** — `OneOrMany<T>`, loaders, `Extractor`'s `T` | **Kept.** Ordinary containers, not dispatch bounds. |

Resulting ECS components, all concrete, `'static`, `Send + Sync`, serde:
`AgentRun`, `AgentConfig`, `ToolRegistrySnapshot`, `PendingCompletion`,
`CompletionResponse`, plus per-provider config/marker components.

---

## 5. Sequencing

Each phase compiles and ships on its own; none requires the next.

1. **Complete the protocol** (Move 1). `CallModel` carries a full
   `CompletionRequest`; move `AgentRunner`'s config fields into run/config data;
   feed tool definitions from `ToolRegistrySnapshot`. *No generics removed yet* —
   but `AgentRun` becomes independently sufficient, which is the precondition for
   everything else. Internal-only change if the old surface is kept.

2. **Split one provider** (Move 3, pilot). Extract OpenAI's
   `build_request`/`parse_response`/`parse_chunk` as free functions and reimplement
   the existing `CompletionModel` impl as a thin wrapper calling them. Behavior
   identical, cassettes unchanged — this is the proof the split is total. Pick
   OpenAI because `GenericCompletionModel<Ext, H>` already backs ~20 providers, so
   one split covers most of the fleet.

3. **Introduce the data-driven runtime.** `AgentConfig` + a non-generic executor
   driving `AgentRun` against the provider functions. Ships alongside `Agent<M>`;
   nothing breaks yet.

4. **Normalize the response** (Move 4). Concrete `CompletionResponse` with
   `finish_reason`. This is the one genuinely breaking data-model change — it
   closes #2090 and #1886 at the same time, so land it with those.

5. **Migrate remaining providers, then delete the generic shell.** `Agent<M>`,
   `AgentRunner<M>`, `CompletionModel`, `CompletionRequestBuilder<M>`,
   `Client<Ext, H>` all go in one major release.

6. **Repeat for Tier 3 modalities**, embeddings first.

---

## 6. What you give up

Stated plainly, because these are real:

- **Compile-time provider/model coupling.** `ModelRef` is a string pair; a typo
  or an unsupported-capability request becomes a runtime error instead of a
  compile error. Descriptor fields (`supports_tools`, …) let you fail fast at
  request-build time, but it is a check, not a proof.
- **Out-of-tree providers lose the convenience layer.** They keep the public
  `AgentRun` protocol but cannot add an arm to the facade enum (§2).
- **The `Ext` code-sharing mechanism.** ~20 OpenAI-compatible providers currently
  share an implementation through `GenericCompletionModel<Ext, H>`. Descriptor
  data plus shared functions replaces it, but the migration is per-provider work
  (35 providers), and provider authors' mental model changes from "implement a
  trait" to "supply a descriptor and two functions."
- **Provider-typed raw responses at the public boundary.** They remain reachable
  through the provider's own parse function, but they are no longer threaded
  through the generic runtime. Affects the candle tests, two Gemini examples, and
  a few cassette tests.
- **The typed builder chain.** `client.agent(GPT_5_2)` returning a
  provider-specialized builder becomes a data-configured one. The ergonomics can
  be preserved with per-provider constructor helpers; the type-level guarantee
  cannot.
- **Migration size.** This is a multi-release rearchitecture across ~35 providers
  and 10 store crates, not a refactor. Phases 1 and 2 are individually valuable
  and reversible; commit to those before the rest.

## 7. Why this is worth it for rig-ecs specifically

The payoff is not that generics are ugly. It is that `Agent<M>` is *N* component
types — one per provider — so a scheduler cannot write a system over "all
agents," and a world cannot hold a heterogeneous set of them. `AgentConfig` +
`AgentRun` is one component pair regardless of provider count.

The second payoff is that the effect protocol is already what an ECS wants:
`next_step()` is a system tick, `CallModel`/`CallTools` are commands spawned as
entities, `model_response`/`tool_results` are the fulfilment write-back, and the
whole run state is serde — so save/load, rollback, and replay come free. That is
why extending `AgentRun` outward is the right shape, and why erasing `M` inside
`Agent<M>` would have left you with an ECS-hostile design wearing a vtable.
