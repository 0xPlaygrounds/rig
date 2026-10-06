# TYPED_OPTIONS: portable options, model catalog, provider extras, raw shape, citations and cost

This reference fixes the design of five additions to `rig-core` and its
companion provider crates. A coding harness must be able to switch provider
and model at runtime without writing provider JSON and without losing type
safety. Each addition makes one bug class impossible to write:

| addition | bug class it removes |
|---|---|
| `GenerationOptions` | a portable knob written as provider JSON, or silently dropped by a wire that cannot send it |
| the model catalog | model facts (levels, limits, prices) scattered through encoders as name patterns |
| `ProviderExtension` options and extras | provider JSON sent to the wrong provider; reply fields read by indexing `raw` |
| one `raw` shape | a streamed `raw` that differs from the unary `raw` of the same turn |
| citations and cost | citations reachable only through native JSON; cost computed by every caller |

The work lands as a stack of seven pull requests. Each later phase is stacked
on the one before it:

| phase | title | breaking | adds |
|---|---|---|---|
| P0 | design and acceptance tests | no | this file; the gated acceptance tests |
| P1 | make room | yes | `#[non_exhaustive]`, constructors, empty new fields, `cost` in `Usage`'s `Add`; drops `Eq` from `Usage`; no behaviour change |
| P2 | `GenerationOptions` on every completion wire | yes: `ReplayTarget` gains a required `map_options`; deletes the caching knobs `cache` fully replaces; the body changes in section 12.0 | guarantees 1 and 2; deletes the knobs in section 12.1 |
| P3 | model catalog and connect-by-reference | no | guarantee 4; `cargo xtask catalog sync` |
| P4 | typed provider options and extras | yes: deletes three provider setters a typed body option replaces | guarantee 3; decision B; deletes the knobs in section 12.3 |
| P5 | streamed replies carry the unary `raw` shape | yes: a completion `Wire` names a `Reassembler`; streamed `raw` values change | guarantee 5; decision D |
| P6 | citations and cost in core types | no | decision E; `Usage.cost` |

The acceptance tests live in `crates/rig-cassette/tests/runtime/typed_options/tests.rs`
(section 13). Each group is compiled out with `#[cfg(any())]`; the phase
named on the gate deletes it, and the tests must then pass unchanged.

Notation: `file:line` is relative to the repository root at `7dfd8a422`
unless it starts with `references/`. "[unverified]" marks a cell no vendor
page or recording confirms. One rule holds for every such cell, including
one that shows JSON: P2 and later phases answer it `Mapping::Unsupported`,
with the reason "unverified for <provider>", until a vendor page or a
recording confirms it. The JSON in such a cell is what the wire sends once
it is confirmed. The `option_matrix` golden expects a refusal for it.

## 1. The five guarantees

Each guarantee has a type-level core: a compiler error, or a private
constructor that only one function can call. A guard or a test backs the
core where the compiler cannot reach, and is never the guarantee on its own.
Each phase shows the attempted code and the error in its PR body. What
in-crate code can still do despite the core is listed in section 1.1, not
hidden in the mechanism.

| # | guarantee | type-level core | backed by | phase | attempted code and the error |
|---|---|---|---|---|---|
| 1 | No wire can silently drop an option. | Every completion wire's `ReplayTarget` must implement `fn map_options(&self, request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap`, which has no default, so a missing one is E0046. `Completion::prepare`, which the driver runs before every completion encode and which already refuses a wire that names no replay target, calls it on the routed target and reports each `Unsupported` slot through `on_unsupported` and fails a set option whose slot is `Nothing`, before `encode` runs. This holds for every wire, SDK-backed and local ones included, whatever its `encode` does. `map_options` destructures `OptionFields` with no `..` and returns an `OptionMap` struct literal with one `Mapping` per option (section 2.1). `GenerationOptions::fields` destructures `GenerationOptions` with no `..` inside rig-core. A new option breaks `fields` (E0027), then every wire's pattern (E0027) and every wire's `OptionMap` literal (E0063). The result is typed per field, so a wire cannot answer for one option in another option's slot. | The `options-mapping` guard rejects `..`, `_` and `_`-prefixed bindings in an `OptionFields` pattern, and a `..base` in an `OptionMap` literal. The `option_matrix` golden checks each slot against its section 6 cell and the body the wire sends, so `Omit` passes only where the cell says "omit", and a `Send` the wire never writes into its request fails. | P2 | Add `pub logprobs: Option<bool>` to `GenerationOptions`: `fields` fails with `error[E0027]: pattern does not mention field 'logprobs'`. Then add it to `OptionFields` and `OptionMap`: every wire fails with `error[E0027]` and `error[E0063]: missing field 'logprobs' in initializer of 'OptionMap'`. `impl ReplayTarget for Converse` without `map_options`: `error[E0046]: not all trait items implemented, missing: 'map_options'`. `OptionFields { reasoning, .. }`: `source-guards` fails `options-mapping: crates/rig-core/src/providers/openai/wire/chat.rs:<line>: an OptionFields pattern uses '..'`. `seed: Mapping::Omit("default")` on OpenRouter, whose cell sends `"seed":n`: `option_matrix` fails `openrouter: seed: expected {"seed":7}, the body did not change`. |
| 2 | Precedence lives in one place. | `options::request_params` is the one merge: base body, mapped options, provider options (P4) and `additional_params`, in that order. It returns a `FinalBody`, whose field is private to `completion::options`, so no wire can construct one. `FinalBody` has no `&mut` access, and `FinalBody::into_body` is how a completion wire gets its request bytes. | The `options-precedence` guard, over the completion-wire files (section 2.1), rejects reading `additional_params` as a field, the `Body::Bytes` and `Body::Multipart` constructors, any `.body(..)` call whose argument is not `FinalBody::into_body()` or `Body::empty()`, and `FinalBody::deserialize` turbofished to `Value` or `Map`. It sees only the listed files, so it does not stop every copy of a `FinalBody` into JSON, nor a `Body` built by a helper in an unlisted file (section 1.1). | P2 (merge, `FinalBody`, guard), P4 (provider layer) | `body.insert("cache_control".into(), top)` on a `FinalBody`: `error[E0599]: no method named 'insert' found for struct 'FinalBody'`. `FinalBody(map)` in a wire: `error[E0423]: cannot initialize a tuple struct which contains private fields`. `let raw = request.additional_params.clone();` in `anthropic/completion.rs`: `options-precedence: crates/rig-core/src/providers/anthropic/completion.rs:<line>: reads additional_params outside request_params`. `Body::Bytes(serde_json::to_vec(&body)?)` in a completion wire: `options-precedence: <file>:<line>: builds a request body without FinalBody::into_body`. |
| 3 | The decoding path cannot depend on typed views. | Each provider's `Options` and `Extras` live in `providers::<p>::extension`, a module no other provider module names. Rust has no visibility that hides a module from its siblings while keeping it public, so the prompt makes a source guard this guarantee's mechanism. | The `extras-off-decode-path` guard parses each file with `syn` and resolves every `use` tree (grouped, aliased, glob, `self`, `super`) before matching. Outside an `extension` module, no non-test file under rig-core's `providers` tree or a companion provider crate's `src` may name a path through an `extension` module, or `ReplyExtras` or `ProviderExtension`. A `pub use` or `pub type` of an extension item is rejected everywhere. The guard's own tests cover each import form (section 2.5). | P4 | `use crate::providers::openrouter::{extension as ext};` in `openai/wire/chat.rs`: `source-guards` fails `extras-off-decode-path: crates/rig-core/src/providers/openai/wire/chat.rs:<line>: names crate::providers::openrouter::extension`. `fn read<P: ProviderExtension>(raw: &Value)` in a decoder: `<file>:<line>: names ProviderExtension`. |
| 4 | Every public model constant has a catalog entry. | The catalog is data, and the constants are `pub const`. No type links them, so this guarantee is a guard test, as the prompt specifies. | A guard test walks every `pub const <NAME>: &str` model constant in rig-core and the companion provider crates and looks it up in `Catalog::builtin()`. | P3 | Add `pub const GPT_7: &str = "gpt-7";` with no entry: `every_public_model_constant_has_a_catalog_entry` fails `openai::GPT_7 ("gpt-7") has no catalog entry`. |
| 5 | Streamed and unary `raw` agree for every API. | `Out::raw` moves behind `Emit = Free`, so a completion decoder cannot write `raw` (E0599). Every `Wire` names an associated `type Reassembler: Reassemble<Self::Frame>`. The driver, not the decoder, feeds it every frame of a stream and records its `finish()` as `raw`, so no decoder can skip a frame. | A whole-document parity test runs over every recorded unary and streamed pair. | P5 | `out.raw(json!({"response_id": "x"}))` in a completion decoder: `error[E0599]: the method 'raw' exists for struct 'Out<'_, Completion>', but its trait bounds were not satisfied`. An `impl Wire` without a `Reassembler`: `error[E0046]: not all trait items implemented, missing: 'Reassembler'`. |

Decisions B and E have type checks of their own (sections 3 and 5). B gives
E0308, E0609 and E0616 on misplaced provider options. E gives E0451 and
E0616 on hand-written spans and citation lists.

### 1.1 Known limits

These are the bypasses the type-level cores leave open inside a crate. Each
is listed with what covers it.

- **Guarantee 1.** The compiler checks that every wire answers for every
  option, and `prepare` reports every refusal, but not that the answer is
  right or that the wire keeps it. A wire may return `Omit` where its cell
  says `Send`, or return `Send` and leave the value out of what it encodes.
  The second matters for the wires `request_params` does not bind (the
  first guarantee 2 bullet below). The `option_matrix` golden catches both for every wire and
  dialect in its table, because it compares the encoded request with the
  cell. P2 extends the table to all of them.
- **Guarantee 1.** `Descriptor::replay` is an `Option`, so a completion wire
  that names no replay target compiles without `map_options`. It cannot send
  anything: `Completion::prepare` refuses every request it is given, as it
  does today.
- **Guarantee 1.** In-crate code can still build a `CompletionRequest`, so a
  wire could encode a copy of the request with default options. The
  `options-mapping` guard rejects `CompletionRequest` struct expressions,
  `CompletionRequest::new`, `.options(..)` calls and assignments to
  `.options` in completion-wire files. A helper outside those files is not
  covered.
- **Guarantee 2.** Nothing forces an SDK-backed completion wire (Bedrock
  Converse, Vertex, Gemini gRPC) or Candle to call `request_params`. They
  must still implement `map_options`, and `prepare` still reports their
  refusals (guarantee 1). An HTTP wire is held to `request_params` because
  the `options-precedence` guard lets its listed files build request bytes
  from `FinalBody::into_body` alone; these wires build an SDK or local
  request type, so that rule does not reach them. For them only the
  `option_matrix` golden catches a mapped value or a raw key that never
  reaches the request, once P2 extends it to every wire.
- **Guarantee 2.** `Encoded` and `http::Request` keep public constructors,
  because every non-completion wire uses them. `Body::Bytes`,
  `Body::Multipart` and any `.body(..)` on an `http::request::Builder` are
  therefore not a compile error in a completion wire. The
  `options-precedence` guard is the barrier there, and it covers only the
  listed completion-wire files.
- **Guarantee 2.** The SDK-backed wires (Vertex, Gemini gRPC) and Candle
  turn the `FinalBody` into their own request type with
  `FinalBody::deserialize::<T>()`, and Bedrock serializes it to bytes for
  its SDK call. The compiler cannot stop a write to that typed value
  afterwards. Section 12.1 moves every write that exists today (Vertex
  `contents` and `model`) into the base builder.
- **Guarantee 2.** A `FinalBody` can still be copied into a mutable JSON
  value. `FinalBody` derives `Serialize` (Bedrock sends it as bytes), so
  `serde_json::to_value(&body)` works, and so does `let v: Value =
  body.deserialize()?`. The guard sees only `deserialize` spelled with a
  turbofish to `Value` or `Map`; it cannot type a binding. Inside the listed
  completion-wire files such a copy cannot become request bytes, because
  the guard rejects every `Body` constructor and every `.body(..)` argument
  but `FinalBody::into_body()` and `Body::empty()`. A helper in an unlisted
  file (for example under `providers/internal`) can still build a `Body` or
  a whole `http::Request` from such a copy and hand it to a listed file
  through `http::Request::new(..)` or `Encoded::new(..)`: at the use site
  that is an opaque call, and the guard does not see it. On the SDK-backed
  wires and Candle the copy can be sent directly. In both cases nothing but
  review stops it.
- **Guarantee 2.** Code reads raw keys through `options::param`,
  `options::mapped_param` and `BaseInput::param` (section 2.1). Reading cannot change rank, and
  `BaseInput::raw_tools` can take out only `tools`, which every wire
  appends today.
- **Guarantee 3.** The guard resolves names within a file. A macro that
  expands to an extension path is not seen. No provider module defines such
  a macro today, and the guard rejects `macro_rules!` that mention
  `extension`.
- **Guarantee 5.** The type ensures every frame reaches the reassembler. It
  cannot ensure the reassembler folds each frame correctly. That is the
  parity test's job, and Bedrock, DeepSeek, Venice, xAI Chat and
  Interactions have no recorded pair. Their reassemblers are checked by unit
  tests only.
- **Guarantee 5.** `CompletionResponse::raw` stays a public field. Code that
  runs after the driver (a session layer, the caller) can still overwrite
  it. The guarantee covers what the driver records.
- **Decision E.** `Text::with_citations` and `Text::span` stay public, so a
  caller can attach a citation built from offsets in the wrong unit. `span`
  takes byte offsets into that text and refuses a range off a character
  boundary, but it cannot know what the offsets meant. Decoders cannot call
  either: the `extras-off-decode-path` guard also rejects them under the
  `providers` tree (section 5).

## 2. Fixed decisions

### 2.1 A: `GenerationOptions`

`GenerationOptions` lives in `rig_core::completion::options` and is
re-exported from `rig::completion`. It holds only knobs `CompletionRequest`
does not already have. `temperature`, `max_tokens`, `tool_choice` and
`output_schema` stay on the request.

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub reasoning: Option<Reasoning>,
    pub cache: Option<CacheRetention>,
    pub service_tier: Option<ServiceTier>,
    pub verbosity: Option<Verbosity>,
    pub parallel_tool_calls: Option<bool>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub on_unsupported: OnUnsupported, // default Error
}
#[non_exhaustive] pub enum Reasoning { Off, Effort(Effort), Budget { tokens: u32 } }
#[non_exhaustive] pub enum Effort { Minimal, Low, Medium, High, XHigh, Max }
#[non_exhaustive] pub enum CacheRetention { None, Short, Long }
#[non_exhaustive] pub enum ServiceTier { Auto, Default, Flex, Priority }
#[non_exhaustive] pub enum Verbosity { Low, Medium, High }
#[non_exhaustive] pub enum OnUnsupported { Error, Ignore }
```

The enums derive `Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize,
Deserialize`. Their serde names are the lower-case wire words (`"xhigh"`, not
`"x_high"`). Every `GenerationOptions` field is `#[serde(default)]`, and
`stop` is skipped when empty.

Builders and the request:

```rust
impl GenerationOptions {
    pub fn reasoning(self, reasoning: impl Into<Reasoning>) -> Self; // From<Effort> for Reasoning
    pub fn cache(self, cache: CacheRetention) -> Self;
    pub fn service_tier(self, tier: ServiceTier) -> Self;
    pub fn verbosity(self, verbosity: Verbosity) -> Self;
    pub fn parallel_tool_calls(self, parallel: bool) -> Self;
    pub fn top_p(self, top_p: f64) -> Self;
    pub fn seed(self, seed: u64) -> Self;
    pub fn stop<S: Into<String>>(self, stop: impl IntoIterator<Item = S>) -> Self;
    pub fn on_unsupported(self, policy: OnUnsupported) -> Self;
    /// Every option but the policy, borrowed. Destructures `self` with no
    /// `..`, so a new field fails to compile here first.
    pub fn fields(&self) -> OptionFields<'_>;
    /// `self` with every field `over` sets put on top: a `Some` option, a
    /// non-empty `stop` list, a non-default `on_unsupported`.
    pub fn overlay(self, over: &GenerationOptions) -> GenerationOptions;
}

// CompletionRequest (field added empty in P1, read in P2)
#[serde(default, skip_serializing_if = "GenerationOptions::is_default")]
pub options: GenerationOptions,
pub fn options(self, options: GenerationOptions) -> Self;
```

`CompletionRequest::additional_params` stays a public field with its
builder, as decision A requires of the escape hatch.

Unsupported options:

```rust
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{option}` is not supported by {provider} model `{model}`: {reason}")]
pub struct UnsupportedOption {
    pub option: &'static str,   // the GenerationOptions field name: "reasoning", "cache", ...
    pub provider: String,       // ReplayTarget::provider()
    pub model: String,          // the resolved request model
    pub reason: String,
}
```

- Under `OnUnsupported::Error` the request fails with
  `EncodeError::UnsupportedOption { option, provider, model, reason }` in the
  form of section 2.4: `ProviderError::UnsupportedOption(UnsupportedOption)`
  carried by `EncodeError`, read with
  `EncodeError::unsupported_option(&self) -> Option<&UnsupportedOption>`. Its
  `ErrorKind` is `Request`, as every encode error's is (`crates/rig-core/src/error.rs:668-671`).
  `Completion::prepare` returns it as the `ProviderError` it converts into,
  before the wire encodes, so a caller of `Model::call` or `stream` matches
  `ProviderError::UnsupportedOption`.
- Under `OnUnsupported::Ignore` the option is skipped with one
  `tracing::warn!` carrying the fields `option`, `provider`, `model` and
  `reason`.
- An option is never silently dropped.

The mapping and the merge (P2), all in `rig_core::completion::options`:

```rust
/// Every option but the policy, as a wire's `map_options` reads it.
/// Not `#[non_exhaustive]`, so wires in companion crates can destructure it
/// whole, which `#[non_exhaustive]` forbids for `GenerationOptions` (E0638).
pub struct OptionFields<'a> {
    pub reasoning: Option<&'a Reasoning>,
    pub cache: Option<&'a CacheRetention>,
    pub service_tier: Option<&'a ServiceTier>,
    pub verbosity: Option<&'a Verbosity>,
    pub parallel_tool_calls: Option<bool>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stop: &'a [String],
}

/// What a wire does with one option.
pub enum Mapping {
    /// The option is unset. Wrong for a set option: `check` fails.
    Nothing,
    /// Deep-merge this JSON object into the body at the mapped rank.
    Send(Value),
    /// Honour the option by sending nothing; the provider default already
    /// does what was asked. Logged at `debug` with the reason.
    Omit(&'static str),
    /// Honour the option through markers the base builder writes inside the
    /// body's arrays (MiniMax block markers, Bedrock `cachePoint`). Valid
    /// for `cache` only.
    Place,
    /// The wire or model cannot honour it: an error under `Error`, a
    /// warning under `Ignore`.
    Unsupported(String),
}

/// One `Mapping` per option, by field name. Not `#[non_exhaustive]` and no
/// `Default`, so a wire must write every field.
pub struct OptionMap {
    pub reasoning: Mapping,
    pub cache: Mapping,
    pub service_tier: Mapping,
    pub verbosity: Mapping,
    pub parallel_tool_calls: Mapping,
    pub top_p: Mapping,
    pub seed: Mapping,
    pub stop: Mapping,
}

/// The base builder's read-only view of the options and the raw params.
pub struct BaseInput<'a> { /* private */ }
impl BaseInput<'_> {
    /// The cache retention the mapping honoured (`Send` or `Place`), for
    /// markers inside the body's arrays.
    pub fn cache(&self) -> Option<CacheRetention>;
    /// A marker the base cannot place (Bedrock after a reasoning turn):
    /// reported through `on_unsupported` under the name `cache`.
    pub fn refuse_cache(&mut self, reason: impl Into<String>) -> Result<(), EncodeError>;
    /// The value `key` gets from the layers above the base, read only:
    /// `param(target, request, key)` below.
    pub fn param(&self, key: &str) -> Option<&Value>;
    /// `additional_params.tools`, taken out of the raw layer for the base to
    /// append after rig's own tools. An error when it is not an array.
    pub fn raw_tools(&mut self) -> Result<Vec<Value>, EncodeError>;
}

/// Where the raw layer merges. Data, not code.
pub enum RawAt {
    /// At the top level of the body (every wire but the two below).
    Top,
    /// Under one object: Bedrock, `"/additionalModelRequestFields"`.
    Under(&'static str),
    /// Ollama native: the keys in `top` (its `TOP_LEVEL` list plus `think`,
    /// `crates/rig-core/src/providers/ollama/chat.rs:37-44`) and the key
    /// `rest` names merge at the top level, every other key under `rest`
    /// (`"options"`), as the wire splits them today.
    Split { top: &'static [&'static str], rest: &'static str },
}

/// The merged body. Its field is private to this module, so only
/// `request_params` builds one. No `&mut` access. `Serialize` is for
/// Bedrock, which hands its SDK the body as bytes.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct FinalBody(Map<String, Value>);
impl FinalBody {
    pub fn is_empty(&self) -> bool;
    pub fn get(&self, key: &str) -> Option<&Value>;
    pub fn pointer(&self, pointer: &str) -> Option<&Value>;
    /// The request bytes of an HTTP completion wire.
    pub fn into_body(self) -> Body;
    /// The request type of an SDK-backed wire (Bedrock, Vertex, gRPC) or
    /// Candle's generation overrides.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error>;
}

/// The writes `request_params` makes after the merge, each one a write a
/// wire makes after its `body.extend(params)` today. Each reads the merged
/// body, raw keys included, as it does today.
#[non_exhaustive]
pub enum Rewrite {
    OutputCapRename,
    DropUnboundThinking,
    ToolChoiceNeedsTools,
    StreamFlag(Mode),
    ReasoningCiphertext,
    CodexStore,
    /// The arms of `Chat::rewrite` that survive P2, for this dialect.
    ChatDialect(BodyRewrite),
    /// Gemini `cachedContent`: the wire's handle, plus the raw
    /// `cachedContent`/`cached_content` handles, which `request_params`
    /// takes out of the raw layer as it takes `tools`.
    GeminiCachedContent(Option<String>),
}

/// The value top-level `key` gets from the layers above the base: the
/// provider options for `target` (P4), then `additional_params`. Read only,
/// so it cannot rank anything. The one sanctioned reader of raw keys.
pub fn param<'a>(target: &dyn ReplayTarget, request: &'a CompletionRequest, key: &str) -> Option<&'a Value>;

/// The value top-level `key` gets from the mapped options and the layers
/// above them, merged as `request_params` merges them, with no base. For a
/// reader that runs before encoding and must agree with the body (Anthropic
/// `drops_unbound_items`). Reports nothing: an `Unsupported` slot adds
/// nothing here.
pub fn mapped_param(target: &dyn ReplayTarget, request: &CompletionRequest, key: &str) -> Option<Value>;

/// Calls `target.map_options` on `request.options.fields()` and reports
/// each `Unsupported` slot through `on_unsupported`: an error under `Error`;
/// under `Ignore` a warning, and the option is cleared on `request`, so it
/// is reported once. A set option answered `Nothing` is an error. Called by
/// `Completion::prepare` before every completion encode.
pub fn check(target: &dyn ReplayTarget, request: &mut CompletionRequest) -> Result<(), EncodeError>;

/// The body `target` sends for `request`. Runs `check` on its own copy (a
/// no-op after `prepare`; it catches an `encode` called without one), calls
/// `target.map_options`, then `base`, then merges the layers, then applies
/// `rewrites`. The only reader of `request.options` and of
/// `additional_params` on the completion path.
pub fn request_params(
    target: &dyn ReplayTarget,
    request: &CompletionRequest,
    base: impl FnOnce(&mut BaseInput<'_>) -> Result<Map<String, Value>, EncodeError>,
    raw_at: RawAt,
    rewrites: &[Rewrite],
) -> Result<FinalBody, EncodeError>;

// rig_core::completion::ReplayTarget, a new required item
/// How this wire answers for each option, for `request` (the resolved
/// model and the typed fields it may conflict with). No default: every
/// completion wire writes one. Destructures `fields` with no `..`.
fn map_options(&self, request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap;
```

`map_options` uses the mechanism P5 uses for `type Reassembler`: a
required trait item, so an implementor without one is E0046. It sits on
`ReplayTarget`, not on `Wire`, because `ReplayTarget` is the trait only
completion wires implement, and
`Completion::prepare` already requires one, so non-completion wires need no
placeholder. A wire that picks its API per request answers through the
target `route` names, the one `prepare` and the fold already use.

- **Order.** `Completion::prepare` runs `check` on the routed target before
  the wire encodes, so every completion wire, whatever its `encode` does,
  reports each `Unsupported` slot and fails a set option whose slot is
  `Nothing` before any request is built. `request_params` then calls
  `map_options` for the values to merge. `base` runs next. It builds the wire's own encoding of
  the request: messages, tools and the typed fields `temperature`,
  `max_tokens`, `tool_choice` and `output_schema`. It places array markers
  from `BaseInput::cache`. The base builder never sees `GenerationOptions`.
- **One rank**, lowest to highest: the base body, mapped options, typed
  provider options, raw `additional_params`. `additional_params` stays the
  documented escape hatch and beats everything below it. A mapped option and
  a typed request field never write the same scalar. Decision A keeps the
  request's fields out of `GenerationOptions`, and where they interact
  (Anthropic `top_p` with `temperature`) the mapping answers `Unsupported`
  rather than overwriting.
- **Deep merge.** Objects merge key by key, recursively. Any other value
  replaces, and so does any array but one. So `additional_params.output_config
  = {"task_budget": ..}` keeps the mapped `output_config.effort` and the base
  `output_config.format`; `parallel_tool_calls: false` adds
  `disable_parallel_tool_use` to the base `tool_choice` object; Responses
  `verbosity` joins the base `text.format`; Gemini options join the base
  `generationConfig`. Today `body.extend(params)` replaces the whole key
  (`crates/rig-core/src/providers/anthropic/completion.rs:256`).
- **`tools` appends.** Every wire appends `additional_params.tools` to rig's
  own tools today: `crates/rig-core/src/providers/anthropic/completion.rs:497-520`,
  `crates/rig-core/src/providers/openai/wire/chat.rs:327-335`,
  `crates/rig-core/src/providers/openai/responses_api/mod.rs:362-370`,
  `crates/rig-core/src/providers/gemini/completion.rs:398-402`,
  `crates/rig-core/src/providers/gemini/interactions_api/mod.rs:380`,
  `crates/rig-core/src/providers/cohere/chat.rs:102` and
  `crates/rig-core/src/providers/ollama/chat.rs:132`. That stays. The base
  builder takes the raw `tools` through `BaseInput::raw_tools` and appends
  them with today's conversions (Responses parses them as
  `ResponsesToolDefinition` and applies strict mode to them). Every wire but
  Interactions already refuses a raw `tools` that is not an array;
  Interactions discards it today, and `raw_tools` makes it an error there
  too (section 12.0). The raw layer
  then holds no `tools` key, so the merge cannot replace rig's tools. The
  `precedence` acceptance test pins this with a function tool plus a raw
  server tool.
- **Raw keys read before the merge.** Some code decides how to encode
  history from a raw key: Responses `store` (`crates/rig-core/src/providers/openai/responses_api/mod.rs:388`),
  the `continues_stored` readers
  (`crates/rig-core/src/providers/openai/responses_api/wire.rs:257-263`,
  `crates/rig-core/src/providers/gemini/interactions_api/mod.rs:136-142`),
  the `declares_tools` readers and the Gemini `cachedContent` handles
  (`crates/rig-core/src/providers/gemini/completion.rs:255-264`, `:403-416`),
  and the Anthropic `container` check (`crates/rig-core/src/providers/anthropic/completion.rs:236`).
  Each reads through `options::param` (or `BaseInput::param` inside the base
  builder), which leaves the key in the merge. Once P4 lands, the same call
  also sees a typed provider option (`OpenAiOptions` `store`), so a key set
  either way gives one answer. Reading cannot change rank, because the base
  is the lowest layer and `continues_stored` and `declares_tools` run in
  `Completion::prepare`, before encoding. The `options-mapping` guard rejects
  `options::param` in a function that destructures `OptionFields`, so a
  mapping cannot defer to a raw key.
- **The WebSocket session writes the raw layer.** It puts the chain's
  `previous_response_id` into `additional_params` before encoding
  (`crates/rig-core/src/providers/openai/responses_api/websocket.rs:371-387`),
  acting as a caller. It is the guard's one allowlisted field access.
- **`null` is a value.** A `null` in a higher layer replaces the lower value
  and is sent as `null`, at any depth. At the top level this is what
  `body.extend` does today on every wire but OpenAI Responses, which skips a
  raw `null` (`crates/rig-core/src/providers/openai/responses_api/mod.rs:464-468`)
  and now sends it (section 12.0). It never clears a key, and there is no way to
  clear one: a caller who relied on `body.extend` replacing a whole object
  to drop a key the wire writes now gets a merge (section 12.0).
- **Post-merge rewrites are a closed list.** Today some wires write keys
  after `body.extend(params)`, so those keys beat `additional_params`. Each
  such write becomes one `Rewrite` variant, applied by `request_params`
  after the merge:
  - `OutputCapRename`: Chat `max_tokens` to `max_completion_tokens`
    (`crates/rig-core/src/providers/openai/wire/chat.rs:391-396`);
  - `DropUnboundThinking`: Anthropic (`crates/rig-core/src/providers/anthropic/completion.rs:257-259`);
  - `ToolChoiceNeedsTools`: Anthropic drops `tool_choice` without tools;
  - `StreamFlag`: `stream: true` and Chat's `stream_options`
    (`crates/rig-core/src/providers/openai/wire/chat.rs:190-198`);
  - `ReasoningCiphertext`: Responses `include`
    (`crates/rig-core/src/providers/openai/responses_api/mod.rs:491-494`);
  - `CodexStore`: `store: false`;
  - `ChatDialect`: `Chat::rewrite` (`crates/rig-core/src/providers/openai/wire/chat.rs:200`,
    `:553-616`), which runs after `body.extend(params)` (`:384`) today:
    Moonshot's `required`-to-`auto` coercion with its appended steering
    message, the DeepSeek and Mistral `tool_choice` relaxation
    (`finalize_deepseek` `:743`, `finalize_mistral` `:764`) and Mistral's
    content chunks, the Perplexity and Mira content flattening (`:598-609`),
    and the llama.cpp, Moonshot and Ollama `/v1` refusals. The OpenRouter
    caching arm and the Ollama `think` rewrite are deleted (section 12.1);
    the Ollama arm keeps its `num_ctx`/`options` refusal until P4;
  - `GeminiCachedContent`: `with_cached_content`, which inserts
    `cachedContent` and checks it against the system instruction, tools and
    tool choice after `body.extend(params)`
    (`crates/rig-core/src/providers/gemini/completion.rs:484-487`,
    `crates/rig-core/src/providers/gemini/cached_content.rs:388-421`).

  Each rewrite reads the merged body, as it does today, so a raw key still
  drives it: a raw `additional_params.reasoning` still adds the ciphertext
  `include`, a raw `thinking: {"type": "disabled"}` still keeps DeepSeek's
  forced `tool_choice`, and a raw `tool_choice` or `messages` is still
  rewritten. The `precedence` acceptance test pins the first two. A closure
  would hand the wire the merged body, which is the wire-local merge
  guarantee 2 rules out. Moving these writes below the merge would let
  `additional_params` override them, a behaviour change. So the list stays,
  as data. Anthropic's top-level `cache_control`, inserted after the
  merge today (`completion.rs:260`), becomes a mapped option instead, so a
  raw `additional_params.cache_control` beats it by rank.
- **Checks on the final body.** A wire may read the `FinalBody` and return an
  error, never write. The Anthropic 1 h-before-5 min and budget-of-4 checks,
  Gemini's cached-content conflicts, and Candle's refusal of hosted `tools`
  are such reads.
- **Completion-wire files.** The guards of guarantees 1 and 2 cover a
  checked-in list in xtask: each file that encodes a completion request,
  with its helpers and the Bedrock, Vertex, gRPC and Candle request builders.
  A second rule fails when a file holding an `impl Wire` with `type Op =
  Completion`, or an `impl ReplayTarget`, is missing from the list. Image, transcription and audio
  requests keep their own `additional_params`
  (`crates/rig-core/src/providers/openai/wire/modality.rs:713-767`,
  `crates/rig-core/src/providers/gemini/image_generation.rs:79`), and those
  files are not on the list.
- **Model-dependent shapes before P3.** Where a cell's JSON depends on the
  model (Anthropic `Off` as `disabled` or `between_tools`; OpenAI `Long` as
  `prompt_cache_retention` or `prompt_cache_options`), P2 chooses the shape
  from the in-code model facts that exist today (section 12.2). P3 replaces
  those facts with catalog lookups. Before P3, a level the model rejects
  reaches the provider and fails there: an error, not a silent drop.
- **Elsewhere.** P2 adds `AgentBuilder::options(GenerationOptions)` in
  rig-agent and an `Options(GenerationOptions)` component in rig-ecs. The
  two layers merge with `GenerationOptions::overlay`: `agent.overlay(&run)`,
  so each field the run sets beats the agent's, and every other field keeps
  the agent's value. A `stop` list replaces; it never concatenates. Because
  `on_unsupported` has no unset state, a run cannot put `Error` back over an
  agent's `Ignore`. The caller sets it on the agent instead. The
  `option_layers` acceptance test pins this (section 13).

### 2.2 C: the model catalog

The catalog lives in `rig_core::catalog`, built on `providers::registry`
(`ProviderId`, `ProviderRef`, `ProviderConfig`) and `completion::ModelRef`.

```rust
#[non_exhaustive]
pub struct ModelSpec {
    pub id: String,
    pub provider: ProviderId,
    pub display_name: String,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub input: Modalities,
    pub reasoning: ReasoningSupport,
    pub caching: CacheSupport,
    pub tools: bool,
    pub structured_output: bool,
    pub pricing: Option<Pricing>,
    pub deprecated: bool,
}
#[non_exhaustive] pub struct Modalities { pub text: bool, pub image: bool, pub audio: bool, pub video: bool, pub pdf: bool }
#[non_exhaustive] pub struct ReasoningSupport {
    pub levels: Vec<Effort>,
    pub budget: Option<RangeInclusive<u32>>,
    pub can_disable: bool,
    pub default: Option<Effort>,
}
#[non_exhaustive] pub struct CacheSupport { pub retention: Vec<CacheRetention> } // the values the model honours
#[non_exhaustive] pub struct Pricing {          // USD per million tokens
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
}

impl Catalog {
    pub fn builtin() -> &'static Catalog;
    pub fn get(&self, provider: ProviderId, model: &str) -> Option<&ModelSpec>;
    pub fn resolve(&self, reference: &str) -> Option<&ModelSpec>; // "anthropic/claude-opus-5-5"
    pub fn iter(&self) -> impl Iterator<Item = &ModelSpec>;
    pub fn from_json(json: &str) -> Result<Catalog, CatalogError>; // a models.dev-style override file
    pub fn merge(self, overrides: Catalog) -> Catalog;
}
impl ModelSpec {
    pub fn validate(&self, options: &GenerationOptions) -> Result<(), UnsupportedOption>;
}
```

- **The data** is a checked-in JSON file under `crates/rig-core/src/catalog/`.
  `cargo xtask catalog sync` produces it from models.dev, and the main
  providers are reviewed by hand (section 8).
- **`validate`** checks `reasoning` against `levels`, `budget` and
  `can_disable`, and `cache` against `caching.retention`. `option` names the
  field, `provider` is `provider.vendor()`, `model` is `id`.
- **`get`** matches `provider.vendor()`. A vendor served over two formats
  (Z.AI, MiniMax, Moonshot, MiMo) has one entry per model, because both
  halves serve the same ids.
- **Connecting.** `registry::connect(spec_or_reference, credentials)` returns
  the existing `DynModel<Completion>` that `ProviderRef::completion_model`
  builds (`crates/rig-core/src/providers/registry.rs:662`).

### 2.3 Cost

```rust
// Usage (field added empty in P1, filled in P6)
#[serde(default, skip_serializing_if = "Option::is_none")]
pub cost: Option<Cost>,

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost { pub input: f64, pub output: f64, pub cache_read: f64, pub cache_write: f64, pub total: f64 } // USD
```

- A cost the provider reports wins.
- Otherwise the driver computes it from the catalog pricing of the resolved
  request model (`Origin::model`).
- Otherwise it is `None`.
- Cost is never part of token arithmetic: no counter is derived from it and
  it is derived from no counter except through `Pricing`.
- A provider that reports only a total gives `total` its figure and the
  parts `0.0`. Section 11 lists which providers split.
- Uncached input is `input_tokens - cached_input_tokens -
  cache_creation_input_tokens`, because `Usage::input_tokens` counts cache
  reads and writes (`crates/rig-core/src/completion/request.rs:470-485`).
  Cache reads without a `cache_read` price are charged at `input`; cache
  writes without a `cache_write` price likewise.

Summing usage (`Add` and `AddAssign`, `crates/rig-core/src/completion/request.rs:527-549`,
which sum only the token counters today). P1 extends them to `cost`:

| `self.cost` | `other.cost` | result |
|---|---|---|
| `Some(a)` | `Some(b)` | `Some`, each of `input`, `output`, `cache_read`, `cache_write` and `total` summed |
| `Some(_)` | `None` | `None` |
| `None` | `Some(_)` | `None` |
| `None` | `None` | `None` |

A side that reports no counter and no cost (`!is_reported()`, the empty
`Usage::default()`) is the identity: the sum is the other side, cost
included. So a fold that starts from `Usage::default()` keeps the cost of
its first turn.

`Some + None` is `None` because the `None` side is a turn whose cost is
unknown. A partial sum would read as the cost of every turn and understate
it. The token counters keep today's rule, where an unreported side adds
nothing. P1 lands this
with `cost` always `None`, so every existing sum is unchanged. The
`usage_cost_sum` acceptance test pins the table (section 13).

### 2.4 Conflicts with the fixed decisions, and their resolutions

| conflict | resolution | phase |
|---|---|---|
| `EncodeError` is a struct wrapping `ProviderError` (`crates/rig-core/src/error.rs:659`), not an enum, so `EncodeError::UnsupportedOption { .. }` cannot be a variant. | `ProviderError::UnsupportedOption(UnsupportedOption)` with the fixed field set, kind `Request`, plus `EncodeError::unsupported_option(&self) -> Option<&UnsupportedOption>` and `EncodeError::unsupported(UnsupportedOption) -> Self`. | P1 |
| `Usage` derives `Eq` (`crates/rig-core/src/completion/request.rs:484`) and `Cost` holds `f64`. | P1 drops `Eq` from `Usage` (keeps `PartialEq`) and from every type that derived it only through `Usage`. Breaking; stated in P1's Migration. `Cost` stays `f64`. | P1 |
| `#[non_exhaustive]` forbids an exhaustive pattern outside rig-core (E0638), so Bedrock, gRPC and Candle cannot destructure `GenerationOptions` without `..`. | `OptionFields<'_>`, a plain struct of borrowed options with no policy, built by `GenerationOptions::fields`. `fields` destructures `self` exhaustively in rig-core, so a new option breaks it first. The new `OptionFields` and `OptionMap` fields then break every wire, in rig-core and companion crates alike: all of them destructure `OptionFields` and build `OptionMap`, never `GenerationOptions`. | P2 |
| `Catalog::resolve("anthropic/claude-opus-5-5")` collides with `ProviderRef`'s grammar `vendor[/format]:model` (`crates/rig-core/src/providers/registry.rs:623-636`), where `/` separates the format. | `resolve` reads `vendor[/format]:model` when the text holds a `:`, and otherwise splits at the first `/` as `vendor/model` (so `openrouter/anthropic/claude-sonnet-4.5` is OpenRouter's model `anthropic/claude-sonnet-4.5`). | P3 |
| `ModelSpec.provider: ProviderId` cannot name `aws_bedrock`, `vertexai`, `gemini-grpc` or `candle`: the registry registers the OpenAI, Anthropic and Gemini formats only (`crates/rig-core/src/providers/registry.rs:63-73`). Guarantee 4 still needs entries for rig-bedrock's constants. | Proposed: P3 adds a catalog-only `ProviderId` kind for providers the registry cannot configure. `ProviderId::new` keeps rejecting them, `get` and `validate` work, and `connect` returns an error naming the companion crate. **For the user to confirm.** | P3 |
| The harness-switch test asks for a long cache on Gemini, and GenerateContent has no request field for it: explicit caching is a separate `cachedContents` resource (`crates/rig-core/src/providers/gemini/cached_content.rs:388-430`). | `CacheRetention::Long` on Gemini GenerateContent, Vertex, gRPC and Interactions is `UnsupportedOption`. The test runs under `OnUnsupported::Ignore` and asserts the warning, and asserts the error under `Error`. Mapping `Long` to the `Caching` transport is a later extension. | P2 |
| Decision D's text reads extras as one `Deserialize` of `raw`; decision B needs per-route dispatch (`OpenAiExtras::{Chat, Responses}`). | B's `ReplyExtras::from_reply(api, raw)` is the contract. A one-route provider implements it as `serde_json::from_value(raw.clone())`. | P4 |
| `Pricing` has one `cache_write` price; Anthropic and Bedrock bill 1 h writes at 2x input and 5 min writes at 1.25x, and Anthropic's reply splits them (`usage.cache_creation`). Tier, fast-mode, US-only and context-tier prices are absent too. | P6 prices every write at `cache_write` and documents catalog cost as a lower bound for those cases. `Pricing` is `#[non_exhaustive]`, so a later PR can add `cache_write_1h` and tiers without a break. | P6 |
| Decision C gives `Pricing` four prices, "input, output, cache_read, cache_write", none optional. models.dev rows often carry `cost.input` and `cost.output` with no `cache_read` or `cache_write`, and a `0.0` would claim cache reads are free. | `Pricing { input: f64, output: f64, cache_read: Option<f64>, cache_write: Option<f64> }`. `None` means unknown, and cost prices those tokens at `input` (section 2.3). **For the user to confirm.** | P3 |
| The prompt fixes P1 as the only breaking phase, and also requires each phase to delete the knobs it replaces. P2 deletes the knobs `GenerationOptions` fully replaces, and only those: `Messages::with_automatic_caching`, `with_automatic_caching_1h` (`crates/rig-core/src/providers/anthropic/wire.rs:410`, `:429`) and the pub fields `automatic_caching`, `automatic_caching_ttl` (`wire.rs:367-369`), replaced by `cache(Short)` and `cache(Long)`; `Chat::with_prompt_caching` (`crates/rig-core/src/providers/openai/wire/chat.rs:266`), `OpenAiWire::with_prompt_caching` (`crates/rig-core/src/providers/openai/wire/route.rs:109`) and `Converse::with_prompt_caching` (`crates/rig-bedrock/src/completion.rs:199`), replaced by `cache(Short)`. P4 deletes the setters a typed body option replaces: `Converse::with_guardrail` (`crates/rig-bedrock/src/completion.rs:210`), `GenerateContent::with_cached_content` (`crates/rig-core/src/providers/gemini/completion.rs:121`) and Cohere's `with_strict_tools` (`crates/rig-core/src/providers/cohere/wire.rs:123-128`). Both also change wire bodies for callers who set no option (section 12.0). | P2 and P4 are titled `feat(...)!` and carry a Migration entry for every deleted item and every body change in section 12.0. Keeping a knob next to the option that replaces it would leave two ways to set one thing. Knobs that choose placement rather than retention (Anthropic `with_prompt_caching`, `with_static_prefix_cache_ttl`) are not replaced, so they stay unchanged. **For the user to confirm.** | P2, P4 |
| The prompt lists "the copied caching state behind `Model::thought_replay`" among the knobs to absorb (`crates/rig-core/src/client/gemini_caching.rs:69`, `:84-92`, `:354-359`). `ThoughtReplay` edits the encoded `contents`; it is neither a portable option nor a body key, and the caching transport sees only the encoded request. | Not absorbed in this stack. `thought_replay` stays a wire setter, and the "call it before `caching`" rule stays documented. Absorbing it needs a transport API that reads per-request wire settings, which none of the five additions provides. **For the user to confirm.** | none |

Conflicts with open pull requests, which this stack does not build on:
- a Gemini rewrite on a generated API mirror stops reading `additional_params`, which contradicts decision A's escape hatch;
- an error-model rewrite renames the error report type that `EncodeError` converts into;
- a change puts each built-in provider behind a Cargo feature, so `catalog::connect` and each `extension` module must allow a `cfg(feature)` gate.

Section 14 names them.

### 2.5 The source guards

Three new checks join `cargo xtask verify --check source-guards`. Each one
parses files with `syn` (already an xtask dependency, `xtask/Cargo.toml:13`),
not with text patterns. Each first resolves every `use` tree in the file
into a map from local name to full path, covering grouped (`{a, b}`),
renamed (`x as y`), glob (`x::*`), `self` and `super` forms. A guard skips
`#[cfg(test)]` items and `tests.rs` files.

| check | phase | files | rejects |
|---|---|---|---|
| `options-mapping` | P2 | the completion-wire files (section 2.1) | in a pattern of type `OptionFields`: `..`, a `_` binding, a binding whose name starts with `_`; in an `OptionMap` struct expression: a `..base`; a `CompletionRequest` struct expression, a call that resolves to `CompletionRequest::new`, a method call `.options(..)`, an assignment to a field `options`; a field access `.options`; a call that resolves to `options::param` inside a function that destructures `OptionFields` |
| `options-precedence` | P2 | the completion-wire files | a field access `.additional_params`, but for the one allowlisted site in `responses_api/websocket.rs`; any path that resolves to `Body::Bytes` or `Body::Multipart` (`Body` has no `From` impl, so these and `Body::empty` are its only constructors); a method call `.body(x)` with one argument, whatever its receiver (`http::Request::builder(..)`, `http::Request::get(..)`, `post(..)` or a stored `http::request::Builder`; the guard cannot type it), unless `x` is a `FinalBody::into_body()` call or the constant `Body::empty()` (a body-less request such as Interactions resume); `FinalBody::deserialize` turbofished to `Value` or `Map`; `serde_json::to_value`, `to_vec` or `to_string` applied to a binding of type `CompletionRequest` |
| `extras-off-decode-path` | P4 | every non-test file under `crates/rig-core/src/providers/` and the `src` of each companion provider crate, outside `providers::<p>::extension` modules | any resolved path through a module named `extension`; the names `ReplyExtras` and `ProviderExtension`; `pub use` or `pub type` of an extension item; a call that resolves to `Text::with_citations` or `Text::span`, and a `Citation` or `Span` struct expression (decision E); `macro_rules!` whose body names `extension` |

The guards have unit tests in `xtask/src/verify/tests.rs`, one per form,
each a source string that must fail:

- `use crate::providers::openrouter::{extension as ext};`;
- `use crate::providers::openrouter::{self, extension::*};`;
- `use super::super::openrouter::extension;` from a sibling provider;
- `use crate::completion::provider_options::ProviderExtension as Ext;`;
- `use crate::completion::options::{OptionFields as F};` and then `F { reasoning, .. }`;
- `let OptionFields { stop: _stop, .. } = fields;`.

They also have one source string per form that must pass, for example a
`use` of `extension` inside an extension module.

## 3. Decision B: provider extensions keyed by provider name, with route sections

**Decided.** One `ProviderOptions` entry per provider name. The name is the
string the wire reports as `ReplayTarget::provider()` and stamps as
`Origin::provider` (`crates/rig-core/src/completion/message/native.rs:61-66`).
Inside an entry, section `"*"` holds fields every route of that provider
spells the same, and a section named by an `Api` string (`"openai.chat"`,
`"openai.responses"`, `"cohere.chat"`, ...) holds route-only fields.

**Mechanism.**
- `ProviderOptions::insert::<P>(&P::Options)` keys the entry by `P::PROVIDER`,
  never by a caller string.
- At encode time `request_params` takes the entry under `target.provider()`,
  merges `"*"`, then the section named `target.api()`, under
  `additional_params`. The route is whatever wire actually encodes; the caller
  never predicts it.
- `CompletionResponse::extras::<P>()` returns `None` unless
  `origin.provider == P::PROVIDER`.
- An entry or section the route actually taken cannot send is reported
  through `on_unsupported` (an error by default), not merely warned. This
  replaces the hand-written refusal in `finalize_ollama`
  (`crates/rig-core/src/providers/openai/wire/chat.rs:692-701`).
- An `Options` type never carries a field `GenerationOptions` or
  `CompletionRequest` owns. A per-extension test serializes a fully-set
  `Options` and checks no reserved key appears.

**Public API.**

```rust
// rig_core::completion::provider_options, re-exported from rig::completion
pub const SHARED: &str = "*";

pub trait ProviderExtension {
    const PROVIDER: &'static str;   // == ReplayTarget::provider()
    type Options: Serialize;        // an object of sections: "*" and Api names
    type Extras: ReplyExtras;
}
pub trait ReplyExtras: Sized {
    fn from_reply(api: &Api, raw: &serde_json::Value) -> Result<Self, serde_json::Error>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderOptions(/* private */ BTreeMap<String, Map<String, Value>>);
impl ProviderOptions {
    pub fn new() -> Self;
    pub fn insert<P: ProviderExtension>(&mut self, options: &P::Options) -> Result<&mut Self, OptionsError>;
    pub fn with<P: ProviderExtension>(self, options: &P::Options) -> Result<Self, OptionsError>;
    pub fn remove<P: ProviderExtension>(&mut self);
    pub fn contains<P: ProviderExtension>(&self) -> bool;
    pub fn is_empty(&self) -> bool;
}
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum OptionsError {
    Serialize { provider: &'static str, source: serde_json::Error },
    NotSections { provider: &'static str },
}

// CompletionRequest
#[serde(default, skip_serializing_if = "ProviderOptions::is_empty")]
pub provider_options: ProviderOptions;
pub fn provider_options(self, options: ProviderOptions) -> Self;

// CompletionResponse
pub fn extras<P: ProviderExtension>(&self) -> Option<Result<P::Extras, serde_json::Error>>;
```

Each provider's marker, `Options` and `Extras` live in
`providers::<p>::extension` (for example `rig::providers::openrouter::extension::{OpenRouter, OpenRouterOptions, OpenRouterExtras}`).
Each dialect constant and each `PROVIDER` take the provider module's one
`PROVIDER_NAME` const, so the option key, the dialect name, the replay
identity and `Origin` cannot drift apart. Today `"cohere"` is written twice
(`crates/rig-core/src/providers/cohere/mod.rs:39`, `crates/rig-core/src/providers/openai/wire/dialects.rs:412-416`),
and so is `"ollama"`.

**The hard cases.**

| case | handling |
|---|---|
| OpenRouter, DeepSeek and the other Chat dialects share one wire | The Chat wire's `provider()` is the dialect name (`crates/rig-core/src/providers/openai/wire/chat.rs:878-880`), so each dialect reads only its own entry. A user gateway declared with `Dialect::gateway("mygw", ..)` gets typed options from a user `impl ProviderExtension { const PROVIDER = "mygw"; .. }`. |
| OpenAI Chat and Responses share one provider | One `"openai"` entry. `"*"` goes to whichever route runs; `"openai.chat"` and `"openai.responses"` go only to their route. OpenAI defaults to Responses (`crates/rig-core/src/providers/openai/wire/dialects.rs:35`) and Copilot picks per model (`crates/rig-core/src/providers/copilot/wire.rs:79-85`). |
| Cohere has two routes | Native (`api "cohere.chat"`) and Compatibility (`api "openai.chat"`) both read `"cohere"`. `ChatRoute::Auto` picks per request (`crates/rig-core/src/providers/cohere/wire.rs:131-137`), and each request takes the right half. |
| Ollama has two routes | `/v1` (`"openai.chat"`) and native `/api/chat` (`"ollama.chat"`) both read `"ollama"`. Native-only `options.num_ctx` on `/v1` goes through `on_unsupported`. |
| Vertex reuses the Gemini wire | The shared body builder already takes the target (`crates/rig-core/src/providers/gemini/completion.rs:388-392`), so REST reads `"gcp.gemini"`, Vertex `"vertexai"` and gRPC `"gemini-grpc"`. Section structs are shared: `VertexOptions` embeds the GenerateContent fields. Only a provider key can say the Developer API takes `serviceTier` and `store` and Vertex takes `modelArmorConfig`. |
| One vendor on two formats | `anthropic::wire::ZAI` and `openai::wire::ZAI` both say `"zai"` (`crates/rig-core/src/providers/anthropic/wire.rs:185`, `crates/rig-core/src/providers/openai/wire/dialects.rs:346`). `ZaiOptions` has an `"openai.chat"` and an `"anthropic.messages"` section. |

**Evidence (spike, uncommitted worktree at `7dfd8a422`).** Sixteen files,
+743/-27. Three encoders routed through `request_params` (Chat, Responses,
Cohere native) and four extension modules (OpenAI with two routes, OpenRouter,
DeepSeek, Cohere with two routes). `cargo clippy -p rig-core --all-features --tests -- -D warnings`
clean; 9 of 9 new tests and 897 rig-core tests passed; no existing test or
snapshot moved. The captured errors:
- `ProviderOptions::new().with::<DeepSeek>(&OpenRouterOptions::default())`: `error[E0308]: mismatched types ... expected '&DeepSeekOptions', found '&OpenRouterOptions'`;
- `options.shared.logit_bias.insert(..)`: `error[E0609]: no field 'logit_bias' on type 'providers::openai::extension::Shared'`;
- `request.provider_options.0.contains_key("openrouter")` in `chat.rs`: `error[E0616]: field '0' of struct 'ProviderOptions' is private`.

The spike read `extras::<OpenRouter>()` from the recorded reply of
`crates/rig-cassette/fixtures/cassettes/openrouter/agent/completion_smoke.yaml`
(`provider = "OpenAI"`, `cost = 0.0000351`, `native_finish_reason = "stop"`),
got `None` for `extras::<DeepSeek>()` on it, and read
`prompt_cache_miss_tokens = 57` from a DeepSeek recording. It found that the
merge must be deep, which section 2.1 adopts.

**Weaknesses.**
- Provider names become API: `"gcp.gemini"`, `"gemini-grpc"`, `"vertexai"`, `"aws_bedrock"`, `"azure.openai"` freeze as they are.
- Gemini REST and gRPC are one API under two names, so a harness inserts the same `GeminiOptions` twice (`GeminiGrpc` declares `type Options = GeminiOptions`).
- A section for a route not taken is skipped; `for_target` warns when an entry has route sections but none for the running route and an empty `"*"`.
- Section names are strings; a typo in a hand-written `#[serde(rename)]` is caught by the per-extension test only.

**Not covered: headers and encoder directives.** `ProviderOptions` is a
body layer: `request_params` merges it into the JSON body and nowhere else.
Options that are request headers or that change how the base builder works
are not body keys. Section 7 does not list them, and P4 does not add them:
- the ones that exist stay setters on the wire: Anthropic betas
  (`Messages::with_beta`, `crates/rig-core/src/providers/anthropic/wire.rs:283`),
  Copilot `with_intent` (`crates/rig-core/src/providers/copilot/wire.rs:261`),
  Anthropic `with_prompt_caching` and `with_static_prefix_cache_ttl`
  (`crates/rig-core/src/providers/anthropic/wire.rs:391`, `:455`), Gemini
  `thought_replay` (`crates/rig-core/src/providers/gemini/completion.rs:113`);
- the ones that do not exist yet are out of this stack: Vertex
  `shared_request_type` (a header), the Anthropic binding-block policy, and
  Bedrock document citations and tool caching (directives on the base
  arrays).

A body field in section 7 marked "+ beta" needs its beta set on the wire
with `with_beta`. P4 does not add the header for it. Without the beta, the
provider rejects the request: an error, not a silent drop. A typed header
channel would need a second merge target, with its own precedence, and no
acceptance test asks for one.

**Alternative considered.** Keying by wire API with a dialect layer
(`openai.chat/openrouter`). Rejected: the fixed `const PROVIDER` would hold a
wire key; the caller must know the route; OpenRouter needs one entry per
route; dialect matching is a runtime string.

## 4. Decision D: streamed replies rebuild the unary document

**Decided: full rebuild.** Unary `raw` stays the provider body byte for byte.
Each API has one reassembler that rebuilds the unary document from the
stream. An `Extras` type is written once, against the unary document, and
reads both paths.

**Mechanism.**
- Every `Wire` names `type Reassembler: Reassemble<Self::Frame>`. In stream
  mode the driver hands each frame to the reassembler before the decoder
  sees it, and at the end of the stream records `finish()` as the reply's
  `raw`. A decoder cannot skip a frame, because it never feeds the
  reassembler. On a truncated or failed stream the driver records the
  partial document, which is more than today's `Null`.
- `Out::raw` moves into the `impl<Op: Operation<Emit = Free>>` block that
  already hides `Out::event` from completions (`crates/rig-core/src/wire.rs:613`).
  A completion decoder has no way to write `raw`. Other operations keep it,
  and their wires name `type Reassembler = Unreassembled`, which records
  nothing.
- On the unary path nothing changes: the transport's whole body is `raw`
  (`crates/rig-core/src/wire.rs:346-357`).
- Reassemblers are plain JSON folds with no typed provider structs, so they
  name no `Extras` type. They live in provider modules, where the
  `extras-off-decode-path` guard applies.
- Shared folding rules (`wire::document`): a non-null value replaces; `null`
  only fills an absent key; stream-only transport fields are dropped from a
  per-API list; the stream's tag is renamed to the unary tag. `null`, an
  absent key and an empty list are equivalent; `Extras` fields are `Option<T>`
  or `#[serde(default)] Vec<T>`, and the parity test normalizes the same way.
- `CompletionResponse::raw` stays a public field (section 1.1).

**Public API.**

```rust
// rig_core::wire::document
pub trait Reassemble<Frame>: Default + WasmCompatSend + 'static {
    fn absorb(&mut self, frame: &Frame);
    fn finish(self) -> serde_json::Value;
}
/// For wires whose operation is not a completion: records nothing.
#[derive(Debug, Default)]
pub struct Unreassembled;
impl<F> Reassemble<F> for Unreassembled { /* finish() is Value::Null */ }

// rig_core::wire
pub trait Wire {
    // ...the existing items, then:
    type Reassembler: Reassemble<Self::Frame>;
}
impl<Op: Operation<Emit = Free>> Out<'_, Op> { pub fn raw(&mut self, raw: serde_json::Value); }
```

One reassembler per API: `openai::wire::chat::document::ChatCompletion`
(`"openai.chat"`), `openai::responses_api::document::Response`,
`anthropic::document::Message`, `gemini::document::GenerateContentResponse`
(shared by Vertex and gRPC), `gemini::interactions_api::document::Interaction`,
`cohere::document::ChatResponse`, `ollama::document::ChatResponse`,
`rig_bedrock::document::ConverseOutput` and an identity reassembler for Candle.
`GenerateContentDecoder::keep_raw` is deleted. P5 is breaking because every
`Wire` impl, companion crates included, must name its `Reassembler`.

**Evidence (spike, uncommitted worktree at `7dfd8a422`).** The Chat
reassembler (263 lines) passed 816 rig-core library tests, including:
- an `Extras`-shaped struct (`provider`, `usage.cost`, `usage.is_byok`, `usage.cost_details`, `choices[].native_finish_reason`) reading equal values from the OpenRouter unary and streamed recordings (`Azure`, `2.7e-6`, `stop`); `native_finish_reason` cannot be read from today's streamed `raw` at all;
- whole-document equality for OpenRouter, and for OpenAI's text pair and tool-call pair.

In the spike the decoder fed the reassembler. P5 moves the feeding into the
driver, so a decoder cannot skip a frame; the fold itself is unchanged.

The first run failed on two real differences that became rules: the stream
sends `obfuscation`, which unary lacks (dropped), and unary has
`annotations: []`, which the stream omits (the null, absent and empty
equivalence).

**Known gaps for P5.** Bedrock, DeepSeek, Venice, xAI and Interactions have no
recording of one prompt answered both ways; their reassemblers are checked by
unit tests only, and the P5 PR says so. Gemini part coalescing and
Interactions `outputs` are the reassemblers most likely to need fixes from
recordings.

**Alternative considered.** A documented per-API subset (the envelope without
content). Rejected: it strips content from unary `raw`, a second breaking
change to recorded data, and keeps two `raw` contracts.

## 5. Decision E: citations live on `message::Text`

**Decided.** A private `citations` field on `message::Text`. Decoders set it
through the fold (`Out::cite`, `Out::set_citations`); a caller building a
`Text` by hand uses `Text::with_citations`. Wire spans carry their unit and
are resolved to byte spans checked against the quoted text. The list is
fingerprinted, so editing the text hides it. The provider's own citation JSON
stays where it is, in `Text.native` or `raw`, and replays unchanged.

**Guarantees.**
- `Span`'s fields are private to its module, so any other module, in rig-core or not, that writes `Span { start: 3, end: 9 }` gets `error[E0451]: fields 'start' and 'end' of struct 'Span' are private`. Decoders hand over a `WireSpan` naming its unit (Gemini bytes, Anthropic and Cohere characters) through `Out::cite`. `Citation` is `#[non_exhaustive]` with public fields, which binds only outside rig-core, and `Text::with_citations` and `Text::span` are public. So the `extras-off-decode-path` guard (section 2.5) rejects `Citation` struct expressions and calls to `Text::with_citations` and `Text::span` under the `providers` tree and in companion provider crates. Section 1.1 lists what remains open.
- A span is kept only when it can be checked or its unit is documented. Where the unit is undocumented and the wire carries no quoted text (OpenAI Responses and Chat annotations, Gemini Interactions annotations), the decoder hands over `span: None`: the citation stays on the block, its offsets stay in `native`, until a recording with non-ASCII text before the span settles the unit and a later PR adds a `SpanUnit` for it. Offsets read in a guessed unit would resolve to wrong but valid bytes that no check could catch.
- `Out::cite` never fails a reply. A span that does not resolve (out of range, not on a character boundary) or whose resolved text differs from `quoted` (Gemini `segment.text`, Cohere `text`) drops that citation with one `tracing::warn!` carrying `provider`, `index` and the reason; the reply decodes as it does today. A `ProviderError` there would turn a reply that decodes today into a failure, an unstated behaviour change.
- Any module but `Text`'s own that writes `text.citations.push(c)` gets `error[E0616]: field 'citations' of struct 'Text' is private`.
- The fold resolves and fingerprints a block's citations when the block closes, not when `Out::cite` is called. Anthropic streams a `citations_delta` before the later `text_delta`s of the same block, so a fingerprint taken at the call would not match the finished text.
- `Text::citations()` returns `&[]` once `text` no longer matches the fingerprint the list was resolved against; reads go through `str::get`, never an index.
- Citations never enter the fingerprint projection (`crates/rig-core/src/completion/message.rs:224-226` stays `["v1","text",text]`), so no stored `native` goes stale and replay is byte for byte.

**Public API.**

```rust
// rig_core::completion::message::citation, re-exported from rig::message
impl Text {
    pub fn citations(&self) -> &[Citation];
    pub fn with_citations(self, citations: impl IntoIterator<Item = Citation>) -> Self;
    pub fn clear_citations(&mut self);
    pub fn cited(&self, citation: &Citation) -> Option<&str>;
    /// A span over `range`, in bytes of this text; `None` off a character boundary.
    pub fn span(&self, range: std::ops::Range<usize>) -> Option<Span>;
}
#[non_exhaustive] pub struct Citation { pub span: Option<Span>, pub sources: Vec<Source> }
pub struct Span { /* private start, end */ }   // start(), end(), range()
#[non_exhaustive] pub struct Source {
    pub location: SourceLocation,
    pub title: Option<String>,
    pub cited_text: Option<String>,
    pub confidence: Option<f32>,
}
#[non_exhaustive] pub enum SourceLocation {
    Document { index: Option<u32>, id: Option<String>, within: Option<DocumentRange> },
    Url { url: String },
    File { file_id: String, filename: Option<String>, container_id: Option<String> },
    SearchResult { index: u32, source: String, blocks: Option<std::ops::Range<u32>> },
    ToolOutput { id: String },
}
#[non_exhaustive] pub enum DocumentRange { Chars(Range<u64>), Pages(Range<u32>), Blocks(Range<u32>) }

// rig_core::wire, for decoders
#[non_exhaustive] pub struct WireCitation { pub span: Option<WireSpan>, pub sources: Vec<Source> }
#[non_exhaustive] pub struct WireSpan { pub start: u64, pub end: u64, pub unit: SpanUnit, pub quoted: Option<String> }
#[non_exhaustive] pub enum SpanUnit { Bytes, Chars, Utf16 }
impl Out<'_, Completion> {
    /// Resolved when the block closes; dropped with a warning when its span
    /// does not resolve or match `quoted`.
    pub fn cite(&mut self, index: usize, citation: WireCitation);
    pub fn set_citations(&mut self, index: usize, citations: Vec<WireCitation>);
}
```

`Text` is `#[non_exhaustive]` from P1. The field is serialized as
`citations`, skipped when empty, and read leniently, so history stored before
P6 loads unchanged.

**Not covered.** Cohere `PLAN` and `THINKING_CONTENT` citations land on blocks
without a `citations` field and stay native-only; uncited source lists
(Perplexity `search_results`, Anthropic `web_search_tool_result`) stay in
`Opaque` blocks and `Extras`.

**Evidence.** No spike. Every span-bearing source already sits on the text
block's provider item (section 10), so the field maps one item to one block.
OpenAI and xAI already record both routes of a web-search citation turn
(`crates/rig-cassette/fixtures/cassettes/openai/web_search_citations/streamed_and_unary.yaml`).

**Alternative considered.** A per-turn citation list anchored to blocks by
position and fingerprint. Rejected: anchors must be remapped in `continued()`
and the runtimes, and callers read two places.

## 6. Option mapping per completion wire and dialect

Each cell is the JSON merged into the request body (`Mapping::Send`), "omit"
(`Mapping::Omit`: the provider default already does what was asked, so an
omitted option is told apart from a dropped one), or "unsupported"
(`Mapping::Unsupported`) with its reason. An unsupported cell is
`UnsupportedOption` under `Error` and a warning under `Ignore`. A cell that
puts markers inside arrays (MiniMax, Bedrock) is `Mapping::Place`, and the
base builder writes the markers from `BaseInput::cache` (section 2.1). Every
cell is a row of the `option_matrix` golden for the wires it covers
(section 13), so a wire may answer `Omit` only where its cell says "omit". "model" means the catalog row decides (section 8). Doc labels are
defined at the head of each subsection.

### 6.1 Anthropic Messages and its dialects

Docs: [AM] https://platform.claude.com/docs/en/api/messages/create,
[AE] https://platform.claude.com/docs/en/build-with-claude/effort,
[AA] https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking,
[AX] https://platform.claude.com/docs/en/build-with-claude/extended-thinking,
[AG] https://platform.claude.com/docs/en/about-claude/models/migration-guide,
[AP] https://platform.claude.com/docs/en/build-with-claude/prompt-caching,
[AS] https://platform.claude.com/docs/en/api/service-tiers,
[AT] https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview.

Model classes (rig constants at `crates/rig-core/src/providers/anthropic/completion.rs:27-57`):

| class | models | thinking on | off | budget | effort levels | default effort | `top_p` | Priority Tier |
|---|---|---|---|---|---|---|---|---|
| A0 | `claude-haiku-4-5` | `{"type":"enabled","budget_tokens":N}` | omit or `disabled` | 1024 <= N < `max_tokens` | none | none | `temperature` or `top_p`, not both | yes |
| A1 | Opus 4.5 (no constant) | budget | `disabled` | yes | low, medium, high | high | one of the two | yes |
| A2 | `claude-opus-4-6`, `claude-sonnet-4-6` | `adaptive`, sent explicitly | `disabled` | deprecated, accepted | low..max (no xhigh) | high | one of the two | yes |
| A3 | `claude-opus-4-7`, `claude-opus-4-8` | `adaptive`, sent explicitly | `disabled` | 400 | low..max with xhigh | high | 400 | yes |
| A4 | `claude-sonnet-5` | adaptive by default | `disabled` | 400 | all five | high | 400 | no |
| A5 | `claude-opus-5` | adaptive by default | `disabled` at effort <= high | 400 | all five | high | 400 | no |
| A6 | `claude-sonnet-5-5` | adaptive by default | `between_tools` at effort <= high (`disabled` is 400) | 400 | all five | high | non-default 400 | no |
| A7 | `claude-opus-5-5` | adaptive by default | 400 at every effort | 400 | all five | medium | 400 | no |
| A8 | `claude-fable-5`, `claude-fable-5-1` | always on | 400 | 400 | all five | high | 400 | Fable 5 yes, 5.1 no |

**Anthropic** (`"anthropic"`):

| option | JSON | notes | doc |
|---|---|---|---|
| `Reasoning::Off` | A0-A5: `"thinking":{"type":"disabled"}`. A6: `"thinking":{"type":"between_tools"}`. A7, A8: unsupported, "thinking cannot be disabled on this model" | A5 and A6: unsupported with effective effort `xhigh` or `max` | [AM], [AG] |
| `Effort(Minimal)` | unsupported on every model: no `minimal` level | no silent map to `low` | [AE] |
| `Effort(Low/Medium/High)` | A2-A8: `"thinking":{"type":"adaptive"},"output_config":{"effort":"<level>"}`. A1: `"output_config":{"effort":..}` only. A0: unsupported | `output_config` deep-merges with `output_config.format`; on binding models the existing `block_binding` joins the same `thinking` object (`crates/rig-core/src/providers/anthropic/completion.rs:168-182`) | [AE], [AA] |
| `Effort(XHigh)` | A3-A8 as above with `"xhigh"`; A0-A2 unsupported | | [AE] |
| `Effort(Max)` | A2-A8 as above with `"max"`; A0, A1 unsupported | | [AE] |
| `Budget { tokens }` | A0-A2: `"thinking":{"type":"enabled","budget_tokens":N}`; A3-A8 unsupported | N >= 1024 and N < `max_tokens`, else unsupported. A2 accepts it though deprecated | [AX] |
| `CacheRetention::None` | no `cache_control` anywhere; with `with_prompt_caching()` or `with_static_prefix_cache_ttl(..)` on the wire, unsupported: "the wire places cache markers" | a caller's tool `cache_control` in `additional_params.tools` is kept | [AP] |
| `CacheRetention::Short` | top-level `"cache_control":{"type":"ephemeral"}` (automatic caching) | byte for byte what `with_automatic_caching()` sends today. The wire's placement knobs stay as they are and are not options: `Messages::with_prompt_caching()` (tool, system and last message block) and `with_static_prefix_cache_ttl(..)` (tool and system markers). Manual markers take their TTL from `BaseInput::cache`, so `with_prompt_caching()` plus `cache(Long)` sends what `with_prompt_caching().with_automatic_caching_1h()` sends today (section 12.1). `with_prompt_caching()` with `cache` unset keeps today's body. 4 markers at most. Below the model's minimum prefix the API declines to cache; that is the provider's choice, not a drop | [AP] |
| `CacheRetention::Long` | top-level `"cache_control":{"type":"ephemeral","ttl":"1h"}`; with `with_prompt_caching()`, the base builder's tool and system markers carry `"ttl":"1h"` too | byte for byte `with_automatic_caching_1h()`. 1 h markers precede 5 min markers (check kept, `crates/rig-core/src/providers/anthropic/completion.rs:771-787`) | [AP] |
| `ServiceTier::Auto` | `"service_tier":"auto"` | | [AS] |
| `ServiceTier::Default` | `"service_tier":"standard_only"` | | [AS] |
| `ServiceTier::Priority` | unsupported: no "priority only" value; `auto` uses Priority capacity only under a commitment. Also unsupported by model on A4-A7 and Fable 5.1 | the served tier is `usage.service_tier` | [AS] |
| `ServiceTier::Flex` | unsupported: no flex tier | Batch is another endpoint | [AS] |
| `verbosity` | unsupported: no verbosity knob | `effort` changes depth, not verbosity | [AM] |
| `parallel_tool_calls: true` | omit (the default) | | [AT] |
| `parallel_tool_calls: false` | `"tool_choice":{"type":"auto","disable_parallel_tool_use":true}`, or the caller's `any`/`tool` choice with the flag added | merges into the `tool_choice` object (`crates/rig-core/src/providers/anthropic/completion.rs:476-490`); with no tools, omit | [AM] |
| `top_p` | A0-A2: `"top_p":p`; A3-A8 unsupported | A0-A2: unsupported when `temperature` is also set | [AG] |
| `seed` | unsupported: no seed parameter | | [AM] |
| `stop` | `"stop_sequences":[..]` | | [AM] |

**Z.AI** (`"zai"`, `https://api.z.ai/api/anthropic`; docs https://docs.z.ai/scenario-example/develop-tools/claude, https://docs.z.ai/guides/capabilities/thinking, https://docs.z.ai/guides/capabilities/cache):

| option | JSON |
|---|---|
| `Off` | `"thinking":{"type":"disabled"}` [unverified]; unsupported on GLM-5.3 |
| `Effort(_)` | unsupported [unverified]: a top-level `reasoning_effort` may pass through, unrecorded |
| `Budget` | unsupported [unverified] |
| `cache None` | omit (caching is implicit and cannot be turned off; "send no markers" is honoured) |
| `cache Short` | omit (implicit) |
| `cache Long` | unsupported: no TTL control |
| `service_tier`, `verbosity`, `seed` | unsupported [unverified] |
| `parallel_tool_calls: false` | unsupported [unverified] |
| `top_p` | `"top_p":p` [unverified] |
| `stop` | `"stop_sequences":[..]` [unverified] |

**MiniMax** (`"minimax"`; docs https://platform.minimax.io/docs/api-reference/text-anthropic-api, https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache):

| option | JSON |
|---|---|
| `Off` | M2.x (every rig constant): unsupported, always-on thinking. M3: `"thinking":{"type":"disabled"}`. M3.1-Flash: unsupported (400) |
| `Effort(_)` | M3.1-Flash: `"output_config":{"effort":"low".."max"}`; every rig constant: unsupported |
| `Budget` | unsupported [unverified] |
| `cache None` | omit (passive caching cannot be disabled) |
| `cache Short` | `Mapping::Place`: the base builder writes block-level `{"type":"ephemeral"}` markers on tools, system and the last message block, at most 4; the top-level marker is undocumented [unverified], so this dialect places explicit breakpoints |
| `cache Long` | unsupported: 5 minute TTL only |
| `ServiceTier::Default` / `Priority` | `"service_tier":"standard"` / `"priority"`; `Auto`, `Flex` unsupported |
| `verbosity`, `seed` | unsupported |
| `parallel_tool_calls: false` | unsupported [unverified] |
| `top_p` | `"top_p":p` |
| `stop` | unsupported: MiniMax documents `stop_sequences` as ignored |

**Moonshot** (`"moonshot"`; docs https://platform.kimi.ai/docs/guide/agent-support, https://platform.kimi.ai/docs/guide/use-thinking-effort):

| option | JSON |
|---|---|
| `Off` | kimi-k2.6: `"thinking":{"type":"disabled"}` [unverified]; kimi-k3, kimi-k2.7-code: unsupported |
| `Effort(Low/High/Max)` | kimi-k3: `"output_config":{"effort":..}` [unverified]; `Medium`, `XHigh`, `Minimal` unsupported |
| `Budget` | unsupported [unverified] |
| `cache` | `None`, `Short`: omit (automatic); `Long`: unsupported |
| `service_tier`, `verbosity`, `seed`, `parallel_tool_calls`, `top_p`, `stop` | unsupported [unverified] |

**Xiaomi MiMo** (`"xiaomimimo"`; doc https://mimo.mi.com/docs/en-US/api/chat/anthropic-api):

| option | JSON |
|---|---|
| `Off` | `"thinking":{"type":"disabled"}` |
| `Effort(_)`, `Budget` | unsupported: only `thinking.type` enabled or disabled |
| `cache` | `None`, `Short`: omit (implicit); `Long`: unsupported |
| `service_tier`, `verbosity`, `seed` | unsupported [unverified] |
| `parallel_tool_calls: false` | `"tool_choice":{"type":"auto","disable_parallel_tool_use":true}` |
| `top_p` | `"top_p":p`, range 0.01 to 1.0 |
| `stop` | `"stop_sequences":[..]` |

No dialect cell above is backed by a recording: only `anthropic/` has
cassettes. MiMo documents unsigned thinking blocks that must be kept, while
its dialect uses `Quirks::gateway()` with `unsigned_thinking: false`
(`crates/rig-core/src/providers/anthropic/wire.rs:216-221`). That is a
correctness note outside this work.

### 6.2 OpenAI Chat Completions and its dialects

One wire (`crates/rig-core/src/providers/openai/wire/chat.rs`) serves every
dialect in `crates/rig-core/src/providers/registry.rs:37-61`. Mapped options
merge between the typed fields (`chat.rs:372-383`) and `additional_params`
(`chat.rs:384`). The dialect body rewrites that run after the merge today
(DeepSeek, Mistral, Moonshot, Perplexity, Mira) stay after it as
`Rewrite::ChatDialect` and read the merged body, so a mapped option and a
raw key drive them alike (section 2.1).

Cross-cutting rules:
- Reasoning levels depend on model and route: Azure documents `max` only on
  GPT-6 and GPT-5.6 with Responses, `xhigh` only on GPT-6, 5.6, 5.5, 5.4 and
  `gpt-5.1-codex-max`, `minimal` only on GPT-5, `none` from GPT-5.1
  (https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning).
- GPT-5.6 and GPT-6 on Chat reject function tools unless `reasoning_effort` is
  `none`, and the default is `medium`. `validate` must see tools and reasoning
  together. The doc comments at `crates/rig-core/src/providers/openai/completion/mod.rs:8-30`
  claim an encoder refusal that no code performs.
- Automatic caching: `Short` = omit where the dialect caches by default;
  `None` = unsupported unless a documented off switch exists; `Long` =
  unsupported unless a longer retention knob exists.
- `stop` limits: OpenAI, OpenRouter, Groq, Venice, Z.AI, MiMo 4; Moonshot 5; DeepSeek 16.
  A longer list is unsupported with the limit as reason.
- Mira rejects every pass-through parameter (`chat.rs:290-294`): every option is unsupported.
- A cache key (`prompt_cache_key`, `session_id`) is not in `GenerationOptions`; it is a provider option (section 7).

**OpenAI** (`"openai"`, Chat route; [OC] https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create, [OK] https://developers.openai.com/api/docs/guides/prompt-caching):

| option | JSON | notes | doc |
|---|---|---|---|
| `Off` | `"reasoning_effort":"none"` | model: GPT-5.1+ only; GPT-5 cannot disable; non-reasoning models: omit | [OC] |
| `Effort(e)` | `"reasoning_effort":"minimal"\|"low"\|"medium"\|"high"\|"xhigh"\|"max"` | model; non-reasoning model unsupported | [OC] |
| `Budget` | unsupported: effort only | | [OC] |
| `cache None` | GPT-5.6+: `"prompt_cache_options":{"mode":"explicit"}` with no breakpoints; earlier: unsupported (always on) | | [OK] |
| `cache Short` | before 5.6: `"prompt_cache_retention":"in_memory"`; 5.6+: omit (30 min is the only TTL) | GPT-5.5 and 5.5-pro take only `24h`: unsupported there | [OK] |
| `cache Long` | before 5.6, on extended-retention models: `"prompt_cache_retention":"24h"`; 5.6+: unsupported | the 24 h default for non-ZDR orgs makes "omit" differ from `Short` | [OK] |
| `service_tier` | `"service_tier":"auto"\|"default"\|"flex"\|"priority"` | `Flex` is model-limited | [OC] |
| `verbosity` | `"verbosity":"low"\|"medium"\|"high"` | GPT-5 family | [OC] |
| `parallel_tool_calls` | `"parallel_tool_calls":bool` | | [OC] |
| `top_p` | `"top_p":p` | reasoning models reject sampling (model) | [OC] |
| `seed` | `"seed":n` | deprecated, accepted | [OC] |
| `stop` | `"stop":[..]`, at most 4 | not on o3 or o4-mini | [OC] |

**Azure OpenAI** (`"azure.openai"`; https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning, https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/prompt-caching):
the OpenAI JSON for every field, with three differences. `Effort(Max)` is
unsupported on Chat. `service_tier` is unsupported [unverified]. rig's default
`api-version` `2024-10-21` (`dialects.rs:21`) predates `reasoning_effort`,
`verbosity` and `prompt_cache_options` [unverified whether it rejects them];
P2 tests or requires `v1`. Azure keeps `OutputCap::Legacy` though its
reasoning models take only `max_completion_tokens`.

**OpenRouter** (`"openrouter"`; [OR] https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request, [ORR] https://openrouter.ai/docs/guides/best-practices/reasoning-tokens, [ORC] https://openrouter.ai/docs/guides/best-practices/prompt-caching, [ORT] https://openrouter.ai/docs/guides/features/service-tiers):

| option | JSON | notes | doc |
|---|---|---|---|
| `Off` | `"reasoning":{"effort":"none"}` | upstream may still reason (model) | [ORR] |
| `Effort(e)` | `"reasoning":{"effort":"<e>"}` | OpenRouter converts effort to an Anthropic budget itself | [ORR] |
| `Budget { tokens }` | `"reasoning":{"max_tokens":N}` | Anthropic and Qwen upstreams; OpenAI and Grok take effort only (model) | [ORR] |
| `cache None` | omit | unsupported on automatic-cache upstreams (OpenAI, DeepSeek, Grok, Groq), by `vendor/` prefix | [ORC] |
| `cache Short` | top-level `"cache_control":{"type":"ephemeral"}`; automatic upstreams: omit | replaces today's system-message marker (`chat.rs:816-843`) | [ORC] |
| `cache Long` | top-level `"cache_control":{"type":"ephemeral","ttl":"1h"}` | meaningful on Anthropic upstreams | [ORC] |
| `ServiceTier::Auto` | omit (requests use the standard tier unless one is named) | OpenRouter accepts only `default`, `flex`, `priority` (alias `fast`) and `ultrafast`; `auto` is not a value | [ORT] |
| `Default` / `Flex` / `Priority` | `"service_tier":"default"\|"flex"\|"priority"` | `ultrafast` is a provider option (section 7) | [ORT] |
| `verbosity` | `"verbosity":"low"\|"medium"\|"high"` | | [OR] |
| `parallel_tool_calls`, `top_p`, `seed` | `"parallel_tool_calls":bool`, `"top_p":p`, `"seed":n` | | [OR] |
| `stop` | `"stop":[..]`, at most 4 | | [OR] |

**DeepSeek** (`"deepseek"`; [DS] https://api-docs.deepseek.com/api/create-chat-completion, https://api-docs.deepseek.com/guides/thinking_mode, https://api-docs.deepseek.com/guides/kv_cache):

| option | JSON | notes |
|---|---|---|
| `Off` | `"thinking":{"type":"disabled"}` | thinking is on by default |
| `Effort(e)` | `"thinking":{"type":"enabled"},"reasoning_effort":"low"\|"high"\|"max"` | catalog levels `[Low, High, Max]`; others unsupported rather than mapped server-side |
| `Budget` | unsupported | |
| `cache None` / `Long` | unsupported: disk cache always on, fixed retention | |
| `cache Short` | omit | |
| `service_tier`, `verbosity`, `parallel_tool_calls`, `seed` | unsupported (not documented) | |
| `top_p` | `"top_p":p` | ignored in non-thinking mode, clamped to >= 0.95 in thinking mode |
| `stop` | `"stop":[..]`, at most 16 | |

**Mistral** (`"mistral"`; https://docs.mistral.ai/api/endpoint/chat, https://docs.mistral.ai/capabilities/reasoning):
`Off` `"reasoning_effort":"none"` (adjustable models only); `Effort(High)`
`"reasoning_effort":"high"`, other levels model; `Budget` unsupported; cache
`Short` omit, `None` and `Long` unsupported; `Auto` `"service_tier":"auto"`,
`Default` `"service_tier":"standard_only"`, `Flex` and `Priority` unsupported;
`verbosity` unsupported; `parallel_tool_calls`, `top_p` as OpenAI; **`seed` is
`"random_seed":n`**; `stop` `"stop":[..]`.

**Groq** (`"groq"`; https://console.groq.com/docs/api-reference, https://console.groq.com/docs/reasoning, https://console.groq.com/docs/prompt-caching):
`Off` `"reasoning_effort":"none"` (Qwen only; GPT-OSS cannot disable);
`Effort(e)` `"reasoning_effort":"<e>"` for every level, model: the API
lists `none`, `default`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max`
and rejects a value outside the model's set with a 400 (GPT-OSS takes
low..high; `qwen/qwen3.8-27b` takes low..high and maps `high` to its native
`xhigh`), so the catalog decides; `Budget` unsupported; cache `Short` omit, `None` and `Long`
unsupported ("cannot be manually disabled"); `Auto` `"auto"`, `Default`
`"on_demand"`, `Flex` `"flex"`, `Priority` `"performance"`; `verbosity`
unsupported; `parallel_tool_calls`, `top_p`, `seed` as OpenAI; `stop` at most 4.

**xAI** (`"xai"`, Chat route; https://docs.x.ai/developers/model-capabilities/text/reasoning, https://docs.x.ai/docs/api-reference):
`Off` unsupported on reasoning models ("cannot be disabled"), omit on
`*-non-reasoning`; `Effort(Low..XHigh)` `"reasoning_effort":..` (grok-4.6+;
grok-4.5 low..high, and xAI silently treats `xhigh` as `high` there, so the
catalog must refuse it); `Max`, `Minimal`, `Budget` unsupported; cache
`Short` omit, `None` and `Long` unsupported; `Default` `"default"`, `Priority`
`"priority"`, `Auto` and `Flex` unsupported; `verbosity` unsupported;
`parallel_tool_calls`, `top_p` as OpenAI; `seed` [unverified]; `stop` on
non-reasoning models only.

**Together** (`"together"`; https://docs.together.ai/reference/chat-completions-1):
`Off` `"reasoning":{"enabled":false}`; `Effort(Low/Medium/High)`
`"reasoning":{"enabled":true},"reasoning_effort":..` (model); `Budget`
unsupported; cache `Short` omit, `None`, `Long` unsupported; `service_tier`,
`verbosity` unsupported; `parallel_tool_calls` [unverified]; `top_p`, `seed`,
`stop` as OpenAI.

**Venice** (`"venice"`; https://docs.venice.ai/api-reference/endpoint/chat/completions, https://docs.venice.ai/overview/guides/prompt-caching):
`Off` `"reasoning_effort":"none"`; `Effort(e)` `"reasoning_effort":"<e>"`
(model); `Budget` unsupported; cache `None` unsupported, `Short`
`"prompt_cache_retention":"default"`, `Long` `"prompt_cache_retention":"24h"`;
`service_tier`, `verbosity` unsupported; `parallel_tool_calls`, `top_p`,
`seed` as OpenAI; `stop` at most 4.

**Moonshot / Kimi** (`"moonshot"`; https://platform.kimi.ai/docs/api/chat):
`Off` kimi-k2.6 `"thinking":{"type":"disabled"}`, others unsupported;
`Effort` kimi-k3 `"reasoning_effort":"low"\|"high"\|"max"`, kimi-k2.6
unsupported (toggle only); `Budget` unsupported; cache `None` unsupported,
`Short` `"prompt_cache_options":{"mode":"implicit","ttl":"5m"}`, `Long`
`"prompt_cache_options":{"mode":"implicit","ttl":"1h"}`; `service_tier`,
`verbosity`, `parallel_tool_calls`, `top_p`, `seed` [unverified]; `stop` at
most 5, 32 bytes each.

**Z.AI** (`"zai"`, also `ZAI_CODING`; https://docs.z.ai/api-reference/llm/chat-completion):
`Off` `"thinking":{"type":"disabled"}`; `Effort` `"thinking":{"type":"enabled"},"reasoning_effort":"low"\|"high"\|"max"`
on models that take it, toggle-only models unsupported; `Budget` unsupported;
cache `Short` omit, `None`, `Long` unsupported; `service_tier`, `verbosity`,
`parallel_tool_calls`, `seed` unsupported; `top_p` 0.01 to 1.0; `stop` at most 4.

**llama.cpp** (`"llamacpp"`; https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):
`Off` `"reasoning_effort":"none"`; `Effort(e)` `"reasoning_effort":"<e>"`
(template-defined; accept all); `Budget` unsupported per request (a server
flag); cache `None` `"cache_prompt":false`, `Short` omit, `Long` unsupported;
`service_tier`, `verbosity` unsupported; `parallel_tool_calls`, `top_p`,
`seed`, `stop` as OpenAI.

**Ollama `/v1`** (`"ollama"`; https://docs.ollama.com/api/openai-compatibility):
`Off` `"reasoning_effort":"none"`; `Effort(e)` `"reasoning_effort":"<e>"`
(model-defined); `Budget` unsupported; cache `Short` omit, `None`, `Long`
unsupported; `service_tier`, `verbosity`, `parallel_tool_calls` unsupported
(in the documented unsupported list); `top_p`, `seed`, `stop` supported.

**Cohere Compatibility** (`"cohere"`, `ChatRoute::Compatibility`; https://docs.cohere.com/docs/compatibility-api):
`Off` `"reasoning_effort":"none"`; `Effort(High)` `"reasoning_effort":"high"`,
other levels unsupported ("only none and high"); `Budget` unsupported on this
route; every `cache` value unsupported (no cache control); `service_tier`,
`parallel_tool_calls`, `verbosity` unsupported; `top_p`, `seed`, `stop`
supported.

**Perplexity** (`"perplexity"`; https://docs.perplexity.ai/api-reference/chat-completions-post):
Sonar Chat Completions support ended on 2026-09-27; every cell is as of the
deprecated API. `Effort(Low/Medium/High)` `"reasoning_effort":..` on
`sonar-deep-research` only, unsupported on every other model;
`Effort(Minimal)`, `Effort(XHigh)` and `Effort(Max)` unsupported:
`sonar-deep-research` takes low, medium and high only; `Off`, `Budget`, every `cache` value,
`service_tier`, `verbosity`, `parallel_tool_calls` unsupported; `top_p`,
`stop` supported; `seed` [unverified].

**MiniMax** (`"minimax"`, OpenAI half; https://platform.minimax.io/docs/api-reference/text-openai-api):
`Off` `"thinking":{"type":"disabled"}` (M3; M3.1-Flash unsupported);
`Effort` `"thinking":{"type":"adaptive"},"reasoning_effort":..`; `Budget`,
cache `None`/`Long`, `service_tier`, `verbosity` unsupported; cache `Short`
omit; `top_p`, `stop` supported; `seed`, `parallel_tool_calls` [unverified].

**Xiaomi MiMo** (`"xiaomimimo"`; https://mimo.mi.com/docs/en-US/api/chat/openai-api):
`Off` `"thinking":{"type":"disabled"}`; `Effort(_)` unsupported (toggle only);
`top_p` unsupported while thinking (fixed 0.95); `Budget`, cache
`None`/`Long`, `service_tier`, `verbosity` unsupported; cache `Short` omit;
`parallel_tool_calls` and `seed` unsupported (not request parameters);
`stop` `"stop":[..]`, at most 4.

**Copilot** (`"copilot"`, Chat route for non-codex models): no public API
reference. `Effort(e)` `"reasoning_effort":"<e>"` for GPT models, levels from
the catalog [unverified]; every other cell unsupported until recorded.

**Hugging Face router** (`"huggingface"`): OpenAI JSON for every field
[unverified]; support depends on the sub-provider, so under the notation rule
P2 answers each `Unsupported` until a recording confirms it.
**Hyperbolic** (`"hyperbolic"`), **Doubleword** (`"doubleword"`): no doc page
lists their parameters. `top_p`, `seed`, `stop` as OpenAI [unverified];
every other cell unsupported [unverified].
**Mira** (`"mira"`): every cell unsupported.

### 6.3 OpenAI Responses and its dialects

One encoder (`crates/rig-core/src/providers/openai/responses_api/mod.rs:341-496`)
serves HTTP and the WebSocket session (`responses_api/websocket.rs:521-533`).
The WebSocket path never calls `Wire::encode`, so the mapping and
`request_params` sit inside `responses_request`, never in
`encode_with_headers`.

Docs: [RC] https://developers.openai.com/api/reference/resources/responses/methods/create,
[RR] https://developers.openai.com/api/docs/guides/reasoning,
[RK] https://developers.openai.com/api/docs/guides/prompt-caching,
[RL] https://developers.openai.com/api/docs/guides/latest-model,
[RF] https://developers.openai.com/api/docs/guides/flex-processing,
[RP] https://developers.openai.com/api/docs/guides/priority-processing,
[XR] https://docs.x.ai/developers/rest-api-reference/inference/responses,
[XT] https://docs.x.ai/developers/model-capabilities/text/reasoning,
[OO] https://openrouter.ai/docs/api/api-reference/responses/create-a-response.

The Responses API has no `seed` and no `stop` parameter.

**OpenAI** (`"openai"`; also Azure v1 and compatible servers [unverified]):

| option | JSON | model dependence | doc |
|---|---|---|---|
| `Off` | `"reasoning":{"effort":"none"}` | models whose levels include `none`; unsupported on gpt-6-astra, gpt-6.1-sol, gpt-5, o-series, `*-pro`; omit on non-reasoning models | [RR] |
| `Effort(Minimal)` | `{"effort":"minimal"}` | gpt-5, gpt-5-mini, gpt-5-nano only | [RR] |
| `Effort(Low/Medium/High)` | `"reasoning":{"effort":"<level>"}`, and the `ReasoningCiphertext` rewrite adds `"include":["reasoning.encrypted_content"]` (as for every effort cell) | `*-pro` and `gpt-5.2-chat-latest` restrict levels | [RR] |
| `Effort(XHigh)` | `{"effort":"xhigh"}` | gpt-5.2 and later | [RR] |
| `Effort(Max)` | `{"effort":"max"}` | gpt-5.6, gpt-6, gpt-6.1-sol | [RR] |
| `Budget` | unsupported: effort only | | [RC] |
| `cache None` | gpt-5.6+: `"prompt_cache_options":{"mode":"explicit"}` with no breakpoints; earlier: unsupported | | [RK] |
| `cache Short` | gpt-5.6+: omit; earlier: `"prompt_cache_retention":"in_memory"`; gpt-5.5, 5.5-pro: unsupported (24 h only) | | [RC], [RK] |
| `cache Long` | before 5.6: `"prompt_cache_retention":"24h"`; 5.6+: `"prompt_cache_options":{"ttl":"30m"}` | on 5.6+ the 30 minute TTL is a minimum and the only value, so `Long` and `Short` coincide; recorded as a P2 decision | [RK] |
| `ServiceTier` | `"service_tier":"auto"\|"default"\|"flex"\|"priority"` | `Flex` beta and model-limited; `Priority` not with EU residency on gpt-6 | [RC], [RF], [RP] |
| `verbosity` | `"text":{"verbosity":..}`, deep-merged with `text.format` (`responses_api/mod.rs:469-478`) | GPT-5 family and later | [RC] |
| `parallel_tool_calls` | `"parallel_tool_calls":bool` | | [RC] |
| `top_p` | `"top_p":p` | unsupported when effective effort is not `none` (gpt-6 rule); o-series and gpt-5 reject it; gpt-5.5 unsupported until its default effort is confirmed [unverified] | [RL] |
| `seed`, `stop` | unsupported: no such parameter | | [RC] |

**xAI** (`"xai"`): `Off` unsupported on grok-4.5, 4.6, 4.7 (`{"effort":"none"}`
on grok-4.3 only; omit on `*-non-reasoning`); `Effort(Low/Medium/High)`
`"reasoning":{"effort":..}`; `XHigh` grok-4.6+ only; `Minimal`, `Max`,
`Budget` unsupported; cache `Short` omit, `None`, `Long` unsupported; `Default`
`"default"`, `Priority` `"priority"`, `Auto`, `Flex` unsupported; `verbosity`,
`seed`, `stop` unsupported; `parallel_tool_calls`, `top_p` as OpenAI. [XR], [XT].

**OpenRouter Responses** (`"openrouter"`, Responses route): `Off`
`"reasoning":{"effort":"none"}`; `Effort(e)` `"reasoning":{"effort":"<e>"}`;
`Budget` `"reasoning":{"max_tokens":N}` (upstream); cache `None` unsupported
on implicit upstreams, `"prompt_cache_options":{"mode":"explicit"}` on gpt-5.6+
upstreams; `Short` `"cache_control":{"type":"ephemeral"}` on Anthropic
upstreams, omit on OpenAI upstreams; `Long` `"cache_control":{"type":"ephemeral","ttl":"1h"}`
on Anthropic upstreams; `Auto` omit, `Default`, `Flex`, `Priority` as
OpenRouter Chat (`auto` is not an OpenRouter value, [ORT]); `verbosity`
`"text":{"verbosity":..}`; `parallel_tool_calls`, `top_p` as OpenAI; `seed`,
`stop` unsupported. [OO].

**ChatGPT / Codex backend** (`"chatgpt"`): no public doc; evidence is
`references/codex/codex-rs/codex-api/src/common.rs:279-304`,
`references/pi/packages/ai/src/api/openai-codex-responses.ts:553-597` and
the recorded echoes in `crates/rig-cassette/fixtures/cassettes/chatgpt/codex_sessions/`.
`Off` `"reasoning":{"effort":"none"}` (model); `Effort(e)` as OpenAI; `Budget`
unsupported; cache `Short`, `None` unsupported, `Long` omit (the backend
echoes `24h`); `Flex` `"service_tier":"flex"`, `Priority`
`"service_tier":"priority"`, `Auto`, `Default` unsupported; `verbosity`
`"text":{"verbosity":..}`; `parallel_tool_calls` `bool`; `top_p`, `seed`,
`stop` unsupported. Today the Codex contract silently deletes
`parallel_tool_calls`, `service_tier` and `text` (`responses_api/mod.rs:483`),
which Codex itself sends. The Codex base builder still leaves out the typed
`max_tokens`, `temperature` and `output_schema`, which the backend does not
accept. They are request fields, not options, so guarantee 1 does not
cover them; P2's Migration line says so (section 12.0).

**Copilot Responses** (`"copilot"`, `*codex*` models): no public doc;
evidence `crates/rig-cassette/fixtures/cassettes/copilot/reasoning_roundtrip/streaming.yaml`.
`Effort(e)` as OpenAI with catalog levels (gpt-5.3-codex has no `none`, so
`Off` unsupported); `Budget` unsupported; cache `Long` omit (echoed default),
`Short`, `None` unsupported; `service_tier`, `verbosity` [unverified];
`parallel_tool_calls`, `top_p` as OpenAI; `seed`, `stop` unsupported.

**WebSocket mode**: the HTTP row of the session's dialect, inside
`{"type":"response.create", ..}`.

### 6.4 Gemini GenerateContent, Vertex, Interactions and gRPC

Docs: [GR] https://ai.google.dev/api/generate-content,
[GT] https://ai.google.dev/gemini-api/docs/generate-content/thinking,
[GF] https://ai.google.dev/gemini-api/docs/generate-content/flex-inference,
[GP] https://ai.google.dev/gemini-api/docs/generate-content/priority-inference,
[GC] https://ai.google.dev/gemini-api/docs/generate-content/caching,
[G38] https://ai.google.dev/gemini-api/docs/latest-model,
[GX] https://ai.google.dev/api/interactions-api,
[GXC] https://ai.google.dev/gemini-api/docs/caching,
[VF] https://docs.cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo,
[VP] https://docs.cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo,
[VC] https://docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview,
[GPR] https://github.com/googleapis/googleapis/blob/master/google/ai/generativelanguage/v1beta/generative_service.proto.

GC is the Developer API (`"gcp.gemini"`, `gemini.generate_content`); VX is
rig-vertexai (`"vertexai"`, same body builder, sent through the Vertex SDK);
IX is Interactions (`"gcp.gemini"`, `gemini.interactions`, snake_case); gRPC
is rig-gemini-grpc (`"gemini-grpc"`, the REST body transcoded).

Model data (with models.dev `reasoning_options`):

| model | levels | default | budget | can disable |
|---|---|---|---|---|
| gemini-3.8-flash, 3.7-flash | low, medium, high (`minimal` is an error) | medium | none | no |
| gemini-3.6-flash, 3.5-flash | minimal, low, medium, high | medium | none | no |
| gemini-3.1-pro(-preview) | low, medium, high | high | none | no |
| gemini-3.5-flash-lite, 3.1-flash-lite | minimal..high | minimal | none | no |
| gemini-3-flash-preview | minimal..high | high | none | no |
| gemini-3-pro-preview | low, high | high | none | no |
| gemini-2.5-pro | none | dynamic | 128..=32768 | no |
| gemini-2.5-flash | none | dynamic | 0..=24576 | yes (`0`) |
| gemini-2.5-flash-lite | none | off | 512..=24576 | yes (`0`) |

| option | GC | VX | IX | gRPC | doc |
|---|---|---|---|---|---|
| `Off` | 2.5 Flash, Flash-Lite: `{"generationConfig":{"thinkingConfig":{"thinkingBudget":0}}}`; 2.5 Pro and Gemini 3: unsupported (`minimal` is not off) | as GC | unsupported: no budget, no off | as GC | [GT], [GX] |
| `Effort(Minimal..High)` | Gemini 3: `{"generationConfig":{"thinkingConfig":{"thinkingLevel":"<level>"}}}` where listed; 2.5: unsupported (no effort-to-budget table in the catalog) | as GC | `{"generation_config":{"thinking_level":"<level>"}}` | as GC; rig's proto declares `thinking_level`, Google's does not [unverified] | [GT], [GX], [GPR] |
| `Effort(XHigh/Max)` | unsupported: the enum stops at `high` | as GC | as GC | as GC | [GT] |
| `Budget { tokens }` | 2.5: `{"generationConfig":{"thinkingConfig":{"thinkingBudget":N}}}` inside the model range; Gemini 3: unsupported | as GC (`i32`) | unsupported | as GC | [GT] |
| `cache None` | unsupported: implicit caching cannot be turned off per request | unsupported: per project only | unsupported | unsupported | [GC], [VC], [GXC] |
| `cache Short` | omit (implicit caching) | omit | omit | omit | [GC] |
| `cache Long` | unsupported: explicit caching is a `cachedContents` resource | unsupported | unsupported: no explicit caching | unsupported | [GC], [GXC] |
| `Auto` | omit | omit | omit | omit | [GF] |
| `Default` | `{"serviceTier":"standard"}` | omit (standard PayGo) | `{"service_tier":"standard"}` | unsupported: no proto field | [GP], [GX] |
| `Flex` | `{"serviceTier":"flex"}` | unsupported: needs request headers the SDK transport cannot set | `{"service_tier":"flex"}` | unsupported | [GF], [VF] |
| `Priority` | `{"serviceTier":"priority"}` | unsupported, as `Flex` | `{"service_tier":"priority"}` | unsupported | [GP], [VP] |
| `verbosity` | unsupported | unsupported | unsupported | unsupported | [GR], [GX] |
| `parallel_tool_calls: true` | omit | omit | omit | omit | [GR] |
| `parallel_tool_calls: false` | unsupported: no switch | unsupported | unsupported | unsupported | [GR] |
| `top_p` | `{"generationConfig":{"topP":p}}` | as GC | `{"generation_config":{"top_p":p}}` [unverified: no longer listed in [GX]] | as GC | [GR], [GX] |
| `seed` | `{"generationConfig":{"seed":n}}`; `n > i32::MAX` unsupported | as GC | `{"generation_config":{"seed":n}}` | as GC | [GR], [GPR] |
| `stop` | `{"generationConfig":{"stopSequences":[..]}}`, at most 5 | as GC | `{"generation_config":{"stop_sequences":[..]}}` | as GC | [GR] |

**Resume** (`InteractionResume`, `"gcp.gemini"`, `gemini.interactions`,
`crates/rig-core/src/providers/gemini/interactions_api/mod.rs:229-274`) is a
completion wire with no request body: it reads a stored interaction by id, and
its `encode` takes `_request: CompletionRequest` and uses none of it. Every
`GenerationOptions` field on it is `UnsupportedOption`, reason "a resumed
interaction is read, not created; its options were fixed when it was
created". `additional_params` or provider options on it are not options, so
they are an `EncodeError` under either policy. Its `map_options`
destructures `OptionFields` with no `..` and answers `Unsupported` for each
set field, so `prepare` refuses a set option before `encode` runs. Its
`encode` still calls `request_params` with an empty base and reads the
result: a non-empty
`FinalBody` (raw or provider keys) is that `EncodeError`. It sends
`Body::empty()`, the one body the `options-precedence` guard accepts besides
`FinalBody::into_body()`, so a `GET` never carries `{}`. So guarantee 1 holds
for it as for any wire: a new field fails to compile there too, and a set
option is an error under `Error` and a warning under `Ignore`, never dropped.

Today typed fields override `additionalParams.generationConfig` and
`generation_config` (`crates/rig-core/src/providers/gemini/completion.rs:428-437`,
`gemini/interactions_api/mod.rs:352-370`). Under `request_params` raw config
keys win. That is a behaviour change, stated in P2's Migration. gRPC checks
`service_tier` itself before `from_rest`, which would otherwise fail with a
generic request error (`crates/rig-gemini-grpc/src/rest.rs:400-411`).
[G38] recommends stripping `temperature`, `top_p` and `top_k` on Gemini 3.x;
that is advice, not a documented 400, so the wire still sends them.

### 6.5 Bedrock Converse, Cohere native, Ollama native and Candle

Docs: [CV] https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html,
[CS] https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html,
[CP] https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html,
[PC] https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html,
[ST] https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ServiceTier.html,
[ET] https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html,
[AD] https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html,
[NV] https://docs.aws.amazon.com/nova/latest/nova2-userguide/extended-thinking.html,
[CO] https://docs.cohere.com/reference/chat,
[COR] https://docs.cohere.com/docs/reasoning,
[OL] https://docs.ollama.com/api/chat,
[OLT] https://docs.ollama.com/capabilities/thinking.

**Bedrock Converse** (`"aws_bedrock"`). Reasoning maps into
`additionalModelRequestFields`, below the caller's `additional_params`
(today the whole object, `crates/rig-bedrock/src/request.rs:133`).

| option | JSON | notes | doc |
|---|---|---|---|
| `Off` | Claude 5.x, Opus 4.6/4.7: `{"additionalModelRequestFields":{"thinking":{"type":"disabled"}}}`; Claude 4.5 and older: omit; Fable, Mythos: unsupported; Nova 2 Lite: `{"reasoningConfig":{"type":"disabled"}}`; always-on models (DeepSeek R1): unsupported | model | [AD], [ET], [NV] |
| `Effort(Low/Medium/High)` | adaptive Claude: `{"thinking":{"type":"adaptive"},"output_config":{"effort":"<level>"}}`; Nova 2: `{"reasoningConfig":{"type":"enabled","maxReasoningEffort":"<level>"}}`; budget-only Claude: unsupported | Nova with `high` requires `temperature`, `topP` and `maxTokens` unset | [AD], [NV] |
| `Effort(Minimal)` | unsupported: no Bedrock model lists it | | [AD] |
| `Effort(XHigh/Max)` | adaptive Claude where the catalog lists the level [unverified: [AD] and pi disagree on which models take `xhigh`] | | [AD] |
| `Budget { tokens }` | Claude with budgets: `{"thinking":{"type":"enabled","budget_tokens":N}}`, N >= 1024 and < `maxTokens`; Opus 4.7, Claude 5, Fable, Mythos, Nova 2: unsupported | | [ET] |
| `cache None` | omit every `cachePoint` (implicit caching cannot be stopped; "no explicit checkpoints") | | [PC] |
| `cache Short` | `Mapping::Place`: the base builder appends `{"cachePoint":{"type":"default"}}` after the system blocks and at the end of the last message; models without explicit caching: unsupported | a request with reasoning in history gets no message checkpoint (`crates/rig-bedrock/src/request.rs:106-114`); the mapping answers `Place`; the base builder places the system checkpoint and reports the skipped message checkpoint through `BaseInput::refuse_cache`, never silently | [CP], [PC] |
| `cache Long` | `Mapping::Place`, the same blocks with `"ttl":"1h"`; Claude 3.7 and 3.5 v2: unsupported | 1 h checkpoints precede 5 min ones | [CP], [PC] |
| `Auto` | omit `serviceTier` | | [ST] |
| `Default` / `Flex` / `Priority` | `"serviceTier":{"type":"default"\|"flex"\|"priority"}` | per-model tier support (model) | [ST] |
| `verbosity` | unsupported | | [CV] |
| `parallel_tool_calls` | unsupported: `toolChoice` has no flag | | [CV] |
| `top_p` | `"inferenceConfig":{"topP":p}`; Claude classes A3-A8 (section 6.1, Sonnet 5 included): unsupported | Claude thinking rejects it ([ET]) | [CV] |
| `seed` | unsupported: `inferenceConfig` has no seed | | [CV] |
| `stop` | `"inferenceConfig":{"stopSequences":[..]}` | | [CV] |

**Cohere native** (`"cohere"`, `api "cohere.chat"`):
`Off` `"thinking":{"type":"disabled"}` on reasoning models (thinking is on by
default), omit otherwise; `Effort(High)` `"thinking":{"type":"enabled"}`,
other levels unsupported; `Budget` `"thinking":{"type":"enabled","token_budget":N}`;
cache `None` omit, `Short`, `Long` unsupported; `service_tier` unsupported
(`priority` is a queue integer, a provider option); `verbosity`,
`parallel_tool_calls` unsupported; `top_p` `"p":p`, 0.01 to 0.99 (1.0 is
unsupported); `seed` `"seed":n`; `stop` `"stop_sequences":[..]`, at most 5.
[CO], [COR].

**Ollama native `/api/chat`** (`"ollama"`, `api "ollama.chat"`):
`Off` `"think":false`; `Effort(Low/Medium/High)` `"think":"<level>"` where
`/api/show` lists the values, boolean-only models unsupported; `Effort(Max)`
model (rig accepts `max` today, [OLT] lists none); `Minimal`, `XHigh`,
`Budget` unsupported; cache `None` omit, `Short`, `Long` unsupported
(`keep_alive` keeps the model loaded, a provider option); `service_tier`,
`verbosity`, `parallel_tool_calls` unsupported; `top_p`
`"options":{"top_p":p}`; `seed` `"options":{"seed":n}`; `stop`
`"options":{"stop":[..]}`. [OL], [OLT].

**Candle** (`"candle"`, local; no wire JSON, cells are `GenerationConfig`
fields): `Off` honoured (Qwen3's template already renders no-thinking mode,
`crates/rig-candle/src/protocol.rs:515-520`; other models do not reason);
`Effort`, `Budget` unsupported; cache `None` honoured, `Short`, `Long`
unsupported; `service_tier`, `verbosity`, `parallel_tool_calls` unsupported;
`top_p` `GenerationConfig.top_p`, 0 < p <= 1; `seed` `GenerationConfig.seed`;
`stop` unsupported (generation stops only on end tokens).

## 7. Provider extensions

Sources: the 0.43 typed types (`git show v0.43.0:<path>`, indexed by the
"#### Providers" table of the 0.44 migration guide) and current provider
docs. Every `Options` type is serialize-only; every `Extras` type reads
`raw`. Sections are `"*"` and the route `Api` names.

| provider key | route sections | Options fields | Extras fields | sources |
|---|---|---|---|---|
| `anthropic` | `"*"` | `thinking_display` (`summarized`, `omitted`, `updates` + beta), `top_k`, `metadata_user_id`, `inference_geo`, `speed` (`fast` + beta), `task_budget` (`output_config.task_budget` + beta), `fallbacks` + beta, `container`, `context_management` + beta, `mcp_servers` + beta, `diagnostics_previous_message_id` + beta | `stop_reason`, `stop_sequence`, `stop_details`, `cache_creation { ephemeral_5m_input_tokens, ephemeral_1h_input_tokens }`, `service_tier`, `inference_geo`, `speed`, `server_tool_use`, `container`, `fallback_model` | v0.43 `crates/rig-core/src/providers/anthropic/completion.rs` 57-121, 212-230, 350-355; [AM] |
| `zai`, `minimax`, `moonshot`, `xiaomimimo` | `"*"`, `"openai.chat"`, `"anthropic.messages"` | Z.AI: `do_sample`, `request_id`, `user_id`, `tool_stream`, `clear_thinking`; MiniMax: `reasoning_split`; Moonshot: `thinking_keep`, `prompt_cache_key`; MiMo: web search tool config | Z.AI `web_search`, `request_id` [unverified keys]; MiniMax `reasoning_details`, `base_resp`; MiMo `message.annotations`; Moonshot choice-level `usage` | dialect docs in 6.1 and 6.2 |
| `openai` | `"*"`, `"openai.chat"`, `"openai.responses"` | `"*"`: `store`, `metadata`, `prompt_cache_key`, `safety_identifier`. Chat: `logit_bias`, `prediction`, `logprobs`, `top_logprobs`, penalties, `n`, `modalities`, `audio`, `web_search_options`, `include_obfuscation`. Responses: `reasoning_summary`, `reasoning_mode`, `reasoning_context`, `include`, `previous_response_id`, `conversation`, `truncation`, `context_management`, `prompt_cache_options { mode, prewarm, comparison_response_id }`, `background`, `max_tool_calls`, `top_logprobs`, `service_tier_extra` (`scale`, `ultrafast`), `access_programs` | `OpenAiExtras::{Chat, Responses}` with shared `service_tier()`. Chat: `system_fingerprint`, `service_tier`, `prompt_tokens_details`, `completion_tokens_details`. Responses: `service_tier`, effective `reasoning`, `prompt_cache_retention`, `prompt_cache_options`, `incomplete_details.reason`, `phase` per item, `billing` (unary only) | v0.43 `crates/rig-core/src/providers/openai/responses_api/mod.rs` 1378-1422, 1563-1610, 1659-1895; [OC], [RC] |
| `azure.openai` | as `openai` | the `openai` set, `data_sources` | `prompt_filter_results`, `content_filter_results`, `message.context.citations` [unverified fields] | Azure docs in 6.2 |
| `openrouter` | `"*"`, `"openai.chat"`, `"openai.responses"` | `"*"`: `provider: ProviderPreferences { order, only, ignore, allow_fallbacks, require_parameters, data_collection, zdr, sort, preferred_min_throughput, preferred_max_latency, max_price, quantizations }`, `models` (fallbacks), `route`, `plugins`, `session_id`, `trace`, `metadata`, `reasoning_exclude`, `reasoning_summary`, `top_k`, `min_p`, `top_a`, `repetition_penalty`, `user`, `service_tier_extra` (`ultrafast`) | `provider`, `native_finish_reason`, `service_tier`, `system_fingerprint`, `cost`, `cost_details`, `is_byok`, `prompt_tokens_details`, `server_tool_use_details`, `openrouter_metadata`, `annotations` | v0.43 `crates/rig-core/src/providers/openrouter/completion.rs` 30-621; [OR], https://openrouter.ai/docs/guides/routing/provider-selection |
| `deepseek` | `"*"` | none: `thinking` and `reasoning_effort` are portable | `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, `reasoning_tokens`, `system_fingerprint` | v0.43 `crates/rig-core/src/providers/deepseek.rs` 20-55; [DS] |
| `mistral` | `"*"` | `prompt_mode`, `safe_prompt`, `guardrails`, `prompt_cache_key`, penalties, `n`, `prediction` | `usage.service_tier`, `prompt_audio_seconds`, `num_cached_tokens`, `prompt_tokens_details` | v0.43 `crates/rig-core/src/providers/mistral/completion.rs` 120-160 |
| `groq` | `"*"` | `reasoning_format`, `include_reasoning`, `search_settings`, `citation_options`, `compound_custom` | `x_groq`, `queue_time`, `prompt_time`, `completion_time`, `total_time`, `usage_breakdown`, `service_tier`, `executed_tools` | https://console.groq.com/docs/api-reference |
| `xai` | `"*"`, `"openai.chat"`, `"openai.responses"` | `prompt_cache_key`, `search_parameters` | `cost_in_usd_ticks`, `num_sources_used`, `num_server_side_tools_used`, `citations` | https://docs.x.ai/developers/cost-tracking, [XR] |
| `together` | `"*"` | `chat_template_kwargs`, `top_k`, `min_p`, `repetition_penalty`, `safety_model` | `warnings`, `message.reasoning` | https://docs.together.ai/reference/chat-completions-1 |
| `venice` | `"*"` | `venice_parameters: VeniceParameters { character_slug, strip_thinking_response, enable_web_search, enable_web_scraping, enable_x_search, enable_web_citations, include_search_results_in_stream, return_search_results_as_documents, include_venice_system_prompt }`, `prompt_cache_key` (`disable_thinking` is `Reasoning::Off`) | `venice_parameters` echo with `web_search_citations`, `cost { usd, diem }`, `cache_creation_input_tokens` | v0.43 `crates/rig-core/src/providers/venice/completion.rs` 40-233 |
| `perplexity` | `"*"` | `search_mode`, `search_domain_filter`, `search_recency_filter`, `return_images`, `return_related_questions`, `search_context_size` | `citations`, `search_results`, `images`, `related_questions`, `usage.cost`, `search_context_size` | https://docs.perplexity.ai/api-reference/chat-completions-post |
| `llamacpp` | `"*"` | `chat_template_kwargs`, `reasoning_format`, `n_probs`, `samplers`, `top_k`, `min_p`, `typical_p`, `mirostat`, `id_slot`, `timings_per_token` | `timings { cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, .. }` | v0.43 `crates/rig-core/src/providers/llamacpp/completion.rs` 20-60 |
| `ollama` | `"*"`, `"openai.chat"`, `"ollama.chat"` | `"*"`: `keep_alive`. `"ollama.chat"`: `num_ctx`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `model_options`, `logprobs`, `top_logprobs`, `truncate` and `shift` [unverified] | native: `model`, `created_at`, `done_reason`, durations, `prompt_eval_count`, `prompt_eval_cached_count`, `eval_count`, `logprobs` | v0.43 `crates/rig-core/src/providers/ollama.rs` 124-294; [OL] |
| `cohere` | `"*"`, `"openai.chat"`, `"cohere.chat"` | `"*"`: `strict_tools`, `frequency_penalty`, `presence_penalty`. `"cohere.chat"`: `citation_mode`, `safety_mode`, `priority`, `top_k` (`k`), `logprobs` | native: `id`, `finish_reason`, `billed_units`, `tokens`, `cached_tokens`, `tool_plan`, `logprobs` | v0.43 `crates/rig-core/src/providers/cohere/completion.rs` 26-305; [CO] |
| `copilot` | `"*"`, `"openai.chat"`, `"openai.responses"` | `copilot_cache_control` [unverified] | `copilot_usage`, `prompt_filter_results` | recorded parity snapshot `crates/rig-cassette/fixtures/parity/copilot.json` |
| `chatgpt` | `"openai.responses"` | `prompt_cache_key`, stable `session_id`, `client_metadata`, `access_programs` | as `openai` Responses | `references/codex/codex-rs/codex-api/src/common.rs:279-304` |
| `gcp.gemini` | `"*"`, `"gemini.generate_content"`, `"gemini.interactions"` | GenerateContent: `include_thoughts`, `safety_settings`, `top_k`, penalties, `response_logprobs`, `logprobs`, `candidate_count` (only 1), `response_modalities`, `image_config`, `speech_config`, `media_resolution`, `enable_enhanced_civic_answers`, `cached_content`, `labels`, `store`. Interactions: `agent`, `agent_config`, `background`, `store`, `previous_interaction_id`, `thinking_summaries`, `response_modalities`, `response_format`, `response_mime_type`, speech, transcription and video config, `safety_settings`, `service_tier_deferred` | `GeminiExtras::{GenerateContent, Interactions}`. GenerateContent: `model_version`, `response_id`, `service_tier`, usage details, `safety_ratings`, `prompt_feedback`, `finish_message`, `citation_metadata`, `grounding_metadata`, `url_context_metadata`, `avg_logprobs`, `logprobs_result`. Interactions: `id`, `status`, `service_tier`, `created`, `updated`, per-modality usage | v0.43 `crates/rig-core/src/providers/gemini/completion.rs` 622-680, 1314-1700, 2138-2155; v0.43 `gemini/interactions_api/mod.rs` 378-430, 1727-1935; [GR], [GX] |
| `vertexai` | `"*"`, `"vertexai.generate_content"` | the GenerateContent fields, `labels`, `model_armor_config`, `routing_config`, `audio_timestamp` | the GenerateContent extras, `traffic_type`, `create_time`, Vertex `citationMetadata.citations` | `google-cloud-aiplatform-v1` 1.11.0 `model.rs` |
| `gemini-grpc` | as `gcp.gemini` (`type Options = GeminiOptions`) | the GenerateContent fields the proto declares; others go through `on_unsupported` | GenerateContent extras minus `grounding_metadata` and `url_context_metadata` (rig's proto omits them, `crates/rig-gemini-grpc/proto/gemini.proto:439-470`) | [GPR] |
| `aws_bedrock` | `"*"` | `guardrail { identifier, version, trace, stream_processing_mode }` (sent on both modes), `performance_latency`, `service_tier_reserved`, `request_metadata`, `additional_response_field_paths`, `claude { thinking_display, betas }`, `top_k` | `stop_reason`, `latency_ms`, `cache_details`, `service_tier`, `performance_latency`, `trace`, `invoked_model_id`, `additional_model_response_fields` | v0.43 `crates/rig-bedrock/src/types/converse_output.rs` 26-297, v0.43 `crates/rig-bedrock/src/completion.rs` 128-189; [CV], [CS] |
| `candle` | `"*"` | `top_k`, `disable_top_p`, `repeat_penalty`, `repeat_last_n` | `CandleCompletionResponse` (already public, `crates/rig-candle/src/types.rs:237-260`) | `crates/rig-candle/src/generation.rs:81-89` |
| `huggingface`, `hyperbolic`, `doubleword`, `mira` | `"*"` | none beyond generic OpenAI fields | none | |

Every field here is a body key. Headers and encoder directives are not
provider options (section 3, "Not covered"). Server-tool definitions (web
search, web fetch, code execution) stay in `additional_params.tools`, which
every wire appends to rig's tools (section 2.1), or a later typed-tools
phase. Per-block cache
breakpoints are out of scope: a content-level marker belongs to neither
`GenerationOptions` nor a request-level `Options` type.

## 8. Catalog schema and the models.dev mapping

The data file is models.dev-shaped JSON: an object of provider keys, each with
a `models` object keyed by model id. `Catalog::from_json` reads the same shape
for overrides, and `merge` lets the override's rows win field by field.

| models.dev field | `ModelSpec` field | rule |
|---|---|---|
| provider key | `provider` | through the key map below |
| `models.<id>.id` | `id` | |
| `name` | `display_name` | |
| `limit.context` | `context_window` | |
| `limit.output` | `max_output_tokens` | |
| `modalities.input` | `input` | `text`, `image`, `audio`, `video`, `pdf` |
| `reasoning_options[type=effort].values` | `reasoning.levels` | every value but `none`; Groq's `default` is dropped |
| `reasoning_options[type=effort].values` contains `none`, or `type=toggle` present | `reasoning.can_disable` | |
| `reasoning_options[type=budget_tokens].{min,max}` | `reasoning.budget` | `max` absent: up to `max_output_tokens` |
| (none) | `reasoning.default` | hand-entered from vendor docs |
| (none) | `caching.retention` | hand-entered per provider and model generation |
| `tool_call` | `tools` | |
| `structured_output` | `structured_output` | absent: `false` |
| `cost.{input,output,cache_read,cache_write}` | `pricing` | USD per million tokens |
| `status == "deprecated"` | `deprecated` | |

Provider key map: `openai` to `openai`, `azure` to `azure.openai`,
`anthropic` to `anthropic`, `google` to `gcp.gemini`, `google-vertex` to
`vertexai`, `amazon-bedrock` to `aws_bedrock`, `openrouter`, `deepseek`,
`groq`, `mistral`, `xai`, `venice`, `zai`, `minimax`, `cohere`, `perplexity`,
`huggingface` to themselves, `togetherai` to `together`, `moonshotai` to
`moonshot`, `xiaomi` to `xiaomimimo`, `ollama-cloud` to `ollama`,
`github-copilot` to `copilot`. models.dev has no rows for `hyperbolic`,
`doubleword`, `mira`, `llamacpp` or `chatgpt`; those entries are hand-written
or absent.

Gaps the research found (P3 lists every unverified entry in its PR body):
- **Default effort** is not in models.dev.
- **`can_disable` is incomplete.** Opus 4.7 and 4.8 accept `disabled` but
  carry no `toggle`; only `claude-sonnet-5` has one. Hand review is required.
- **Anthropic's own `GET /v1/models` is a better source** for the Anthropic
  rows: it returns `capabilities.effort.<level>.supported`,
  `capabilities.thinking.types.{adaptive,enabled}.supported`,
  `max_input_tokens` and `max_tokens` (recorded in
  `crates/rig-cassette/fixtures/cassettes/anthropic/models/list_models_smoke.yaml`;
  https://platform.claude.com/docs/en/api/models/list). That recording lists
  no `xhigh` for `claude-opus-4-7` where the docs and models.dev do, so sync
  treats a missing key as unknown, not unsupported.
- **Constants missing from models.dev**: `minimax::MINIMAX_M2_1_HIGHSPEED`
  (`crates/rig-core/src/providers/minimax.rs:39`), `zai::GLM_4_6_AIR`
  (`crates/rig-core/src/providers/zai.rs:30`), `zai::GLM_4_6_X`
  (`zai.rs:32`), `zai::GLM_4_5_AIRX` (`zai.rs:40`). DeepSeek's models.dev rows
  are `deepseek-flash`, `deepseek-v4-flash`, `deepseek-v4-pro` and
  `deepseek-v4-flash-vision-exp`; any other id the corpus uses
  (`deepseek-chat`, `deepseek-reasoner`) is hand-written or absent. The other
  families are not yet checked [unverified]; guarantee 4 lists them.
- **Local Ollama models** are not in models.dev; `/api/show` returns
  `thinking.values` and `default` at runtime (https://docs.ollama.com/capabilities/thinking).
- **Facts the fixed `ModelSpec` lacks**, proposed as P3 additions behind
  `#[non_exhaustive]`: supported `service_tiers`; `verbosity`; levels per
  route (Azure `max` only on Responses); sampling allowed only at effort
  `none` (models.dev `temperature: false`); the interleaved reasoning field
  (models.dev `interleaved.field`, 1229 rows, replacing the hard-coded list at
  `crates/rig-core/src/providers/openai/wire/chat.rs:225-239`); Anthropic's
  forced tool choice, mid-conversation system and context binding
  (`crates/rig-core/src/providers/anthropic/completion.rs:81-140`).
- **Prices `Pricing` cannot hold**: two cache-write prices, context tiers
  (models.dev `cost.tiers`, `context_over_200k`), reasoning-token prices
  (`cost.reasoning`), audio input (`cost.input_audio`), fast mode
  (`experimental.modes.fast.cost`), service-tier multipliers, explicit-cache
  storage per token-hour.

## 9. Streamed raw: the reassembly plan per API

| API | unary `raw` | streamed `raw` today | rebuild |
|---|---|---|---|
| OpenAI Chat, every dialect | `chat.completion` body | `{usage, finish_reason (rig's enum), response_id, model, logprobs, additional_params}` (`crates/rig-core/src/providers/openai/wire/chat.rs:1687-1708`); drops `native_finish_reason`, `annotations`, `refusal`; repeated arrays are extended (Perplexity: 68 citations, 17 distinct) | top-level keys last-wins, `object` renamed `chat.completion`; `delta.content`, `refusal`, `reasoning`, `reasoning_content` append; `tool_calls` merge by `index` (`arguments` append), `index` dropped at finish; `reasoning_details` merge with the decoder's own merge (`chat.rs:1748`); `annotations`, `images`, `audio.data`, `logprobs.content` append; choices kept by `index`; `obfuscation` dropped; OpenRouter's terminal `usage.cost`, `provider`, `native_finish_reason` land where unary has them; a whole `chat.completion` frame replaces the choice's `message`; a bare-string reply (Mira) has no document |
| Anthropic Messages | `Message` | `{usage, stop_reason, stop_sequence, message_id, model}` (`crates/rig-core/src/providers/anthropic/streaming.rs:549-558`), pinned by a key-set test | `message_start.message` is the skeleton; `content_block_start` placed at its index; `text_delta`, `thinking_delta`, `signature_delta` append; `input_json_delta` accumulates and is parsed at `content_block_stop` (`{}` when empty); `citations_delta` appends; `message_delta` sets `stop_reason`, `stop_sequence`, `stop_details`, `container`, and merges its cumulative `usage` over the start usage; server-tool results and `fallback` blocks kept whole |
| OpenAI Responses (HTTP and WebSocket) | `Response` | the terminal event's `response` (`crates/rig-core/src/providers/openai/responses_api/streaming.rs:795`), already the unary shape | take the terminal `response`; fill `output` from `response.output_item.done` items when the terminal `output` is empty (the Codex backend sends `output: []`); `billing` is unary-only and stays `Option` in `Extras` |
| Gemini GenerateContent, Vertex, gRPC | `GenerateContentResponse` | REST: a snake_case summary (`crates/rig-core/src/providers/gemini/streaming.rs:195-207`); gRPC and Vertex: the last chunk through `keep_raw` (`gemini/streaming.rs:73`, `crates/rig-gemini-grpc/src/streaming.rs:52`, `crates/rig-vertexai/src/types/completion_response.rs:34`) | `candidates[i].content.parts` append; adjacent text parts with equal `thought` and no `thoughtSignature` coalesce (derived from recordings, not docs); `citationMetadata.citationSources` append; `groundingMetadata`, `urlContextMetadata`, `safetyRatings`, `finishReason`, `finishMessage`, `usageMetadata`, `modelVersion`, `responseId` last non-null; `promptFeedback` first. Vertex's "stream" re-emits the unary reply, so it is already equal |
| Gemini Interactions | interaction resource | `{usage, interaction, model_version}` (`crates/rig-core/src/providers/gemini/interactions_api/streaming.rs:342-350`); `interaction.completed` carries no `steps` | the completed interaction with `steps` (or `outputs`) rebuilt from the content events, as the decoder already folds them (`interactions_api/streaming.rs:300-323`) |
| Cohere native | chat response | the `message-end` event (`crates/rig-core/src/providers/cohere/streaming.rs:493`) | `id` from `message-start`; `content-start`/`content-delta` build `message.content[i]`; `tool-plan-delta` appends `message.tool_plan`; `tool-call-*` build `message.tool_calls[i]`; `citation-start` appends `message.citations` with `content_index`; `message-end.delta` gives `finish_reason` and `usage`. The Compatibility route uses the Chat reassembler |
| Ollama native | `/api/chat` body | the final `done` record (`crates/rig-core/src/providers/ollama/streaming.rs:158`) | the last record with `message.content` and `thinking` appended, `tool_calls` and `images` collected, `logprobs` appended |
| Bedrock Converse | `ConverseOutput` | `{messageStart, messageStop, metadata}` (`crates/rig-bedrock/src/streaming.rs:412-427`) | `messageStart.role`; `contentBlockStart`/`contentBlockDelta` by index build `output.message.content[i]` (`text`, `toolUse.input` parsed at stop, `reasoningContent`, `citation`); `messageStop` gives `stopReason`, `additionalModelResponseFields`; `metadata` gives `usage`, `metrics`, `trace`, `performanceConfig`, `serviceTier` |
| Candle | the serialized response | the same record (`crates/rig-candle/src/model.rs:450`) | identity |

Recorded pairs exist for `anthropic`, `cohere`, `gemini`, `ollama`, `openai`,
`openrouter` and `perplexity`. DeepSeek has unary and streamed recordings but
none of one prompt answered both ways; Bedrock, Venice, xAI Chat and
Interactions have no pair.

## 10. Citations per provider

| provider and API | wire shape | span unit | mapping to `Citation` |
|---|---|---|---|
| Anthropic Messages | a list on each `text` block: `char_location`, `page_location`, `content_block_location`, `search_result_location`, `web_search_result_location`; streamed as `citations_delta` (https://platform.claude.com/docs/en/build-with-claude/citations) | whole block | `span: None`; `Document { index, within: Chars/Pages/Blocks }` with `title`, `cited_text`; `SearchResult { index, source, blocks }`; `Url { url }`. `encrypted_index` stays native |
| Bedrock Converse (Claude) | `citationsContent` blocks; streamed `contentBlockDelta.delta.citation` (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Citation.html) | whole block | as Anthropic: `documentChar`, `documentPage`, `documentChunk`, `searchResultLocation`, `web`. rig never sets `citations.enabled` today (`crates/rig-bedrock/src/request.rs:344-366`); turning them on is an encoder directive, out of this stack (section 3) |
| Cohere native | `message.citations[] { start, end, text, sources, content_index, type }` (https://docs.cohere.com/reference/chat) | characters, with the quoted `text` | span checked against `text`; `Document { id }`, `ToolOutput { id }`. `PLAN` and `THINKING_CONTENT` stay native |
| Cohere Compatibility | none: documents go as text | | |
| OpenAI Responses (and xAI, OpenRouter, Copilot, ChatGPT) | `output_text.annotations[]`: `url_citation`, `file_citation`, `container_file_citation`, `file_path` ([RC]) | undocumented, and no quoted text | `span: None` until a non-ASCII recording settles the unit; then `url_citation` span + `Url`, `container_file_citation` span + `File { container_id }`, `file_citation` and `file_path` an empty span at `index` + `File`, with later parts shifted by the earlier parts' lengths (rig concatenates parts). Until then: `Url` and `File` sources only |
| OpenAI Chat, OpenRouter web plugin, MiMo | `message.annotations[].url_citation { start_index, end_index, url, title }` | undocumented, no quoted text | `span: None` on the turn's text block until a non-ASCII recording settles the unit |
| Perplexity, xAI Chat live search, Venice, Z.AI | URL lists with `[n]` or `[REF]` markers, no spans | | one `Citation { span: None }` per URL on the text block |
| Gemini GenerateContent, Vertex | candidate-level `groundingMetadata.groundingSupports[].segment { partIndex, startIndex, endIndex, text }` and `citationMetadata` (Vertex: `citations[]`) ([GR]) | bytes within `parts[partIndex]` | the decoder records each part's byte offset in its block; span + quoted `segment.text`; one `Url` source per `groundingChunkIndices`, with `confidence`. `webSearchQueries`, `searchEntryPoint` stay in `Extras` |
| Gemini gRPC | `citationMetadata` only; `groundingMetadata` is dropped at protobuf decode (rig's proto omits it) | bytes | grounding needs the proto to gain `GroundingMetadata` |
| Gemini Interactions | text item `annotations[]`: `url_citation`, `file_citation`, `place_citation` ([GX]) | undocumented, no quoted text | `span: None` with `Url` or `Document` until a non-ASCII recording settles the unit. Streamed `text_annotation_delta` today becomes a separate `Opaque` block (`crates/rig-core/src/providers/gemini/interactions_api/streaming.rs:226-239`) [unverified, unrecorded]; P6 merges it into the text item |
| Mistral | `reference` content chunks | | already `Opaque { replay: true }`; out of scope |
| Ollama, Candle, DeepSeek, Groq | none | | |

Replay never reads `citations`: an unedited turn replays its `native` item
byte for byte, and an edited or foreign turn is rebuilt from `text` alone.

## 11. Cost per provider

| provider | reported in the reply | path and unit | parts reported | otherwise |
|---|---|---|---|---|
| OpenRouter (Chat and Responses) | yes | `usage.cost`, credits (1 credit = 1 USD [unverified]); `usage.cost_details.upstream_inference_*` | total; upstream prompt and completion in `cost_details` | |
| xAI (Chat and Responses) | yes | `usage.cost_in_usd_ticks`, an integer, 10^10 ticks per USD (https://docs.x.ai/developers/cost-tracking) | total | |
| Venice | yes | top-level `cost.usd` (and `cost.diem`) | total | |
| Perplexity | yes | `usage.cost { input_tokens_cost, output_tokens_cost, request_cost, total_cost }`, USD | input, output, total; request fees are not token-derivable | |
| an OpenAI-compatible gateway's top-level `cost` (string or number) | yes [unverified: undocumented, seen in one capture] | top-level `cost` | total | |
| Anthropic and its dialects | no | tokens only; `usage.cache_creation` splits 5 min and 1 h writes | | catalog |
| OpenAI, Azure, ChatGPT, Copilot | no | tokens only | | catalog (none for ChatGPT and Copilot subscriptions) |
| DeepSeek, Mistral, Groq, Together, Moonshot, Z.AI, MiniMax, MiMo, llama.cpp | no | tokens only | | catalog |
| Gemini (GenerateContent, Interactions, Vertex, gRPC) | no | tokens; served tier in `usageMetadata.serviceTier` / `trafficType` | | catalog |
| Bedrock | no | tokens; `usage.cacheDetails` splits write TTLs | | catalog |
| Cohere | no | `usage.billed_units` (billed) and `usage.tokens` | | catalog over `billed_units` |
| Ollama, Candle | no | local | | `None` (no pricing row) |

The existing caller-side calculator `CacheRates`/`CacheCost`
(`crates/rig-core/src/completion/cache_cost.rs:23-68`) prices cache reads,
writes and storage from caller rates; P6 builds `Cost` from catalog `Pricing`
and keeps `CacheCost` for storage token-hours, which `Pricing` cannot hold.
Pricing by the served tier (pi multiplies by 2 for priority and 0.5 for flex,
`references/pi/packages/ai/src/api/openai-responses.ts:389-415`) is a later
extension.

## 12. Existing knobs each phase absorbs and deletes

A phase that replaces a knob deletes it in the same PR; there is never a
second way to set the same thing.

### 12.0 Wire-body changes for callers who set no option

These change what an existing caller sends without touching
`GenerationOptions`. Each is a wire-body change: its phase lists it under
Migration and names every cassette, golden or snapshot that moves (none is
expected; the corpus sets none of these keys in the affected shapes, and the
phase's `--check` runs prove it). The ChatGPT cassettes that set
`max_tokens`, `temperature` or `output_schema` (under
`crates/rig-cassette/tests/providers/chatgpt/`: `cassette/codex_sessions.rs:125`,
`cassette/raw_completion_parity_matrix.rs:75`, `:81`,
`cassette/raw_capture_matrix.rs:53`, `reasoning_tool_roundtrip.rs:17`,
`streaming.rs:28`, and the extractor tests) keep their recorded
bodies, because the Codex base builder still omits those fields (the Codex
row below).

| phase | change | who is affected | Migration line |
|---|---|---|---|
| P2 | Gemini GenerateContent and Interactions: raw `additionalParams.generationConfig` / `generation_config` keys now beat the typed fields (`crates/rig-core/src/providers/gemini/completion.rs:428-437`, `gemini/interactions_api/mod.rs:352-370`) | callers who set a typed field and the same key in `additional_params` | "`additional_params` now overrides `temperature`, `max_tokens` and other typed fields on Gemini; remove the duplicate key to keep the typed value." |
| P2 | OpenAI Responses flips from base-wins to raw-wins (`crates/rig-core/src/providers/openai/responses_api/mod.rs:464-468`): `additional_params.temperature`, `tool_choice`, `text` and any other key now override the typed field. The same loop skips a raw `null` today (`!value.is_null()`, `:465`); it is now sent as `null`, as on every other wire | Responses callers whose `additional_params` repeat a typed field or hold a `null` | "On OpenAI Responses, `additional_params` now overrides typed request fields, and a `null` in it is sent as `null`, as on every other wire. Remove a key to send nothing." |
| P2 | Shallow `extend` becomes a deep merge on Anthropic (`completion.rs:256`), Chat (`chat.rs:384`), Cohere native (`cohere/chat.rs:146-151`) and Bedrock (`crates/rig-bedrock/src/request.rs:133`): an object in `additional_params` such as `output_config`, `thinking`, `tool_choice` or `generationConfig` now merges into the wire's object instead of replacing it | callers who relied on an object in `additional_params` replacing the wire's whole object (for example to drop `output_config.format`) | "`additional_params` objects now merge key by key into the request body. A `null` is still sent as `null`. To drop a key the wire writes from a typed field, leave the field unset (for `output_config.format`, set no `output_schema`)." |
| P2 | Ollama `/v1`: the `think` to `reasoning_effort` rewrite is deleted (`crates/rig-core/src/providers/openai/wire/chat.rs:688-722`), so an existing `additional_params.think` is sent raw | `/v1` callers using `think` | "On Ollama's OpenAI-compatible route, set `GenerationOptions::reasoning` instead of `additional_params.think`, which is now sent as written." |
| P2 | OpenAI Responses, Codex: the `UNACCEPTED` strip (`responses_api/mod.rs:479-490`) is deleted, but not what it does to typed fields. The Codex base builder keeps omitting the typed fields the backend does not accept, `max_output_tokens` (from `max_tokens`), `temperature` and `text.format` (from `output_schema`), so a request with no `additional_params` sends today's body. Two things change: the mapped `parallel_tool_calls`, `service_tier` and `text.verbosity` are sent when the option is set (section 6.3), and raw `additional_params` keys (`temperature`, `max_output_tokens`, `top_p`, `metadata`, `user`, `background`, `parallel_tool_calls`, `service_tier`, `text`) reach the backend instead of being stripped | ChatGPT callers who put those keys in `additional_params` | "The ChatGPT backend no longer strips `additional_params` keys; remove `temperature`, `top_p`, `metadata`, `user`, `background` and `max_output_tokens` from them or expect the backend's error. Typed `max_tokens`, `temperature` and `output_schema` are still not sent to ChatGPT." |
| P2 | Anthropic: `with_automatic_caching()` and `with_automatic_caching_1h()` are deleted (section 12.1). `with_prompt_caching()` and `with_static_prefix_cache_ttl(..)` are unchanged; their markers take a 1 h TTL from `cache(Long)` | callers of the two deleted methods | "Replace `Messages::with_automatic_caching()` with `GenerationOptions::cache(CacheRetention::Short)` on the request, and `with_automatic_caching_1h()` with `CacheRetention::Long`." |
| P2 | rig-vertexai: the `contents` rewrites and the `model` key, written after the body is built today (`crates/rig-vertexai/src/completion.rs:66-76`), move into the base builder, so a raw `additional_params.model` or `contents` now overrides them | Vertex callers who put `model` or `contents` in `additional_params` | "On Vertex AI, `additional_params.model` and `additional_params.contents` now override the request's own values, as raw params do on every wire." |
| P2 | Bedrock: `cache(Short/Long)` after a reasoning turn reports the skipped message checkpoint through `on_unsupported`, an error by default, where `with_prompt_caching` skipped it silently (`crates/rig-bedrock/src/request.rs:106-114`) | Bedrock callers caching a conversation with reasoning in its history | "On Bedrock, a cache checkpoint that cannot follow a reasoning turn is an `UnsupportedOption` error; set `OnUnsupported::Ignore` to skip it with a warning." |
| P2 | OpenRouter Chat: `cache(Short)` sends the documented top-level `cache_control` where `with_prompt_caching` marked the system message (`chat.rs:816-843`) | OpenRouter callers of `with_prompt_caching` | "OpenRouter caching now uses the top-level `cache_control` marker." |
| P2 | Ollama native `/api/chat`: the `think` validation is deleted (`crates/rig-core/src/providers/ollama/chat.rs:46-47`, `:117-126`, `:257-269`), so a raw `additional_params.think` is sent as written under `RawAt::Split`: `"HIGH"` is no longer lowercased, and `"bogus"` is the daemon's error, not a local `EncodeError`. The warning on a raw `reasoning_effort` goes; the key still lands in `options` | native callers who set `think` in `additional_params` | "On Ollama's native route, set `GenerationOptions::reasoning` instead of `additional_params.think`, which is now sent as written; use a lower-case level." |
| P2 | Gemini Interactions: a raw `additional_params.tools` that is not an array is an `EncodeError` through `BaseInput::raw_tools`, where it is discarded today (`crates/rig-core/src/providers/gemini/interactions_api/mod.rs:380`) | Interactions callers whose `additional_params.tools` is not an array | "On Gemini Interactions, `additional_params.tools` must be an array, as on every other wire." |
| P2 | Gemini Interactions resume: a set option is refused through `on_unsupported`, and `additional_params` or provider options are an `EncodeError`, where `InteractionResume::encode` ignores the whole request today (`crates/rig-core/src/providers/gemini/interactions_api/mod.rs:247-274`) | callers who reuse a request with options or `additional_params` to resume an interaction | "`InteractionResume` now refuses a request that sets options or `additional_params`; resume with a request that sets neither." |
| P4 | Bedrock: the guardrail is sent on streams as well as unary requests (`crates/rig-bedrock/src/request.rs:135` filters it to unary today) | streaming callers of `with_guardrail` | "A Bedrock guardrail now also applies to streamed requests." |

### 12.1 P2: options

| knob | where | replaced by |
|---|---|---|
| Anthropic `automatic_caching`, `with_automatic_caching`, `with_automatic_caching_1h`, `automatic_caching_ttl` | `crates/rig-core/src/providers/anthropic/wire.rs:367-369`, `:410-413`, `:429-433` | `CacheRetention::Short`, `Long` |
| Anthropic `CacheTtl` as the automatic-caching TTL | `crates/rig-core/src/providers/anthropic/completion.rs:63-73` | `CacheRetention`; `CacheTtl` stays public as the type of `with_static_prefix_cache_ttl` |
| Anthropic `top_level_cache_control` (a wire-local precedence merge) | `crates/rig-core/src/providers/anthropic/completion.rs:661-692` | `request_params`; the 1 h-before-5 min and budget-of-4 checks run after it |
| Anthropic `body.extend(params)` | `crates/rig-core/src/providers/anthropic/completion.rs:256` | `request_params` (deep merge of `output_config`, `thinking`, `tool_choice`) |
| Anthropic `drops_unbound_thinking` reading raw JSON; `drops_unbound_items` reading `additional_params["thinking"]` | `crates/rig-core/src/providers/anthropic/completion.rs:157-182`, `:257-259`; `crates/rig-core/src/providers/anthropic/wire.rs:523-530`, `:646-653` | `drops_unbound_thinking` and `Rewrite::DropUnboundThinking` read the merged `FinalBody`'s `thinking`, as today. `drops_unbound_items` runs in `prepare`, before any body exists, and reads `options::mapped_param(target, request, "thinking")`: the mapped `reasoning` with the provider and raw layers on top. So with `reasoning(Off)` replay sees the `disabled` the body sends, and a raw `thinking` still wins, so replay and encoding agree |
| Chat `prompt_caching` field and `with_prompt_caching` (a no-op on every dialect but OpenRouter) | `crates/rig-core/src/providers/openai/wire/chat.rs:49-51`, `:248`, `:265-269` | `cache`; `UnsupportedOption` where a dialect cannot cache on request |
| `OpenAiWire::with_prompt_caching` (a no-op on the Responses route) | `crates/rig-core/src/providers/openai/wire/route.rs:107-111` | `cache` |
| OpenRouter `BodyRewrite::OpenRouter` caching rewrite | `crates/rig-core/src/providers/openai/wire.rs:165-167`, `crates/rig-core/src/providers/openai/wire/chat.rs:613`, `:816-843` | the documented top-level `cache_control` |
| Ollama `/v1` `think` to `reasoning_effort` rewrite | `crates/rig-core/src/providers/openai/wire/chat.rs:688-722`, doc `crates/rig-core/src/providers/openai/wire.rs:168-171`, `crates/rig-core/src/client/ollama.rs:40-43` | `reasoning`; the `num_ctx`/`options` refusal becomes the P4 route-section check |
| DeepSeek thinking detection from JSON | `crates/rig-core/src/providers/openai/wire/chat.rs:743-760` | kept as `Rewrite::ChatDialect(DeepSeek)`, reading the merged body's `thinking`, so a mapped `reasoning(Off)` and a raw `thinking: {"type": "disabled"}` both keep a forced `tool_choice` |
| Chat `body.extend(params)` | `crates/rig-core/src/providers/openai/wire/chat.rs:384` | `request_params` |
| Responses `additional_params` merged only where absent (inverted precedence) | `crates/rig-core/src/providers/openai/responses_api/mod.rs:464-468` | `request_params` |
| Responses `additional_params.text` replacing structured output | `crates/rig-core/src/providers/openai/responses_api/mod.rs:469-478` | deep merge of `text` |
| Responses Codex `UNACCEPTED` strip list | `crates/rig-core/src/providers/openai/responses_api/mod.rs:479-490` | the Codex base builder omits `max_output_tokens`, `temperature` and `text.format`, as the strip does today; options answer per field (section 6.3: `parallel_tool_calls`, `service_tier`, `verbosity` sent, `top_p` `UnsupportedOption`); raw keys are sent (section 12.0) |
| Responses `include` added from raw `reasoning` | `crates/rig-core/src/providers/openai/responses_api/mod.rs:330-337`, `:491-494` | `Rewrite::ReasoningCiphertext`, reading the merged body's `reasoning`, as today: a mapped or a raw `reasoning` both add the ciphertext |
| Gemini typed fields overriding `generationConfig`; `body.extend(params)` | `crates/rig-core/src/providers/gemini/completion.rs:417-437`, `:484` | `request_params` (behaviour change in Migration) |
| Interactions body built over `additional_params` | `crates/rig-core/src/providers/gemini/interactions_api/mod.rs:336-370` | `request_params` |
| Bedrock `prompt_caching`, `with_prompt_caching` | `crates/rig-bedrock/src/completion.rs:155`, `:199-202`; `crates/rig-bedrock/src/request.rs:62-64`, `:106-115` | `cache` (`Mapping::Place`); the skipped message checkpoint goes through `BaseInput::refuse_cache` |
| Bedrock `additional_params` as `additionalModelRequestFields`; `inferenceConfig` without `topP`, `stopSequences` | `crates/rig-bedrock/src/request.rs:125-133` | `request_params`; `top_p`, `stop` |
| Cohere native `body.extend(params)` | `crates/rig-core/src/providers/cohere/chat.rs:146-151` | `request_params` |
| Ollama native `think` validation and `THINK_LEVELS`; `reasoning_effort` moved into `options` with a warning | `crates/rig-core/src/providers/ollama/chat.rs:46-47`, `:117-126`, `:257-269` | `reasoning`; a raw `think` is sent as written (section 12.0) |
| Candle `top_p` and `seed` read from `additional_params`; the hosted-`tools` check reading `additional_params` | `crates/rig-candle/src/generation.rs:81-112`; `crates/rig-candle/src/protocol.rs:213-221` | `top_p`, `seed`. Candle calls `request_params` with an empty base and keeps `FinalBody` in its payload: it refuses hosted tools when `FinalBody::get("tools")` is present, and reads the remaining overrides with `FinalBody::deserialize` (`deny_unknown_fields` stays) |
| rig-vertexai writes to the built body: thought-signature and media rewrites of `contents`, the `model` key | `crates/rig-vertexai/src/completion.rs:66-76` | the Vertex base builder, below the merge (section 12.0) |
| Docs that steer reasoning to `additional_params` | `crates/rig-core/src/providers/groq.rs:3-4`, `crates/rig-core/src/providers/openai/completion/mod.rs:8-29` | `GenerationOptions::reasoning` |

Each existing caching placement and how P2 expresses it. Every row sends
the same body as today, so no cassette moves. Error texts that name a deleted
method change, and the tests asserting them
(`crates/rig-cassette/tests/providers/anthropic/cassette/prompt_caching.rs:337-359`)
are updated with them.

| today | markers sent | P2 | callers |
|---|---|---|---|
| `with_automatic_caching()` | top-level, no TTL | `cache(Short)` | `crates/rig-cassette/tests/providers/anthropic/cassette/prompt_caching.rs:61`, `long_run_workloads.rs:43`, `strict_schema_streaming.rs:174`, `test-support/rig-test-support/src/model_session.rs:685`, `crates/rig-cassette/tests/common/ecs_matrix/world/tests.rs:30` |
| `with_automatic_caching_1h()` | top-level `1h` | `cache(Long)` | `prompt_caching.rs:64`, `ecs_matrix_long_loop.rs:38`, `strict_schema_integrations.rs:97`, `crates/rig-cassette/tests/common/ecs_matrix/long_tasks/cache/tests.rs:23` |
| `with_prompt_caching()` | final tool, last system block, last message block; no top-level | unchanged, `cache` unset | `prompt_caching.rs:57`, `ecs_prompt_caching.rs:20`, `ecs_matrix_long_loop.rs:24`, `long_tasks/cache/tests.rs:16` |
| `with_prompt_caching().with_automatic_caching()` | top-level; tool and system (with a top-level marker, manual placement adds no message marker, `completion.rs:761`) | `with_prompt_caching()` plus `cache(Short)` | `prompt_caching.rs` |
| `with_prompt_caching().with_automatic_caching_1h()` | top-level `1h`; tool and system `1h` | `with_prompt_caching()` plus `cache(Long)` | `prompt_caching.rs` `ManualAutomatic1h` |
| `with_automatic_caching().with_static_prefix_cache_ttl(t)` (or `_1h()`) | top-level; tool and system with `t` | `with_static_prefix_cache_ttl(t)` plus `cache(Short)` (or `Long`) | `prompt_caching.rs:69`, `strict_schema_integrations.rs:57-58`, `ecs_matrix_long_loop.rs:53`, `long_tasks/cache/tests.rs:30` |
| `with_static_prefix_cache_ttl(t)` alone | tool and system with `t` | unchanged, `cache` unset | none in the workspace |
| Bedrock `Converse::with_prompt_caching()` | `cachePoint` after the system blocks and at the end of the last message, skipped after a reasoning turn | `cache(Short)`; the skip goes through `BaseInput::refuse_cache` (section 12.0) | `crates/rig-cassette/tests/providers/bedrock/cassette/agent.rs:61`, `crates/rig-bedrock/tests/history_conformance.rs:625`, `tests/integrations/bedrock/adaptive_thinking.rs:28` |
| Chat `with_prompt_caching()` | OpenRouter: a system-message marker; elsewhere nothing | `cache(Short)`: OpenRouter's documented top-level marker, `UnsupportedOption` elsewhere (section 12.0) | `crates/rig-cassette/tests/common/ecs_matrix/world/tests.rs:68`, on OpenAI where it is a no-op; the call is removed. No OpenRouter cassette uses it |

`tests/core/history_conformance/anthropic.rs:161` builds `Messages` by
struct literal and drops the two deleted fields.

### 12.2 P3: the catalog

| knob | where |
|---|---|
| Anthropic private model table and its readers | `crates/rig-core/src/providers/anthropic/completion.rs:81-140` (also read by `crates/rig-bedrock/src/completion.rs:308-310`) |
| `OutputCap::OpenAiReasoningFamilies`, `is_openai_reasoning_model` | `crates/rig-core/src/providers/openai/wire.rs:130-138`, `crates/rig-core/src/providers/openai/wire/chat.rs:391-396`, `crates/rig-core/src/providers/openai/completion/mod.rs:144-167` |
| hard-coded reasoning replay field by model | `crates/rig-core/src/providers/openai/wire/chat.rs:225-239` |
| per-vendor image-input rules | `crates/rig-core/src/providers/openai/wire/chat.rs:1050-1078` |
| GPT-6 sampling rule claimed in docs | `crates/rig-core/src/providers/openai/completion/mod.rs:8-30` |

### 12.3 P4: provider options

| knob | where | replaced by |
|---|---|---|
| Gemini `with_cached_content` and raw `cachedContent` | `crates/rig-core/src/providers/gemini/completion.rs:68-70`, `:121-124`, `:403-416` | `GeminiOptions.cached_content` |
| Gemini `safetySettings` always `null` | `crates/rig-core/src/providers/gemini/completion.rs:479` | `GeminiOptions.safety_settings` |
| Interactions `previous_interaction_id`, `agent`, `agent_config` read from raw | `crates/rig-core/src/providers/gemini/interactions_api/mod.rs:136-142`, `:346`, `:396-399` | `GeminiOptions` Interactions section; the readers go through `options::param`, which sees the typed option and the raw key alike |
| Responses `store`, `previous_response_id`, `conversation` read from raw JSON | `crates/rig-core/src/providers/openai/responses_api/mod.rs:386-389`; `crates/rig-core/src/providers/openai/responses_api/wire.rs:257-263` | `OpenAiOptions` Responses section; the readers go through `options::param`. The WebSocket session keeps writing the chain's `previous_response_id` to `additional_params` (section 2.1) |
| Bedrock `guardrail`, `with_guardrail` (sent on unary only: a silent drop on streams) | `crates/rig-bedrock/src/completion.rs:156-158`, `:210-222`; `crates/rig-bedrock/src/request.rs:135` | `BedrockOptions.guardrail`, sent on both modes |
| Cohere `with_strict_tools` | `crates/rig-core/src/providers/cohere/wire.rs:123-128` | `CohereOptions` `"*"` `strict_tools` |
| Candle `RequestGenerationOverrides` non-portable fields | `crates/rig-candle/src/generation.rs:81-89` | `CandleOptions` |

### 12.4 P5: one raw shape

The terminal records in section 9: `crates/rig-core/src/providers/openai/wire/chat.rs:1687-1708`,
`crates/rig-core/src/providers/anthropic/streaming.rs:549-558`,
`crates/rig-core/src/providers/gemini/streaming.rs:195-207` and `keep_raw` at `:73`
(callers `crates/rig-gemini-grpc/src/streaming.rs:52`, `crates/rig-vertexai/src/types/completion_response.rs:34`),
`crates/rig-core/src/providers/gemini/interactions_api/streaming.rs:342-350`,
`crates/rig-core/src/providers/cohere/streaming.rs:493`,
`crates/rig-core/src/providers/ollama/streaming.rs:158`,
`crates/rig-bedrock/src/streaming.rs:412-427`. The public `Out::raw` for
completions goes away (`crates/rig-core/src/wire.rs:600`). The generic JSON
decoder that calls `out.raw(document)` for every `Op`
(`crates/rig-core/src/wire.rs:736`) stops doing so for completions, whose
`raw` the driver now records from the wire's `Reassembler` (section 4).

### 12.5 P6: citations and cost

| knob | where | replaced by |
|---|---|---|
| reading citations from `Text.native` JSON | `crates/rig-cassette/tests/providers/cohere/cassette/native.rs:36-46` and every caller indexing `native` | `Text::citations()` (`native` keeps working) |
| `AutoCache` hard-coded Gemini 3.8 Flash price ratios | `crates/rig-core/src/providers/gemini/caching.rs:126-160` | catalog `Pricing` |

## 13. Acceptance tests

All live in `crates/rig-cassette/tests/runtime/typed_options/tests.rs`, in the
`runtime` target of `rig-cassette`: it is the cross-provider replay target,
and its `rig` dev-dependency includes Bedrock. Each group is an inline module
gated by `#[cfg(any())]` with a comment naming its phase.

| test | phase | what it pins |
|---|---|---|
| `harness_switch::one_options_value_drives_five_wires` | P2 | one `GenerationOptions` (reasoning `High`, cache `Long`, tier `Default`) encoded for Anthropic `claude-opus-4-8`, OpenAI Responses `gpt-5.5`, Gemini `gemini-3-flash-preview`, Bedrock `us.anthropic.claude-sonnet-5` and OpenRouter `anthropic/claude-sonnet-4.5`; asserts each body against section 6 (Anthropic: one top-level 1 h marker, none placed by hand) and that the only warning is Gemini's `cache`. No JSON is written on the request side |
| `harness_switch::long_cache_on_gemini_is_an_error_under_the_default_policy` | P2 | Gemini `Long` under `Error` returns `UnsupportedOption { option: "cache", provider: "gcp.gemini", .. }` |
| `no_silent_drop::caching_on_a_dialect_that_cannot_cache_is_refused` | P2 | `Short` on Cohere's Compatibility route and `Long` on DeepSeek return `UnsupportedOption` naming `cache` and the provider |
| `no_silent_drop::under_ignore_the_option_is_skipped_with_a_warning` | P2 | the same under `Ignore`: the body has no cache field and one warning names `cache` and the provider |
| `no_silent_drop::the_driver_refuses_an_option_before_any_wire_encodes` | P2 | on Bedrock Converse, an SDK-backed wire, `Completion::prepare` alone refuses `seed` with `UnsupportedOption` naming `seed` and `aws_bedrock`; under `Ignore` it warns once and clears the option |
| `option_matrix::every_option_alone_gives_its_section_6_cell` | P2 | a table-driven golden: each `GenerationOptions` field set alone on Anthropic, Responses, Gemini, Bedrock, OpenRouter, Cohere and DeepSeek gives exactly its section 6 cell for that wire, field and value: the body equals the baseline body with the cell's object deep-merged in (or a marker pushed onto an array), equals the baseline for an "omit" cell, or fails with `UnsupportedOption` naming that field and provider. So `Omit` passes only where the cell says "omit", and a set option answered `Nothing` fails it. P2 extends the table to every completion wire and dialect, `InteractionResume` included |
| `option_layers::a_run_field_beats_the_agent_field_and_leaves_the_rest` | P2 | `GenerationOptions::overlay`, the one merge rig-agent and rig-ecs use: an agent's `reasoning(High)` survives a run that sets only `cache(Long)`; the run's `seed` and `stop` win; an empty run changes nothing |
| `precedence::raw_tools_are_appended_to_rig_tools` | P2 | a function tool plus a raw server tool in `additional_params.tools` both reach the Anthropic body, rig's first; a raw function tool joins rig's on OpenRouter Chat. The merge never replaces rig's tools |
| `precedence::null_is_sent` | P2 | a `null` in `additional_params` over a mapped `top_p` is sent as `"top_p": null`, as `body.extend` sends it today; on OpenAI Responses a raw `"user": null`, skipped today, is sent (section 12.0) |
| `precedence::interactions_refuses_raw_tools_that_are_not_an_array` | P2 | a raw `tools` object on Gemini Interactions, discarded today, is an encode error that names no option (section 12.0) |
| `precedence::codex_still_omits_the_typed_fields_the_backend_refuses` | P2 | ChatGPT with `max_tokens`, `temperature` and `output_schema` sends no `max_output_tokens`, `temperature` or `text.format`, as today, so its recordings keep their bodies; set `parallel_tool_calls`, `service_tier(Flex)` and `verbosity(Low)` and a raw `metadata` are sent |
| `precedence::post_merge_rewrites_read_raw_keys` | P2 | a raw `additional_params.reasoning` still adds `include: ["reasoning.encrypted_content"]` on Responses, and a raw `thinking: {"type": "disabled"}` still keeps a forced `tool_choice: "required"` on DeepSeek: the post-merge rewrites read the merged body, as today |
| `usage_cost_sum::cost_sums_only_when_every_turn_has_one` | P1 | the table in section 2.3: two priced turns sum each part, a priced and an unpriced turn give no cost, a fold from `Usage::default()` keeps the first turn's cost; token counters sum as today |
| `catalog_validation::validate_rejects_a_reasoning_level_the_model_lacks` | P3 | Gemini 3.8 Flash rejects `Minimal` and takes `High`; Claude Haiku 4.5 rejects any effort and takes a budget inside 1024 and up |
| `typed_extras_unary::openrouter_and_deepseek_extras_from_unary_recordings` | P4 | `extras::<OpenRouter>()` gives `provider = "Azure"`, `cost = 2.7e-6`, `native_finish_reason = "stop"`; `extras::<DeepSeek>()` gives 256 cache hits; each is `None` on the other's reply |
| `typed_extras_streamed::the_same_extras_from_streamed_recordings` | P5 | the same fields from the streamed recordings, and OpenRouter's agree across paths; DeepSeek gives 4864 hits |
| `citations_and_cost::a_cohere_citation_reply_decodes_with_citations` | P6 | the recorded Cohere native reply has one citation spanning "Dock Seven" from document `harbor-record-1`, and replay sends the recorded native citation unchanged |
| `citations_and_cost::reported_cost_wins` | P6 | OpenRouter's `usage.cost` becomes `Cost.total` on both paths |
| `citations_and_cost::catalog_cost_when_none_is_reported` | P6 | a DeepSeek reply's cost is computed from catalog pricing, and a model outside the catalog has none |

Fixtures read (all existing; none is recorded or edited by P0):
`openrouter/raw_capture_matrix/raw_round_trips_openrouter_type.yaml`,
`openrouter/raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider.yaml`,
`deepseek/portability_matrix/from_openai_responses.yaml`,
`deepseek/prompt_caching/streaming_probe.yaml`,
`cohere/native/a_conversation_switching_routes_replays.yaml`.

## 14. Open pull requests: supersession

| PR | what it does | this stack | phase | tests to port |
|---|---|---|---|---|
| #2616 | Opus 5.5 and GPT-6 typed settings (the constants already landed) | supersedes the rest: Anthropic effort and thinking with one `output_config` merge (P2); GPT-6 sampling and effort rules as catalog data (P3); `ThinkingDisplay` with its beta and Responses `prompt_cache_options.mode`/`prewarm` (P4). `Messages::with_effort`/`with_thinking` are not ported: they would be a second way to set reasoning | P2, P3, P4 | 8 Anthropic wire tests, 6 OpenAI completion tests, 1 Responses test |
| #1480 | Claude adaptive thinking | supersedes: the mapping (P2) and per-model validation (P3) | P2, P3 | 3 |
| #1833 | Bedrock cache point beside reasoning | supersedes the behaviour: the checkpoint is placed or reported through `on_unsupported`, never skipped silently | P2 | 1 |
| #2171 | OpenAI explicit cache breakpoints | supersedes `prompt_cache_options.ttl` (P2) and `mode`/`prewarm` (P4); per-block breakpoints are out of scope | P2, P4 | 1 |
| #1867 | OpenRouter model fallbacks and routed model | supersedes `models` beside `ProviderPreferences`; the routed model already exists as `Origin::response_model`. Precedence reverses: raw `additional_params` wins over the typed option | P4 | 5, the precedence test inverted |
| #2243 | provider-reported streaming cost | supersedes with `Cost` (not `f64`), no token arithmetic, no `Eq` on floats; reads OpenRouter `usage.cost` and the undocumented top-level `cost` [unverified] | P6 | 2 |
| #2169 | Anthropic per-block `cache_control` | not superseded: out of scope; its carrier field is gone from `Text` | | |
| #2625 | Gemini on a generated API mirror | not built on: it stops reading `additional_params`, against decision A | | |
| #2612 | one error type | not built on: P1 adds `UnsupportedOption` so it survives either error model | | |
| #2517 | providers opt-in by Cargo feature | not built on: `extension` modules and `connect` arms must allow a `cfg(feature)` gate | | |
| #2634, #2192, #2190 | new presets and routes | independent; each adds rows to sections 6 and 7 if it lands | | |
| #2722, #2524, #2588, #2624, #2620, #2281, #2203 | unrelated or rebase-only | ignored | | |

## 15. Prior art

- **pi**: one portable reasoning level, `SimpleStreamOptions.reasoning: ThinkingLevel`
  (`references/pi/packages/ai/src/types.ts:352`), and the catalog fields
  `reasoning`, `thinkingLevelMap`, `contextWindow`, `maxTokens`,
  `promptCache`, `cost`, `compat` (`references/pi/packages/ai/src/types.ts:1100-1135`);
  `Usage.cost` (`references/pi/packages/ai/src/types.ts:429`). pi keys
  per-API options by API (`references/pi/packages/ai/src/types.ts:259-270`)
  and must then put OpenRouter routing on the model's compat record
  (`references/pi/packages/ai/src/types.ts:829-832`), which is why decision B
  keys by provider. pi maps `minimal` to `low` on Anthropic
  (`references/pi/packages/ai/src/api/anthropic-messages.ts:916-918`); this
  design refuses instead.
- **opencode**: `variants(model)` (`references/opencode/packages/opencode/src/provider/transform.ts:790`)
  and provider-keyed `providerOptions` (`references/opencode/packages/opencode/src/provider/transform.ts:1421-1478`);
  its key-remapping shim (`:486-497`) and Azure double-write (`:1473-1477`)
  show what happens when the key is not the runtime identity.
- **Codex**: `ModelInfo.supported_reasoning_levels` and
  `default_reasoning_level` (`references/codex/codex-rs/protocol/src/openai_models.rs:405`),
  the model of `ReasoningSupport`; provider settings owned by the provider
  (`references/codex/codex-rs/model-provider-info/src/lib.rs:139-170`).
- **hermes**: rebuilds Anthropic `Message` and Chat completions from streams
  (`references/hermes/agent/relay_llm.py:606-670`,
  `references/hermes/agent/auxiliary_client.py:7249-7370`), the decision D
  angle; keeps citations on the text block
  (`references/hermes/agent/anthropic_message_convert.py:49-53`), the decision
  E angle; and chooses provider JSON by base-URL host
  (`references/hermes/agent/reasoning_params.py:88-120`), the bug decision B
  removes.
