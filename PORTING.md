# Porting a provider to the wire model (working notes — deleted before the PR)

Worktree: `/Users/kisaczka/Desktop/code/many_rigs/rig-wire-ffe543`. Branch
`wire/unification-ffe543`. **Never** work in `~/Desktop/code/many_rigs/rig`
or any other `rig-wire-*` worktree.

The shared surface is frozen and green (`cargo test -p rig-core --lib`, 1832
passing). Read these before writing anything:

- `crates/rig-core/src/wire.rs` — `Wire`, `Operation`, `Decoder`, `Encoded`,
  `Body`, `Framing`, `Secret`, `ObservationSink`, `WireError`, `HasCompletion`.
  Its module docs contain a complete fake provider.
- `crates/rig-core/src/driver.rs` — `call`, `stream`, `WireDriver`, plus
  `driver/bound.rs` (`Bound`) and `driver/consumers.rs` (the seven consumer
  impls and the `Has*` construction traits).
- `crates/rig-core/src/driver/tests.rs` — a worked fake wire with a unary
  variant, observation projection, and paging.
- `crates/rig-core/src/operation.rs` + `operation/*.rs` — the nine
  operations.

## The shape of a port

```rust
// providers/<name>/mod.rs — the provider's shared configuration: DATA.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anthropic {
    pub api_key: Secret,
    pub base_url: String,
    pub version: String,
    pub betas: Vec<String>,
    pub dialect: Dialect,
}

impl Anthropic {
    pub fn new(api_key: impl Into<Secret>) -> Self { … }        // defaults
    pub fn from_env() -> Result<Self, EnvError> { … }            // same env vars as today
    pub fn messages(&self, model: impl Into<String>) -> Messages { … }
}

// One wire per operation, named after the API endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Messages { pub provider: Anthropic, pub model: String, /* options */ }

impl Wire for Messages {
    type Op = Completion;
    type Decoder = MessagesDecoder;
    fn name(&self) -> &str { self.provider.dialect.name }
    fn model(&self) -> Option<&str> { Some(&self.model) }
    fn encode(&self, request: CompletionRequest) -> Result<Encoded, CompletionError> { … }
    fn decoder(&self) -> Self::Decoder { … }
    fn capabilities(&self) -> ProviderCapabilities { … }
}

// The provider config also implements the construction trait for each
// operation it offers, so `Bound<Anthropic, H>` gets `completion(model)`,
// `model_listing()`, `verify()` … from `driver/consumers.rs`.
impl HasCompletion for Anthropic { type Wire = Messages; fn completion(&self, m: impl Into<String>) -> Messages { self.messages(m) } }
```

## Rules that are not negotiable

1. **No `.await`, no `Arc`, no `Box<dyn`, no `impl Future`, no `async`, and
   no `H`/`T` transport type parameter anywhere under
   `crates/rig-core/src/providers/` except `openai/responses_api/websocket.rs`.**
   `cargo xtask check-wires` enforces it.
2. **One decoder per wire family, serving unary and streaming.** The unary
   reply shape is one more `WireEvent` variant whose `interpret`
   *synthesizes the stream events* (message start, one block per content
   part, terminal with usage and finish reason) and lets the existing block
   code do the rest. Delete the provider's `NormalizeCompletionResponse`
   impl and every duplicated `Content -> AssistantContent` mapping: do not
   move them. If the unary and stream content types genuinely differ,
   convert the unary type to the stream type inside `classify`.
3. **Dialects are `const` values, never types or traits.** Delete every
   `Ext` / `*CompatibleProvider` trait and every `Generic*Model<Ext, H>`. A
   quirk a flag cannot express is a `match dialect.quirks` inside that
   wire's `encode`/`classify`, in one file.
4. **Observation is `Decoder::project`.** The provider's `observation.rs`
   `payload` fn becomes the `project` body, writing through
   `ObservationSink` (`emit`, `provider`, `scrub`). Nothing attaches a
   per-request observer any more: the driver projects the unary reply and
   every stream frame through the decoder itself.
5. **Telemetry, request-id capture, the non-success funnel, `Accept`, the
   status split and the span are the driver's.** Delete the provider's
   `ProviderResponseExt` impl, its span builders, its `record_token_usage`
   calls, its `_observed` siblings and its `raw_completion`/`raw_stream`
   inherent methods. Callers read `CompletionResponse::raw` instead.
6. **Every `with_*` option moves onto the wire**, taking and returning the
   wire by value. `Bound::map_wire` is the one forwarder.
7. **Secrets live in `Secret`**; a wire serialized to JSON never contains a
   key. One unit test per provider config asserts it.
8. Sibling test modules only: `#[cfg(test)] mod tests;` plus `tests.rs`.
9. Comments explain *why*.

## Build order (important)

Ports are **additive** for now: add the new `Wire` types and keep the
provider's existing `Client`/`Has*`/model types compiling beside them. The
integration owner deletes the whole client layer in one later commit, which
is what makes the final diff have exactly one model. This is the only way
each slice can compile and test itself while the others are in flight.

So: **add** `Wire`, `Decoder`, the config struct and the `Has*` impls;
**leave** the old types in place; **do not** edit
`crates/rig-core/src/lib.rs`, `prelude.rs`, `providers/mod.rs`,
`providers/internal/mod.rs`, `wire.rs`, `driver*.rs`, `operation*.rs`, or
any other provider's directory. Ask the integration owner (`Main`) through
`hub` if you need a change there.

## Verifying a slice

- `cargo check -p rig-core --lib` must be clean for your files.
- `cargo test -p rig-core --lib <provider>::` must pass.
- Add byte-fixture unit tests: feed a recorded body from
  `tests/cassettes/<provider>/**` through `wire.decoder()` +
  `WireDriver` and assert the folded response. **Never edit a cassette.**
- Do **not** run `cargo fmt`, clippy, or the workspace suite; the
  integration owner runs those once at the end.
