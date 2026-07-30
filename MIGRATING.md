# Migrating Rig

This guide covers every breaking change from 0.30 through 0.41. Releases 0.36,
0.37, 0.40 and 0.41 were the disruptive ones; 0.40 alone carried 31 breaking
changes, and 0.37 renamed `rig-core`'s library target.

## Which sections apply to you

Sections run newest-first. Find the version you are on and read every section
above it, in order. Each one is self-contained.

| You are on | Start at |
| --- | --- |
| 0.41 | [0.41 → 0.42](#041--042-unreleased) |
| 0.40 | [0.40 → 0.41](#040--041) |
| 0.39 | [0.39 → 0.40](#039--040) |
| 0.38 | [0.38 → 0.39](#038--039) |
| 0.37 | [0.37 → 0.38](#037--038) |
| 0.36 | [0.36 → 0.37](#036--037) |
| 0.35 | [0.35 → 0.36](#035--036) |
| 0.34 | [0.34 → 0.35](#034--035) |
| 0.33 | [0.33 → 0.34](#033--034) |
| 0.32 | [0.32 → 0.33](#032--033) |
| 0.31 | [0.31 → 0.32](#031--032) |
| 0.30 | [0.30 → 0.31](#030--031) |

**Everyone should read [Silent behavior changes](#silent-behavior-changes)
first.** Those are the changes that leave your code compiling and make it do
something different — the ones a compiler upgrade will not point at.

---

## Silent behavior changes

Nothing in this section produces a compile error. Grouped by the release that
introduced it, oldest-first — read from the version you are on upwards, and
check each entry against your code before upgrading.

### 0.31

#### The default TLS backend is now rustls

`rig-core`'s default feature switched from `reqwest-tls` (native TLS) to
`reqwest-rustls`, alongside the upgrade to reqwest 0.13. Unless you opt back in
with `reqwest-native-tls`, HTTPS now goes through rustls.

rustls does not read the platform trust store the way native TLS does. Private
CAs installed in the macOS Keychain or the Windows certificate store, corporate
TLS-inspecting proxies, and servers still on legacy cipher suites can start
failing handshakes on a build that changed nothing but the Rig version.

#### Persisted `Reasoning` JSON no longer round-trips

`Reasoning` changed from `{ id, reasoning: [String], signature: Option<String> }`
to `{ id, content: [ReasoningContent] }`, where each block is typed (`Text` with
an optional signature, `Summary`, `Encrypted`, `Redacted`). The struct change
is a compile error, but the **serde shape change is not**: `Message` derives
`Serialize`/`Deserialize`, and there are no field aliases for the old names.

If you store conversation history as JSON, records written by 0.30 or earlier
that contain reasoning blocks fail to deserialize on 0.31. Migrate the stored
rows, or drop reasoning content from history you replay.

### 0.32

#### `gemini::EMBEDDING_001` points at a different model

The constant kept its name and changed its value from `"embedding-001"` to
`"gemini-embedding-001"`. Embeddings written before and after the upgrade come
from different models and are not comparable — re-embed the corpus, or pin the
literal `"embedding-001"` if the old model is what you want.

See also [gemini embedding dimensions](#4-gemini-embedding-dimensions-come-from-the-model)
in 0.32 → 0.33, which moves the reported dimension count for this model.

#### Proxy environment variables are now honored

reqwest's `system-proxy` feature was enabled (#1442). `HTTP_PROXY`,
`HTTPS_PROXY`, and `NO_PROXY` in the environment now route provider traffic;
previously they were ignored. On machines that set those variables for unrelated
reasons, provider calls start going through the proxy.

### 0.33

#### Preamble is now a system message, and `CompletionRequest.preamble` is empty

`CompletionRequestBuilder::build` no longer populates `CompletionRequest.preamble`.
It inserts the preamble as a leading `Message::System` in `chat_history` and
leaves `preamble` as `None`. The field survives only as a legacy carrier for
callers that construct `CompletionRequest` by hand.

**If you implement `CompletionModel` yourself and read `request.preamble`, the
system prompt silently vanishes.** Read the leading `Message::System` from
`chat_history` instead. Bundled providers were all updated.

#### `max_tokens` is forwarded on Chat Completions

`max_tokens` was dropped when building Chat Completions requests (#1495) and is
now sent. A limit your code has been setting with no effect starts applying, so
responses that ran to their natural stop can begin truncating.

### 0.35

#### String tool outputs are no longer JSON-encoded

A tool whose `Output` serializes to a JSON string used to be handed to the model
as `serde_json::to_string(&output)` — wrapped in quotes, with newlines escaped
as `\n`. It is now passed through verbatim (#1608). Non-string outputs are
unchanged.

This is the intended behavior, but a tool returning `String` now presents to the
model as raw text rather than a quoted JSON literal. Prompts and few-shot
examples tuned against the quoted form are worth re-checking.

### 0.36

#### rustls became the default for websockets, middleware, and companion crates

0.31 switched `rig-core`'s own HTTP client to rustls; 0.36 finished the job
(#1682). `websocket` is now an alias for `websocket-rustls`,
`reqwest-middleware` for `reqwest-middleware-rustls`, and the `rustls` /
`native-tls` features on the workspace fan out to the companion crates.

If you were relying on those paths still using native TLS, the trust-store
caveat from [0.31](#the-default-tls-backend-is-now-rustls) now applies to
them as well. Opt back in with `websocket-native-tls` /
`reqwest-middleware-native-tls`.

### 0.38

#### Hallucinated tool calls now fail the run

Tool calls are validated against the registered tools and against `tool_choice`
before dispatch (#1823). A call naming a tool that does not exist — or any tool
call at all under `ToolChoice::None` — returns `PromptError::UnknownToolCall`
instead of being attempted.

Runs that previously limped along on a provider's occasional invented tool name
now stop. 0.38 also adds the recovery path: implement
`PromptHook::on_invalid_tool_call` and return
`InvalidToolCallHookAction::{Retry, Repair, Skip, Fail}`, and bound the retries
with `.max_invalid_tool_call_retries(n)`.

#### `#[rig_tool]` advertises a different schema

Parameter schemas are generated by schemars instead of the previous hand-rolled
type mapping (#1576). The macro's surface is unchanged, but what reaches the
model is not:

- Integer parameters advertise `"type": "integer"`; they were `"number"`.
- Structs, enums, and other non-primitive parameters get real schemas with
  `$defs`, instead of a bare `{"type": "object"}`.
- Doc comments on the function and on individual parameters become the tool and
  parameter descriptions, replacing the `Function to <name>` /
  `Parameter <name>` defaults.
- `Option<T>` parameters get `#[serde(default)]`, so a model that omits them
  deserializes to `None` rather than failing.

Better schemas usually mean better tool calls, but they are different input to
the model. There is also a compile-time consequence — see
[section 1 of 0.37 → 0.38](#1-rig_tool-parameters-must-implement-jsonschema).

### 0.40

#### `max_turns` counts differently

The highest-impact change in this release. `max_turns` and
`default_max_turns` now bound the **exact total number of model calls**,
including the initial call, tool continuations, and retries.

| Budget | Before | After |
| --- | --- | --- |
| `0` | initial call + 2 | no model call at all |
| `1` | initial call + 2 | only the initial call |
| `n` | effectively `n + 2` | exactly `n` |

An unconfigured tool-then-answer flow now needs an explicit budget of `2`. To
preserve the maximum allowance of an old explicit budget `n`, account for the
old effective `n + 2`; otherwise set the literal total you actually intend.

If you set `max_turns` at all, re-derive the number. A budget that used to
permit a tool round-trip may now stop after the first model call.

#### Providers that used to return empty text now error

Responses with empty assistant content and no tool calls surface the shared
path's "empty response" error on **hyperbolic, perplexity, and huggingface**.
Previously each returned an empty text completion. Code that treated an empty
string as a normal outcome now takes an error branch.

#### Moonshot drops reasoning-only history turns

The shared conversion drops assistant messages carrying neither text nor tool
calls. Reasoning attached to a text or tool-call turn still round-trips via
`reasoning_content`; reasoning-only turns no longer survive in history.

#### Request contents changed for several providers

Consequences of the `GenericCompletionModel` consolidation. None of these
change your source, all of them change what goes over the wire:

- `max_tokens` is now **forwarded** by deepseek, together, hyperbolic, and
  azure. It was silently dropped before — these providers will now respect a
  limit your code has been setting all along with no effect.
- together's streaming uses standard `stream`/`stream_options` rather than
  `stream_tokens`, and a rig-level `ToolChoice::Required` serializes as
  `required` instead of erroring.
- perplexity's non-streaming endpoint drops a stray `/v1` prefix.
- mira sends its preamble as a `system` message instead of `user`.
- `response_format` derived from `output_schema` is deferred while tools are
  pending a result. groq, mistral, and azure previously applied it
  unconditionally.
- groq's streaming usage no longer falls back to the legacy `x_groq.usage`
  envelope.

#### Rig `Video` content now serializes instead of erroring

On the shared conversion, `Video` user content serializes a `video_url` content
part (an OpenRouter/gateway extension) rather than returning a client-side
conversion error. Providers without video support now reject it **server-side**
— the failure moved from your process to theirs, and moved later.

#### Telemetry span names and fields

- Migrated providers' streaming spans are named `chat` with
  `gen_ai.operation.name = "chat"`, previously `chat_streaming`. Dashboards and
  alerts keying on `chat_streaming` go blank.
- `gen_ai.input.messages` / `gen_ai.output.messages` are intentionally left
  empty rather than recording serialized messages.
- `gen_ai.request.model` reports the per-request model override when one
  applies.
- minimax, zai, and xiaomimimo spans stop reporting as `"openai"` and report
  their own provider name.

### 0.41

#### Ollama now honors `max_tokens`

Ollama's native `/api/chat` has no top-level `max_tokens` field, so the value
was serialized into a field the server does not define and silently discarded.
It is now sent as the `num_predict` model parameter inside `options`, where
Ollama actually reads it.

Nothing to change — but if you set `max_tokens` on an Ollama agent at any point
and moved on when it appeared to do nothing, **it starts applying now**.
Responses that had been running to their natural stop will truncate at the
budget you configured, possibly long ago. Check the value is one you still want.
`temperature` is unaffected: it was already being sent inside `options` and only
a redundant top-level copy was removed.

#### Multipart tool results reach OpenAI intact

Tool results carrying several `ToolResultContent` blocks were flattened before
being sent to the Responses and Chat Completions APIs. Individual blocks are now
preserved — retained as multipart when array-form tool results are enabled, and
flattened only where string-form content is required.

Tools that return mixed text/JSON/rich output now present to the model as
distinct blocks rather than one merged blob. That is the intended behavior, but
it does change what the model sees, so prompts tuned against the flattened shape
are worth re-checking.

#### `if_wasm!` / `if_not_wasm!` key on the target

These macros are `#[macro_export]`ed, and a `cfg` inside a macro expansion is
evaluated in the **calling** crate. The old expansion therefore tested whether
*your* crate had a feature named `wasm`, not whether `rig-core` did. Any caller
without such a feature took the `if_not_wasm!` branch on every target, browser
wasm included.

They now key on `all(target_arch = "wasm32", target_os = "unknown")`. If you
defined a `wasm` feature and expected it to drive these macros, you now get the
target's answer instead. Gate on the target directly if you need the old
association.

---

## 0.41 → 0.42 (unreleased)

The agent runtime is now *data-oriented*: providers are plain configuration,
and the classic `Agent` lost its model type parameter.

### `rig-mcp` is renamed to `rig-rmcp`

The crate is named after the `rmcp` SDK it wraps. Only a direct dependency
needs changing:

```toml
# before
rig-mcp = "0.41"
# after
rig-rmcp = "0.42"
```

```rust
// before
use rig_mcp::McpToolset;
// after
use rig_rmcp::McpToolset;
```

Nothing else moves: the facade re-export is still `rig::tool::mcp`, and the
`mcp` feature (with `rmcp` as its legacy alias) is unchanged.

### Device-code prompts are data, not a callback

`chatgpt::auth::DeviceCodeHandler` and `copilot::auth::DeviceCodeHandler` —
each an `Option<Arc<dyn Fn(DeviceCodePrompt) + Send + Sync>>` — are replaced
by a concrete `DeviceCodePrompter` enum:

```rust
// before
let auth = Authenticator::new(source, token, api_key,
    DeviceCodeHandler::new(|p| my_ui.show(p.user_code)), true);

// after — receive the prompt as an owned event
let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
let auth = Authenticator::new(source, token, api_key,
    DeviceCodePrompter::Channel(tx), true);
tokio::spawn(async move {
    while let Some(p) = rx.recv().await { my_ui.show(p.user_code); }
});
```

`DeviceCodePrompter::Stdout` is the default and prints exactly what the
callback-less handler printed, so `DeviceCodeHandler::default()` becomes
`DeviceCodePrompter::default()` with no behaviour change.
`DeviceCodePrompter::Silent` is new, for unattended services.

A full channel or dropped receiver is ignored rather than failing the sign-in,
matching the old contract where a misbehaving callback was the host's problem.

### Vector-store filters: the `SearchFilter` trait is gone

`SearchFilter` was a tagless-final constructor trait whose only job was to
make `Filter::interpret::<F>()` generic. Both are deleted. Each backend now
exposes its constructors as inherent methods plus a concrete `from_filter`.

The practical difference is that `SearchFilter::gt(...)` used to *infer* the
backend filter type from context. Name the type you want instead:

```rust
// before — the trait method inferred Neo4jSearchFilter from the request type
use rig_core::vector_store::request::SearchFilter;
let req = VectorSearchRequest::new(embedding, 5)
    .with_filter(SearchFilter::gt("node.year", 1990.into()));

// after — a native filter
use rig_neo4j::Neo4jSearchFilter;
let req = VectorSearchRequest::new(embedding, 5)
    .with_filter(Neo4jSearchFilter::gt("node.year", serde_json::json!(1990)));

// after — the portable filter, translated by the backend
use rig_core::vector_store::request::Filter;
let portable = Filter::gt("node.year", serde_json::json!(1990));
let native = Neo4jSearchFilter::from_filter(portable);
```

`VectorSearchRequest<F>` keeps its type parameter, so backend-native filters
with richer operators — `VectorizeFilter::in_values`,
`milvus::Filter::array_contains`, `PgSearchFilter::member`,
`LanceDBFilter::array_has_any` — still reach a query unchanged.
`Filter::interpret` is replaced by the backend's own
`from_filter(Filter<serde_json::Value>)`, which is fallible only where the
operand type can reject a JSON value (milvus, scylladb, surrealdb, qdrant).

### Document loaders: no more boxed iterators

`FileLoader`, `PdfFileLoader` and `EpubFileLoader` no longer hold a
`Box<dyn Iterator>`, and their `'a` lifetime parameter is gone.

- `FileLoader<'a, T>` becomes `FileLoader<I>`, generic over its iterator. It
  stays lazy. `read`/`read_with_path` are now one impl bounded on
  `loaders::Readable` (newly `pub`), rather than three item-specific impls.
- `PdfFileLoader<'a, T>` becomes `PdfFileLoader<T>` and
  `EpubFileLoader<'a, T, P>` becomes `EpubFileLoader<T, P>`.
- `loaders::IntoIter` (all three) is deleted; `IntoIterator` now yields the
  underlying iterator.

**Behavior change:** `PdfFileLoader` and `EpubFileLoader` are now **eager** —
each stage materialises a `Vec` before the next runs, so a whole corpus is
held in memory rather than streamed. `FileLoader` is unaffected. If you load
large PDF or EPUB corpora, chunk the glob.

### `EmbeddingConfig::ndims` replaces hardcoded widths

`openai::functions::EmbeddingConfig::ndims()` reports the vector width of the
configured model — an explicit `dimensions` override if set, else the model's
native width. Use it to size a vector-store index instead of restating a
literal:

```rust
// before
const EMBEDDING_DIMS: usize = 1536;
let store = SqliteVectorStore::with_distance_metric(conn, EMBEDDING_DIMS, metric).await?;

// after
let dims = embed_cfg.ndims().ok_or_else(|| anyhow!("unknown model width"))?;
let store = SqliteVectorStore::with_distance_metric(conn, dims, metric).await?;
```

`dimensions` is unchanged and stays opt-in: it is the *request* field, and
`text-embedding-ada-002` rejects it.

### Anthropic prompt caching is reachable again

`anthropic::functions::Config` gains `prompt_caching`, `automatic_caching` and
`automatic_caching_ttl`, with `with_prompt_caching()`,
`with_automatic_caching()` and `with_automatic_caching_1h()` mirroring the
deleted `CompletionModel` builders. 0.41 shipped the caching machinery with no
way to switch it on; requests were always built with caching off.

```rust
let cfg = anthropic::functions::Config::new(CLAUDE_SONNET_4_6)
    .with_automatic_caching();
```

Defaults are unchanged (all off), so existing request bodies are identical.

### Custom OpenAI-compatible providers: the trait stack is inverted

If you implemented `providers::openai::completion::OpenAICompatibleProvider`
for your own extension type, the five hook methods and the capability consts
are gone. The trait is now a four-item lookup shim, and *your dialect lives in
your own free functions*:

```rust
// before
impl OpenAICompatibleProvider for MyExt {
    const PROVIDER_NAME: &'static str = "mine";
    const SUPPORTS_RESPONSE_FORMAT: bool = false;
    type StreamingUsage = openai::Usage;
    type Response = openai::CompletionResponse;
    fn finalize_request_body(&self, body: &mut Value) -> Result<(), CompletionError> { … }
}

// after — the quirk is straight-line code you own
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("mine");
pub const STREAM_DIALECT: ChatCompletionsDialect =
    ChatCompletionsDialect::from_descriptor(&DESCRIPTOR);

pub fn build_body(
    model: &str,
    request: &CompletionRequest,
    options: CompletionModelOptions,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    let typed = compatible_typed_request(model, request, &DESCRIPTOR, options)?;
    let mut body = compatible_body_value(&typed, &DESCRIPTOR, stream)?;
    /* your former finalize_request_body body, here */
    Ok(serde_json::to_vec(&body)?)
}
```

Mapping of the deleted items:

| deleted | replacement |
| --- | --- |
| `PROVIDER_NAME` | `ProviderDescriptor::name` |
| `SUPPORTS_TOOLS` | `ProviderDescriptor::supports_tools` |
| `SUPPORTS_RESPONSE_FORMAT` | `ProviderDescriptor::supports_response_format` |
| `STREAM_INCLUDE_USAGE` | `ProviderDescriptor::stream_include_usage` |
| `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` | `ProviderDescriptor::emits_complete_single_chunk_tool_calls` |
| `type StreamingUsage` | `ChatCompletionsDialect::usage` (a `ChatCompletionsUsageDialect` arm) |
| `build_completion_request` | call `compatible_typed_request` (or your own conversion) inside `build_body` |
| `prepare_request` | straight-line code on the typed request inside `build_body` |
| `finalize_request_body(_with_options)` | straight-line code on the serialized body inside `build_body` |
| `decorate_streaming_tool_call` | a `ChatCompletionsDialect` flag plus a free function in your module |
| `completion_path` | still a trait method; the body forwards to your free function |

`AnthropicCompatibleProvider` shrank the same way: `PROVIDER_NAME` and
`default_max_tokens` are now the two fields of one plain-data
`anthropic::completion::AnthropicDialect`, supplied by a single
`const DIALECT`. On the data-oriented path there is nothing to supply at all —
`anthropic::functions::Config` already stores the resolved
`default_max_tokens` as a field.

### `openai::send_compatible_streaming_request` is gone

Chat-completions stream parsing no longer goes through a profile trait. The
public `send_compatible_streaming_request` free function, the
`CompatibleStreamProfile` trait, and `OpenAICompatibleProfile` are deleted.
Streaming is now: open the transport-edge event stream, then drive the shared
sans-IO state machine with a dialect.

```rust
// after (what every provider's `functions::open_stream` does)
let req = build_request(cfg, &request, true)?;
Ok(compatible_open_stream(rt, req, STREAM_DIALECT))
```

If you drove the shared machinery yourself, box your own event source with
`http_client::sse::boxed_event_source(client, req, allow_missing_content_type)`.
Provider stream drivers are no longer generic over `HttpClientExt`; they take
`http_client::sse::BoxedEventSource`.

### `Agent<M>` is now `Agent`

Every classic runtime type lost its `M: CompletionModel` parameter:
`Agent<M>` → `Agent`, `AgentBuilder<M>` → `AgentBuilder`,
`AgentRunner<M>` → `AgentRunner`, `PromptRequest<S, M>` → `PromptRequest<S>`,
`StreamingPromptRequest<M>` → `StreamingPromptRequest`,
`Extractor<M, T>` → `Extractor<T>`, `ExtractorBuilder<M, T>` →
`ExtractorBuilder<T>`, and the `StreamingPrompt<M>`/`StreamingChat<M>` traits
are now plain `StreamingPrompt`/`StreamingChat`.

If you only ever wrote `client.agent(model)…`, delete the type annotations
and you are done — construction, the builder surface, hooks, memory, and the
tool server are unchanged:

```rust
// before
let agent: Agent<openai::responses_api::ResponsesCompletionModel> =
    openai.agent(openai::GPT_5_2).preamble("…").build();
// after
let agent: Agent = openai.agent(openai::GPT_5_2).preamble("…").build();
```

### Agents hold a `ProviderConfig`, not a model

Internally an `Agent` now stores a `rig::provider::ProviderConfig` (a serde
enum with one arm per bundled provider) plus an `Arc<rig::provider::Runtime>`
(the process-local transport handles). `client.agent(model)` still works: the
new `rig::client::ToProviderConfig` trait (in the prelude) captures the
client's connection details — base URL, headers, API-key placement — as plain
configuration. You can also skip clients entirely:

```rust
use rig::agent::AgentBuilder;
use rig::provider::ProviderConfig;
let provider = ProviderConfig::OpenAiResponses(
    rig::providers::openai::responses_api::functions::Config::new("gpt-5.2"),
);
let agent = AgentBuilder::new(provider).preamble("…").build();
```

### `AgentBuilder::new` takes a `ProviderConfig`

`AgentBuilder::new(model)` became `AgentBuilder::new(provider_config)`.
Migrate `client.completion_model(m)` at agent-construction sites to
`client.provider_config(m)` (from `ToProviderConfig`) or just
`client.agent(m)`. The builder gained `.runtime(Arc<Runtime>)` for sharing
one transport across agents (a fresh default `Runtime` is built otherwise).

### `AgentModelExt` is gone

`model.into_agent_builder()` was removed — a portable completion model no
longer identifies a provider. Use `client.agent(model)` or
`AgentBuilder::new(provider_config)`.

### Providers that cannot be plain configuration

`rig-candle` (in-memory model weights) and `rig-vertexai` (interactive OAuth)
cannot be captured as serde configs, so they no longer construct classic
agents. Drive the sans-IO protocol directly — `AgentRun::new(prompt)` +
`prepare_request(…)` + the provider's own completion call; see
`rig-vertexai/examples/tool_vertexai.rs` for the full loop. ChatGPT/Copilot
clients bridge only credentials that are already cached (non-interactively);
interactive OAuth flows still work through the classic clients themselves.

### Prompting is inherent methods; extraction is a free function (R4)

The `Prompt`, `Chat`, `TypedPrompt`, `StreamingPrompt`, and `StreamingChat`
traits are gone, along with the `PromptRequest`/`TypedPromptRequest`
typestate, `Extractor`/`ExtractorBuilder`, `stream_to_stdout`, and
`client.extractor::<T>()`. `Agent` carries the same operations as inherent
methods, so most call sites only lose an import:

```rust
// before
use rig::completion::{Chat, Prompt};
let answer = agent.prompt("hi").await?;
// after — delete the import; the call is unchanged
let answer = agent.prompt("hi").await?;
```

| Before | After |
| --- | --- |
| `agent.prompt(p).max_turns(3).await?` | `agent.runner(p).max_turns(3).run().await?.output` |
| `agent.prompt(p).extended_details().await?` | `agent.run(p).await?` (or `agent.runner(p)….run().await?`) |
| `agent.prompt_typed::<T>(p).max_turns(3).await?` | `agent.runner(p).max_turns(3).run_typed::<T>().await?` |
| `stream_to_stdout(&mut stream).await?` | drain the stream yourself (it was example sugar) |
| `ChatBotBuilder::new().agent(a)` | `ChatBotBuilder::new(a)` |

`agent.runner(prompt)` is now the fluent per-request surface (it already
carried every setter `PromptRequest` had).

**Extraction** moves to a free function with an options record:

```rust
use rig::extract::{ExtractOptions, extract_with_options};
let outcome = extract_with_options::<Person>(
    AgentConfig::new(), client.provider_config(MODEL), Arc::new(Runtime::new()),
    text, ExtractOptions::classic_extractor().with_retries(1),
).await?;
let person = outcome.value;          // was `.data`
```

`ExtractOptions::classic_extractor()` reproduces the old builder exactly
(output tool `submit`, the extraction preamble, `ToolChoice::Required`,
prompt-repeating retries), so recorded exchanges keep replaying;
`ExtractOptions::new()` is the leaner default. `ExtractionResponse<T>` is
now `ExtractionOutcome<T>`. Extraction has no hook stack — port hook-driven
extractors to `agent.runner(..).output_tool(..)`.

### One driver: the classic engine is deleted (single-architecture R5)

`AgentRunner`, `StreamingPromptRequest`, `MultiTurnStreamItem`,
`StreamingResult`, and `StreamingError` are gone, and with them the second
agent engine (`drive_agent`/`TurnSource`). `AgentSession` and `AgentStream`
are now the only drivers; `Agent` is a thin record over them, and the R1
`SessionAgent` merged into `Agent` (`rig::agent_api::SessionAgent` survives
only as a deprecated alias).

**Blocking code is unchanged.** `agent.prompt/run/chat/prompt_typed` and the
fluent `agent.runner(p)….run()` keep every method name and behavior — only
the runner's *type* was renamed:

| Before | After |
| --- | --- |
| `rig::AgentRunner` | `rig::SessionRunner` (`rig::agent::SessionRunner`) |
| `agent.runner(p).max_turns(3).run().await?` | unchanged |
| `agent.runner(p).run_typed::<T>().await?` | unchanged |

**Streaming call sites change.** The old surface was a future returning a
stream of `MultiTurnStreamItem`; the new one is a stream of
`AgentStreamItem` that must be pinned:

```rust
// before
let mut stream = agent.stream_prompt("hi").max_turns(3).await;
while let Some(item) = stream.next().await { … }

// after
let stream = agent.runner("hi").max_turns(3).stream_run();
futures::pin_mut!(stream);
while let Some(item) = stream.next().await { … }
```

With no per-request setters, `agent.stream_run("hi")` is the whole call.
`Agent::stream_prompt`/`stream_chat` now return the **host-driven**
[`AgentStream`] instead (pull items with `next_item`/`next_item_with_tools`
and answer the decision inboxes yourself) — use `stream_run()` for the
classic fire-and-forget behavior, in which hooks are dispatched and tools
executed for you.

Item mapping:

| `MultiTurnStreamItem` | `rig::stream::AgentStreamItem` |
| --- | --- |
| `StreamAssistantItem(x)` | `Assistant(x)` |
| `StreamUserItem(x)` | `User(x)` |
| `CompletionCall(c)` | `CompletionCall(c)` |
| `ToolExecutionCommitted { tool_call, internal_call_id }` | same fields |
| `ModelTurnRetried { turn }` | same field |
| `FinalResponse(r)` | `Final(r)` |

`AgentStreamItem` carries additional decision variants
(`BeforeModelCall`, `TurnFinished`, `InvalidToolCall`, `ToolCallPending`,
`ToolCallsReady`, `ToolResultReady`) that a `stream_run()` stream never
yields, so an exhaustive `match` needs a `_ => {}` arm. The stream's error
type is now `PromptError` directly: `StreamingError::Completion(e)` was
`PromptError::CompletionError(e)` and `StreamingError::Prompt(b)` was `*b`.

`PromptResponse`, `CompletionCall`, and the shared history/tool-result
helpers moved from the private `agent::prompt_request` module to
`rig::agent::response`; their public paths (`rig::agent::PromptResponse`,
`rig::agent::CompletionCall`) are unchanged.

### Hooks are records; memory is host-owned (single-architecture R3)

The `AgentHook` trait, `HookStack`, `HookContext`, `Scratchpad`, and
`StepEventKind` are gone. Hooks remain attach-and-forget, but a hook is now
a `HookEntry` record instead of a trait impl:

```rust
// before
struct Logger;
impl AgentHook for Logger {
    async fn on_completion_call(&self, ctx: &HookContext, e: CompletionCall<'_>)
        -> CompletionCallAction { ... }
}
agent_builder.add_hook(Logger)
// after — one entry, matching owned events
use rig::hooks::{HookDecision, HookEntry, HookEvent};
fn logger() -> HookEntry {
    HookEntry::new("logger", |event| Box::pin(async move {
        match event {
            HookEvent::BeforeModelCall { turn, prompt, .. } => { /* … */
                HookDecision::Continue }
            _ => HookDecision::Continue,
        }
    }))
}
agent_builder.add_hook(logger())
```

The decision vocabulary (`RequestPatch`, `ToolCallAction`,
`InvalidToolCallAction`, …) is unchanged and still lives at
`rig::agent::hook`. Three behavior notes: delta observers must opt in with
`HookEntry::new(..).observing_deltas()` or they never fire; run identity
(`ctx.run_id()`/`is_streaming()`/`agent_name()`) and the run-scoped
scratchpad are gone — capture your own state in the closure (`turn` is a
field on the events that have it); and tool-call argument rewrites chain as
`serde_json::Value` rather than JSON-encoded strings.

**Memory** is no longer wired into the agent. `ConversationMemory`,
`MessageFilter`, `DemotionHook`, and `Compactor` are deleted along with
`AgentBuilder::memory`/`conversation` and `AgentRunner::without_memory`.
Load before the run and append after it:

```rust
let memory = InMemoryConversationMemory::new();
let history = memory.load(conversation_id)?;          // fatal if it fails
let response = agent.runner(prompt).history(history).run().await?;
if let Err(error) = memory.append(conversation_id, response.messages.clone()) {
    tracing::warn!(%error, "memory append failed");   // warn and proceed
}
```

`InMemoryConversationMemory` is a concrete store with plain synchronous
`load`/`append`/`clear`. `rig-memory`'s policies are now enums
(`MemoryPolicy`, `TokenCounter`, `Compactor`) over one concrete
`PolicyMemory`, whose `append` returns an owned
`AppendOutcome { stored, demoted, compaction }` for the host to act on —
replacing the demotion-hook and compactor callbacks.

### The tool system is records, not traits (single-architecture R2)

`ToolContext`, `ToolSet`, `ToolServer`, `DynamicTool`, `ToolEmbedding`, and
`Agent::into_tool` are gone. `rig::tool::Tool` is now an alias for the
portable contract:

```rust
// before
impl Tool for Adder {
    async fn call(&self, _context: &mut ToolContext, args: Args) -> ... {}
}
// after — drop the context parameter; move state into the struct
impl Tool for Adder {
    async fn call(&self, args: Args) -> ... {}
}
```

Dynamic tools are `PortableDynamicTool::new(name, desc, params, |args| async
{...})`; tool collections are `rig::executor::ToolExecutor::new()
.register(...)`; `#[rig_tool]` functions lose their `&mut ToolContext`
parameter (a targeted compile error guides you) and gain a `.portable()`
record constructor; custom `Serialize` outputs need a one-line
`impl IntoToolOutput` via `serialize_to_tool_output` (the `Any`-based
blanket impl is gone). Sub-agents-as-tools are a `PortableDynamicTool`
closing over an inner agent. MCP moved to `rig::tool::mcp` (`McpToolset`,
host-polled `refresh()` instead of push updates); the `rmcp` cargo feature
now aliases `mcp`.

### Mocking moved to `MockScript`

`rig_core::test_utils::MockCompletionModel` no longer plugs into agents. Use
the scripted mock provider (`test-utils` feature):

```rust
use rig::provider::{MockScript, ProviderConfig};
let script = MockScript::from_responses(vec![/* CompletionResponse per turn */]);
let agent = AgentBuilder::new(ProviderConfig::Mock(script.clone())).build();
// `script` (clone shares the cursor) exposes .calls() and .requests().
```

Custom HTTP transports injected via `.http_client(...)` do not survive the
bridge; inject them at the runtime instead:
`AgentBuilder::new(provider).runtime(Arc::new(Runtime::with_http(HttpRuntime::recording(client))))`.

### Concrete completion and streaming payloads

`CompletionResponse<T>` is now the concrete `CompletionResponse` and the
provider-typed streaming final is the normalized `StreamFinal`. Consequences:

- **`raw_response` is gone.** If you need wire-typed provider data, call the
  provider's own conversion on the raw HTTP payload instead of reading it off
  the Rig response.
- **`FinishReason` is normalized.** Every provider maps its stop/finish
  vocabulary onto one shared `FinishReason` enum
  (`Stop`/`Length`/`ToolCalls`/`ContentFilter`/`Other`), and responses carry
  `provider` and `model` metadata.
- **`GetTokenUsage` is removed.** Usage is a plain field on
  `CompletionResponse` and `StreamFinal`; read `.usage` directly.
- The streaming vocabulary lost its type parameters:
  `RawStreamingChoice`/`StreamingResult`/`StreamingCompletionResponse`/
  `StreamedAssistantContent`/`MultiTurnStreamItem` are all concrete, and
  `CompletionModel` no longer has `type Response`/`type StreamingResponse`.
  Provider-specific streaming-final aliases (deepseek, groq, mistral,
  openrouter, gemini, copilot) are removed with nothing to replace them.

### `CompletionRequestBuilder` is gone

`CompletionRequestBuilder` and `model.completion_request(...)` were removed.
`CompletionRequest` is plain data: build it with the
`CompletionRequest::with_history` / `CompletionRequest::from_prompt`
constructors plus struct-update syntax for everything else.

```rust
// before
let request = model
    .completion_request("Who are you?")
    .preamble("You are a concise assistant.")
    .temperature(0.5)
    .build();

// after
let request = CompletionRequest {
    temperature: Some(0.5),
    ..CompletionRequest::with_history(
        Some("You are a concise assistant."),
        Vec::new(),          // prior history
        "Who are you?",
    )
};
let response = model.completion(request).await?;
```

### Modalities are free functions

Embedding, transcription, image-generation, audio-generation, and rerank calls
are extracted to per-provider free functions over plain configs: each provider
exposes a `functions` module (e.g.
`rig::providers::openai::functions::EmbeddingConfig` with free
`embed`/`embed_batches` in `rig::provider`). The classic
`EmbeddingsBuilder`/`EmbeddingModel` client surface still works and now routes
through the same functions.

### Vector stores: the shared traits are deleted

`VectorStoreIndex`, `VectorStoreIndexDyn`, and `InsertDocuments` are removed,
and **no shared trait replaces them**. Each store crate now exposes concrete
inherent async methods — `top_n`, `top_n_ids`, `top_n_as::<T>`, `insert`,
`insert_as::<T>` — over a *pre-embedded* vocabulary in
`rig::vector_store`:

- `VectorSearchRequest` now carries query **embeddings**
  (`OneOrMany<Embedding>`), not text: you embed first, then search.
- Results are `SearchHit { id, score, payload }`; inserts take
  `StoreRecord { id, payload, embeddings }`.
- Backend-specific filter types stay per store; score direction and id
  handling are store-defined (see each store's docs).

Store constructors no longer take an `EmbeddingModel` parameter. See
`examples/custom_vector_store` for the canonical pre-embedded pattern.

### `dynamic_context` and `retrieved_tools` are removed (again)

`AgentBuilder::dynamic_context`, `ExtractorBuilder::dynamic_context`, and
`AgentBuilder::retrieved_tools` — restored in 0.41 — are gone for good along
with the shared store traits they depended on. Passive RAG is now an explicit
hook recipe: implement `AgentHook::on_completion_call`, embed the prompt,
query your concrete store's `top_n`, and inject the hits with
`RequestPatch::extra_context`. A complete, copy-pasteable hook lives in the
`rig::agent` module docs (the "Passive RAG agent example") and in
`examples/rag`. Dynamic tool retrieval is per-turn
`RequestPatch::active_tools`.

---

## 0.40 → 0.41

### 1. The crate split

The monolithic core is now a portable contracts crate (`rig-core`) plus the
classic agent runtime (`rig-agent`), presented behind the `rig` facade.

**If you depend on the `rig` facade — almost nothing changes.** These keep
working unchanged:

```rust
use rig::prelude::*;
use rig::tool::{Tool, ToolContext};   // still the classic contextual trait
use rig::completion::Prompt;
```

**If you depend on `rig-core` directly**, it is now portable-only. Agent
construction — `AgentBuilder`, `ExtractorBuilder`, contextual tools, hooks, the
run loop — moved to `rig-agent`. Depend on `rig-agent` or the facade.

**If you depend on `rig-agent` directly**, its root no longer re-exports all of
`rig-core`. The previous `pub use rig_core::*;` made it an implicit second
facade. Reach portable items through the explicit namespace:

```rust
use rig_agent::core::OneOrMany;   // was: rig_agent::OneOrMany
```

The `impl From<rmcp::model::Tool> for ToolDefinition` and its `&`-borrow variant
were removed — with `ToolDefinition` in `rig-core` and `rmcp::model::Tool`
foreign, the orphan rule forbids the impl in `rig-agent`. Normal MCP use is
unchanged; if you relied on the direct conversion, build the `ToolDefinition`
from the tool's `name`, `description`, and `schema_as_json_value()`.

### 2. Client construction needs traits in scope

Provider clients no longer carry inherent `.agent()` / `.extractor()` methods.
There is one canonical `CompletionClient` (in `rig-core`, providing
`completion_model`); the classic constructors live on a new `AgentClientExt`.

```rust
use rig::prelude::*;                  // brings both into scope

let agent     = client.agent(model).build();          // AgentClientExt
let extractor = client.extractor::<T>(model).build(); // AgentClientExt
let m         = client.completion_model(model);       // CompletionClient
```

Or import explicitly: `use rig::client::{CompletionClient, AgentClientExt};`

### 3. The portable tool contract is `PortableTool`

The context-free tool contract is now `PortableTool` (with
`PortableToolEmbedding`, `PortableDynamicTool`, `portable_tool_definition`).
The `rig_core::tool::Tool` alias is removed.

On the facade, `rig::tool::Tool` remains the classic *contextual* trait, so
existing facade code is unchanged. Portable contracts are available as
`rig::tool::PortableTool`, and in full under `rig::tool::portable`. A
`PortableTool` still registers with the classic runtime — it blanket-implements
the contextual `Tool`.

### 4. Tool authoring and dispatch, reworked

The largest change in this release. Typed tools now implement only:

```rust
Tool::call(&mut ToolContext, Args) -> Result<Output, Error>
```

Author-facing errors stay typed until private runtime erasure normalizes them
into `ToolExecutionError`.

**Tool implementations.** Keep one typed `type Error` for ordinary `?`
propagation. Remove `classify_error`, `call_with_extensions`, and
`call_structured`. The optional `map_error` classifies domain failures at the
erased boundary; its default preserves the source as `Other`. Return refusals
through `map_error` with `ToolExecutionError::refused`. Attach host-only result
metadata with `ToolContext::insert_result`.

**Context.** `ToolCallExtensions` and `ToolResultExtensions` are replaced by
`ToolContext`; `.tool_extensions(...)` becomes `.tool_context(...)`.

**Dynamic tools.** `ToolDyn` is removed from the public API — use
`DynamicTool`. Rig's erased dispatch trait is now private. Typed tools use
`Tool::NAME` as their sole identity; runtime-named agents convert explicitly
with `Agent::into_tool()`.

**Registration:**

| Before | After |
| --- | --- |
| `AgentBuilder::tools(Vec<Box<dyn ToolDyn>>)` | repeated `.tool(...)`, or `dynamic_tools(Vec<DynamicTool>)` |
| `dynamic_tools(sample, index, toolset)` | `retrieved_tools` *(removed again in 0.42 — see above)* |
| `ToolSetBuilder::dynamic_tool(ToolEmbedding)` | `retrieved_tool` |
| `ToolSetBuilder::dynamic_tool(...)` (callbacks) | `dynamic_tool(DynamicTool)` |

**Dispatch:**

| Before | After |
| --- | --- |
| `ToolSet::{call, call_with_extensions, call_structured}` | `ToolSet::execute` |
| `ToolServerHandle::{call_tool, call_tool_with_extensions, call_tool_structured}` | `ToolServerHandle::execute` |

**Errors.** Replace `ToolError`, `ToolFailure`, `ToolFailureKind`, `ToolReturn`,
`ToolReturnOutcome`, `ToolExecutionResult`, and `ToolOutcome` with
`ToolExecutionError`, `ToolErrorKind`, and the read-only `ToolResult` that hooks
observe. `ToolSetError` is removed.

Explicit `ToolExecutionError` constructors keep actionable diagnostics
model-visible. The generic `from_error` path preserves operator diagnostics and
the concrete source but defaults to safe kind-level model feedback — use
`with_model_feedback` for deliberate replacement text, or `with_model_output`
for JSON/multimodal feedback.

**Model presentation.** Serializable outputs convert once into canonical
`ToolOutput` content blocks. Strings stay literal text, explicit
`serde_json::Value` stays JSON, and multimodal tools use `ToolOutput::content` /
`ToolOutput::one` or return typed `ToolResultContent`. Rig never reparses
strings to infer rich content. Inspect with `as_text` / `as_json`, and decode
explicitly with `deserialize_json`.

**Registry.** `ToolSet` is the single ordered registry and records whether each
tool is always advertised or retrieval-only. `ToolSet::{get_tool_definitions,
documents}` are now synchronous and infallible, and `ToolServerHandle`
registration/removal no longer returns an artificial `Result`.

### 5. Hooks are event-specific and provider-independent

`AgentHook::on_event`, `StepEvent`, and `Flow` are replaced by event-specific
`AgentHook` methods with their own action types: `CompletionCallAction`,
`ToolCallAction`, `ToolResultAction`, `InvalidToolCallAction`,
`ObservationAction`. This makes invalid event/action combinations
unrepresentable.

`AgentHook`, `HookStack`, and the internal erased-hook interface no longer carry
a completion-model type parameter. `CompletionResponseEvent` and
`StreamResponseFinish` expose canonical Rig content, usage, prompt, and message
ID fields instead of typed provider responses. Direct `CompletionModel`
completion and streaming APIs still return typed raw provider responses.

Invalid-tool hooks return `None` to defer; every explicit action, including
`Fail`, is terminal for that hook stack. The atomically surfaced post-batch
streaming event is named `ToolExecutionCommitted` — for live host lifecycle
events, observe `on_tool_call` / `on_tool_result`.

### 6. `AgentRunner` is the only execution path

The raw `Completion` and `StreamingCompletion` traits and their `Agent`
implementations are removed; agent execution state is private.

```rust
// before
agent.completion(prompt, history).await?.send().await?;
agent.stream_completion(prompt, history).await?.stream().await?;

// after — pick a turn budget large enough for tool follow-ups
agent.runner(prompt).history(history).max_turns(3).run().await?;
agent.runner(prompt).history(history).max_turns(3).stream().await;
```

The runner consumes tool calls rather than returning the first raw model
response. For intentionally hook-free transport, start from
`model.completion_request(prompt).messages(history)` then `.send()` or
`.stream()`. *(In 0.42 the request builder is removed — use the
`CompletionRequest::with_history`/`from_prompt` constructors instead; see
above.)*

`AgentRun::new(prompt).with_history(history)` remains a sans-I/O state machine
for custom drivers. It holds no configured model, tools, memory, or hooks and is
not an alternate execution path for configured agents.

An `Agent`'s model is fixed and private. Former per-call `.model(...)` /
`.model_opt(...)` users should retain the provider `CompletionModel` and use its
raw request API, or construct a separate `Agent`.

`Extractor` now routes through the full hook lifecycle.

### 7. `dynamic_context` is back, but it is a hook now

*(Removed again in 0.42 — see the 0.41 → 0.42 section above for the
replacement hook recipe. The notes below describe the 0.41 behavior.)*

`AgentBuilder::dynamic_context` and `ExtractorBuilder::dynamic_context` were
removed in #2174 and **restored in #2219**. If you are tracking `main`, you may
have seen the gap; if you are upgrading from a release, the call still exists
and **your `.dynamic_context(samples, index)` calls need no change**.

What changed underneath: the separate retrieval pipeline in agent request
construction is gone for good, and the internal `DynamicContextStore` with it.
The helper is now a thin wrapper over a private `AgentHook` on the ordinary
completion-call lifecycle. Behavior that is deliberately preserved: retrieval on
every model call, current-prompt query selection with latest-textual-history
fallback, sample-count forwarding, pretty-JSON document formatting,
static-context-before-retrieved-context ordering, failure raised before provider
I/O, and support across blocking, streaming, and extractor execution.

Two consequences of it being an ordinary hook are worth checking:

- **Registration order matters.** Retrieval and the injected documents now
  follow hook registration order relative to your own hooks. Register a stop
  policy *before* `dynamic_context` if it should be able to suppress retrieval —
  previously the side pipeline ran regardless.
- **Multiple registrations run sequentially.** Several `dynamic_context` calls
  now execute in order through `HookStack` rather than concurrently through the
  former side pipeline. If you registered several against independent indexes
  and depended on the concurrency, expect added latency.

If you want control beyond that — filtering, reranking, caching, per-turn policy
— write your own `AgentHook`; that is the only passive-RAG execution path now,
and `dynamic_context` is simply a prepackaged one.

### 8. `#[rig_tool]` required-ness follows the parameter types

Required-ness now has one source of truth, and the advertised schema always
agrees with the deserializer. Previously a parameter left out of an explicit
`required(...)` was advertised optional while the generated deserializer still
demanded it — failing at runtime whenever the model legitimately omitted it.

- **No `required(...)`:** non-`Option` parameters are required; `Option<T>` is
  optional and deserializes to `None` when absent. Previously `Option`
  parameters were *advertised* as required. If a provider needs everything
  marked required, list them explicitly.
- **Explicit `required(...)`:** listed parameters are required. Omitted ones get
  `#[serde(default)]`, so their types must be `Option<T>` or implement
  `Default` — a type that is neither is now a **compile error** instead of a
  runtime deserialization failure.
- Listing an `Option<T>` in `required(...)` is a compile error: schemars
  excludes `Option` fields from `required` and serde deserializes a missing
  `Option` to `None`, so the directive would be silently ignored on both sides.
- Names in `params(...)` and `required(...)` must match actual parameters.
  Malformed or duplicate entries are compile errors rather than silently
  ignored.
- A wildcard context binding (`#[rig(context)] _: &mut ToolContext`) is
  rejected — name it `_context`.

Two dependency improvements need no action but may let you tidy up: crates using
`#[rig_tool]` / `#[derive(Embed)]` no longer need direct `serde` or `serde_json`
dependencies, the `Embed` trait no longer needs importing where it is derived,
and fully qualified `&mut rig::tool::ToolContext` parameters are recognized
without `#[rig(context)]` even under renamed dependencies.

### 9. Core errors are `#[non_exhaustive]`

`PromptError`, `StructuredOutputError`, and `VectorStoreError` are now
`#[non_exhaustive]`. Downstream `match` expressions need a wildcard arm.

Conversation memory load failures surface as the typed
`PromptError::MemoryError` instead of `CompletionError::RequestError`.

### 10. Every wasm feature flag is gone

`rig-core`'s `wasm`, `rig-agent`'s `wasm`, and the facade's `wasm` are all
removed. **Building for `wasm32-unknown-unknown` is the entire opt-in** — there
are no wasm feature flags anywhere in the workspace.

Drop `features = ["wasm"]` from any dependency line. Nothing replaces it; Cargo
rejects the unknown feature at resolution, so this fails loudly.

Relaxing the bounds cannot break *implementors* — the relaxed markers are
blanket-implemented (`impl<T> WasmCompatSend for T {}`), so every type that
satisfied the strict form satisfies the relaxed one. The one exception is a
generic *consumer* on browser wasm that wrote `T: WasmCompatSend` and then
relied on `T: Send` internally, and only if it was previously building with the
feature off.

**`rmcp` is native-only.** It never compiled for wasm — rmcp's `ClientHandler`
requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot
satisfy — but it used to fail with a wall of `dyn ErasedTool` trait errors. It
now fails with a single explanatory `compile_error!`.

WASI (`wasm32-wasip1` / `wasip2`) is **not supported**; its dependency graph has
never built. See `crates/rig-agent/README.md` for the full target matrix.

---

## 0.39 → 0.40

### 1. `max_turns` — see [Silent behavior changes](#max_turns-counts-differently)

The single most likely change to alter your program's behavior without a
compile error.

### 2. `Tool` / `ToolDyn` metadata is flat

Tool authors implement `description()` and `parameters()` directly.
`Tool::definition(prompt)` and `ToolDyn::definition(prompt)` are removed.

`ToolDefinition` remains a provider/request artifact generated from registered
tools. `Tool::NAME` / `Tool::name()` / `ToolDyn::name()` are the single source of
truth for advertised and dispatched tool names.

### 3. Structured tool-execution results, and hook system v2

Two large additions that also broke existing surfaces (#2015, #2012). Hooks
became composable middleware; tool execution gained structured results. If you
implemented hooks or tools against 0.39, expect to rewrite against the new
shapes — and note that both were reworked *again* in 0.41 ([section
4](#4-tool-authoring-and-dispatch-reworked) and [section
5](#5-hooks-are-event-specific-and-provider-independent) above). If you are
jumping 0.39 → 0.41, migrate straight to the newer shape and skip this
intermediate form.

### 4. `PromptResponse` and `FinalResponse` are one type

Unified in #2056. `FinalResponse` and its accessors (`content`,
`assistant_content`, `completion_calls`) no longer exist as a separate type.

### 5. Providers consolidated onto `GenericCompletionModel<Ext>`

groq, deepseek, mistral, together, moonshot (OpenAI side), perplexity,
hyperbolic, mira, azure, huggingface, and llamafile all lose their hand-rolled
`CompletionModel` structs, request types, and `TryFrom<message::Message>`
conversions. `CompletionModel` in each module is now a **type alias** for the
generic model, and provider-specific `StreamingCompletionResponse` types are
replaced by the shared OpenAI one.

A new `OpenAICompatibleProvider` trait (mirroring `AnthropicCompatibleProvider`)
is now **required** by `GenericCompletionModel`'s `Ext` parameter. It carries the
telemetry provider name and an `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` flag for
llama.cpp-style streaming tool calls, plus `SUPPORTS_RESPONSE_FORMAT`,
`STREAM_INCLUDE_USAGE`, and `SUPPORTS_TOOLS` consts. Provider wire dialects live
in its `completion_path`, `prepare_request`, and `finalize_request_body` hooks.

Also in this area:

- `openai::ToolChoice` gains a `Function { name }` variant and is now
  `#[non_exhaustive]`.
- `openai::CompletionRequest` fields are now public; `OpenAIRequestParams` gains
  `supports_response_format`.
- `GenericCompletionModel`'s `strict_tools` / `tool_result_array_content` fields
  are private — use the `with_*` builder methods. The redundant `with_model`
  constructor is removed; use `new`.
- `StreamingCompletionResponse` is generic over the provider's streaming usage
  payload (`StreamingCompletionResponse<U = Usage>`).
- perplexity, hyperbolic, and mira set `SUPPORTS_TOOLS = false`;
  `tools`/`tool_choice` are dropped with a warning during request conversion.

### 6. OpenRouter de-forked

`openrouter::{Message, UserContent, ImageUrl}` are re-exports of the shared
OpenAI types. The fork's `FileContent` / `VideoUrlContent` are replaced by shared
`FileData` / `VideoUrl`. `ReasoningDetails` / `ResponseImage` move into the
openai module (re-exported from openrouter).

The shared OpenAI types gained OpenRouter's extensions to support this:
`UserContent::Video`, `ImageUrl.detail` becomes `Option<ImageDetail>`, and
`Message::Assistant` gains `reasoning_details`, an inbound-only `images` field,
and a deserialize-only `role: "model"` alias.

Message conversion goes through `openrouter::messages_from_rig_message`.
OpenRouter's `UserContent` builder helpers (`image_url`, `file_base64`,
`video_url`, …) are removed — construct the shared `openai` content variants
directly. `ToolChoice::Specific` with multiple function names now errors
client-side.

### 7. Error types gained a `ProviderResponse` variant (#1944)

> Not recorded in the CHANGELOG. Found by diffing the public API.

Every provider HTTP-error path is routed through a shared
`from_http_response` / `from_provider_body` funnel, so `provider_response_*`
helpers recover the raw status and body instead of flattening into
`ProviderError(String)`.

Practical impact: **`RerankError` becomes `#[non_exhaustive]` and gains a
`ProviderResponse` variant**, and public `from_http_response` /
`from_provider_body` constructors are added on every capability error. An
exhaustive `match` on `RerankError` will no longer compile — add a wildcard arm.

### 8. `Output::Unknown` carries a payload (#1950)

> Only its sibling PR (#1951) is in the CHANGELOG.

In the OpenAI Responses API, `Output::Unknown` was a fieldless variant, so every
unrecognized output item decoded to a unit and its payload was discarded.
Provider-native hosted tools (`web_search_call`, `file_search_call`,
`computer_use_call`, `code_interpreter_call`) arrive as exactly these items, so
their data was destroyed at the typed-decode boundary.

`Output::Unknown` is now `Output::Unknown(serde_json::Value)`. Any `match` arm
binding that variant needs updating — and the data you were previously losing is
now available.

### 9. Removed outright

| Removed | Replacement |
| --- | --- |
| `providers::galadriel` (whole integration) | none |
| `evals` module + `experimental` feature | none |
| experimental `pipeline` module | none |
| `rig_derive::ProviderClient` derive | none |
| `Extractor::{get_inner, into_inner}` | none |
| `TryFrom<String> for Nothing` (always failed) | none |
| `streaming::stream_completion_to_stdout` | `agent::stream_to_stdout` |
| `AudioGeneration<M>` / `ImageGeneration<M>` / `Transcription<M>` wrapper traits | the corresponding `*Model` APIs and request builders |
| `providers::anthropic::decoders` | shared SSE machinery |
| `SpanCombinator::record_model_output` | none |
| `together::ToolChoice`, `together::ToolChoiceFunctionKind`, `moonshot::ToolChoice` | shared `openai::ToolChoice` |
| `groq::send_compatible_streaming_request`, `deepseek::send_compatible_streaming_request` | `openai::send_compatible_streaming_request` |
| raw response types of perplexity / hyperbolic / huggingface (`Message`, `Choice`, `Usage`, `Delta`, `Role`) | each module keeps a `CompletionResponse` alias to the shared OpenAI payload |

---

## 0.38 → 0.39

Only two breaking changes.

### 1. Sans-I/O `AgentRun` state machine (#1899)

Both agent loops became thin drivers over a sans-I/O `AgentRun` state machine.
If you drove the agent loop yourself rather than calling `prompt` / `chat`, you
are affected. Note that the surrounding execution API changed again in 0.41 —
see [`AgentRunner` is the only execution
path](#6-agentrunner-is-the-only-execution-path).

### 2. Deterministic, duplicate-safe tool registration (#1913)

Tool registration became order-deterministic and duplicate-safe, with `ToolSet`
backed by an `IndexMap`. Code relying on the previous registration order or on
duplicate-name behavior may see different tools advertised.

---

## 0.37 → 0.38

### 1. `#[rig_tool]` parameters must implement `JsonSchema`

The macro now derives `schemars::JsonSchema` on the generated parameters struct
(#1576), so **every parameter type must implement `JsonSchema`**. Primitives,
`String`, `Vec<T>`, and `Option<T>` are covered; your own types need
`#[derive(schemars::JsonSchema)]`. Previously an unknown type was quietly
advertised as `{"type": "object"}` with no fields, so this converts a class of
runtime tool-call failures into a compile error.

You do not need a direct `schemars` dependency — the macro refers to
`rig_core::schemars`.

For what the model now sees, see
[`#[rig_tool]` advertises a different schema](#rig_tool-advertises-a-different-schema).

### 2. Invalid tool calls are validated and recoverable

Tool calls are checked against the registered tools and the request's
`tool_choice` before dispatch (#1823). The new `PromptError::UnknownToolCall`
variant carries the offending `tool_name`, the `available_tools`, the
`allowed_tools`, and the `chat_history` at the point of failure. `PromptError`
is not `#[non_exhaustive]`, so an exhaustive `match` over it needs the new arm.

Recovery is a hook (#1840). `PromptHook` gains `on_invalid_tool_call`, taking an
`InvalidToolCallContext` and returning `InvalidToolCallHookAction`:

| Action | Effect |
| --- | --- |
| `Retry(feedback)` | re-prompt the model with your feedback text |
| `Repair(tool_name)` | rewrite the call to name a real tool |
| `Skip` | drop the call and continue |
| `Fail` | surface `PromptError::UnknownToolCall` |

Bound the loop with `.max_invalid_tool_call_retries(n)` on either prompt builder.

### 3. The streaming final response carries content, not a string

`MultiTurnStreamItem::final_response` and `final_response_with_history` take
`OneOrMany<AssistantContent>` where they took `&str`:

```rust
// before
MultiTurnStreamItem::final_response("some text", usage)

// after
MultiTurnStreamItem::final_response(OneOrMany::one(AssistantContent::text("some text")), usage)
```

`FinalResponse` keeps `response()` for the concatenated text and adds
`content()` / `assistant_content()` for the structured form. `MultiTurnStreamItem`
is `#[non_exhaustive]`, so its new `CompletionCall` variant does not break
existing matches.

### 4. Per-completion-call usage on responses

`PromptResponse` and `TypedPromptResponse` gain
`completion_calls: Vec<CompletionCall>` (#1787), one entry per model call in the
turn, each with a `call_index` and optional `Usage`. Streaming emits the same
data as `MultiTurnStreamItem::CompletionCall`. Aggregate `usage` is unchanged.

This is additive unless you construct those responses yourself — use
`.with_completion_calls(...)` if you do.

### 5. New enum variants and struct fields

None of these are `#[non_exhaustive]`, so exhaustive matches and struct literals
break:

| Type | Change |
| --- | --- |
| `RawStreamingChoice` | new `TextStart` and `TextAdditionalParams` variants |
| `message::Text` | new `additional_params` field — use `Text::new(text)` |
| `completion::Usage` | new `tool_use_prompt_tokens` field |
| `anthropic::Content` | new `ServerToolUse` and `WebSearchToolResult` variants |
| `anthropic::ContentDelta` | new `CitationsDelta` and `Unknown` variants |
| `gemini::FinishReason` | new `MalformedResponse`, `MissingThoughtSignature`, `TooManyToolCalls`, `UnexpectedToolCall` variants |

`openai::responses_api`'s `ArgsTextChunk.content_index` and
`DeltaTextChunkWithItemId.content_index` became `Option<u64>`, and
`DeltaTextChunkWithItemId` lost its `item_id` field (#1828).

### 6. Embedding token usage is additive

`EmbeddingModel` gains `embed_text_with_usage` / `embed_texts_with_usage`
returning `EmbeddingResponse { embeddings, usage }`, plus
`EmbeddingsBuilder::build_with_usage` (#1791). Both trait methods have default
implementations that delegate to the existing ones and report zero usage, so
custom embedding models keep compiling.

### 7. `AgentBuilder::hook` no longer resets tool state

`.hook(...)` returned `AgentBuilder<M, P2, NoToolConfig>`, discarding the
typestate that records how tools were registered. It now returns
`AgentBuilder<M, P2, ToolState>`. Builders that called `.hook(...)` after
`.tool(...)` and then hit a type error can drop the workaround.

### 8. 0.38.1 unified the workspace versions

Every crate in the workspace moved to the workspace version (#1853). Companion
crates that were on their own numbering — `rig-mongodb` was at `0.4.7`, for
example — jump to `0.38.1`. Nothing about their APIs changed; the version line
in your `Cargo.toml` does.

From 0.38.1 onward, a companion crate's version tracks the `rig-core` release it
was built against.

---

## 0.36 → 0.37

### 1. `rig-core`'s library is now `rig_core`

`rig-core` built a library target named `rig`, so the idiomatic code was
`cargo add rig-core` followed by `use rig::...`. In 0.37 the library target is
named `rig_core`, and the name `rig` belongs to a new facade crate (#1699).

**This breaks every `use rig::...` in a crate that depends on `rig-core`.** Two
ways out:

```toml
# preferred: depend on the facade, keep `use rig::...` unchanged
rig = "0.37"
```

```toml
# or: stay on rig-core and rewrite the imports to `use rig_core::...`
rig-core = "0.37"
```

The facade re-exports `rig-core` and puts every companion crate behind one
feature each (`mongodb`, `qdrant`, `lancedb`, `memory`, …), so a workspace that
was juggling `rig-core` plus several `rig-*` dependencies can collapse to one.

Related: `#[rig_tool]` and `#[derive(Embed)]` used to emit hardcoded `rig::`
paths, which is why depending on `rig-core` under any other name failed. They
now resolve `rig-core` or `rig` through `proc-macro-crate`, so both layouts work
and renamed dependencies do too.

### 2. `Chat::chat` takes `&mut Vec<Message>` and appends to it

```rust
// before — history passed by value, response only
fn chat<I, T>(&self, prompt: impl Into<Message>, chat_history: I) -> Result<String, PromptError>
where I: IntoIterator<Item = T>, T: Into<Message>;

// after — history borrowed and updated in place
fn chat(&self, prompt: impl Into<Message>, chat_history: &mut Vec<Message>)
    -> Result<String, PromptError>;
```

The prompt and every assistant and tool message produced during the turn are
appended to the vector you pass (#1733). **Do not push the user prompt yourself
before calling** — you will send it twice. Callers that were manually
reconstructing history after each turn should delete that code.

### 3. Conversation memory

A new `rig::memory` module (#1702) with the `ConversationMemory` trait,
`MemoryError`, a `MessageFilter` trait, and an `InMemoryConversationMemory`
backend. `AgentBuilder` gains `.memory(...)` and `.conversation_id(...)`; both
prompt builders gain `.conversation(id)` and `.without_memory()`.

`MemoryError` is `#[non_exhaustive]` from the start, and `load` failures are
fatal while `append` failures are logged and swallowed. Later releases added
`DemotionHook` (#1737) and `Compactor` (#1748) for eviction and rolling
summaries, with the named policies living in the `rig-memory` companion crate.

Entirely additive — agents without `.memory(...)` behave as before.

### 4. `ClientBuilder`'s HTTP client slot is a typestate

`ClientBuilder<Ext, ApiKey = Missing, H = Missing>` — the `H` parameter defaulted
to `reqwest::Client` and now defaults to `Missing`, with every provider's
`ClientBuilder` alias following. Building without supplying a client still
produces a `reqwest`-backed one, so ordinary `Client::builder().api_key(k).build()`
chains are unaffected. Code that named `H` explicitly needs updating.

### 5. Test doubles moved behind a feature

`MockStreamingClient`, `MockResponse`, and friends moved to
`rig_core::test_utils` behind the new `test-utils` feature (#1745). Add
`rig-core = { version = "0.37", features = ["test-utils"] }` to your
`[dev-dependencies]` if you used them.

`rig-core`'s `default` feature also picked up `derive`, so `#[rig_tool]` and
`#[derive(Embed)]` are available without opting in. The `all` feature was
removed; name `derive`, `pdf`, and `rayon` individually.

### 6. Smaller surface changes

- `DocumentSourceKind::FileId` and `anthropic::DocumentSource::File` support
  provider-side file IDs (#1740). `DocumentSourceKind` is `#[non_exhaustive]`,
  so matches are safe.
- `completion::Usage` gains `reasoning_tokens`, and gemini's `UsageMetadata`
  gains per-modality token detail — breaking for struct literals.
- Ollama's `think` additional-parameter accepts `"low"`/`"medium"`/`"high"` as
  well as a bool (#1747). Bools keep working.

---

## 0.35 → 0.36

The largest release in this range. Sections 1 and 3 touch every provider
integration.

### 1. `DynClientBuilder` and the `*Dyn` traits are gone

Deprecated since 0.25, removed in #1633. All of this no longer exists:

| Removed | Replacement |
| --- | --- |
| `DynClientBuilder`, `AnyClient`, `ProviderFactory`, `DefaultProviders` | construct the provider client directly |
| `CompletionClientDyn`, `EmbeddingsClientDyn`, `TranscriptionClientDyn`, `ImageGenerationClientDyn`, `AudioGenerationClientDyn`, `VerifyClientDyn` | the non-`Dyn` client traits |
| `CompletionModelDyn`, `EmbeddingModelDyn`, `TranscriptionModelDyn`, `ImageGenerationModelDyn`, `AudioGenerationModelDyn` | the corresponding `*Model` traits |
| `CompletionModelHandle`, `TranscriptionModelHandle`, `ImageGenerationModelHandle`, `AudioGenerationModelHandle` | the concrete model types |

If you were selecting a provider at runtime through `DynClientBuilder`, you now
own that dispatch — a `match` over your own provider enum returning a boxed
`Agent`, or an enum of concrete clients.

### 2. Request typestate builders are gone — plain constructors instead

The request builders that encoded required-ness in the type (`Missing` /
`Provided<T>`, #1611) are deleted. Each request struct now takes its required
fields in a constructor and its optional fields through `with_*` setters, the
same shape `CompletionRequest` already uses. `Provided<T>` is removed;
`markers::Missing` survives only inside `ClientBuilder`.

| Deleted builder | Replacement constructor |
| --- | --- |
| `VectorSearchRequestBuilder` | `VectorSearchRequest::new(query: OneOrMany<Embedding>, samples: u64)` |
| `TranscriptionRequestBuilder` | `TranscriptionRequest::new(data: Vec<u8>)` |
| `ImageGenerationRequestBuilder` | `ImageGenerationRequest::new(prompt: impl Into<String>)` |
| `AudioGenerationRequestBuilder` | `AudioGenerationRequest::new(text: impl Into<String>, voice: impl Into<String>)` |

The builder-returning trait methods `TranscriptionModel::transcription_request`,
`ImageGenerationModel::image_generation_request`, and
`AudioGenerationModel::audio_generation_request` are removed with them, as is the
builders' `send()`. Build the request, then call the model's
`transcription`/`image_generation`/`audio_generation` method (or the provider's
`functions::transcribe`/`generate_image`/`generate_audio` free function).

```rust
// before
let response = model
    .transcription_request()
    .data(bytes)
    .filename(Some("audio.mp3".to_string()))
    .temperature(0.5)
    .send()
    .await?;

// after
let request = TranscriptionRequest::new(bytes)
    .with_filename("audio.mp3")
    .with_temperature(0.5);
let response = model.transcription(request).await?;
```

```rust
// before
let req = VectorSearchRequest::builder()
    .query(query_embedding)
    .samples(10)
    .threshold(0.7)
    .build();

// after
let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 10)
    .with_threshold(0.7);
```

Renames to note: `queries(..)` is `with_queries(..)` (and `with_query(..)` takes
a single `Embedding`); `filter(..)` is `with_filter(..)`;
`additional_params(..)` is `with_additional_params(..)` and no longer returns a
`Result`, so drop the `?`; `TranscriptionRequestBuilder::load_file(path)` is
`TranscriptionRequest::from_file(path)`, still returning `io::Result`;
`additional_params_opt` is `with_additional_params_opt`.

`ChatBotBuilder` and `ClientBuilder` keep their typestate for now.

### 3. `ProviderClient` construction is fallible

```rust
// before
fn from_env() -> Self;              // panicked on a missing/invalid variable
fn from_val(input: Self::Input) -> Self;

// after
type Error;
fn from_env() -> Result<Self, Self::Error>;
fn from_val(input: Self::Input) -> Result<Self, Self::Error>;
```

Bundled providers use `ProviderClientError` (`EnvironmentVariable`, `Http`,
`InvalidConfiguration`) via the `ProviderClientResult<T>` alias, and
`required_env_var` / `optional_env_var` are available for your own
implementations. `llamafile::from_url` became fallible for the same reason.

Add `?` at every `Client::from_env()` call site — this is the most common
mechanical edit in this release.

### 4. Provider models moved onto `GenericCompletionModel`

`openai::CompletionModel`, `openai::ResponsesCompletionModel`,
`openai::EmbeddingModel`, and `anthropic::CompletionModel` became type aliases
over generic structs parameterized by a provider extension. The aliases keep
their old parameter lists and their fields stay public, so ordinary use is
unaffected. This is the groundwork that 0.40 extended to eleven more providers.

### 5. Smaller surface changes

- `DynamicContextStore` dropped its `RwLock` — it is now `Arc<Vec<...>>`
  (#1641). Immutable after construction, so tool lookups no longer contend.
- `cohere::CompletionResponse::message` returns
  `Result<AssistantMessageParts, CompletionError>` instead of a three-element
  tuple.
- Ollama's client builder takes an `OllamaApiKey` instead of `Nothing`, so a
  base URL and key can be set programmatically (#1511).
- `openai::responses_api::Output` gained an `Unknown` catch-all (#1552) —
  exhaustive matches need an arm. (0.40 later gave it a payload.)
- `deepseek::DEEPSEEK_CHAT` and `DEEPSEEK_REASONER` are `#[deprecated]` in favor
  of the `deepseek-v4-flash` names (#1664).
- `json_utils::empty_or_none` was removed.
- `#[rig_tool]` accepts `name = "..."`, validated against the
  1–64 character, ASCII-alphanumeric-plus `_`/`-` rule providers enforce (#1619).

---

## 0.34 → 0.35

### 1. Legacy Anthropic model constants removed

`CLAUDE_3_5_HAIKU`, `CLAUDE_3_5_SONNET`, `CLAUDE_3_7_SONNET`, `CLAUDE_4_OPUS`,
and `CLAUDE_4_SONNET` are gone (#1616), replaced by `CLAUDE_OPUS_4_6`,
`CLAUDE_SONNET_4_6`, and `CLAUDE_HAIKU_4_5`. Model ids are plain strings — pass
the literal (`"claude-3-5-sonnet-latest"`) if you need a retired model.

### 2. `ToolServer` internals are private

`ToolServerRequest`, `ToolServerResponse`, `ToolServerRequestMessageKind`,
`ToolServer::run`, `ToolServer::handle_message`, and
`ToolServerHandle::get_tool_definitions` left the public API, along with the
`ToolServerError::{Canceled, InvalidMessage, SendError}` variants (#1607). The
lock-free rework behind them removed contention during tool lookup.

`ToolServerHandle` remains the supported interface. If you were driving the
message enum directly, move to the handle.

### 3. Provider examples became integration tests

Roughly 60% of `rig-core`'s examples moved under `tests/`, organized by provider
(#1603), so `cargo run --example <name>` stopped resolving for those. They run
as ignored tests instead:

```
cargo test -p rig-core --test <provider> -- --ignored --test-threads=1
```

`--test-threads=1` matters: the provider suites share rate limits.

### 4. rmcp 1.3

The rmcp integration was upgraded from 0.x to 1.3 (#1596). If you pass
`rmcp::model::Tool` or `rmcp::service::ServerSink` values into
`AgentBuilder::rmcp_tool(s)`, your own rmcp dependency has to move in lockstep.

### 5. `response_format` is deferred on tool turns

When a request carries tools that have not produced a result yet, the
`response_format` derived from `output_schema` is withheld until the tool round
trip completes (#1622). Structured output still applies to the final answer;
providers stop rejecting the intermediate tool turns.

---

## 0.33 → 0.34

### 1. History is generic and immutable

`with_history` no longer borrows a `&'a mut Vec<Message>` (#1563):

```rust
// before
agent.prompt("hi").with_history(&mut history).await?;

// after — anything that iterates into messages
agent.prompt("hi").with_history(history.clone()).await?;
```

`PromptRequest` and `TypedPromptRequest` lost their `'a` lifetime parameter as a
result — `PromptRequest<'a, S, M, P>` is now `PromptRequest<S, M, P>`. The
streaming builder's `with_history` changed the same way, and
`CompletionRequestBuilder::{documents, messages}` now take
`impl IntoIterator<...>`.

Because the history is no longer borrowed mutably, the updated conversation
comes back on the response rather than being written into your vector. Read
`PromptResponse::messages` (see [0.31 → 0.32](#031--032)).

### 2. TLS features were restructured

| Before | After |
| --- | --- |
| `reqwest-rustls` | `reqwest` + `rustls` |
| `reqwest-native-tls` | `reqwest` + `native-tls` |
| — | `websocket`, `reqwest-middleware-native-tls` |

`default` is `["reqwest", "rustls"]`. The old composite names are gone, so a
dependency line naming them fails to resolve — which is the loud failure you
want here.

### 3. Anthropic automatic prompt caching

`CompletionModel::with_automatic_caching()` and `with_automatic_caching_1h()`
add a top-level `cache_control` to the request and let the API place the
breakpoint (#1572). The existing `with_prompt_caching()` — explicit breakpoints
on the system prompt and messages — is unchanged. `CacheTtl::{FiveMinutes, OneHour}`
and `Usage::cache_creation_input_tokens` came along with it.

### 4. Custom `Authorization` headers win

A header set through `http_headers()` is no longer overwritten by the provider's
generated auth header (#1553). This is what lets OpenAI-compatible endpoints
that want a non-`Bearer` scheme work. If you were setting an `Authorization`
header expecting the provider's key to take precedence anyway, it no longer
does.

---

## 0.32 → 0.33

### 1. `Message::System` and provider-native tools

`Message` gained a `System` variant (#1527) — it is not `#[non_exhaustive]`, so
exhaustive matches need the arm. See
[Preamble is now a system message](#preamble-is-now-a-system-message-and-completionrequestpreamble-is-empty)
for the behavioral half of this change.

Separately, `ProviderToolDefinition` and
`CompletionRequestBuilder::{provider_tool, provider_tools}` expose hosted tools
that run on the provider's side — OpenAI's `web_search`, `file_search`,
`computer_use`, and `code_interpreter` (#1430).

### 2. `McpTool` was replaced by the rmcp handler

`tool::McpTool`, `tool::McpToolError`, and `McpTool::from_mcp_server` were
removed in favor of `McpClientHandler` and the `rmcp_tool(s)` builder methods
(#1525).

### 3. Raw embedding payloads use `serde_json::Number`

Provider embedding response types — cohere, gemini, mistral, openai, openrouter,
together — changed their vector fields from `Vec<f64>` to
`Vec<serde_json::Number>` so they deserialize under
`serde_json/arbitrary_precision` (#1518, #1526). The `embeddings::Embedding`
type you actually consume still holds `Vec<f64>`; only the raw provider structs
changed. Call `.as_f64()` if you were reading them directly.

### 4. Gemini embedding dimensions come from the model

`gemini::EmbeddingModel::{new, with_model}` take `ndims: usize` instead of
`Option<usize>` (#1513). `ndims()` used to return a hardcoded `768` for every
model while `output_dimensionality` went to the API as `null`; it now reports
the value you passed, or the model's documented default — 3072 for
`gemini-embedding-001`, 768 for `text-embedding-004`.

**If you sized a vector-store column from `ndims()`, the number changes.** For
`gemini-embedding-001` the vectors were already 3072-wide; the reported count
was simply wrong.

### 5. Other provider changes

- `GenerateContentRequest.tools` is `Option<Vec<Value>>`, and
  `ThinkingConfig.thinking_budget` is `Option<u32>` alongside the new
  `thinking_level: Option<ThinkingLevel>` for Gemini 3 (#1520).
- `gemini::Client::{generate_content_api, interactions_api}` select between the
  two Gemini surfaces (#1230).
- `openai::Client::responses_websocket` opens a stateful Responses session
  behind the `websocket` feature (#1500).
- New `llamafile` provider (#1519).

---

## 0.31 → 0.32

### 1. `PromptResponse::total_usage` is `usage`

Renamed in #1453, and the response gained `messages: Option<Vec<Message>>` —
the full conversation for the turn, populated with `.extended_details()`
(#1450). `PromptResponse::new(output, usage)` keeps its shape.

### 2. Prompt requests carry a response-detail typestate

`TypedPromptRequest<'a, T, M, P>` became `TypedPromptRequest<'a, T, S, M, P>`,
where `S: PromptType` records whether `.extended_details()` was called (#1446).
`TypedPromptResponse<T>` is the extended form, with `output`, `usage`, and later
`completion_calls`. Chained builder expressions are unaffected; explicit type
annotations need the extra parameter.

### 3. Custom providers: `ProviderBuilder` reshaped

```rust
// before
trait ProviderBuilder: Sized {
    type Output: Provider;
    // Provider::build did the work
}

// after
trait ProviderBuilder: Sized + Default + Clone {
    type Extension<H>;
    type ApiKey;
    fn build(self, ...) -> ...;
}
```

`Provider::build` was removed, `Client::builder()` returns
`ClientBuilder<Ext::Builder, NeedsApiKey, reqwest::Client>`, and `Client::new`
takes `<Ext::Builder as ProviderBuilder>::ApiKey` instead of a bare key type
(#1436). Only affects crates implementing their own provider.

### 4. The SSE event source was reified

`GenericEventSource` gained a retry-policy parameter
(`GenericEventSource<HttpClient, RequestBody, Retry = ExponentialBackoff>`) and
a `with_retry_policy` constructor (#1428). `ReadyState` and `ready_state()` were
removed, and `last_event_id()` returns `Option<&str>` rather than `&str`.

### 5. New media variants

`AudioMediaType` gained `M4A`, `PCM16`, and `PCM24`; `VideoMediaType` gained
`MOV` and `WEBM`; openrouter's `UserContent` gained `InputAudio` and `VideoUrl`
(#1413). None are `#[non_exhaustive]` — exhaustive matches need the new arms.

### 6. `rig-eternalai` and `rig-wasm` were archived

Both moved to `archived/` and out of the workspace (#1472). They are no longer
published or built. `rig-eternalai` had already stopped publishing in an earlier
release; 0.32 is where the source left the tree.

### 7. Extractors gained retrieval and usage

`Extractor::dynamic_context(sample, index)` mirrors the agent builder, and
`extract_with_usage` / `extract_with_chat_history_with_usage` return
`ExtractionResponse<T> { data, usage }` (#1447). Both additive.

---

## 0.30 → 0.31

### 1. `AgentBuilderSimple` is gone; `AgentBuilder` is a typestate

`AgentBuilder<M>` became `AgentBuilder<M, P = (), ToolState = NoToolConfig>`,
and the separate `AgentBuilderSimple` that `.tool()` used to return no longer
exists. The `ToolState` parameter is `NoToolConfig`, `WithBuilderTools`, or
`WithToolServerHandle`.

The practical effect is that **mixing builder-registered tools with a
`ToolServerHandle` is now a compile error.** In 0.30, `.tool()` converted the
builder into an `AgentBuilderSimple` that had no field for the handle — so
`.tool_server_handle(h).tool(t)` dropped `h` on the floor, along with any
`dynamic_context` and `dynamic_tools` configured up to that point. Pick one
registration style; the compiler now enforces it.

Chained expressions otherwise need no change; explicit `AgentBuilder<M>`
annotations do.

`Agent<M>` became `Agent<M, P = ()>`, which still resolves for existing code.

### 2. One `PromptHook` for blocking and streaming

`StreamingPromptHook` was removed and its methods folded into `PromptHook`
(#1352): `on_text_delta`, `on_tool_call_delta`, and
`on_stream_completion_response_finish` joined `on_completion_call`,
`on_completion_response`, `on_tool_call`, and `on_tool_result`. Every method has
a default, so a hook that only cares about one event stays small.

The types also moved from `agent::prompt_request` to
`agent::prompt_request::hooks`; `rig::agent::{PromptHook, HookAction,
ToolCallHookAction}` re-exports them either way.

Hooks can now be attached to the agent instead of the call (#1356):
`AgentBuilder::hook(h)` sets a default that every `prompt` and `stream_prompt`
picks up, and `StreamingPrompt`/`StreamingChat` gained a `type Hook: PromptHook<M>`
associated type — use `type Hook = ();` if you implement them and want none.

`PromptRequest::new` was replaced by `PromptRequest::from_agent(&agent, prompt)`.

### 3. Reasoning content is typed

`Reasoning { reasoning: Vec<String>, signature: Option<String> }` became
`Reasoning { content: Vec<ReasoningContent> }`, with variants `Text { text,
signature }`, `Summary(String)`, `Encrypted(String)`, and `Redacted { data }`
(#1395, #1396). This is what makes reasoning traces survive a round trip across
providers.

Constructors and accessors cover the common cases: `Reasoning::new`,
`new_with_signature`, `summaries`, `encrypted`, `redacted`, and
`display_text` / `first_text` / `first_signature` / `encrypted_content`.

Stored histories need attention — see
[Persisted `Reasoning` JSON no longer round-trips](#persisted-reasoning-json-no-longer-round-trips).

### 4. Structured outputs and per-request model override

`CompletionRequest` gained `output_schema: Option<schemars::Schema>` and
`model: Option<String>` (#1382, #1374), with
`AgentBuilder::{output_schema, output_schema_raw}`,
`CompletionRequestBuilder::{output_schema, output_schema_opt, model, model_opt}`,
the `TypedPrompt` trait, `TypedPromptRequest`, and `StructuredOutputError`.

Additive unless you build `CompletionRequest` with a struct literal, which the
new fields break — use the builder.

### 5. Model listing

A new `ModelLister` / `ModelListingClient` pair with `model::listing::{Model,
ModelList, ModelListingError}` (#1243). Providers that support it can enumerate
models at runtime.

**Breaking for custom providers:** `client::Capabilities` gained a
`type ModelListing: Capability;` associated type. Set it to the unsupported
marker if your provider cannot list models.

### 6. reqwest 0.13, rustls by default

The HTTP stack moved to reqwest 0.13 (#1218). Feature names moved with it:

| Before | After |
| --- | --- |
| `reqwest-tls` | `reqwest-native-tls` |
| `reqwest-rustls` (opt-in) | `reqwest-rustls` (**default**) |

`reqwest/macos-system-configuration` is no longer pulled in. If you pass a
`reqwest::Client` into a Rig client, it has to be a 0.13 one. For the runtime
consequences, see
[The default TLS backend is now rustls](#the-default-tls-backend-is-now-rustls).

### 7. Anthropic source types became enums

`ImageSource` and `DocumentSource` changed from structs to enums, and
`ImageSourceData` was removed:

```rust
// before — media_type and type carried even for URLs, where they mean nothing
ImageSource { data: ImageSourceData::Url(url), media_type, r#type: SourceType::URL }

// after
ImageSource::Url { url }
ImageSource::Base64 { data, media_type }
```

`DocumentSource` gained `Base64` and `Text` variants (`Url` followed in 0.32),
`PlainTextMediaType` was added, and `Content` gained `RedactedThinking` — the
enums model what Anthropic actually accepts for URL-backed images and plain-text
documents (#1403, #1377).

### 8. Streaming message ids

`RawStreamingChoice` gained a `MessageId` variant and
`StreamingCompletionResponse` a `message_id: Option<String>` field.
`RawStreamingChoice` is not `#[non_exhaustive]`, so an exhaustive match over it
needs the new arm.

---

## Appendix: symbol reference

Renamed or relocated items, for searching.

| Old | New | Version |
| --- | --- | --- |
| `rig_core::tool::Tool` (portable) | `rig_core::tool::PortableTool` | 0.41 |
| `rig_agent::<item>` (portable re-export) | `rig_agent::core::<item>` | 0.41 |
| `client.agent(...)` inherent method | `AgentClientExt::agent` (via `rig::prelude::*`) | 0.41 |
| `ToolCallExtensions` / `ToolResultExtensions` | `ToolContext` | 0.41 |
| `.tool_extensions(...)` | `.tool_context(...)` | 0.41 |
| `ToolDyn` (public) | `DynamicTool` | 0.41 |
| `ToolSet::{call, call_with_extensions, call_structured}` | `ToolSet::execute` | 0.41 |
| `ToolServerHandle::call_tool*` | `ToolServerHandle::execute` | 0.41 |
| `ToolError` / `ToolFailure` / `ToolReturn` / `ToolOutcome` / `ToolExecutionResult` | `ToolExecutionError` / `ToolErrorKind` / `ToolResult` | 0.41 |
| `AgentHook::on_event` + `StepEvent` + `Flow` | event-specific `AgentHook` methods + action types | 0.41 |
| `agent.completion(...)` / `agent.stream_completion(...)` | `agent.runner(...).run()` / `.stream()` | 0.41 |
| `AgentBuilder::dynamic_context` | unchanged call, now hook-backed (removed in #2174, restored in #2219) | 0.41 |
| `DynamicContextStore` | none — the side retrieval pipeline is gone for good | 0.41 |
| `dynamic_tools(sample, index, toolset)` | `retrieved_tools` | 0.41 |
| `ToolSetBuilder::dynamic_tool(ToolEmbedding)` | `retrieved_tool` | 0.41 |
| `features = ["wasm"]` | nothing — target is the opt-in | 0.41 |
| `Tool::definition(prompt)` | `description()` + `parameters()` | 0.40 |
| `FinalResponse` | `PromptResponse` | 0.40 |
| `streaming::stream_completion_to_stdout` | `agent::stream_to_stdout` | 0.40 |
| `groq`/`deepseek`::`send_compatible_streaming_request` | `openai::send_compatible_streaming_request` | 0.40 |
| `Output::Unknown` | `Output::Unknown(Value)` | 0.40 |
| provider-specific `StreamingCompletionResponse` | shared `openai::StreamingCompletionResponse` | 0.40 |
| `GenericCompletionModel::with_model` | `GenericCompletionModel::new` | 0.40 |
| `MultiTurnStreamItem::final_response(&str, ..)` | `final_response(OneOrMany<AssistantContent>, ..)` | 0.38 |
| `DeltaTextChunkWithItemId.item_id` | none | 0.38 |
| library target `rig` in `rig-core` | `rig_core`, or the new `rig` facade crate | 0.37 |
| `Chat::chat(prompt, impl IntoIterator)` | `Chat::chat(prompt, &mut Vec<Message>)` | 0.37 |
| `rig_core::http_client::MockStreamingClient`, `streaming::MockResponse` | `rig_core::test_utils::*` (feature `test-utils`) | 0.37 |
| `all` feature | `derive` + `pdf` + `rayon` | 0.37 |
| `DynClientBuilder` / `AnyClient` / `ProviderFactory` | none — dispatch yourself | 0.36 |
| `CompletionModelDyn` / `EmbeddingModelDyn` / `TranscriptionModelDyn` / `ImageGenerationModelDyn` / `AudioGenerationModelDyn` | the corresponding `*Model` traits | 0.36 |
| `CompletionClientDyn` / `EmbeddingsClientDyn` / `TranscriptionClientDyn` / `ImageGenerationClientDyn` / `AudioGenerationClientDyn` / `VerifyClientDyn` | the non-`Dyn` client traits | 0.36 |
| `client::NeedsApiKey` | `markers::Missing` | 0.36 |
| `ProviderClient::from_env() -> Self` | `-> Result<Self, Self::Error>` | 0.36 |
| `VectorSearchRequestBuilder` | `VectorSearchRequest::new(query, samples)` + `with_*` | 0.37 |
| `TranscriptionRequestBuilder` / `TranscriptionModel::transcription_request` | `TranscriptionRequest::new(data)` + `with_*` | 0.37 |
| `ImageGenerationRequestBuilder` / `ImageGenerationModel::image_generation_request` | `ImageGenerationRequest::new(prompt)` + `with_*` | 0.37 |
| `AudioGenerationRequestBuilder` / `AudioGenerationModel::audio_generation_request` | `AudioGenerationRequest::new(text, voice)` + `with_*` | 0.37 |
| `markers::Provided<T>` | none — no typestate builders remain over it | 0.37 |
| `VectorSearchRequestBuilder::build() -> Result<_, _>` | infallible `build()` | 0.36 |
| `json_utils::empty_or_none` | none | 0.36 |
| `anthropic::{CLAUDE_3_5_HAIKU, CLAUDE_3_5_SONNET, CLAUDE_3_7_SONNET, CLAUDE_4_OPUS, CLAUDE_4_SONNET}` | `CLAUDE_OPUS_4_6` / `CLAUDE_SONNET_4_6` / `CLAUDE_HAIKU_4_5` | 0.35 |
| `ToolServerRequest` / `ToolServerResponse` / `ToolServerRequestMessageKind` | `ToolServerHandle` | 0.35 |
| `reqwest-rustls` / `reqwest-native-tls` features | `reqwest` + `rustls` / `native-tls` | 0.34 |
| `PromptRequest<'a, S, M, P>` | `PromptRequest<S, M, P>` | 0.34 |
| `tool::McpTool` / `McpToolError` / `McpTool::from_mcp_server` | `McpClientHandler` + `rmcp_tool(s)` | 0.33 |
| `CompletionRequest.preamble` | leading `Message::System` in `chat_history` | 0.33 |
| `gemini::EmbeddingModel::new(.., Option<usize>)` | `new(.., usize)` | 0.33 |
| `PromptResponse.total_usage` | `PromptResponse.usage` | 0.32 |
| `Provider::build` | `ProviderBuilder::build` | 0.32 |
| `ProviderBuilder::Output` | `ProviderBuilder::Extension<H>` | 0.32 |
| `http_client::sse::ReadyState` / `ready_state()` | none | 0.32 |
| `rig-eternalai`, `rig-wasm` | none — archived | 0.32 |
| `AgentBuilderSimple` | `AgentBuilder<M, P, ToolState>` | 0.31 |
| `StreamingPromptHook` | `PromptHook` | 0.31 |
| `PromptRequest::new` | `PromptRequest::from_agent` | 0.31 |
| `Reasoning.reasoning` / `Reasoning.signature` | `Reasoning.content: Vec<ReasoningContent>` | 0.31 |
| `anthropic::ImageSourceData` | `anthropic::ImageSource` enum variants | 0.31 |
| `reqwest-tls` feature | `reqwest-native-tls` | 0.31 |
