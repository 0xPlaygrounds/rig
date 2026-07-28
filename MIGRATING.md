# Migrating Rig

This guide covers every breaking change from 0.38 through the current
unreleased version. Releases 0.39 and 0.40 were large, and 0.40 in particular
carried 31 breaking changes.

## Which sections apply to you

Work upwards from the version you are on. Each section is self-contained.

| You are on | Read |
| --- | --- |
| 0.40 | [0.40 → unreleased](#040--unreleased) |
| 0.39 | [0.39 → 0.40](#039--040), then [0.40 → unreleased](#040--unreleased) |
| 0.38 or earlier | [0.38 → 0.39](#038--039), then both sections above |

**Everyone should read [Silent behavior changes](#silent-behavior-changes)
first.** Those are the changes that leave your code compiling and make it do
something different — the ones a compiler upgrade will not point at.

---

## Silent behavior changes

Nothing in this section produces a compile error. Check each one against your
code before upgrading.

### `max_turns` counts differently (0.40)

The highest-impact change in either release. `max_turns` and
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

### Providers that used to return empty text now error (0.40)

Responses with empty assistant content and no tool calls surface the shared
path's "empty response" error on **hyperbolic, perplexity, and huggingface**.
Previously each returned an empty text completion. Code that treated an empty
string as a normal outcome now takes an error branch.

### Moonshot drops reasoning-only history turns (0.40)

The shared conversion drops assistant messages carrying neither text nor tool
calls. Reasoning attached to a text or tool-call turn still round-trips via
`reasoning_content`; reasoning-only turns no longer survive in history.

### Request contents changed for several providers (0.40)

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

### Rig `Video` content now serializes instead of erroring (0.40)

On the shared conversion, `Video` user content serializes a `video_url` content
part (an OpenRouter/gateway extension) rather than returning a client-side
conversion error. Providers without video support now reject it **server-side**
— the failure moved from your process to theirs, and moved later.

### Telemetry span names and fields (0.40)

- Migrated providers' streaming spans are named `chat` with
  `gen_ai.operation.name = "chat"`, previously `chat_streaming`. Dashboards and
  alerts keying on `chat_streaming` go blank.
- `gen_ai.input.messages` / `gen_ai.output.messages` are intentionally left
  empty rather than recording serialized messages.
- `gen_ai.request.model` reports the per-request model override when one
  applies.
- minimax, zai, and xiaomimimo spans stop reporting as `"openai"` and report
  their own provider name.

### `if_wasm!` / `if_not_wasm!` key on the target (unreleased)

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

## 0.40 → unreleased

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
| `dynamic_tools(sample, index, toolset)` | `retrieved_tools` |
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
`.stream()`.

`AgentRun::new(prompt).with_history(history)` remains a sans-I/O state machine
for custom drivers. It holds no configured model, tools, memory, or hooks and is
not an alternate execution path for configured agents.

An `Agent`'s model is fixed and private. Former per-call `.model(...)` /
`.model_opt(...)` users should retain the provider `CompletionModel` and use its
raw request API, or construct a separate `Agent`.

`Extractor` now routes through the full hook lifecycle.

### 7. `dynamic_context` is removed

`AgentBuilder::dynamic_context`, `ExtractorBuilder::dynamic_context`, and the
internal `DynamicContextStore` passive-retrieval pipeline are gone. Static
builder context remains.

For passive RAG, applications now own query selection, retrieval, filtering,
reranking, formatting, caching, failure handling, and per-turn policy in a local
`AgentHook`.

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

### 1. `max_turns` — see [Silent behavior changes](#max_turns-counts-differently-040)

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
shapes — and note that both were reworked *again* before the unreleased version
([section 4](#4-tool-authoring-and-dispatch-reworked) and
[section 5](#5-hooks-are-event-specific-and-provider-independent) above). If you
are jumping 0.39 → unreleased, migrate straight to the newer shape and skip this
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
If you drove the agent loop yourself rather than calling `prompt` / `chat`,
you are affected. Note that the surrounding execution API changed again in the
unreleased version — see
[`AgentRunner` is the only execution path](#6-agentrunner-is-the-only-execution-path).

### 2. Deterministic, duplicate-safe tool registration (#1913)

Tool registration became order-deterministic and duplicate-safe, with `ToolSet`
backed by an `IndexMap`. Code relying on the previous registration order or on
duplicate-name behavior may see different tools advertised.

---

## Appendix: symbol reference

Renamed or relocated items, for searching.

| Old | New | Version |
| --- | --- | --- |
| `rig_core::tool::Tool` (portable) | `rig_core::tool::PortableTool` | unreleased |
| `rig_agent::<item>` (portable re-export) | `rig_agent::core::<item>` | unreleased |
| `client.agent(...)` inherent method | `AgentClientExt::agent` (via `rig::prelude::*`) | unreleased |
| `ToolCallExtensions` / `ToolResultExtensions` | `ToolContext` | unreleased |
| `.tool_extensions(...)` | `.tool_context(...)` | unreleased |
| `ToolDyn` (public) | `DynamicTool` | unreleased |
| `ToolSet::{call, call_with_extensions, call_structured}` | `ToolSet::execute` | unreleased |
| `ToolServerHandle::call_tool*` | `ToolServerHandle::execute` | unreleased |
| `ToolError` / `ToolFailure` / `ToolReturn` / `ToolOutcome` / `ToolExecutionResult` | `ToolExecutionError` / `ToolErrorKind` / `ToolResult` | unreleased |
| `AgentHook::on_event` + `StepEvent` + `Flow` | event-specific `AgentHook` methods + action types | unreleased |
| `agent.completion(...)` / `agent.stream_completion(...)` | `agent.runner(...).run()` / `.stream()` | unreleased |
| `AgentBuilder::dynamic_context` | own it in an `AgentHook` | unreleased |
| `dynamic_tools(sample, index, toolset)` | `retrieved_tools` | unreleased |
| `ToolSetBuilder::dynamic_tool(ToolEmbedding)` | `retrieved_tool` | unreleased |
| `features = ["wasm"]` | nothing — target is the opt-in | unreleased |
| `Tool::definition(prompt)` | `description()` + `parameters()` | 0.40 |
| `FinalResponse` | `PromptResponse` | 0.40 |
| `streaming::stream_completion_to_stdout` | `agent::stream_to_stdout` | 0.40 |
| `groq`/`deepseek`::`send_compatible_streaming_request` | `openai::send_compatible_streaming_request` | 0.40 |
| `Output::Unknown` | `Output::Unknown(Value)` | 0.40 |
| provider-specific `StreamingCompletionResponse` | shared `openai::StreamingCompletionResponse` | 0.40 |
| `GenericCompletionModel::with_model` | `GenericCompletionModel::new` | 0.40 |
