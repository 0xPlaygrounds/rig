# Migrating Rig

<!-- MIGRATING-GUIDE-INSTRUCTIONS:START -->

## Maintainers: generate this guide for every release

> [!IMPORTANT]
> This section is the immutable preamble. Keep it at the top of this file,
> unchanged and between these markers, when updating the migration guide. Edit
> the release material below the end marker. CI pins both its placement and its
> contents.

`MIGRATING.md` is an editorial synthesis, not the output of one generator. Use
the public API diff as its exhaustive spine, the changelogs for release context,
and the relevant pull requests for migration details. This is the process used
to produce the guide in [#2216](https://github.com/0xPlaygrounds/rig/pull/2216).

1. Choose the previous release tag and the release ref (`HEAD` while preparing a
   release, or the new tag when auditing a completed release). Run API diffs in
   a separate, up-to-date, clean worktree and do not edit the guide there.
   `cargo public-api` checks out both refs in place and restores the original
   checkout, so uncommitted changes will prevent a tag-to-ref diff.
2. Install [`cargo-public-api`](https://github.com/cargo-public-api/cargo-public-api)
   and ensure `jq` is available. With the clean worktree checked out at each ref
   in turn, enumerate the publishable library and proc-macro packages. Take the
   union of both lists so packages added or removed during the release are not
   missed:

   ```console
   cargo metadata --no-deps --format-version 1 | \
     jq -r '.packages[] | select(.publish != []) | select(any(.targets[]; any(.kind[]; . == "lib" or . == "proc-macro"))) | .name'
   ```

   Diff every package in that union, including re-exporting facades such as
   `rig`:

   ```console
   cargo install cargo-public-api --locked
   cargo public-api -p PACKAGE --simplified diff PREVIOUS_TAG..RELEASE_REF
   # Example while preparing the release after v0.41.0:
   cargo public-api -p rig-core --simplified diff v0.41.0..HEAD
   ```

   Repeat the diff with `--all-features` or the relevant `--features` when the
   release changed feature-gated APIs. Review and classify every added, removed,
   or changed item, even when the diff is only one line: additions such as enum
   variants, public fields, or required trait items can also break downstream
   code. If a package exists at only one ref, inspect its complete public API and
   document the package-level addition or removal instead of expecting a range
   diff to work.
3. Read the matching entries in the root and affected crate `CHANGELOG.md`
   files. Use their breaking-change entries to explain intent and identify the
   replacement API, but do not assume the changelogs are exhaustive.
4. Inspect the pull request, commits, tests, and documentation for each change
   found by the API diff or changelogs. Record the old form, the new form, and
   the smallest useful migration example. Do not summarize every merged pull
   request; investigate the changes that affect downstream users.
5. Review the release range for behavior changes that a public API diff cannot
   detect, including changed defaults, serialization or wire formats, provider
   behavior, feature semantics, and error handling. Put these under **Silent
   behavior changes** because compiling successfully does not make them safe.
6. Add the newest release section first, update **Which sections apply to
   you**, and update the old-to-new symbol appendix. Keep each release section
   self-contained. When an API changes again in a later release, tell users who
   are skipping versions which intermediate form they can skip.
7. Before release, verify every named symbol and feature against the release
   tree, check every internal Markdown link, and compile or test code snippets
   where practical. Clearly label snippets that are illustrative or transcribed
   instead of compiled. Rebase, update or recreate the clean API-diff worktree,
   repeat the diffs for newly merged changes, and replace `next` with the final
   version number immediately before tagging.

The public API diff finds compiler-visible changes; the changelogs and targeted
history explain why and how to migrate; the final behavior review catches the
changes that still compile. All three inputs are required for each release.

<!-- MIGRATING-GUIDE-INSTRUCTIONS:END -->

This guide covers every breaking change from 0.30 through the unreleased changes
after 0.41. Releases 0.36, 0.37, 0.40 and 0.41 were the disruptive ones; 0.40
alone carried 31 breaking changes, and 0.37 renamed `rig-core`'s library
target.

## Which sections apply to you

Sections run newest-first. Find the version you are on and read every section
above it, in order. Each one is self-contained.

| You are on | Start at |
| --- | --- |
| 0.41 | [0.41 → next](#041--next) |
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

### next

#### An assistant turn that carried nothing is empty, not a fabricated empty text part

Message content was a non-empty container, so a turn the model ended without
text and without tool calls — a tool-call-only turn whose calls were all
dropped, a content-filtered turn, a truncated stream, a local model emitting
EOS immediately — could not be represented. Six production sites papered over
that by pushing an `AssistantContent::text("")` the model never produced, and
that fabricated part reached history and the wire indistinguishably from a real
empty text block. (A seventh `text("")`, in the streaming accumulator's
`ensure_text_block`, is not a sentinel: it opens a slot that deltas fill, and
`finish` drops it if nothing ever arrives. That one stays.)

Content is a `Vec` now, so the honest representation exists and the fabrication
is gone. What changes for you:

- A streamed turn that produces nothing yields `choice == []` where it used to
  yield one empty-text part. Code matching `choice.first()` for a `Text` block
  to detect "the model said nothing" needs to check `choice.is_empty()` as
  well — both spellings appear: a blocking wire can still deliver a turn whose
  only part is an empty text block, and histories persisted before this
  release encode empty turns that way too.
- The agent loop neutralizes empty turns in **both** spellings: a model turn
  that is a zero-part list or a single empty, unannotated text block is kept
  out of history. That is the loop curating its own turns, not a filter on
  yours — history **you** supply is never rewritten, so an empty text block
  you replay goes to the wire exactly as this release's predecessor sent it
  (and some providers reject it there). If you carry pre-`Vec` histories with
  the fabricated empty-text part, drop those turns yourself before replaying.
- Three internal guards no longer fire: two that cancelled a run on "lost
  assistant content" and one in `rig-candle`. All three were unreachable while
  the padding existed, and reachable they would have failed runs that previously
  succeeded.
- Anthropic's empty `end_turn` follow-up — documented by Anthropic, and the
  reason the sentinel existed on that path — now normalizes to an empty choice
  rather than one empty-text part. The recorded provider response is unchanged;
  only rig's spelling of it is.

#### Decoding accepts two shapes it used to reject: `[]` and `null`

The removed container's `Deserialize` implemented only `visit_seq`. Its JSON was
otherwise byte-identical to `Vec`'s, which is why no recorded provider fixture
changes in this release — but two inputs that used to raise a local parse error
now decode to an empty `Vec`:

- an **empty array** (`[]`), which the container rejected outright;
- **`null`**, on the fields that moved from the deleted `string_or_one_or_many`
  onto `json_utils::string_or_vec` — anthropic `Message.content` and
  `Content::ToolResult.content`, plus the OpenAI chat and Responses-API
  system/user content. `string_or_vec` carries `visit_none`/`visit_unit` arms
  its predecessor did not, and those arms cannot simply be dropped: the OpenAI
  assistant-content field that already used this helper depends on them, because
  OpenAI sends `"content": null` for a message whose only payload is tool calls.

If you fed rig a history with a null or empty content field and relied on the
decode to reject it, that check is now yours. The value survives as an empty
content list; on the request path it is caught by
`CompletionRequest::validate_message_content` before the network — including a
tool result whose own block list decoded to empty, which the validator rejects
by tool name even though the user message carrying it is non-empty — but a
value you hand straight to a provider is re-emitted as `"content": []`, which
providers generally reject at the API.

#### A tool whose `Output` is `Vec<ToolResultContent>` now sends rich content

This is the change in this release that alters a wire payload with no compile
error to announce it, so check any tool that returns a list of
`ToolResultContent`.

`IntoToolOutput` preserves the canonical rich-content types ahead of its
`Serialize` fallback, so returning an image does not silently become a JSON
object. That guard used to name `OneOrMany<ToolResultContent>`; it now names
`Vec<ToolResultContent>`. Three consequences:

- `type Output = OneOrMany<ToolResultContent>` no longer compiles. Change it to
  `Vec<ToolResultContent>` and the behaviour you had is preserved.
- `type Output = Vec<ToolResultContent>` **compiles unchanged and behaves
  differently.** It used to miss the guard and fall through to serialization,
  reaching the model as a single JSON tool result
  (`[{"type":"text",...}, ...]`). It now takes the rich path and reaches the
  model as N ordered content blocks, with images sent as image parts. For nearly
  every tool that is the intended result — it is what the `OneOrMany` guard
  always did — but it is a payload change, and a prompt that parsed that JSON
  array out of the tool result needs updating.
- An **empty** `Vec<ToolResultContent>` is now an eager `ToolExecutionError`
  where it used to produce `json([])`. A zero-block tool result cannot be sent
  (the request boundary rejects it), so the error fires at the tool instead —
  as an ordinary tool failure the agent feeds back to the model, not a
  run-aborting request error one turn later. Return an explicit
  `serde_json::json!([])` for the old shape, or one empty text block for a
  genuinely empty result. (An empty **MCP** result is different: it is
  protocol-legal and outside the tool author's control, so the MCP path
  normalizes it to one empty text block instead of erroring.)
- The check lives in `ToolOutput::content`, which is now **fallible**:
  `content(Vec<ToolResultContent>) -> Result<ToolOutput, ToolExecutionError>`,
  and `impl From<Vec<ToolResultContent>>` becomes `TryFrom`. On 0.41 the
  argument type (`OneOrMany`) made the empty case unrepresentable, so the
  constructor could not fail; the `Vec` argument moves that guarantee into the
  return type. `text`, `json`, and `one` stay infallible — they construct
  exactly one block.

#### A local generation that produces nothing keeps succeeding

- A degenerate local generation on `rig-candle` — a model that emits EOS
  immediately, or only whitespace, which the parser trims — used to be padded to
  non-empty and succeed. Removing the padding exposed an emptiness check that
  would have turned it into a hard `CandleError::Inference`. That check is
  removed: an empty turn is legal here exactly as it is everywhere else, so this
  case keeps succeeding, now with genuinely empty content.
- Requests are validated before the network: an empty `chat_history`, or a user
  or assistant message with no content, is rejected locally with a message
  naming the offending index instead of becoming a provider 400. System content
  is not checked — it is a `String` and the removed container never constrained
  it.

#### Cohere token counts come from `tokens`, not `billed_units`

`GetTokenUsage for cohere::completion::Usage` read `usage.billed_units`, while
the non-streaming response conversion and the streaming final read
`usage.tokens`. The same response therefore reported two different numbers
depending on which surface you asked. All three now share one mapping, sourced
from `tokens`.

`billed_units` excludes cached input and system-prompt overhead, so the counters
it produced were much smaller: a short prompt that reported 27 input / 66 output
now reports 556 / 69. Nothing to change, but **telemetry dashboards and any
cost estimates keyed on Cohere token counts will step up**, by roughly 10-20x on
the input side. `cached_input_tokens` is also populated now, from Cohere's
`usage.cached_tokens`, which was previously discarded.

#### Cohere now honors `max_tokens`

`CohereCompletionRequest` had no `max_tokens` field, so the value was dropped
before the request was built and never reached `/v2/chat`, which defines it as a
top-level parameter. It is now sent.

Nothing to change — but if you set `max_tokens` on a Cohere agent at any point
and moved on when it appeared to do nothing, **it starts applying now**, and
generations that previously ran to the model's own output limit will be cut off
at your value.

#### Persisted Cohere `ToolResultContent` JSON no longer round-trips

`cohere::completion::ToolResultContent` was missing the `#[serde(tag = "type")]`
its sibling content enums carry, so it serialized externally-tagged as
`{"Text":{"text":"-3"}}` — a shape Cohere rejects with a 422, which is why tool
calling failed on the second turn. It now serializes as
`{"type":"text","text":"-3"}`.

The type is public. If you persisted it directly, records written by 0.41 or
earlier fail to deserialize; re-encode them to the tagged form.

#### Cohere HTTP errors surface as `InvalidStatusCodeWithMessage`

`CompletionModel::completion` boxed transport failures into
`http_client::Error::Instance`, which hid the status and body from
`CompletionError::provider_response_status()` and `provider_response_body()` —
both returned `None` on every real HTTP failure. The error is now propagated
unwrapped.

If you matched on the inner `http_client::Error` variant, the payload for a
non-success response moves from `Instance(..)` to
`InvalidStatusCodeWithMessage(status, body)`. Code that only used the
`provider_response_*` helpers starts getting answers instead of `None`.

---

## 0.41 → next

### xAI uses the shared Responses wire response

`providers::xai::completion::CompletionResponse` is now an alias for
`providers::openai::responses_api::CompletionResponse`. The xAI model still
sends xAI's request shape to `/v1/responses`, preserves xAI error envelopes and
request ids, and emits completed streamed tool calls at the same boundary; only
the duplicated response and streaming implementation is gone.

Code inspecting `raw_completion()` results should use the shared field names
and types: `created_at` replaces `created`, `status` is a `ResponseStatus`
instead of `Option<String>`, and the complete Responses metadata surface is
available. To normalize a raw value explicitly, import
`completion::NormalizeCompletionResponse` and call `raw.normalize("xai")`;
the xAI-specific `TryFrom` implementation no longer exists.

`ResponseStatus` also gains `Other(String)`. This lets OpenAI-compatible
providers preserve a newly introduced status instead of failing wire
deserialization, but exhaustive matches need an `Other` arm.

### `OneOrMany<T>` is gone; lists are `Vec<T>`

`rig_core::OneOrMany` and `rig_core::EmptyListError` are removed, along with the
`one_or_many` module and both prelude re-exports. Every use becomes `Vec<T>`.
There is no replacement crate and no bespoke non-empty type.

The type promised something untrue. It asserted "at least one item" on a
response path where zero items is a real outcome, so the code fabricated data to
satisfy it (see [Silent behavior changes](#next)). Its `is_empty()` returned a
hardcoded `false`, which meant every caller asking a list whether it was empty
got the wrong answer with no compile error — two live defects were hiding behind
that, one of them a provider guard that could never fire.

**The serialized format is unchanged**, so persisted histories and stored
embeddings need no migration: the container already serialized as a plain
sequence. This is a source-only break, and it is why not one recorded provider
fixture changes.

The migration at your call sites:

| Was | Now |
| --- | --- |
| `OneOrMany<T>` | `Vec<T>` |
| `OneOrMany::one(x)` | `vec![x]` |
| `OneOrMany::many(xs)` → `Result<_, EmptyListError>` | the `Vec` itself; use `message::require_non_empty` where you were relying on the rejection |
| `OneOrMany::merge(xs)` → `Result<_, EmptyListError>` | `xs.into_iter().flatten().collect()` — the `?` is gone: `merge` returned `Err(EmptyListError)` when the flattened result was empty, this yields an empty `Vec`. Add your own check if you relied on the rejection |
| `OneOrMany::from_iter_optional(xs)` | `message::non_empty(xs)`, or inline `Some(xs).filter(\|items\| !items.is_empty())` |
| `.first()` / `.last()` → owned `T` | `.first()` / `.last()` → `Option<&T>` |
| `.first_ref()` / `.last_ref()` → `&T` | `.first()` / `.last()` → `Option<&T>` |
| `.rest()` → `Vec<T>` | `.iter().skip(1)` or `.get(1..)` |
| `.is_empty()` → always `false` | `.is_empty()` → the real answer |
| `one_or_many::string_or_one_or_many` | `json_utils::string_or_vec` |

`.len()`, `.iter()`, `.iter_mut()`, `.into_iter()`, `.push()` and `.insert()`
carry over unchanged.

Two conversions could not stay trait impls: with both sides now foreign types,
`impl TryFrom<Vec<..>> for Vec<..>` violates the orphan rule. They are `pub`
free functions in `providers::openai::completion`, so the surface is not
narrowed:

| Was | Now |
| --- | --- |
| `<Vec<Message>>::try_from(v)` where `v: Vec<message::UserContent>` | `user_content_to_messages(v)` |
| `<Vec<Message>>::try_from(v)` where `v: Vec<message::AssistantContent>` | `assistant_content_to_messages(v)` |
| `impl From<OneOrMany<String>> for Vec<ReasoningSummary>` | `providers::openai::responses_api::reasoning_summaries(v)` |

`<Vec<Message>>::try_from(m)` for a whole `message::Message` is unaffected.

### Where the container's enforcement went

The container was doing two jobs in opposite directions, and they separate:

- **Outbound (request path).** `CompletionRequest::validate_message_content`
  rejects an empty `chat_history` and any user or assistant message with no
  content, before the request is sent. `CompletionRequestBuilder::send`/`::stream`
  call it, which is also how both agent surfaces issue their requests, so agent
  traffic is covered without a second check. Handing a request straight to a
  `CompletionModel` bypasses it — call it yourself there.
- **Inbound (response path).** Per-wire guards route through the new
  `message::require_non_empty(items, || error)`, each naming its own error
  rather than sharing the constructor's context-free one. Most keep the message
  they already had verbatim; the exception is bedrock, whose guards previously
  surfaced `EmptyListError` — a message naming the deleted container — and now
  say which message came back empty, as `ResponseError` rather than
  `RequestError`. These reject a provider that returned nothing where its
  protocol promises content — a provider defect, which is a different claim from
  "empty assistant content is illegal". It is not: on the response path an empty
  turn is legal, which is the whole reason the container had to go.

If you implement `CompletionModel` yourself, nothing is required of you. If you
built messages by hand and relied on the constructor to reject empties, call
`message::require_non_empty` at that point, or let the request boundary catch it.


### The raw grammar is a part lifecycle: Start / Delta / End per content kind

Every streamed part is now an **entity with a lifecycle** — open, mutate in
place, close — instead of an id-keyed event sequence (review 84a43e9e root
cause A; vercel's stream-part triples and pydantic-ai's
`_stop_tracking_vendor_id` are the reference designs):

- `RawStreamingChoice` gains `ReasoningStart { id, provider_id }`,
  `ReasoningEnd { id, reasoning, signature }` and `TextEnd { id }`. The
  whole-block `Reasoning` event remains as shorthand for
  open + authoritative restatement + close. `ReasoningSignature` (added
  earlier in this PR's lineage) is **deleted** — a trailing signature is
  just an `End` arriving late, and one end primitive covers a signature
  closing an open block, a trailing signature after interleaved output, and
  a signature-only stream, with no per-case branch for an adapter to
  forget.
- The accumulator reduces to open-maps into an arrival-ordered part list:
  open registers the part once (fixing order), deltas mutate through the
  handle, end applies the authoritative payload and moves the key to a
  finished-set. **Idempotence belongs to the entity**: the finished-set is
  populated by every finalization route (fragment ends,
  authoritative-payload ends, whole-call adoption), so a repeated
  `ToolInputEnd` — even one carrying an authoritative name and arguments —
  finalizes nothing. Key reuse after finishing opens a new part; that
  lenient rule replaces the ordinal machinery.
- Wires that never announce boundaries (gemini thought parts, ollama and
  chat-compat reasoning, cohere thinking) have their adapters *synthesize*
  the end events at the boundaries they already detect — one grammar
  instead of per-wire lifecycle re-derivation. Deleted with the old
  machinery: `close_minted_reasoning`, the ordinal maps, the
  `DeltaBuilt`/`Complete` part tags, `closed_by_full_call`, and every
  adapter-side thought/restatement buffer (gemini REST/interactions/gRPC,
  anthropic's thinking-text buffer).
- Consumers: `StreamedAssistantContent::Reasoning` now fires when the
  **wire said something at the block's end** — an end carrying an
  authoritative payload (a restatement or a signature), or a bare end frame
  the wire itself sent (`ReasoningEnd { wire_sent: true }`, e.g. anthropic's
  `content_block_stop` on an *unsigned* thinking block, which keeps firing a
  completed event exactly as before this refactor). Only a bare end a rig
  adapter *synthesized* at an interleaving boundary stays silent: consumers
  already received every delta, and no fabricated completion event changes
  what history builders observe.

### Stream keys are opaque; durable ids and correlators are separate values

The raw-event identity type is now `rig_core::streaming::StreamPartId` — an
**opaque accumulation key**: `Eq + Hash + Clone + Debug` and nothing else. It
has no serialization, no rendering, and no accessor into the durable id
space — and its representation is private, so pattern-matching cannot
extract the wire string either (the `identity_leak` compile-fail suite pins
all four). Construction goes through `StreamPartId::wire` /
`StreamPartId::minted` / `MintKind::for_wire_index`. The durable provider handle
travels separately as `WireId` (`Reasoning`/`ReasoningDelta` gain
`provider_id: Option<WireId>`; `RawStreamingToolCall` gains
`tool_id: Option<WireId>`; `ToolInputEnd::tool_id` is `Option<WireId>`).
`WireId::new` rejects the empty string, so an absent handle is `None` by
construction — the fabricated-empty-id class (and every per-serializer
`.filter(|id| !id.is_empty())` compensating for it) is gone.

Public stream items change accordingly (breaking):

- `StreamedAssistantContent::ReasoningDelta.id` is now a **rig-generated
  correlator** — stable per reasoning part, unique per run — plus a new
  `provider_id: Option<String>` carrying the provider-issued item id when
  one exists. The previous `rig:reasoning:0`-style renderings (and their
  one-stream uniqueness caveat) are gone; the caveat documented for
  `0.41 → next` is superseded.
- `StreamedAssistantContent::ToolCallDelta` loses its `id` field:
  `internal_call_id` is the correlator, and provider ids arrive on the
  completed `ToolCall`. The agent hook payload `ToolCallDelta` loses
  `tool_call_id` for the same reason.
- Consumers reconstructing durable ids from delta ids must use
  `provider_id`: the assembled `Reasoning::id` carries only provider-issued
  values (this closes a latent leak where a minted rendering could enter
  history through the agent's delta-only assembly path).
- `StreamedAssistantContent::Reasoning` becomes a struct variant
  `Reasoning { reasoning, id }`: the completed block restates the
  rig-generated correlator its deltas carried, mirroring the completed
  `ToolCall`'s `internal_call_id`. On wires with no provider handle
  (anthropic unsigned thinking, gemini signature-only ends) the correlator
  is the *only* way to associate the completed replacement with its prior
  deltas — `reasoning.id` stays the durable provider handle and the
  correlator never enters replayable history. Matchers change from
  `Reasoning(reasoning)` to `Reasoning { reasoning, .. }`.

### Tool-call identity is typed: `ToolCallId` + `ProviderCallId`

The message layer receives the same identity decomposition the streaming
layer already has (`StreamPartId` vs `WireId`): rig's correlation handle and
the provider's wire handles are now different types, and the empty-string
sentinel is unrepresentable.

```rust
pub struct ToolCall {
    /// Rig's correlation handle. Always present; minted when the provider issued none.
    pub id: ToolCallId,
    /// What the provider issued, if anything — the only ids that go back on its wire.
    pub provider: Option<ProviderCallId>,
    pub function: ToolFunction,
    pub signature: Option<String>,
    pub additional_params: Option<serde_json::Value>,
}

/// Dual-identifier wires need both: OpenAI Responses issues an item id
/// (`fc_…`) *and* a `call_id` (`call_…`).
pub struct ProviderCallId {
    pub call_id: String,
    pub item_id: Option<String>,
}

pub struct ToolResult {
    /// Which call this answers — correlation, always present.
    pub call: ToolCallId,
    pub provider: Option<ProviderCallId>,
    /// The *executed* tool's name. Required, not `Option`.
    pub name: String,
    pub content: Vec<ToolResultContent>,
}
```

`ToolCallId` is non-empty by construction (`new` returns `None` for `""`;
`mint()` generates a fresh 21-character handle; `new_or_mint` is the
boundary guard). Every provider decode path adopts the wire's id when one
was issued (`ToolCall::from_wire`, `ToolCall::from_dual_wire`) and mints
when none was — so comparing or keying by `ToolCall::id` is now correct on
id-less wires (older Ollama daemons, Gemini REST), where every call
previously carried `id: ""` and collided.

What this changes for you:

- **Construction**: `ToolCall::new` takes a `ToolCallId`; provider decode
  paths use `ToolCall::from_wire(wire_id, function)` (empty mints) or
  `ToolCall::from_dual_wire(item_id, call_id, function)`.
  `with_call_id` is replaced by `with_provider(ProviderCallId)`.
  `UserContent::tool_result(call, name, content)` takes the answered
  call's correlation handle (echo `ToolCall::id`) and the executed
  tool's name (required); the string is recorded as the handle only,
  never as a provider-issued id — a bare string cannot prove provider
  provenance, so echoing a minted handle no longer sends it upstream on
  optional-id wires. When you hold provider identifiers, use
  `tool_result_from_wire(wire_id, name, content)` (single-identifier
  wire echo), `tool_result_with_call_id` (dual-identifier), or
  `tool_result_for(call, provider, name, content)` — the agent-driver
  form; `tool_result_named` is gone.
- **Serialization to providers**: wires that require a call id (OpenAI
  chat/Responses, Anthropic, Bedrock, Gemini Interactions, xAI) receive
  `provider.call_id` when the provider issued one, else rig's minted id —
  never an empty string, and the Interactions/Responses "requires
  `call_id`" request errors are gone (unrepresentable). Wires with optional
  ids (Gemini REST/gRPC) omit the id unless the provider issued one:
  minted handles never travel upstream there.
- **Persisted histories**: the canonical serde shape changed (`ToolCall`
  gains `provider`, loses `call_id`; `ToolResult` renames `id` → `call`,
  requires `name`). Pre-provider-split `ToolCall` JSON is **not migrated
  on load**: a legacy `call_id` key is an unknown field and is ignored, so
  a dual-identifier payload loses its correlator and a single-identifier
  payload's `id` is read as rig's handle with no provider provenance.
  Migrate the JSON before upgrading if you need those identifiers
  (`call_id` → `provider.call_id`; for old dual-identifier payloads the
  `fc_…` `id` becomes `provider.item_id`, **and the top-level `id` becomes
  the `call_…` correlator** — `id` is required, and rig pairs a
  `ToolResult.call` against `ToolCall.id`, so leaving the `fc_…` handle
  there breaks the pairing the adjacent `ToolResult` recipe produces). Legacy `ToolResult` JSON does
  **not** deserialize (no `call`, no `name`); re-run the conversation or
  migrate the JSON by hand (`id`/`call_id` → `call` + `provider.call_id`,
  add the executed tool's `name`). Empty-string ids in old JSON are
  rejected by construction.
- **The back-compat pairing shim is deleted**:
  `providers::internal::resolve_tool_result_names` (and the name-in-id
  legacy encodings it supported) no longer exist — `ToolResult::name` is
  required data, and every name-keyed serializer (Gemini
  `functionResponse.name`, Ollama, Vertex AI, gemini-grpc) reads it
  directly. One typed-id descendant remains: rig's own inbound converters
  cannot supply a name (Anthropic/OpenAI-chat/Cohere/Bedrock wires carry
  none), so name-keyed request assembly fills an *empty* name from the
  result's paired call, matched by identifier only
  (`providers::internal::resolve_empty_tool_result_names`). Established
  names are never overwritten, and nothing pairs by position or name.
- **Hooks and telemetry**: tool-call ids surfaced to hooks
  (`ToolCallEvent`/`ToolResultEvent`/`InvalidToolCallContext.tool_call_id`)
  and telemetry are now the durable id — the provider's when issued, else
  rig's minted handle — never `Some("")` and never absent for a real call.

### `ToolResult` carries the executed tool's name

> **Superseded in this release**: `name` is now **required** (`String`, not
> `Option<String>`) and the `resolve_tool_result_names` shim is deleted —
> see the `ToolCallId` section above.

`rig::message::ToolResult` gains `name: Option<String>` — the name of the
tool that actually executed (which can differ from the model's call when a
hook repaired it). Struct-literal constructions need the new field
(`name: None` preserves the old shape); `UserContent::tool_result_named` is
the constructor the agent drivers use, and serde skips the field when
absent, so persisted histories round-trip unchanged.

Why: several wires require the function *name* on a replayed tool result
(Gemini's `functionResponse.name`, Ollama's tool messages), and rig used to
smuggle it through the tool-call id — which collided two calls to the same
tool and, for histories sourced from OpenAI-Chat-shaped providers, replayed
the literal identifier (`call_abc`) as the function name. The name is now
data on the result; the history-pairing heuristic
(`resolve_tool_result_names`) survives only as a back-compat shim for
results with `name: None`, and an id that matches a paired call's identity
now resolves to that call's *name* instead of leaking the identifier.



### Stream parse policy and aggregation are centralized

Choice aggregation is one component (`PartsAccumulator`); a full reasoning
block supersedes only its own deltas, and multi-part reasoning items keep
every part. Parse policy is uniform across providers: unknown event types are
skipped with a warning; corrupt or schema-defective known frames surface as
`Err` items while the stream keeps consuming (the buffered ChatGPT unary path
fails the completion instead — it has no stream to carry errors, and the
request/response OpenAI Responses *websocket* turn likewise fails the whole
session on a corrupt frame — its conformance suite records both as sanctioned
`xfail`s). Consumers
that drain to `None` see identical content to before, plus error items where
frames were previously dropped silently.

Copilot's `response.failed` handling is unified with the shared Responses
interpreter: the raw event body is now always attached to the surfaced error
(`provider_response_body()` is `Some`), including when the event carries no
error object. Previously Copilot kept a two-tier shape where an object-less
`response.failed` produced a `ProviderError` with no attached body; code that
distinguished the failure mode by `provider_response_body().is_none()` should
inspect the preserved event JSON instead.

### Streaming reasoning events carry mandatory identity

`RawStreamingChoice::Reasoning` and `RawStreamingChoice::ReasoningDelta` now
carry `id: PartId` instead of `id: Option<String>`, and the public
`StreamedAssistantContent::ReasoningDelta` carries a rendered `id: String`.
Reasoning interleaves with other output on real wires (OpenAI Responses emits
the completed item after tool calls), so aggregation keys by identity rather
than guessing by adjacency; an optional id made the correct behavior
unimplementable.

Provider authors: propagate the wire's item identity as `PartId::wire(...)`
(the Responses `item_id`) when it exists; when the wire has none, mint one
via `SyntheticIds` / `MintKind::for_wire_index` (a per-stream constant minted
identity preserves merge-into-one-block behavior for non-interleaving
protocols). All deltas of one block must share one id with that block's
completed form — that id is what lets a full block supersede its deltas.

Consumers matching on `ReasoningDelta { id, .. }` drop the `Option` unwrap.
A full `Reasoning` event supersedes prior deltas with the same id — render it
as a replacement, not an addition.

On the OpenAI Responses wire, a reasoning delta arriving without `item_id`
(ChatGPT's replayed envelope-less bodies) is keyed by a minted
`output-{output_index}` identity, and the slot's `response.output_item.done`
full blocks **adopt that minted identity** instead of the item's real `rs_*`
id, so the restated content supersedes the delta-built part rather than
duplicating beside it. Consequence: because minted identities are
Rig-authored, such a turn's reasoning (including any encrypted content) is
not replayed to the provider on follow-up requests — the wire itself dropped
the correlation the provider would need. Streams whose deltas carry `item_id`
keep the exact `rs_*` collapse and replay behavior.

### Stream errors mentioning "aborted" are no longer swallowed

The stream error path had a special case: a `CompletionError::ProviderError` whose
message *contained* the substring `"aborted"` terminated the stream as a clean
end-of-stream. That discarded the error **and** every item streamed before it,
so a gateway reporting "request aborted" surfaced as a successful empty turn.
The case is gone; such errors are delivered like any other.

Cancellation is unaffected and needs no changes: `StreamingCompletionResponse::cancel()`
aborts through `Abortable`, which ends the stream normally. If you relied on the
substring to signal cancellation from a custom provider, switch to `cancel()`.

Two related streaming fixes need no action but are worth knowing: re-polling an
already-drained stream no longer replaces the aggregated choice with an empty
text part, and a wire that restates a fragmented tool call as a complete
`ToolCall` now reuses the `internal_call_id` its deltas published (a trailing
`ToolInputEnd` for that id is a no-op rather than a duplicate call).

### Gemini gRPC surfaces tool-protocol failures

The gRPC surface now returns an error and stops the stream when Gemini reports
`MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL` or `TOO_MANY_TOOL_CALLS`,
matching the REST surface; the provider's `finish_message` is included. Code
that treated those turns as complete (they previously arrived as a normal
finish with no content) will now see the failure. This affects both the
streaming and unary gRPC paths.

### Streaming text blocks carry mandatory identity

`RawStreamingChoice::TextStart` now carries `id: PartId` alongside its
optional `additional_params`. The contract mirrors [reasoning
identity](#streaming-reasoning-events-carry-mandatory-identity): distinct wire
output items must aggregate as distinct text parts, so aggregation keys text
blocks by identity instead of tracking a single active block. On the OpenAI
Responses wire, two `message` output items now aggregate as two text parts in
wire order — previously their deltas concatenated boundary-less into one.

Provider authors: propagate the wire's item identity (the Responses
`item_id`, Anthropic's content-block index as `block-{index}`) when it
exists. A wire that never announces text-block boundaries needs no
`TextStart` at all — a bare `Message` with no open block opens a block under
a boundary-minted `text-{n}` identity, preserving the previous
single-text-block aggregation exactly. A `TextStart` whose id was already
seen on the stream reactivates that block (the keyed collapse reasoning ids
get); an unseen id closes the active block and opens the new one lazily on
its first delta, so a content-less `TextStart` leaves no empty part behind.

Minted text ids are internal bookkeeping only: aggregated
`AssistantContent::Text` parts carry no id, so nothing changes downstream for
consumers — except that multi-item streams now produce the correct number of
text parts.

### Streaming part identity carries provenance (`PartId`)

Every identity-bearing raw streaming event — `TextStart`, `ToolCallDelta`,
`Reasoning`, `ReasoningDelta`, `ToolInputEnd::id`, and
`RawStreamingToolCall::id` — now uses `rig_core::streaming::PartId` instead
of a `String`:

- `PartId::Wire(String)` is an identifier the provider put on the wire. It is
  the only identity that becomes a durable provider handle
  (`Reasoning::id`, `ToolCall::id`) and round-trips upstream.
- `PartId::Minted { kind, index }` is an identity rig fabricated at a stream
  boundary because the wire supplied none. It keys accumulation for the life
  of the stream and structurally cannot reach a request: `PartId` implements
  no `Serialize`, and the only request-serializable form (`WireId`) is
  constructible solely from `PartId::Wire`. The reserved string namespaces
  (`reasoning-{n}`, `block-{n}`, `output-{n}`, `tool-{index}`, `text-{n}`)
  and the `is_boundary_minted_id` provenance gate are gone — provenance
  travels in the type, so there is nothing to parse and no serializer gate
  to keep in sync.

Provider authors: wrap wire identities with `PartId::wire(...)` (a plain
`String`/`&str` also converts via `From`, always to `Wire`); mint via
`SyntheticIds` (now `rig_core::streaming::SyntheticIds`, minting `PartId`
values; the old string-based helper in `providers::internal::adapter` is
re-exported from its new home) or `MintKind::for_wire_index`.

Public stream consumers: `StreamedAssistantContent::ReasoningDelta` /
`ToolCallDelta` ids are the rendered form. A wire id renders verbatim; a
minted id renders namespaced (`rig:reasoning:0`) and is **unique within one
stream only** — it restarts on every turn of a multi-turn run, so never key
across streams by it (correlate across a run with `internal_call_id`
instead). Aggregated `Reasoning` parts from minted-identity streams now carry
`id: None` (previously the minted string leaked into history and, on some
providers, upstream).

Behavior change on id-less wires (gemini REST/interactions, ollama,
chat-compat gateways): rig no longer fabricates a durable tool-call id — not
from an index, and not from the tool name (two calls to the same tool in one
turn no longer collide). The durable `ToolCall::id` is the absent (empty)
value and serializers omit it; Gemini's `functionResponse.name` is resolved
by pairing each result with its assistant-turn call in the history (by
`call_id` when the wire supplied one, else in wire order).

### The streaming wire-adapter surface is public and contractual

`WireAdapter`, `run_wire_stream`, `run_wire_buffered`, `SyntheticIds`
(`rig_core::providers::internal::adapter`) and `ToolCallBridge`
(`rig_core::providers::internal::tool_call_bridge`) are `pub`: out-of-tree
provider crates (rig-bedrock, rig-gemini-grpc, rig-candle are the in-repo
consumers) implement their streaming pipelines against them. The contract an
implementation must uphold:

- **`classify` delegates**: frame decoding goes through a `wire.rs`-style
  classifier (`classify_tagged_frame`, `classify_chat_completions_frame`,
  `classify_untyped_line`, `classify_typed_event`) — never raw serde — so
  decode-then-validate policy is stated once per wire family.
- **The driver owns policy**: `Unknown` frames warn, skip the semantic path,
  and surface verbatim as `RawStreamingChoice::Unknown` /
  `StreamedAssistantContent::Unknown` passthrough events (the openai-agents
  raw-event precedent) — never folded into the aggregated assistant choice;
  buffered mode has no stream to carry them, so there they remain a warned
  skip. `Corrupt` frames surface as in-band `Err` items (buffered mode: fail
  the operation), transport errors end the stream with truncation semantics.
  Adapters contain no `match WireEvent`; `interpret` maps `Known` events
  only. (For websocket consumers: `ResponsesWebSocketEvent` gained an
  `Unknown` variant carrying the same raw passthrough — exhaustive `match`es
  over that enum need a new arm.)
- **The unknown payload is `UnknownPayload`, not a bare `Value`.**
  `RawStreamingChoice::Unknown` and `StreamedAssistantContent::Unknown` carry
  `streaming::UnknownPayload` — a transparent-serde wrapper whose `Debug`
  is **redacted** (structural metadata only). Unmodeled frames can carry
  model output, and `warn!(?value)`-style Debug captures were a recurring
  leak class; with the payload unable to Debug-print its content, the class
  is closed by the type rather than policed by review. Consumers who want
  the content call `.value()` (serialization is unchanged
  — the wrapper is `#[serde(transparent)]`); construct one with
  `UnknownPayload::new(value)` or `value.into()`.
- **Behavioral note — `Unknown` events can now occur on every network
  provider.** `StreamedAssistantContent::Unknown` is not a new variant (it
  has carried unmodeled Responses output items since #1950), but it
  previously appeared only on the OpenAI Responses wire. Every network wire
  family now forwards unrecognized-but-valid frames as `Unknown` passthrough
  events — copilot heartbeats, gateway extras, future provider event types.
  The one exception is rig-candle's local generation: its events are
  constructed in-process, never decoded off a wire, so that family produces
  no `Unknown` frames (its changelog says so explicitly). Aggregation is
  unaffected (`Unknown` is never folded into the choice), but a match arm
  like `other => panic!(..)` that never fired before will fire now: treat
  `Unknown` as an ignorable passthrough unless you deliberately consume raw
  frames.
- **Identity is mandatory**: every `Reasoning`/`ReasoningDelta`,
  `ToolCallDelta`, and `TextStart` event carries a non-empty id — the wire's
  own identity when it exists, else an id minted via `SyntheticIds` in the
  reserved namespaces (`reasoning-{n}`, `block-{n}`, `output-{n}`,
  `tool-{index}`, `text-{n}`). Wire ids must never use these shapes.
  Aggregation treats minted ids as per-stream constants (other output closes
  the open block). Upstream, the rule is per-wire: providers that validate
  identity server-side gate minted ids out of requests (the Responses
  reasoning path drops boundary-minted items); wires where identity is
  structurally required — the chat `tool_call_id` pair — replay minted ids
  self-consistently, which id-omitting gateways accept since they had no
  server-side id to check against.
- **Finish/flush obligations**: `finish` runs only on EOF without a terminal
  and closes open blocks — it must never synthesize a terminal record
  (deferring a terminal the wire already signaled is allowed).
  `flush_before_terminal_error` yields fully-delivered content (buffered tool
  calls) before a terminal error item, and must not push a terminal record.
- **`is_finished`**: return `true` after `interpret` consumed the wire's own
  in-band terminal failure; the driver then stops without running `finish` —
  the adapter has already pushed its flush-then-`Err` sequence.

### Raw streaming surface: smaller, stricter events

Four adjacent breaking changes to the raw streaming vocabulary (none of these
types are `#[non_exhaustive]`, so exhaustive matches and constructions
break):

- `RawStreamingChoice::ToolCallDelta` lost its `internal_call_id` field. The
  shared accumulator now mints the internal correlation id when a call's
  assembly opens and returns it on every fragment; adapters no longer track
  per-call state. Consumers read it from
  `StreamedAssistantContent::ToolCallDelta::internal_call_id`, unchanged.
- `RawStreamingChoice` gained a `ToolInputEnd` variant — the shared
  assembler's signal to finalize a fragmented tool call. Exhaustive matches
  over the enum need a new arm.
- `OpenAICompatibleProvider::decorate_streaming_tool_call` changed from
  mutating a `&mut HashMap<usize, RawStreamingToolCall>` in place to
  returning `Option<ToolCallDecoration>` — an event rewrite the adapter
  applies, instead of a hook into assembler state.
- `OutputFunctionCall::arguments` (OpenAI Responses) is a
  `FunctionCallArguments` newtype over the raw string, so unparseable
  restated arguments can be routed through assembly instead of failing the
  decode. Call `.parse()` / `.as_str()` where a `String` was read before.

Separately, the Anthropic and OpenAI Responses streaming event enums no
longer carry a `#[serde(other)]` catch-all: an unrecognized event now decodes
through the wire classifier as `Unknown` (warn-and-skip, surfaced verbatim)
instead of being silently absorbed into a unit variant — and a frame that
matches a modeled event's tag but not its shape is a decode **error** rather
than a silent absorb. Code deserializing those enums directly must handle
the error case.

### Completion responses are concrete and normalized

`CompletionResponse<T>` is now `CompletionResponse`. The provider-native
`raw_response` field is gone; the normalized response carries the metadata that
callers actually reached into `raw_response` for:

```rust
pub struct CompletionResponse {
    pub choice: Vec<AssistantContent>,
    pub usage: Usage,
    pub message_id: Option<String>,
    pub response_id: Option<String>,
    pub finish_reason: Option<FinishReason>,
    pub provider: String,
    pub model: Option<String>,
}
```

| Before | After |
| --- | --- |
| `response.raw_response.model` | `response.model` |
| provider stop/finish reason off `raw_response` | `response.finish_reason` |
| provider/message identity off `raw_response` | `response.provider`, `response.message_id` |
| response-scoped ID (`chatcmpl-*`, `responseId`, …) off `raw_response` | `response.response_id` |
| a genuinely provider-specific field | `model.raw_completion(request).await?` |

`usage` is unchanged, including the rule that all-zero values mean the provider
supplied no metrics. `model` is the identifier the *wire response* reported, not
the one you requested — it is `None` when the provider omits it. `provider` is
always populated, including on a response derived from a stream that ended
before its terminal record.

`message_id` and `response_id` are distinct on purpose. `message_id` holds only
identifiers the provider would recognize on a *replayed assistant message* (an
OpenAI Responses output-message `msg_*` ID, an Anthropic `msg_*` ID); it is what
agent history promotes into `Message::Assistant`'s `id`. `response_id` holds
identifiers that name the response as a whole (an OpenAI chat `chatcmpl-*` ID, a
Gemini `responseId`, a Cohere generation ID) — useful for logging and support,
never echoed back to a provider. Code that previously read a chat provider's
`message_id` should read `response_id` instead; for those providers
`message_id` is now `None`.

`CompletionResponse` is `#[non_exhaustive]`; build it with
`CompletionResponse::new(choice, usage, provider)` plus the `with_*` helpers.
Use `with_finish_reason` / `with_optional_finish_reason` rather than assigning
the field: the setters apply `FinishReason::reconcile_with_output`, which
upgrades a reported `Stop` to `ToolCalls` when the turn actually carried tool
calls. Several OpenAI-compatible gateways report `stop` on a tool-calling turn,
so code branching on `ToolCalls` would otherwise miss the call.

Tests (or other code) holding a provider's raw response can re-derive the
normalized fields via the additive `NormalizeCompletionResponse::normalize`
bridge — the same conversion the provider's normalized path uses.

### Provider-native responses moved to `raw_completion` / `raw_stream`

Every built-in provider model exposes both:

```rust
let native = model.raw_completion(request).await?;   // the provider's own type
let native_stream = model.raw_stream(request).await?; // RawStreamingResult<TheirTerminal>
```

These share one request builder, transport call, parser, telemetry path, and
error-preservation path with the normalized methods — the normalized method
calls the raw one and maps the result, so there is still exactly one network
request.

The trade: raw access now requires the concrete provider model rather than any
`CompletionResponse`. Code that was generic over `CompletionModel` could never
touch `raw_response` without a bound anyway, so in practice this affects code
that had already committed to a provider.

### Normalized finish reasons

```rust
pub enum FinishReason { Stop, Length, ToolCalls, ContentFilter, Other(String) }
```

Unrecognized provider values are preserved verbatim in `Other` — in the
provider's own spelling, so Gemini's `RECITATION` stays `RECITATION`. A provider
adding a new terminal reason surfaces it rather than reading as a natural stop.
`None` means the provider genuinely reported no reason.

### Ordinary streaming types no longer carry a response parameter

`StreamingCompletionResponse<R>`, `StreamingResult<R>`,
`StreamedAssistantContent<R>`, and the downstream agent streaming types are
concrete. Their terminal record is `StreamFinal`, which carries normalized
usage, finish reason, provider, provider-reported model, message ID, and
response ID.

A full `Reasoning` stream event supersedes prior `ReasoningDelta` events with
the same reasoning `id` — UIs that render deltas incrementally should replace
the accumulated text when the full block arrives, mirroring what the
aggregated `choice` already does.

`GetTokenUsage` is deleted — read `StreamFinal::usage` (or
`StreamingCompletionResponse::usage()`) directly. A stream that ends without a
terminal record still reports `Usage::new()`, the documented zero sentinel.
With `GetTokenUsage` gone, the telemetry helper
`SpanCombinator::record_token_usage` takes `&Usage` instead of a
`GetTokenUsage`-bounded generic.

A terminal record is now emitted only when the provider signaled genuine
completion (its own end-of-response event). Previously, several provider
streams synthesized a default-usage terminal record when the connection ended —
including streams cut off mid-response. A stream that ends without a terminal
record was truncated; treat the missing record as an incomplete turn, not a
zero-usage success.

The agent surface enforces this: `agent.stream_prompt(...)` now yields
`Err("provider stream ended without a terminal record; treating the turn as
truncated")` for a stream the provider never confirmed complete, where it
previously finished "successfully" with zero usage. If you see this error
behind a flaky provider or proxy, the connection was cut mid-response — retry
the turn rather than trusting the partial content.

`StreamingCompletionResponse::stream` takes the provider descriptor name first:

```rust
StreamingCompletionResponse::stream(PROVIDER_NAME, normalized_stream)
```

Provider implementations keep their native terminal type behind
`RawStreamingResult<Native>` and map it once:

```rust
let raw = self.raw_stream(request).await?;
let normalized = rig_core::streaming::normalize_stream(raw, |native| {
    Ok(StreamFinal::new(PROVIDER_NAME, native.usage)
        .with_optional_finish_reason(map_finish_reason(native.finish_reason)))
});
Ok(StreamingCompletionResponse::stream(PROVIDER_NAME, normalized))
```

`normalize_stream` applies the same `Stop` → `ToolCalls` reconciliation as the
unary path, using the tool calls it actually saw on the stream.

`StreamingPrompt<M, R>` and `StreamingChat<M, R>` lost their `R` parameter:
`StreamingPrompt<M>`, `StreamingChat<M>`.

### `CompletionModel` no longer owns response or construction types

Remove `Response`, `StreamingResponse`, `Client`, and `make` from custom
implementations. A custom model implements only the normalized operations, and
optionally `capabilities`:

```rust
impl CompletionModel for MyModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> { /* ... */ }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> { /* ... */ }
}
```

Construction is a separate, optional opt-in. `CompletionClient::completion_model`
is now required and calls your model's own constructor:

```rust
impl CompletionClient for MyClient {
    type CompletionModel = MyModel;

    fn completion_model(&self, model: impl Into<String>) -> MyModel {
        MyModel::new(self.clone(), model.into())
    }
}
```

`client.completion_model(model)` and `client.agent(model)` are unchanged at call
sites. A model type with no client at all is now expressible: implementing
`CompletionModel` no longer drags in a client associated type.

A provider extension built on the generic `rig::client::Client<Ext, H>` cannot
implement `CompletionClient` for that foreign type itself (orphan rule).
Instead, implement the public `ConstructCompletionModel<Client<Ext, H>>` hook
on your model type; the blanket `CompletionClient` implementation over
`Client<Ext, H>` then supplies `completion_model` for you.

`CompletionModel` also no longer requires `Clone` — the trait demands only
async service behavior, in the spirit of `tower::Service`; cloning or sharing
a model is the caller's concern — and wrapping in an `Arc` genuinely works:
`CompletionModel` is implemented for `Arc<M>` by forwarding, so `Arc<M>`
passes through every generic API (`CompletionRequestBuilder`, agent
construction), and `completion_request` on an `Arc` clones the `Arc`, never
the model. Implementors can drop `Clone` derives they only carried for the
bound (keeping them is harmless). Generic code that cloned a model through
the trait must now bound `M: CompletionModel + Clone` explicitly or take the
model by value. The `completion_request` convenience gates on `Self: Clone`
individually; every built-in provider model, `Arc<M>`, and `ModelHandle`
satisfy it, so call sites on concrete types compile unchanged.

`CompletionResponse::finish_reason` is now a private field with a
`finish_reason()` getter: every write flows through `with_finish_reason` /
`with_optional_finish_reason`, so the `Stop` → `ToolCalls` reconciliation can
no longer be bypassed by direct assignment. Replace field reads with the
getter call.

The identifier and model setters on both `CompletionResponse` and
`StreamFinal` now treat an empty string as absent: gateways that echo `""`
produce `None`, matching the streaming paths, and the rule lives in the
setters rather than at provider call sites.

Both invariants also hold through `Deserialize`: the two types deserialize
via a wire-shape mirror that funnels through `new(...)` and the setters, so
a persisted `"finish_reason": "stop"` alongside a tool-call choice comes
back as `ToolCalls` and a persisted `""` identifier comes back as `None`.
The serialized wire format is unchanged.

Corrupt stream frames (payloads that are not valid JSON) are now surfaced as
`Err` items on the stream instead of being logged and silently skipped; the
stream keeps consuming, and a later genuine terminal still completes it.
Valid-JSON events whose shape this client doesn't recognize are still skipped
(with a warning) for forward compatibility with new provider event types.
Consumers that drained to `None` see the same content as before plus any
error items; consumers that stopped at the first `Err` should drain to
`None` — see the emission-contract table on `StreamFinal`.

### Provider behavior is reported through capabilities

`CompletionModel::composes_native_output_with_tools()` is replaced by
`CompletionModel::capabilities()`:

```rust
fn capabilities(&self) -> ProviderCapabilities {
    ProviderCapabilities::default().with_native_output_tool_composition(true)
}
```

`ProviderCapabilities` is public and `#[non_exhaustive]`; start from `Default`
or `ProviderCapabilities::new()` and enable what you support. Capabilities are
plain data, so a runtime can snapshot them instead of holding a callback into
the concrete model.

### Agents erase the model type at construction (runtime model swapping)

`Agent<M>`, `AgentBuilder<M>`, `AgentRunner<M>`, the prompt/stream request
types, and `Extractor<M, T>` lost their model parameter: `AgentBuilder::new`
takes any `CompletionModel + 'static` and erases it once into a concrete
`ModelHandle` (which itself implements `CompletionModel`). Update type
annotations by deleting the parameter — `Agent<openai::CompletionModel>`
becomes `Agent`; `Extractor<M, T>` becomes `Extractor<T>`. Construction call
sites are unchanged.

Because the stored model is a handle, it can now change at runtime:
`Agent::set_model`, per-run `runner(...).using_model(...)`, or an
`AgentHook::on_model_select` hook receiving `ModelSelection` (which sees the
merged `RequestPatch` and the previous model, and may pick a different handle
per model call). `CompletionModel::capabilities()` is captured by value when
the handle is created.

### Assistant content is tagged and provider extras are a named field

`AssistantContent` serializes with a `"type"` tag, exactly like `UserContent`
always has:

```json
{"type": "text",      "text": "hello"}
{"type": "toolcall",  "id": "call_1", "function": {"name": "add", "arguments": {}}}
{"type": "reasoning", "id": null, "content": [...]}
{"type": "image",     "data": ...}
```

The tag is **required** on deserialize, and there is no untagged fallback.
**This breaks released data**: 0.41 serialized assistant content untagged, so
a history or run persisted under 0.41 carries the bare shape
(`{"text": "hello"}`) and fails to load with ``missing field `type` `` (the
internally-tagged enum's wording — grep your logs for that, not for an
untagged-enum error). Migrate stored assistant blocks by inserting the tag:

- text block (`"text"` key) → add `"type": "text"`
- tool call (`"id"` + `"function"` keys) → add `"type": "toolcall"`
- reasoning (`"content"` list of reasoning blocks) → add `"type": "reasoning"`
- image (`"data"` key, assistant side) → add `"type": "image"`

The tag alone is not always enough: 0.41's flatten also wrote provider extras
as top-level siblings on **assistant** text (anthropic citations, raw server
tool content) and **assistant** images (openrouter response images always
carry an `"openrouter"` extras object). Under this release those keys are
**silently dropped on load** — an unknown key on a content block is ignored,
never captured, never an error — so re-nest them under `additional_params` in
the same pass, exactly as the user-content instructions below describe —
`{"text": "…", "citations": […]}` becomes `{"type": "text", "text": "…",
"additional_params": {"citations": […]}}`, and `{"data": …, "openrouter":
{…}}` becomes `{"type": "image", "data": …, "additional_params":
{"openrouter": {…}}}`. The missing *tag* is the loud migration signal; the
un-re-nested extras are the quiet one. Verify your migration with the
round-trip recipe at the end of this section.

`additional_params` on every content block (`Text`, `Image`, `Audio`, `Video`,
`Document`) is now a **named** field instead of a serde flatten, typed
`Option<message::AdditionalParams>` — a newtype that is a non-empty JSON
object *by construction* (build one with `AdditionalParams::from_entries`,
`::new`, or `::try_from_value`; read with `::get`; `Some` always carries
data). The wire shape is unchanged:

```json
{"type": "text", "text": "", "additional_params": {"citations": [...]}}
```

Two defect classes die with the flatten: a stray key can no longer be
silently captured into `additional_params` and replayed to providers, and an
absent field round-trips as `None` instead of the flatten's `Some({})`
artifact — so turn-emptiness classification is identical before and after a
persist/restore. The unknown-key policy is now **uniform and tolerant**
across every content block (the five structs above plus `ToolCall` and
`Reasoning`): a known field with the wrong shape is a loud decode error, an
unknown key is ignored, and an unknown content-block *tag* is a loud error.
Histories written by a newer rig therefore stay loadable by this one. The
params remain provider-specific: a serializer replays only params it
recognizes as its own wire's. The 0.41 helper family
(`non_empty_params`, `params_carry_data`) is gone — the newtype carries the
whole contract, and plain `is_none()`/`is_some()` are always correct.

**Released *user*-content blocks need the same re-nesting.** 0.41's flatten
wrote provider extras at the block's top level — e.g. an Anthropic document
block serialized as `{"type": "document", "data": …, "title": "t",
"citations": …}`. Those keys load silently dropped; re-nest every non-schema
key under `additional_params`: `{"type": "document", "data": …,
"additional_params": {"title": "t", "citations": …}}`. Cover the nested
blocks too: a tool result's content list (`UserContent::ToolResult` →
`ToolResultContent::Text`/`Image`) reuses these same structs, so flattened
extras inside a persisted tool result drop the same way and need the same
re-nesting. (An empty `"additional_params": {}` or `null` is fine — it
canonicalizes to absent on load. Any other non-object value is a decode
error: extras are a keyed namespace, so a re-nesting script that writes
`"additional_params": []` or a bare string fails loudly instead of loading
as an annotation no extractor can read.)

**Verify the migration** with the round-trip recipe: load each persisted
message tolerantly, re-serialize it, and ask
[`message::keys_lost_in_round_trip`] for every key that did not survive —
an empty result means the history migrated whole. (The canonical copy of
this snippet is the compiled doc example on `keys_lost_in_round_trip`
itself, so the recipe and the behavior cannot drift.)

```rust
let loaded: message::Message = serde_json::from_value(original.clone())?;
let round_tripped = serde_json::to_value(&loaded)?;
let lost = message::keys_lost_in_round_trip(&original, &round_tripped);
assert!(lost.is_empty(), "keys dropped by tolerant load: {lost:?}");
```

Run it once over your store at migration time; the runtime load path stays
tolerant on purpose (loudness at the migration boundary, not on every
restore).

**Streaming events**: `StreamedAssistantContent` is a tolerant decode with an
`Unknown` catch-all. A stream item whose text block carries stray sibling
keys — 0.41's flatten shape, or a relay stamping bookkeeping keys onto text
items — decodes as stream *text* with the stray keys dropped: the text is
assembled, nothing is excluded. A replayed **tagged** assistant block
(`toolcall`/`reasoning`/`image` — the tagged `AssistantContent`
serialization is not a stream-item shape) decodes as `Unknown` and is
excluded from assembly, as is a text item whose `additional_params` is
malformed (a non-object — the strict decode rejects the known field); the
**agent assembler** — the one rig component that ingests replayed stream
events — counts both kinds of exclusion and logs a single `tracing` warning
per turn, on every termination path
(`StreamedTurnAssembler::excluded_assistant_content` exposes the count). A consumer assembling self-deserialized events with its
own logic gets no warning and should apply the same check itself.

### Two pre-`Vec` serde accommodations are gone

Backwards compatibility with data persisted before this release is no longer
carried:

- `PromptResponse` JSON serialized **before the `content` field existed** no
  longer deserializes — `content` is a required field. (On self-describing
  formats like JSON the serialized shape is unchanged; a non-self-describing
  format sees `content` as a bare list where the old shadow repr encoded an
  `Option`.) This reaches further than standalone response values: `AgentRun`
  embeds a `PromptResponse` in its `Done` state, so a **persisted run** that
  reached `Done` before the field existed fails to load too — migrate stored
  runs (add `"content": [{"type": "text", "text": <output>}]` to the embedded
  response) before upgrading. Assistant content is tagged with `"type"` in
  this release, exactly like user content — see "Assistant content is tagged
  and provider extras are a named field" below.
- Pre-provider-split `ToolCall` JSON is no longer migrated on load — see the
  "Persisted histories" bullet in the tool-call identity section above for
  what a legacy `call_id` key now means and how to migrate the JSON by hand.

### Errors carry the transport request id; two error-shape changes (#2314)

Failed calls now preserve the provider's transport request id. Two breaks:

- **`http_client::Error` gains the `InvalidStatusCodeWithDetails { status, body, headers }` variant** (the reqwest transport now reports non-success through it, preserving the failed response's headers). Exhaustive matches on `http_client::Error` need a new arm; its Display is identical to `InvalidStatusCodeWithMessage`, and `provider_response_status()`/`provider_response_body()` read both.
- **`ProviderResponseError` is `#[non_exhaustive]`** with a new `provider_request_id` field. Construct via `ProviderResponseError::new(status, body)` / `::without_status(body)` instead of a struct literal; read the id via the `provider_request_id()` accessor on the error enums (also forwarded through `PromptError`).

Behavior: providers with a request-id contract (anthropic, openai, xai, groq, copilot) classify non-success HTTP responses as `CompletionError::ProviderResponse` instead of `HttpError`. Matchers on `CompletionError::HttpError(_)` for those providers' 4xx/5xx need updating; the `provider_response_*` accessors are shape-independent and keep working. Contract-less providers are unchanged.

### Response identity metadata reaches agent observers (#2265)

Completed model calls now report a `rig_core::completion::ResponseIdentity`
(message-scoped id, response-scoped id, and the provider's transport request
id) on hook events and `PromptResponse.completion_calls`. Three source-level
breaks come with it:

- **`CompletionCall` is no longer `Copy`** — it carries owned identity
  strings. Replace `.copied()` with `.cloned()` (or borrow):

  ```rust
  // Was
  let last = response.completion_calls().last().copied();
  // Now
  let last = response.completion_calls().last().cloned();
  ```

  Persisted call records are unaffected: the new `message_id`, `response_id`,
  and `provider_request_id` fields are serde-defaulted, so pre-identity JSON
  still loads (with each field `None`).

- **`AgentRun::record_streamed_completion_call` takes the attempt's identity
  as a second argument.** Hand-driven streaming drivers pass the identity
  read from their stream's terminal record; pass
  `ResponseIdentity::default()` when the provider reported none:

  ```rust
  // Was
  run.record_streamed_completion_call(usage)?;
  // Now
  run.record_streamed_completion_call(usage, ResponseIdentity {
      message_id: stream.message_id.clone(),
      response_id: terminal.and_then(|t| t.response_id.clone()),
      provider_request_id: terminal.and_then(|t| t.provider_request_id.clone()),
  })?;
  ```

- **The `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished`
  hook events gain an `identity: &ResponseIdentity` field.** Hooks that only
  *read* events are unaffected; code constructing these events by hand (test
  harnesses) must supply the field — `&ResponseIdentity::default()` preserves
  the old no-identity behavior. `ModelTurnFinished` now carries identity for
  every accepted turn on both surfaces, so an observer of that one event
  records identity for every completed call.

### The terminal finish reason reaches the caller, and empty truncated turns error (#2322)

The streamed assembler used to discard the provider's finish reason, so a turn
cut short at the output-token limit was undetectable through the agent surface
and finalized as a successful empty string. Four source-level breaks:

- **`AgentRun::record_streamed_completion_call` takes the finish reason as a
  third argument.** Pass `None` when the provider reported none:

  ```rust
  // Was
  run.record_streamed_completion_call(usage, identity)?;
  // Now — from your stream's terminal record
  run.record_streamed_completion_call(usage, identity, terminal.and_then(|t| t.finish_reason.clone()))?;
  ```

- **`StreamedTurnEvent::Completed` gains a `finish_reason` field.** Drivers
  matching it exhaustively must bind or ignore it (`..` keeps working).

- **`ModelTurn` and `StreamedTurn` gain `finish_reason`.** Both are
  `#[non_exhaustive]`, so `ModelTurn::new(..)` is unchanged; attach the reason
  with `.with_finish_reason(resp.finish_reason())`. A `StreamedTurn` built as a
  struct literal needs the new field. It is serde-defaulted, so persisted run
  JSON still loads.

- **A turn that delivered no answer and reports `Length` or `ContentFilter`
  now fails** with a `CompletionError::ResponseError` instead of finalizing as
  `""`. "No answer" means no tool call and no non-empty text — **reasoning
  does not count**, which is the case most likely to affect you: providers
  bill thinking tokens against the output limit, so a thinking model that
  exhausts its budget mid-thought produces reasoning and no text, and that
  shape used to report success with an empty string. This matches what the
  blocking Gemini path already did for a content-less candidate.

  Unchanged: partial output followed by truncation is still a valid answer
  (read the reason from `PromptResponse::completion_calls`), any turn
  reporting `Stop`/`ToolCalls` still finalizes whatever its shape, and
  `FinishReason::Other` is not treated as truncation. The turn is still
  recorded to history before the error, so partial reasoning remains available
  for debugging. If you were relying on an empty string from a truncated turn,
  handle the error — or inspect `completion_calls` and raise `max_tokens`.

  **This also narrows `OutputMode::Tool` recovery.** An agent with an output
  tool that received an answerless turn used to consume an output-retry and
  re-prompt with corrective feedback. When that turn reports `Length` or
  `ContentFilter` it now fails immediately instead. The re-prompt could not
  have helped — the budget or the filter, not the phrasing, is what stopped
  the turn, so the retry would truncate again and report a less specific
  failure at the end. Re-prompting is unchanged for answerless turns with any
  other finish reason.

  `CompletionCall.finish_reason` is serde-defaulted; pre-#2322 run JSON loads
  with it `None`.

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

An `Agent`'s default model is set at construction. Per-run overrides now go
through `runner(...).using_model(...)`, `Agent::set_model`, or a
`ModelSelection` hook (see the "runtime model swapping" section for the
current release).

`Extractor` now routes through the full hook lifecycle.

### 7. `dynamic_context` is back, but it is a hook now

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

`MultiTurnStreamItem::final_response` takes
`OneOrMany<AssistantContent>` where it took `&str`:

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

### 2. Required builder fields are enforced by types

Builders that used to accept a field and fail at `build()` now encode
required-ness in the type (#1611), using the `Missing` / `Provided<T>` markers:

| Builder | Required field(s) |
| --- | --- |
| `VectorSearchRequestBuilder` | `query`, `samples` |
| `TranscriptionRequestBuilder` | `data` |
| `ImageGenerationRequestBuilder` | `prompt` |
| `AudioGenerationRequestBuilder` | `text`, `voice` |
| `ChatBotBuilder` | the chatbot impl |
| `ClientBuilder` | `api_key` (`NeedsApiKey` is now `Missing`) |

Two consequences. `VectorSearchRequestBuilder::build()` returns
`VectorSearchRequest<F>` instead of `Result<_, VectorStoreError>` — drop the `?`.
And `TranscriptionRequestBuilder::load_file` returns
`io::Result<TranscriptionRequestBuilder<M, Provided<Vec<u8>>>>`, so it needs a
`?` where it previously did not.

Naming these builder types explicitly requires the marker parameters; chained
builder expressions need no change.

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
| `rig_core::OneOrMany<T>` (and the `one_or_many` module, both prelude re-exports) | `Vec<T>` — no replacement type; see the conversion table in "0.41 → next" | next |
| `rig_core::EmptyListError` | none — use `message::require_non_empty` where you relied on the rejection | next |
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
| `MultiTurnStreamItem::final_response(&str, ..)` | `final_response(OneOrMany<AssistantContent>, ..)`; if you are skipping straight past 0.41, `OneOrMany` is itself removed in the next release — go directly to `Vec<AssistantContent>` | 0.38 |
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
