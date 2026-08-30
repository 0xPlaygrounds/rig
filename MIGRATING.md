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

#### `AgentRun::advertised_tools()` now carries the definitions the request sent

rig-agent's driver recorded the advertised tools *after* the per-turn request
assembly had already moved the definitions out of the registry snapshot, so
`AgentRun::advertised_tools()` (and the serialized run's `turn_tools`) always
held an empty list. With request preparation in `rig_run::prepare_request`
the driver advertises `PreparedRequest::tools` — the executable tools after
any `active_tools` allow-list plus, in Tool output mode, the synthetic output
tool — i.e. exactly what the provider received. Nothing on the wire changes.

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
  yours — the loop never rewrites history **you** supply. The serializers do,
  where their wire cannot take the block: Anthropic and OpenAI Responses both
  drop an empty assistant text block that carries no extras for that wire
  before the request is built, so it never reaches the API, and on Anthropic
  a turn left with no blocks at all then fails locally with `Assistant
  message did not contain Anthropic-compatible content` rather than as a wire
  400. If you carry pre-`Vec` histories with the fabricated empty-text part,
  drop those turns yourself before replaying.
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

#### Cohere rejects `ToolChoice::Specific` before the request is built

The 0.41 conversion passed every non-`Auto` choice straight through, so
`ToolChoice::Specific { function_names }` was serialized externally-tagged as
`{"specific":{"function_names":[…]}}` and sent to `/v2/chat`, which accepts only
the bare strings `"REQUIRED"` and `"NONE"`. It now fails locally:

```text
the Cohere API cannot be forced to call specific tools by name;
use ToolChoice::Required and restrict the tools you pass instead
```

The remedy is in the message. Pass only the tools you want the model to be able
to call and set `ToolChoice::Required`; the request never leaves the process
otherwise, so the failure arrives as `CompletionError::RequestError` instead of
a provider response.

#### Gemini now honors `temperature` and `max_tokens`

`create_request_body` applied both fields through `generation_config.map(..)`,
and `Option::map` is a no-op on `None` — so unless the request already carried a
`generationConfig` (supplied through `additional_params`, or built by an
`output_schema` turn), both values were dropped and the body went out with
`"generationConfig": null`. The blocking and streaming surfaces share that
builder, so neither ever sent them. The config is now created whenever either
field is set: `.temperature(..)` reaches `generationConfig.temperature` and
`.max_tokens(n)` reaches `generationConfig.maxOutputTokens`.

Nothing to change — but if you set either value on a Gemini agent at any point
and moved on when it appeared to do nothing, **it starts applying now**.
Generations that had been running to Gemini's own output limit truncate at the
budget you configured and report `FinishReason::Length`, and sampling you set
long ago takes effect. Only the field you set is sent: `GenerationConfig`'s
`Default` is all-`None` and every field is `skip_serializing_if`, so setting one
does not silently acquire the other.

#### Gemini stops injecting `maxOutputTokens: 4096` and `temperature: 1.0`

`gemini::completion::GenerationConfig`'s hand-written `Default` set
`temperature: Some(1.0)` and `max_output_tokens: Some(4096)`. It is derived now,
so a default config is all-`None` and every field is
`skip_serializing_if = "Option::is_none"` — a default config puts nothing on the
wire and Gemini applies each model's own documented limit (65,536 output tokens
for `gemini-2.5-flash`, ~16x the old cap).

Two request paths seeded themselves from that `Default` and therefore change
shape with no compile error:

- **native structured output** — the `output_schema` arm of
  `create_request_body`, i.e. every `output_schema` turn;
- **image generation** — `gemini::image_generation`'s `..Default::default()`
  request literal.

Both used to ship a 4096-token cap and a pinned temperature no caller asked for,
so a 16k-token structured-output request was truncated at 4096 — and because a
`MAX_TOKENS` turn with no content finalized as a successful empty string on the
streaming agent surface, the truncation surfaced there as an unexplained empty
response. The blocking path failed instead — `CompletionError::ResponseError`
carrying "Gemini candidate missing content (finish_reason=MaxTokens, …)" — so
the cap was at least named there. Only callers who relied on the model default
were affected: an explicit `max_tokens` was applied afterwards and overwrote the
injected value. To keep the 0.41 wire values, set them yourself with
`.temperature(1.0).max_tokens(4096)` on the agent builder. One further
consequence: passing `GenerationConfig::default()` through `additional_params`
now serializes to `{}` instead of those two fields.

#### A Gemini stream that fails after a `finishReason` no longer reports a completed turn

`streamGenerateContent` sends an *intermediate* `finishReason` when a built-in
tool (code execution) runs a round and then keeps streaming, and the REST
adapter treated the first one as the provider completing the turn — the shared
driver stops reading as soon as a terminal record appears, so the model's whole
answer after that chunk was dropped while the stream still reported a clean
`STOP`. The terminal record is held until EOF now.

What changes for you: a stream whose *transport* fails after a `finishReason`
has been seen surfaces that error and yields **no** terminal record, where it
previously finished cleanly with one. On this wire a `finishReason` is not proof
the turn ended, so a record can no longer be treated as guaranteed once a finish
reason has gone by — handle the error case in code that reads the raw stream.
Also changed: EOF with no `finishReason` at all is treated as truncation and
yields no terminal record either, where 0.41 closed such a stream with a clean
record carrying `finish_reason: None` and whatever usage had gone by — this is
the workspace-wide [terminal-record
rule](#ordinary-streaming-types-no-longer-carry-a-response-parameter) reaching
Gemini. Unchanged: an in-band tool-protocol failure still short-circuits the
stream without a terminal record. The record you do get now carries the *last*
reason, usage and metadata the stream reported rather than the first.

#### Gemini listing failures are `ApiError`, not `RequestError`

`gemini::model_listing`'s lister built its own request and let the transport
failure convert straight through `From<http_client::Error>` into
`ModelListingError::RequestError`. Because the reqwest transport reports a
non-2xx *as* an error before handing back a response, that was the path every
real listing failure took, and Gemini's own status-check branch never ran. The
listing now goes through the shared paginated fetch, which classifies a
non-success response as `ModelListingError::ApiError { status_code, message }`,
with the provider label, request path, status and a body preview in `message`.

A `match` arm on `ModelListingError::RequestError` still compiles and simply
stops firing for Gemini. Move that handling to `ApiError`, which now carries the
status you were reconstructing from the message text. `RequestError` still means
what it always did — the request never got an HTTP response at all.

#### OpenRouter now honors `max_tokens`

OpenRouter's request builder hardcoded `max_tokens: None`, so the caller's
value was dropped before the body was built and never reached the API, which
accepts the field exactly as OpenAI's Chat Completions does. It is now sent.

Nothing to change — but if you set `max_tokens` on an OpenRouter agent at any
point and moved on when it appeared to do nothing, **it starts applying now**,
and generations that previously ran to the model's own output limit will be cut
off at your value.

#### `additional_params.tools` merges with the builder's tools on Chat Completions

`additional_params` is flattened into the Chat Completions request *after* the
typed `tools` field, and the body is built with `serde_json::to_value` — so a
raw `tools` array left in the params used to overwrite every tool the agent
builder had registered. A turn with `tool_choice: "required"` and a builder tool
went out carrying only the params tool.

Entries shaped as function tools (`{"type": "function", …}`) now merge onto the
typed list, builder tools first; anything else stays in `additional_params` for
the provider's `prepare_request` hook (Groq folds its native tools into
`compound_custom` from there).

If you used `additional_params.tools` as an *override*, the builder's tools now
go out **too**, and a forced `tool_choice` may select one of them — drop the
tools from the builder instead of shadowing them. A malformed payload (`tools`
that is not an array, or a `"function"` entry that is not a valid tool
definition) is now a local `CompletionError::RequestError` naming the key rather
than something the provider rejects. Only Chat Completions changes: the
Responses, Anthropic and Gemini paths already merged, and OpenRouter builds its
own request.

#### OpenAI image generation honors `additional_params` and no longer sends `response_format`

Two changes to the `/v1/images/generations` body, neither of which produces a
compile error.

`ImageGenerationRequestBuilder::additional_params` was silently dropped for
OpenAI (xAI and Gemini already merged it). It is now merged last, so it can also
override `model`, `prompt` and `size`. Nothing to change — but parameters you
set once and moved on from when they appeared to do nothing **start applying
now**, and a value the endpoint rejects now fails the request instead of being
ignored. Check what you are passing.

Rig also stopped adding `"response_format": "b64_json"` for models outside a
hardcoded `gpt-image-1`/`1.5`/`2` allowlist: the endpoint now rejects that field
for every model (`400 Unknown parameter: 'response_format'`), which is why
`gpt-image-1-mini`, `chatgpt-image-latest` and dated snapshots could not generate
an image at all. If you point the OpenAI client at an OpenAI-*compatible* images
endpoint that only returns base64 when asked, rig no longer asks — and the
response type requires it (`ImageGenerationData { pub b64_json: String }`), so a
URL response fails to deserialize. Pass `response_format` yourself through
`additional_params`, which now reaches the body. Azure OpenAI's image body is
untouched: it still hardcodes `response_format` and still drops
`additional_params`.

#### The shared text-to-speech body honors `additional_params`

The default text-to-speech request body never merged
`AudioGenerationRequest::additional_params`, so the field was inert for every
provider that inherited it — OpenAI included. Providers that override the body
(xAI, OpenRouter, Venice) already merged it and are unchanged; Azure OpenAI
overrides it and still drops the field.

Nothing to change — but the parameters demonstrably change the response, so if
you set them on an OpenAI TTS request and moved on, **they start applying now**:
`response_format: "wav"` returns a RIFF payload where the default returned MP3,
and `instructions` steers delivery on the `gpt-4o-mini-tts` family. Downstream
code that assumed the returned bytes were MP3 should be checked.

#### Mistral requests carry their attachments, and content Mistral cannot carry now fails

Mistral's request finalizer used to flatten every message's `content` with the
text-only helper, which keeps only parts carrying a `text`/`refusal` key — so an
attached image, audio clip or document was removed from the request and the
model answered a prompt it never saw. Mistral's own content chunks are emitted
for real now: images and audio as `image_url` and `input_audio`, documents as
`document_url` (inline base64, filename in `document_name`) or Mistral's `file`
chunk. Two consequences, neither of which fails to compile:

- **A prompt with an attachment can now be rejected.** The attachment reaches
  the API, so a model without the matching capability surfaces Mistral's own
  error where it previously returned a plausible completion built from the text
  alone. Send such a request to a model that accepts the modality — the
  deprecation notes on the retired Pixtral constants name `MISTRAL_SMALL` and
  `MINISTRAL_3B` as the vision-capable replacements (`PIXTRAL_LARGE`'s note
  also names a `MISTRAL_MEDIUM`, which the crate does not define — there is no
  such constant to reach for) — or drop the attachment.
- **Content Mistral has no chunk for fails before the request is built.** A rig
  `Video` part (it serializes as OpenAI's `video_url`), and any part type a
  future conversion adds, now returns `CompletionError::RequestError` wrapping
  `message::MessageError::ConversionError` — "Mistral cannot carry …" — where
  the part used to be dropped silently and the rest of the turn sent. Convert
  such content to text, an image, audio or a document first.

Text-only content is unaffected: it still flattens to Mistral's plain string,
and a malformed text part carrying no string payload is still dropped exactly
as it always was.

#### An empty Anthropic `stop_sequence` turn succeeds instead of erroring

When the matched stop sequence is the first thing the model emits, Anthropic
strips it and answers `200` with `content: []`. The empty-content carve-out in
Anthropic's `normalize` covered only `end_turn`, so that turn became
`CompletionError::ResponseError("Response contained no message or tool call
(empty)")` and its usage, message id, transport request id and finish reason
went with it — while the streamed twin of the same request already finished
cleanly with an empty choice.

`stop_sequence` now joins `end_turn` as a legal empty terminal, so such a call
returns `Ok` with `choice == []` where 0.41 returned `Err`. Code that treated
that error as "the provider misbehaved" should check `choice.is_empty()` and
read `stop_sequence` off the response instead. The carve-out is narrow: it
applies only when the response also *names* the sequence that fired
(`response.stop_sequence.is_some()`). A response claiming `stop_sequence` while
naming none — the shape an Anthropic-compatible gateway is likeliest to send —
is still an error, and every other empty response is still guarded.

#### A missing or null `finish_reason` decodes instead of erroring

`openai::completion::Choice` declared `index: usize` and `finish_reason: String`
with no serde attributes, so an OpenAI-compatible response that omitted either
field — or sent an explicit `null` — failed to deserialize and the entire turn
came back as an error. Both now read through `json_utils::null_or_default`, as
do `CompletionResponse::object` and `::created` (previously `#[serde(default)]`,
which tolerated a missing key but not a `null`).

A response that used to be a decode error now succeeds. `index` falls back to
`0`, and the empty `finish_reason` normalizes to absent, so
`CompletionResponse::finish_reason()` is `None` where a turn that decoded at all
previously always had a reason. This reaches every OpenAI-compatible provider
that decodes through the shared `openai::completion` types — Copilot's
multi-vendor chat route is the wire that sends these shapes. Providers that keep
their own copies of the response types are untouched and still reject a missing
or null field: `mistral::completion::Choice` and `deepseek::Choice` both declare
a bare `index: usize` / `finish_reason: String`, and
`mistral::completion::CompletionResponse` a bare `object` / `created` pair with
no `default`. If you branch on the finish reason, read `None` as "the provider
did not say" rather than as a normal stop.

#### `EmbeddingsBuilder` results are ordered, and two of its errors changed

`build` / `build_with_usage` return their `(document, embeddings)` pairs in the
order the documents were added, and each document's embeddings in the order its
`Embed` impl produced the texts. Neither held before. Documents were merged out
of a `HashMap<usize, T>`, so the pair sequence came back in arbitrary hash
order; and concurrent batches appended to a per-document list as they finished,
so a document whose texts straddled a `MAX_DOCUMENTS` batch boundary got its own
embeddings shuffled — with a batch cap of 5, a six-text document could come back
as `[t5, t0, t1, t2, t3, t4]`.

Nothing to change, and the workarounds can go: a `sort_by` over the result, or
an id lookup used because positional zipping was unreliable.
`InMemoryVectorStore::add_documents`, which mints `doc{n}` ids from this
sequence's position, now gives the same document the same id on every run — a
store whose ids were assigned by 0.41 or earlier is not reproducible from the
same inputs, so re-index rather than assume the ids still line up.

Two errors on this path changed with it. A document that embedded no text still
fails the whole build, but its `EmbeddingError::ResponseError` message moves
from `missing embedding for document after batch merge` to `document {index}
produced no text to embed, …`; match the variant, not the string. And a provider
that answers a batch with fewer embeddings than the texts it was sent is now
that same error, naming the document and its slot range, where it previously
handed back a silently short list.

---

## 0.41 → next

### Built-in providers are opt-in Cargo features

No concrete provider is compiled unless you name it. `rig`'s default feature
set is unchanged in every other respect, but it no longer pulls in any
provider, so existing code fails to compile on upgrade:

```
error[E0433]: failed to resolve: could not find `openai` in `providers`
```

Add the feature for each provider module you use:

```toml
rig = { version = "0.42", features = ["openai", "anthropic"] }
```

The features are `anthropic`, `azure`, `chatgpt`, `cohere`, `copilot`,
`deepseek`, `doubleword`, `gemini`, `groq`, `huggingface`, `hyperbolic`,
`llamacpp`, `minimax`, `mira`, `mistral`, `moonshot`, `ollama`, `openai`,
`openrouter`, `perplexity`, `together`, `venice`, `voyageai`, `xai`,
`xiaomimimo`, and `zai` — the same name on `rig`, `rig-core` and
`rig-reqwest`, so a direct `rig-core` dependency needs the same addition.
`providers-all` is an explicit aggregate for documentation, CI, and
applications that genuinely need every provider; prefer naming the two or
three you use, since that is the point of the change.

Paths, types, and behavior are otherwise identical once the feature is on,
with one rename below. The capability features (`image`, `audio`) and the
companion crates are unaffected — a companion crate enables whatever provider
it needs itself.

`mistral::completion::MistralStreamingCompletionResponse` is now
`mistral::completion::StreamingCompletionResponse`, the name every other
provider uses for the same thing. It is the same type; only the name changed.

The OpenAI- and Anthropic-compatible provider families share one protocol
implementation, reachable as `providers::openai_compatible` and
`providers::anthropic_compatible`. Both are `#[doc(hidden)]` and **not
public API**: they exist so a build that enables, say, only `groq` still has
the shared tree compiled, and their contents may move in any release. Name
the wire types through the provider you use — `groq::CompletionResponse`,
`azure::TranscriptionResponse`, `minimax::AnthropicCompletionResponse` —
rather than through the shared path.

### Typed ids: `InternalCallId` is a counter, `ConversationId` is a newtype

Two identifiers that were bare `String`s are now dedicated types in
`rig_core::id`, alongside `RunId`:

- **`InternalCallId`** — the rig-generated correlator tying one tool call's
  stream items together (argument deltas → completed call → execution →
  result) — is now a `Copy + Hash + Ord` `NonZeroU64` counter id with
  transparent serde, minted by the streaming assembler (it was a 21-char
  nanoid `String`). It changes type everywhere it appears:
  `StreamedAssistantContent::{ToolCall, ToolCallDelta}`,
  `StreamedUserContent::ToolResult`, `RawStreamingToolCall`,
  `MultiTurnStreamItem::ToolExecutionCommitted`,
  `rig_run::PendingToolCall::internal_call_id`,
  `rig_run::InvalidToolCallContext`, and the hook events
  (`ToolCall`/`ToolResultEvent`/`ToolCallDelta`, where it was `&'a str` and is
  now a by-value `InternalCallId`). Code that compared it to string literals
  should compare ids; code that displayed it still can (`Display` renders the
  decimal). **Serialized stream items and persisted runs now carry it as a
  number** — run-state has no cross-version stability contract, so re-persist
  runs with this release. Persisted ids loaded by a new process advance the
  mint counter (`InternalCallId::advance_past`, wired into
  `PendingToolCall`'s deserialize) so fresh mints cannot collide with ids
  consumers already saw.
- **`ConversationId`** — the key scoping `ConversationMemory` — is a
  string-backed `Hash + Eq` newtype with transparent serde (`From<&str>`,
  `From<String>`, `Display`, `as_str`). The `ConversationMemory` trait
  methods, the rig-memory adapters (`on_demote`, `compact`, `forget`),
  `InMemoryConversationMemory`'s keys, and the `conversation(..)` setters on
  `AgentBuilder`/`AgentRunner`/prompt requests all use it; the setters take
  `impl Into<ConversationId>`, so call sites passing strings compile
  unchanged — only `ConversationMemory` implementors and callers of `load`/
  `append`/`clear` with `&str` must wrap (`&"thread-1".into()`).

### The protocol surface is fully data; owned-future entry points

Groundwork for stepping `AgentRun` from a host scheduler (the upcoming Bevy
plugin), all additive or derive-only:

- `rig_run::ModelTurn::from_response(resp, &PreparedRequest)` (and
  `from_response_parts`) is now the one blessed `CompletionResponse →
  ModelTurn` conversion; rig-agent's runner uses it, and any external driver
  must too — it settles the two inputs hand-assembly gets wrong (tool-name
  sets from the *prepared request*; the normalized `finish_reason()`
  accessor).
- `AgentRunStep`, `PreparedRequest`, `RequestPatch`,
  `InvalidToolCallContext`, `RetryRequest`, `InvalidToolCallAction`, and
  `StreamedTurnEvent` derive `Serialize + Deserialize`; `ModelTurnOutcome` is
  additionally `Clone`. `StreamedTurnAssembler` is now `Clone + Serialize +
  Deserialize`, so a mid-stream streamed turn persists and resumes like a
  blocking `AgentRun` (same no-cross-version caveat).
- `ToolCatalog::execute_owned(self, name, args, context) → (ToolResult,
  ToolContext)` and `ConversationMemoryExt::{load_owned, append_owned,
  clear_owned}` (blanket-implemented, `Arc<Self>` + owned `ConversationId`)
  provide `Send + 'static` futures for hosts that spawn tasks; the borrowed
  methods remain the implementation surface.

### The `discord-bot` feature is gone

`rig`'s `discord-bot` feature, rig-agent's `discord-bot` feature, and
`rig_agent::integrations::discord_bot` (`DiscordExt`, `DiscordBotError`) are
removed. `serenity` 0.12.5 is the newest published release and pins `rustls`
0.22, whose `rustls-webpki` 0.102.8 carries four unpatched advisories — a
reachable CRL-parsing panic plus three name-constraint/CRL-authority
weaknesses — with no version to bump to. A demo-grade integration is not worth
putting that in the dependency graph of everyone who builds the facade with
`--all-features`.

The code lived on briefly as `examples/discord_bot` and is now removed
outright, so `serenity` is out of the repository entirely. If you were using
`DiscordExt`, recover it from git history (`git show
7fd476a15:examples/discord_bot/src/discord_bot.rs`) and depend on `serenity`
directly — it is ~230 lines over the public `Agent` API and needs nothing
internal.

### Run lifecycle hooks and transport middleware

The agent hook surface gained a pre-run and a terminal event, and the erased
HTTP transport gained a middleware seam. What breaks:

- **`StepEventKind` gained `RunStart` and `RunSettled`.** Exhaustive `match`es
  over `StepEventKind` (e.g. in an `AgentHook::observes` implementation) must
  add the two arms. `AgentHook` itself is unaffected: the new `on_run_start`
  and `on_run_settled` methods are default-implemented.
- **`BoxedHttpClient` now applies attached `HttpMiddleware`** (new, opt-in via
  `BoxedHttpClient::with_middleware`). Behavior without middleware is
  unchanged. `BoxedHttpClient::ptr_eq` still compares the underlying transport
  only, so two handles differing only in middleware compare equal.
- **`AgentRun` gained an append-only entry log** — `rig_run::RunEntry`
  (`kind`/`turn`/`value`) with `append_entry`, `entries`, `entries_of`, and
  `last_entry_of` — plus `initial_prompt`, `rewrite_initial_prompt`, and
  `input_chat_history`. Runs serialized before this release deserialize
  unchanged (the field defaults to empty). Entries are protocol data like
  `TurnTools`: never interpreted by the run and never part of a provider
  request.

New, non-breaking: durable hook state is **event-sourced**. Hooks persist by
appending entries to the run's record via `HookContext::append_entry` (stamped
with the current turn, flushed into the `AgentRun` at each step boundary) and
reconstruct by replaying `HookContext::entries(kind)` — the documented default
pattern is snapshot + last-wins via `HookContext::last_entry`. State that
rides the record travels, rewinds, and forks with the record. Also new: the
`RunStart`/`RunStartAction`/`RunSettled`/`SettledOutcome` hook vocabulary. A
`RunStartAction::Stop` terminates the run before any provider call with
`PromptError::PromptCancelled`. (A `Scratchpad::put_durable` /
`ScratchpadSnapshot` / `AgentRun::hook_state` snapshot API existed briefly
between two unreleased PRs and was replaced by the entry log before release;
`Scratchpad` remains as the in-process, non-serialized cross-hook channel.)

### The run protocol is its own crate: `rig-run` (`rig::run`)

`AgentRun` — the sans-IO, serializable state machine behind every agent run —
and everything needed to step it now live in **`rig-run`**, which depends on
`rig-core` only (no async runtime, no hooks, no tool registry; a guard test pins
this). `rig-agent` is the futures driver over it; an ECS plugin can be another.
Every old path still resolves through re-exports, so existing code compiles
unchanged unless it names one of the items below:

- Moved (re-exported at the old paths `rig_agent::agent::run::*`,
  `rig_agent::agent::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome,
  PendingToolCall, OutputMode, PromptResponse, CompletionCall}`,
  `rig_agent::completion::PromptError`,
  `rig_agent::agent::hook::{InvalidToolCallAction, InvalidToolCallContext,
  RetryRequest}`): `AgentRun`, `AgentRunStep`, `ModelTurn`, `ModelTurnOutcome`,
  `PendingToolCall`, the streamed-turn assembler types, `OutputMode`,
  `PromptResponse`, `CompletionCall`, `PromptError`, the three invalid-call /
  retry data types, and the transcript helpers (`rig_run::transcript`).
  The facade exposes the crate as `rig::run` independent of the `agent` feature.
- `PromptError::prompt_cancelled(..)`, `PromptResponse::{with_output_tool_calls,
  output_tool_calls}` and the `AgentRun` driver methods
  (`set_output_tool_name`, `output_tool_name`, `accepted_turn_choice`,
  `ignore_invalid_tool_call`, …) were crate-private to rig-agent and are now
  public on rig-run: they are the protocol's driver API.
- `RunId` (`rig_core::id::RunId`, re-exported from `rig_run` and
  `rig_agent::agent::hook`) is now a `NonZeroU64` counter id — `Copy + Hash +
  Ord + Serialize`, `to_raw()`/`from_raw()`, decimal `Display`/`FromStr`,
  `Option<RunId>` is `u64`-sized — instead of an opaque `String` newtype.
  `RunId::as_str()` is gone (use `to_string()`); `HookContext::run_id()` returns
  it by value.

New, additive:

- **`RunSpec`** (`rig_run::RunSpec`): the protocol-facing half of an agent
  definition as plain `Serialize + Deserialize` data — preamble, static
  context, sampling params, additional params, turn budget, tool choice,
  structured-output policy. `AgentRun::from_spec(&spec, prompt, history)`;
  `Agent::run_spec()` reads it off an agent; `AgentBuilder::apply_spec(&spec)`
  layers one under imperative builder calls (model, tools, hooks, memory are
  untouched).
- **`AgentRun::advertise_tools(turn, Vec<ToolDefinition>)` /
  `advertised_tools() -> Option<&TurnTools>`**: what the request offered the
  model, recorded as run data (serialized with the run) so a resumed run or a
  second driver can re-pair returned calls with the advertised set. rig-agent's
  driver records it before every model call.
- **`rig_run::transcript::validate_canonical(&[Message])`** and
  **`AgentRun::with_validated_history(..)`**: the canonical-transcript rules the
  protocol produces (no consecutive assistant messages; every assistant tool
  call answered in the next message; no orphan tool results), as a checkable
  function for histories that come from outside (memory, a resumed run).
  `with_history` stays unchecked.

### The erased model and the erased tool set are rig-core; request preparation is rig-run

The second step of "one protocol, two drivers". Everything a driver that does
*not* depend on `rig-agent` needs is now in `rig-core` (handles, tools) and
`rig-run` (the pure request step). Every old path still resolves through
re-exports; behavior is unchanged (the recorded provider suites replay the same
request bodies, and a golden test pins the driver's requests for a scripted
tool turn). What moved, and what is new:

- **`ModelHandle`** (`rig_core::completion::ModelHandle`, was
  `rig_agent::agent::model::ModelHandle`; still at `rig_agent::ModelHandle`,
  `rig_agent::agent::ModelHandle`, the prelude): unchanged API, same private
  `ErasedModel`/`ModelDriver` shape, same `compile_fail` pins (not
  `Serialize`). `ModelHandle::named(label, model)` now takes
  `impl Into<ModelRef>` — `&str`/`String` still work — and `label()` still
  returns `Option<&str>`.
- **`ModelRef`** (new, `rig_core::completion::ModelRef`): the serializable
  string identity a `RunSpec`, an asset, or a registry names a model by
  (`Arc<str>`, transparent serde, `Deref<Target = str>`, `Display`,
  `From<&str>/From<String>`). `ModelHandle::model_ref()` reads the typed label.
- **The contextual tool API is rig-core's** (`rig_core::tool::{Tool,
  ToolEmbedding, ErasedTool, ErasedEmbeddingTool, DynamicTool, RegisteredTool,
  ToolDispatch, dispatch_tool, tool_definition, ToolSet}`, module
  `rig_core::tool::contextual`; re-exported unchanged at `rig_agent::tool::*`
  and, with or without the `agent` feature, at `rig::tool::*`). `Tool` and
  `ToolEmbedding` keep their blanket impls over `PortableTool` /
  `PortableToolEmbedding`. `#[rig_tool]` now expands contextual tools against
  `rig_core::tool::{Tool, ToolContext}` too, so a crate that depends on
  `rig-core` alone can author a `&mut ToolContext` tool — the "contextual tools
  require `rig`/`rig-agent`" macro error is gone, and a fully qualified
  `&mut rig_core::tool::ToolContext` is recognised without `#[rig(context)]`.
- **`ToolCatalog`** (new, `rig_core::tool::ToolCatalog`): the pinned,
  retrieval-free view of a tool set — provider definitions plus dispatch by
  name (`definitions`, `names`, `contains`, `execute`, `dispatch`,
  `retain_names`, `take_definitions`). Build one with `ToolSet::catalog()`
  (always-exposed tools, registration order) or
  `ToolCatalog::from_registered(IndexMap<String, RegisteredTool>)`.
  rig-agent's `ToolRegistrySnapshot` is now `pub type ToolRegistrySnapshot =
  ToolCatalog` — same methods, same name, so nothing to change. `ToolSet`
  gained the reads the registry used to get by reaching into its map: `names`,
  `len`, `is_empty`, `get`, `always_exposed_names`, `add_retrievable_tools`,
  `move_to_end`, `catalog`. `ToolServer` / `ToolServerHandle` (retrieval
  indexes, managed remote tool sources, `get_tool_defs(prompt)`, the per-turn
  snapshot) stay in `rig-agent`, layered over these types.
- **`rig_run::prepare_request`** (new): the pure `(RunSpec, ProviderCapabilities,
  history, tools, committed output tool, RequestPatch) -> PreparedRequest`
  step — preamble augmentation, static + extra context, output-mode resolution,
  synthetic output-tool synthesis and naming, `active_tools` narrowing,
  tool-choice validation — as owned data with `PreparedRequest::apply(builder)`
  to bind it to any `CompletionRequestBuilder<M>`. `PrepareError` carries the
  same local, pre-IO messages rig-agent raised before and converts into
  `CompletionError::RequestError`. rig-agent's driver now does only the IO
  around it: retrieve the turn's tools, `prepare_request`, bind the selected
  model's builder.
- **`RequestPatch`** (`rig_run::policy::RequestPatch`, was
  `rig_agent::agent::hook::RequestPatch`; still at the old path and
  `rig_agent::agent::RequestPatch`): plain per-turn data, unchanged fields and
  builder methods; `is_empty()` and `merge(later)` are now public. The hook
  that produces it (`CompletionCallAction::patch`) stays in rig-agent.

A driver over `rig-core` + `rig-run` alone can now erase a model, build a
`ToolSet`/`ToolCatalog` from `PortableDynamicTool`s, construct an `AgentRun`
from a `RunSpec`, `prepare_request`, and dispatch a tool by name — the guard
`tests/core/core_run_driver.rs` runs exactly that fixture and checks its
dependency graph has no `rig-agent`.

### `BoxedHttpClient`: an erased transport, and `Client<Ext>` now means `Client<Ext, BoxedHttpClient>`

`rig_core::http_client::BoxedHttpClient` wraps any `H: HttpClientExt + 'static`
behind one `Arc<dyn …>` and implements `HttpClientExt` itself, so a client can
hold a transport without naming it. `Clone` is a reference-count bump; boxing
an already boxed transport is a clone, not a second layer; `Debug` prints only
the type name. It is for hosts that *hold* one transport for many providers
(worker pools, ECS resources, registries built at startup) — keep the generic
`H` when writing a provider or when you want a monomorphized transport.

- `Client<Ext, H>`'s type default is now `H = BoxedHttpClient` (it was the
  `Missing` typestate placeholder). In **type** position `Client<Ext>` is
  "any transport"; nothing changes in expression position, where defaults never
  applied. The one break: `Client::<Ext>::builder()` no longer resolves —
  `builder()` lives on `Client<Ext, Missing>`, so spell it
  `Client::<Ext, Missing>::builder()` (or go through a provider's
  `ClientBuilder`/the rig-reqwest prelude as before).
- `ProviderFromEnv::from_env_boxed(http)` / `from_val_boxed(input, http)`
  return `Client<Self, BoxedHttpClient>`; `Client::boxed(self)` erases a built
  client's transport; `rig_reqwest::ReqwestClient::boxed()` /
  `impl From<ReqwestClient> for BoxedHttpClient`.

```rust
use rig::client::ProviderFromEnv as _;
use rig::http_client::{BoxedHttpClient, ReqwestClient};
use rig::providers::openai;

let transport: BoxedHttpClient = ReqwestClient::default().boxed();
let client: openai::Client<BoxedHttpClient> =
    openai::OpenAIResponsesExt::from_env_boxed(transport.clone())?;
// …and the same `transport` for every other provider the host talks to.
```

### Synchronous, retrieval-free registry reads: `ToolServerHandle::{snapshot, static_tool_defs, toolset}`; `ToolRegistrySnapshot` is public; `ToolSet: Clone`

Additive. The registry lock has been a `std::sync::RwLock` since 0.41 → next's
runtime-agnostic work, and registration was already synchronous; the read side
now is too. Pick the path by what you need:

- **The registry as it stands** (no dynamic-tool selection): the new sync
  `ToolServerHandle::snapshot() -> ToolRegistrySnapshot` and
  `static_tool_defs() -> Vec<ToolDefinition>` — plain `fn`s, no executor, safe
  to call from a frame/tick loop or a plain `#[test]`. Callers of
  `get_tool_defs(None)` can switch to `static_tool_defs()` and drop the
  `.await` (the async form stays and returns the same definitions).
- **Dynamic-tool selection for a prompt** (vector-store retrieval): the async
  `get_tool_defs(Some(prompt))`, unchanged.

`ToolRegistrySnapshot` — the per-turn pinned view the agent loop already uses —
is now public (`rig_agent::tool::server::ToolRegistrySnapshot`): `definitions()`,
`names()`, `len()`/`is_empty()`, and `execute(name, args, &mut ToolContext)`,
which runs the implementation pinned at snapshot time regardless of later
registry changes. `ToolServerHandle::toolset() -> ToolSet` forks the registry,
and `ToolSet` now derives `Clone` (shallow: shared `Arc` implementations, copied
names/order/exposure). All read paths retire disconnected remote tools first.

### `AgentRunner::run_channel` / `Agent::run_channel`: a future plus an event feed

Additive. Beside `run()` (fold to a `PromptResponse`) and `stream()` (a
`Stream` of `MultiTurnStreamItem`), the agent loop now has a third, runtime-
agnostic shape: `AgentRunner::run_channel(self)` (also on `Agent` and a
configured `StreamingPromptRequest`) returns `(impl Future<Output =
Result<PromptResponse, PromptError>>, RunEvents)`. Spawn the future on any
executor — tokio, `bevy_tasks`, a thread — and consume `RunEvents` wherever the
events are needed: it implements `Stream` and offers a non-blocking
`try_next()` / `is_done()` for tick-driven hosts. The feed is bounded
(`RUN_EVENTS_CAPACITY`, back-pressure rather than drops), and dropping it
does not cancel the run. `rig_agent::agent::{RunEvents, RUN_EVENTS_CAPACITY}`;
`RunEvents` is also in the rig-agent and `rig` preludes. See
`examples/agent_no_tokio` for a `bevy_tasks` host.

### MCP tool support moves from rig-agent's `rmcp` feature to the `rig-rmcp` crate

rig-agent no longer has an `rmcp` feature or an rmcp dependency; MCP lives in
the new **`rig-rmcp`** crate, which depends on **rig-core only**. The
dependency graph is `rig-core ← rig-rmcp` and `rig-core ← rig-agent`, with the
`rig` facade gluing them behind its `rmcp` feature. Through the facade,
`rig::tool::rmcp::*` (`McpClientHandler`, `McpTool`, `Meta`, …) keeps resolving
and `McpClientHandler::new(client_info, tool_server_handle.clone())` works
unchanged. What changes:

- **The registry contract is a rig-core abstraction.** `rig_core::tool::ManagedToolSink`
  (`add_managed_tools` / `reconcile_managed_tools`, generation-tokened,
  last-registration-wins, retire-on-disconnect) with `ManagedToolToken`; rig-agent's
  `ToolServerHandle` implements it, and `McpClientHandler<S: ManagedToolSink>` refreshes
  into any sink. `PortableDynamicTool` gained `with_liveness` / `is_live` so a sink can
  retire tools whose remote backing disconnected; rig-agent's `DynamicTool::from_portable`
  forwards it.
- **MCP tools are rig-core portable tools.** `McpTool` converts `From<McpTool> for
  rig_core::tool::PortableDynamicTool` (`rig_rmcp::tools_from_server` for a whole list).
  Register them wherever portable tools go: `builder.portable_dynamic_tool(tool.into())`,
  `ToolServer::new().portable_dynamic_tool(..)`, `ToolServerHandle::add_portable_dynamic_tool`.
  The `rmcp_tool` / `rmcp_tools` / `…_with_timeout` builder methods on `AgentBuilder` and
  `ToolServer` are **removed** — no replacement method; use the portable registration above
  (nothing is lost: the portable adapter is context-aware).
- **`ToolContext` lives in rig-core, and portable dynamic tools can receive it.**
  `rig_core::tool::{ToolContext, MissingToolContext}` (module `rig_core::tool::context`,
  which also exposes the `TypeMap` primitive and the dispatch helpers `for_dispatch` /
  `accept_dispatch_result` / `clear_dispatch_result` any runtime needs); `rig_agent::tool::ToolContext`
  and `rig::tool::ToolContext` are re-exports, so existing paths and `#[rig(context)]` keep
  working. `PortableDynamicTool::new` stays context-free; `new_with_context` /
  `execute_with` give a dynamic tool the per-call context, and rig-agent's
  `DynamicTool::from_portable` now threads the agent's context through instead of
  discarding it. MCP `_meta` passthrough (an `rmcp::model::Meta` placed in the context)
  and result preservation (`structuredContent`, response `Meta`, raw `CallToolResult`
  on `context.result::<T>()`) therefore work through the portable path exactly as before.
- **Direct rig-agent users** depend on `rig-rmcp`; runtimes other than rig-agent implement
  `ManagedToolSink` for their registry to get live refresh.
- rig-agent's registry support for externally managed tools is public and ungated:
  `ErasedTool` (and `is_live`), `ToolSet::add_erased`, `ToolServer::erased_tool`,
  `AgentBuilder::erased_tool`, `ToolServerHandle::{add_managed_erased_tools,
  reconcile_managed_erased_tools}`, `Agent::tool_server_handle()`. rig-agent's `tokio` is
  optional, enabled only by `test-utils`.

### rig-core has no default transport; the bundled reqwest transport is the new `rig-reqwest` crate

`rig-core` no longer depends on `reqwest` or `tokio` and no longer names a
default HTTP transport anywhere: every `H` type parameter that used to default
to `reqwest::Client` (provider `Client`/`CompletionModel`/`EmbeddingModel`/…
aliases, `Capabilities<H>`, `ModelLister<H>`, `Client<Ext, H>`) now has no
default, and the reqwest-pinned constructors (`Client::new`, the default
`ClientBuilder::build`, the per-provider `ProviderClient` impls) are gone from
rig-core. The transport lives in **`rig-reqwest`**:

- `rig_reqwest::ReqwestClient` — a newtype over `reqwest::Client` implementing
  `HttpClientExt` (a newtype because the orphan rule forbids implementing
  rig-core's trait for reqwest's type from a third crate; it derefs to the
  inner client and converts `From<reqwest::Client>`). `ReqwestMiddlewareClient`
  wraps `reqwest_middleware::ClientWithMiddleware` behind `reqwest-middleware`.
- `rig_reqwest::client::DefaultTransportClient` / `DefaultTransportBuilder` —
  the traits that give every rig-core provider client `new(api_key)`,
  `from_env()`, `from_val(input)` and `builder().…build()` over the bundled
  transport. They are implemented exactly once, for `Client<Ext, ReqwestClient>`,
  which is what lets `openai::Client::from_env()` infer `H`.
- `rig_reqwest::providers::*` — the familiar provider module tree with every
  transport-generic type aliased to `…<ReqwestClient>` for type position
  (`Agent<openai::CompletionModel>`, `let c: openai::Client = …`).
- `rig_reqwest::openai_websocket` — the OpenAI Responses websocket mode
  (feature `websocket`), with `ResponsesWebSocketExt` supplying
  `client.responses_websocket(..)`.
- It works without a tokio runtime (Bevy task pools, smol, `futures::executor`):
  reqwest futures are driven on a lazily started fallback runtime and the
  caller only ever polls runtime-agnostic futures.

**Through the `rig` facade nothing changes for the common path** — the default
`reqwest` feature re-exports all of the above, so with `use rig::prelude::*`
these keep working unchanged:

```rust
use rig::prelude::*;
let client = rig::providers::openai::Client::from_env()?;
let client = rig::providers::openai::Client::new("key")?;
let client = rig::providers::openai::Client::builder().api_key("key").build()?;
let agent: Agent<rig::providers::openai::CompletionModel> = client.agent("gpt-4o").build();
```

What does change:

- Code that imported `rig::client::ProviderClient` (or `rig_core::client::ProviderClient`)
  to call `from_env()` on a **core** provider client must import the new traits
  instead: `use rig::client::{DefaultTransportClient, DefaultTransportBuilder};`
  (or just `use rig::prelude::*;`). `ProviderClient` itself stays in rig-core
  for companion crates' own client types (rig-bedrock, rig-vertexai,
  rig-gemini-grpc), which still implement it.
- Direct `rig-core` users (no facade) must name the transport:
  `Client::new_with(key, http)`, `<OpenAIResponsesExt as ProviderFromEnv>::from_env_with(http)`
  / `from_val_with(input, http)`, or `Client::builder().api_key(..).http_client(http).build()`,
  with any `HttpClientExt` implementation (e.g. `rig_reqwest::ReqwestClient::default()`).
  `llamacpp::Client::from_url(url)` is `from_url_with(url, http)`.
- `impl Capabilities for X` / `impl ModelLister for X` relied on the trait
  defaults and must spell `Capabilities<H>` / `ModelLister<H>`.
- Type annotations that named the nested generic aliases without `H`
  (`openai::responses_api::ResponsesCompletionModel`) use the facade's
  top-level aliases (`rig::providers::openai::ResponsesCompletionModel`) or
  spell `<ReqwestClient>`.
- `rig::http_client::ReqwestClient` / `from_reqwest` still resolve (re-exported
  from `rig-reqwest`); `Error::instance(err)` is the public constructor for a
  transport's response-less failures and `Error::non_success_with_details`
  for status-bearing ones.
- The `websocket*`, `rustls`, `native-tls`, `socks`, `reqwest-middleware*`
  features on the `rig` facade now forward to `rig-reqwest`; rig-core has none
  of them. rig-core's default features are just `["derive"]`.
- `Client::http_client()` borrows a client's transport; `ProviderFromEnv` on
  the provider extension types replaces the per-client `ProviderClient` impls
  for anyone who implemented `ProviderClient` for a core `Client`.

### `AuthError::Http` carries `http_client::Error`, and `Authenticator::auth_context` takes the transport

`rig::providers::copilot::auth::AuthError` / `rig::providers::chatgpt::auth::AuthError`
(`providers::internal::auth::AuthError`) wrapped a raw `reqwest::Error`. The
OAuth/device-code flows now run through the client's own `HttpClientExt`
transport instead of an ad-hoc `reqwest::Client`, so the variant carries the
transport-agnostic `http_client::Error`: a non-success response is one of its
status-bearing variants, a response-less failure is `Instance`.

```rust
// before
Err(AuthError::Http(e)) => e.status()

// after
Err(AuthError::Http(e)) => match e {
    http_client::Error::InvalidStatusCode(status)
    | http_client::Error::InvalidStatusCodeWithMessage(status, _)
    | http_client::Error::InvalidStatusCodeWithDetails { status, .. } => Some(status),
    _ => None,
}
```

`Authenticator::auth_context()` accordingly takes the transport to use:
`auth.auth_context(client.http_client()).await`. The provider clients'
`authorize()` helpers are unchanged. `Client::http_client()` is new: it
borrows the transport a `Client<Ext, H>` sends through.

### `http_client::ReqwestClient` / `from_reqwest` live in a reqwest-only module

Both still resolve at `rig::http_client::ReqwestClient` and
`rig::http_client::from_reqwest`; they are re-exported from the bundled
reqwest transport module, which is now the only place rig-core names a reqwest
type. `http_client::Error::non_success_with_details(status, headers, body)` is
the new transport-agnostic constructor for the headers-preserving non-success
error — custom `HttpClientExt` implementations should build their errors with
it so `non_success_headers()` keeps working for retry policies.

### `VectorStoreError::ReqwestError` is now `VectorStoreError::Http(http_client::Error)`

The variant carried a raw `reqwest::Error`, which tied rig-core's public
error surface to one HTTP backend. It now carries the transport-agnostic
[`rig::http_client::Error`]: a non-success response arrives as one of its
status-bearing variants (so the status code is still inspectable), and a
response-less transport failure (connect, decode, timeout) arrives as
`http_client::Error::Instance`.

```rust
// before
Err(VectorStoreError::ReqwestError(e)) => {
    if let Some(status) = e.status() { /* … */ }
}

// after
Err(VectorStoreError::Http(e)) => match e {
    http_client::Error::InvalidStatusCode(status)
    | http_client::Error::InvalidStatusCodeWithMessage(status, _)
    | http_client::Error::InvalidStatusCodeWithDetails { status, .. } => { /* … */ }
    http_client::Error::Instance(source) => { /* transport failure */ }
    _ => { /* … */ }
},
```

Code that relied on `?` converting a `reqwest::Error` straight into
`VectorStoreError` must map it first; `rig::http_client::from_reqwest`
applies the routing above:

```rust
let res = client.send().await.map_err(http_client::from_reqwest)?;
```

### `ClientBuilderError` is removed

The enum (`HttpError(reqwest::Error)` / `InvalidProperty`) had no remaining
constructor or match site in the workspace — `Client::builder()`/`build()`
return `http_client::Result` — so it is deleted rather than ported off
reqwest. Nothing to migrate unless you named the type yourself; use
`http_client::Error` in its place.

### `ToolServerHandle` registration methods are now synchronous

`add_tool`, `add_dynamic_tool`, `add_portable_dynamic_tool`, `append_toolset`,
and `remove_tool` only ever took a short registry lock that is never held
across an await; the lock is now a `std::sync::RwLock` and the methods are
plain `fn`. Drop the `.await`:

```rust
// before
handle.add_tool(MyTool).await;

// after
handle.add_tool(MyTool);
```

Execution and snapshot paths (`execute`, `get_tool_defs`) are unchanged and
remain async. This removes the last `tokio::sync` primitive from the
tool-server hot path; streaming pause/resume and the copilot/chatgpt auth
caches likewise moved off tokio primitives (no API change), so neither
rig-core nor rig-agent needs a tokio runtime for these paths.

### Telemetry getters borrow: `ProviderResponseExt::get_response_id` / `get_response_model_name` return `Option<&str>`

Both getters exist to hand a value to `tracing::Span::record`, which takes
`&str`; the owning return forced every one of the sixteen provider impls to
clone a `String` per completion turn just to have it read once and dropped.
They now return `Option<&str>`. `get_text_response` and `get_usage` are
unchanged.

```rust
// before
fn get_response_id(&self) -> Option<String> { Some(self.id.clone()) }

// after
fn get_response_id(&self) -> Option<&str> { Some(self.id.as_str()) }
```

Callers that need an owned value add `.map(str::to_owned)`.

### `ReasoningSummary::text()` returns `&str`

The one-variant enum's accessor cloned the summary text; it now borrows.
Callers that need ownership add `.to_owned()`.

### Request assembly accessors that need the history by value now consume the request

`rig_vertexai::types::completion_request::VertexCompletionRequest::contents`
and `rig_bedrock::types::completion_request::AwsCompletionRequest::messages`
took `&self` and cloned the entire chat history per request. Each now takes
`self`. Every other accessor on both types still borrows — call the borrowing
accessors first and the consuming one last, which is the order the completion
paths already used.

```rust
// before
let contents = vertex_request.contents()?;          // cloned the history
let system_instruction = vertex_request.system_instruction();

// after
let system_instruction = vertex_request.system_instruction();
let contents = vertex_request.contents()?;          // moves the history
```

### `TextToImageGeneration::width`/`height` are chainable builders

The two setters took `&mut self` and returned `&Self`, which chained with
nothing else in the tree; every other rig builder is `mut self -> Self`. They
now follow the house idiom.

```rust
// before
let mut request = TextToImageGeneration::new(prompt);
request.width(1024);
request.height(1024);

// after
let request = TextToImageGeneration::new(prompt).width(1024).height(1024);
```

### `rig-s3vectors`: `set_bucket_name` / `set_index_name` removed

Both setters had zero callers anywhere in the workspace and re-allocated from
`&str`. Construct the store with the right names instead; the read accessors
(`bucket_name()`, `index_name()`) are unchanged.

### Vector-store filter constructors take `impl Into<String>` keys

`MongoDbSearchFilter`, `SqliteSearchFilter`, `ScyllaSearchFilter`,
`S3VectorsSearchFilter`, and `PgSearchFilter` constructors that took
`key: String` now take `key: impl Into<String>`, matching the rest of the
workspace (and the other parameter of the same functions, which already did).
`filter::gte("price".to_string(), v)` still compiles; `filter::gte("price", v)`
now also does. The only source break is a caller passing `"key".into()`, which
becomes ambiguous — pass the literal.

### Loosened bounds (no action needed)

These accept strictly more code than before:

- Transport/HTTP-client generic chains across the providers no longer demand
  `Default` or `Debug` (`mistral`/`openrouter` transcription, `hyperbolic`,
  `voyageai`, `ollama`, `gemini` image-generation and Interactions API,
  `anthropic`, `copilot`, `chatgpt`, `openai` completions/responses/websocket).
  Nothing constructed or formatted the client; the minimal chain is
  `HttpClientExt + Clone (+ WasmCompatSend/WasmCompatSync) + 'static`.
- `GenericCompletionModel::new` (and `with_strict_tools`,
  `with_tool_result_array_content`) no longer bound `Client<Ext, H>` or `Ext`
  at all.
- `TypeMap`/`ToolContext` read-side methods (`get`, `get_mut`, `remove`,
  `contains`, `require`, `result`, `require_result`) need only `T: 'static`
  (pure `Any` lookups); the write side keeps
  `Clone + WasmCompatSend + WasmCompatSync + 'static`.
- `SqliteVectorStore<T>`/`SqliteVectorIndex<T>` struct declarations no longer
  carry `T: SqliteVectorStoreTable + 'static`; the `'static` survives only on
  the impl blocks whose `conn.call` closures actually need it.
  `EmbeddingsBuilder`, `InMemoryVectorStoreBuilder`,
  `TranscriptionRequestBuilder`, and `ImageGenerationRequestBuilder` likewise
  drop their struct-level where clauses.
- `HelixDBVectorStore::new`/`client()` require no bounds; the store impls no
  longer restate `C::Err: std::error::Error` (declared on the trait).
- The `VectorStoreIndexDyn` blanket impl no longer restates `WasmCompatSend +
  WasmCompatSync + 'static` on the filter type (implied by
  `VectorStoreIndex::Filter`).
- `ToolSchema::try_from` and gemini's `ConstructEmbeddingModel` impl drop a
  `'static` neither uses.
- Extractor/vector-store generics spell `DeserializeOwned` instead of the
  equivalent `for<'de> Deserialize<'de>`.

### Provider usage types are `Copy`

The scalar-only usage payloads (`openai` chat + responses, `anthropic`,
`deepseek`, `openrouter`, `cohere`, gemini `InteractionUsage`, and their
detail structs) now derive `Copy`. Mistral's `Usage` is not `Copy` — it
carries `service_tier: Option<String>`.


### `providers::llamafile` is now `providers::llamacpp`

The provider named after Mozilla's single-file distribution is gone; the one
named after the upstream everyone actually runs replaces it. `llamafile` the
project serves the same OpenAI-compatible API from the same llama.cpp core, so
one provider genuinely covers both: point `llamacpp::Client` at a running
`.llamafile` and everything works exactly as it did.

```rust
// before
let client = rig::providers::llamafile::Client::from_url("http://localhost:8080")?;

// after
let client = rig::providers::llamacpp::Client::from_url("http://localhost:8080")?;
```

Every type moves with it: `llamafile::CompletionModel` → `llamacpp::CompletionModel`,
`llamafile::EmbeddingModel` → `llamacpp::EmbeddingModel`,
`llamafile::LlamafileExt` → `llamacpp::LlamacppExt`,
`llamafile::LLAMA_CPP` → `llamacpp::LLAMA_CPP`. There is no deprecated alias:
the old module is deleted, so this is a compile error rather than a silent
change.

Six things behave differently, and none of them is a rename:

1. **An API key is now possible.** `llamafile`'s `ApiKey` type was `Nothing`,
   which cannot produce an `Authorization` header at all, so a
   `llama-server --api-key <key>` deployment was unreachable. `llamacpp` takes
   an optional key and sends `Authorization: Bearer <key>` when one is set —
   and still sends no header at all when none is, so an unsecured local server
   keeps working unchanged.

   ```rust
   let secured = rig::providers::llamacpp::Client::builder()
       .api_key("hunter2")
       .base_url("http://localhost:8080")
       .build()?;
   ```

2. **`from_env` no longer requires a base URL.** `LLAMAFILE_API_BASE_URL` was
   mandatory; `LLAMACPP_API_BASE_URL` is optional and defaults to
   `http://localhost:8080`. `LLAMACPP_API_KEY` is read when present.

3. **A base URL ending in `/v1` is no longer doubled.** `llamafile` appended
   `/v1` unconditionally, so passing the URL from its own doc line
   (`http://localhost:8080/v1`) produced `/v1/v1/chat/completions` and a 404.
   `llamacpp` adds the prefix only when the base URL lacks it. llama.cpp's
   own operational routes (`/props`, `/health`, `/slots`, `/tokenize`, …) live
   outside the `/v1` namespace and are addressed there from either spelling.

4. **A specific-function `tool_choice` is now refused instead of silently
   ignored.** `llama-server` reads `tool_choice` as a *string* and understands
   only `auto`, `none` and `required`; an OpenAI-shaped
   `{"type": "function", "function": {"name": "…"}}` type-mismatches that read
   and is served as `auto`, so the model returns whichever tool it liked.
   `ToolChoice::Specific` now returns a `CompletionError::ProviderError` naming
   the tool and pointing at `ToolChoice::Required`. If you were relying on the
   old behaviour you were not getting the tool you asked for; advertise only
   that tool in `tools` to force it.

5. **`verify()` targets `/props` instead of `/models`.** `GET /v1/models` and
   `/health` are the only two routes `llama-server` serves *without* its
   API-key check, so verifying against `/models` returned `Ok(())` for a wrong
   key — or no key — against a server that would reject every real request.
   `/props` is behind the check and is served by every configuration. If your
   deployment is behind a proxy that only forwards `/v1/*` — a common nginx
   `location /v1/` — `verify()` will now 404 where it used to succeed, and you
   need to forward `/props` as well. Note that llama.cpp serves its
   operational routes off the server root, *not* under `/v1`.

6. **`embedding_model_with_ndims` can now fail.** See the standalone section
   below; it applies to every OpenAI-compatible provider, not only this one.

`llamacpp` additionally declares two capability slots `llamafile` did not:
`model_listing` (`GET /v1/models`) and `rerank` (`POST /v1/rerank`, which needs
a server started with `--reranking` and a cross-encoder loaded).

### `llamacpp::raw_completion` returns a llama.cpp response type, not OpenAI's

`llamafile`'s `OpenAICompatibleProvider::Response` was
`openai::CompletionResponse`, so `raw_completion` returned that. `llamacpp`'s
is `llamacpp::CompletionResponse`: the same OpenAI payload `#[serde(flatten)]`ed
under a public `openai` field, plus llama.cpp's own `timings` — the server-side
latency accounting (`prompt_ms`, `predicted_per_second`, `cache_n`) that the
OpenAI type has nowhere to put and was therefore dropping.

```rust
// before
let raw: openai::CompletionResponse = model.raw_completion(request).await?;
let id = raw.id;

// after
let raw: llamacpp::CompletionResponse = model.raw_completion(request).await?;
let id = raw.openai.id;
let tokens_per_second = raw.predicted_tokens_per_second();
```

The normalized `CompletionModel::completion` surface is unchanged, and so is
the streaming path — `raw_stream` already preserved `timings` under
`additional_params`, because the shared streaming chunk type carries a
catch-all and the blocking one does not.

### A declared embedding width the provider ignores is now an error

**This is not llama.cpp-specific.** It affects every provider on the shared
OpenAI-compatible embeddings path — openai, azure, together, openrouter,
venice, doubleword, mistral and llamacpp.

`ndims()` is what a vector store sizes its index from. When a caller declared a
width *explicitly* and the provider returned vectors of a different one, rig
kept reporting the declared number, so the disagreement surfaced far from the
call that caused it — as an index that could not hold its own vectors. The
providers where this happens are the ones that **ignore** `dimensions` rather
than rejecting it, so nothing in the response says anything is wrong.

```rust
// before: Ok, with 1536-wide vectors and ndims() == 512
// after:  Err(EmbeddingError::MismatchedDimensions { requested: 512, returned: 1536, .. })
let model = client.embedding_model_with_ndims("text-embedding-ada-002", 512);
let embeddings = model.embed_texts(["hello"]).await?;
```

The check applies only to a width that was **set explicitly** — through
`EmbeddingsClient::embedding_model_with_ndims` or
`openai::embedding::GenericEmbeddingModel::new` / `with_model` /
`with_encoding_format`. A handle built with `embedding_model` reports whatever
the provider's own table says and is untouched.

A width of **zero is unaffected**: zero is rig's sentinel for *unknown* — it is
what `default_ndims` returning `None` produces — not a declaration, so
`GenericEmbeddingModel::new(client, model, 0)` behaves exactly as before.

If you hit this, you were already getting vectors of a width `ndims()` did not
describe. Either drop to `embedding_model` and let the provider's width stand,
or pass the width the model actually returns.

### `RerankResult::relevance_score` is no longer documented as 0..1

No code change; the type is the same `f64`. The doc comment claimed a 0-to-1
range, which was true of the only implementation that existed and is false of
llama.cpp's: it returns the cross-encoder's raw logit, so negative scores are
normal (measured: `0.8225`, `-4.7583`, `-8.3761` for three documents against one
query). Use the field to *order* results within one response; code that
thresholded it as a probability was already wrong on any logit-scoring
provider.

### Raw OpenAI-compatible streaming terminals gain `logprobs`

`openai::completion::StreamingCompletionResponse<U>` — the provider-native
terminal record returned by `raw_stream` for OpenAI Chat Completions and its
compatible providers — gains one public field:

```rust
pub logprobs: Option<serde_json::Value>
```

It contains the primary choice's per-chunk log-probability objects, deep-merged
in arrival order. In particular, token arrays under `content` and
`reasoning_content` are concatenated rather than overwritten. The normalized
`CompletionModel::stream` surface is unchanged; use `raw_stream` when you need
this provider-native metadata.

Code that only reads the terminal record needs no change. A full struct literal
must add `logprobs: None` to reproduce the old value, and an exhaustive
destructure must name the field or add `..`. The field is serde-defaulted and
omitted when absent, so terminal records persisted before this change still
deserialize and serialization is unchanged when log probabilities were not
requested.

The same terminal record also gains:

```rust
pub additional_params: Option<rig::message::AdditionalParams>
```

This is where otherwise-unmodeled top-level SSE chunk metadata now survives.
It is accumulated across the stream, so OpenAI and OpenRouter raw terminals no
longer lose `service_tier`, `system_fingerprint`, routed `provider`, or a new
compatible-provider extension merely because the shared chunk type did not yet
name it. As with `logprobs`, a full struct literal must add
`additional_params: None`; old serialized terminals still load, and the field
is omitted when empty.

### Four blocking provider response types retain more native metadata

The live cross-provider cassette sweep for #2359 found four blocking fields
that serde was silently discarding:

```rust
openai::completion::CompletionResponse::service_tier: Option<String>
openrouter::completion::CompletionResponse::provider: Option<String>
openrouter::completion::CompletionResponse::service_tier: Option<String>
openrouter::completion::Choice::logprobs: Option<serde_json::Value>
mistral::Usage::service_tier: Option<String>
```

These are provider-native raw-response fields; normalized choices and usage are
unchanged. Every field is serde-defaulted and omitted when absent, so old
payloads remain compatible. These public structs are not non-exhaustive,
however, so full struct literals must add `None` and exhaustive destructures
must name the field or use `..`.

### `ProviderResponseError` gains a `headers` field

A failed provider response now carries its headers onto the error, so
rate-limit metadata (`Retry-After`, `x-ratelimit-*`) is recoverable — read it
with `provider_response_headers()` on any capability error, or
`http_client::Error::non_success_headers()` when you hold the transport error
directly (a custom `RetryPolicy` does).

The only breaking part is the new public field. Construction through
`ProviderResponseError::new` / `::without_status` plus the `with_*` setters —
which the type's docs already steer you toward — is unaffected. Only a full
struct literal has to change:

```rust
// Before
let error = ProviderResponseError {
    status: Some(StatusCode::TOO_MANY_REQUESTS),
    body: body.to_string(),
    provider_request_id: None,
};

// After: add the field...
let error = ProviderResponseError {
    status: Some(StatusCode::TOO_MANY_REQUESTS),
    body: body.to_string(),
    provider_request_id: None,
    headers: None,
};

// ...or switch to the constructors, which do not need revisiting when
// transport metadata grows again.
let error = ProviderResponseError::new(StatusCode::TOO_MANY_REQUESTS, body);
```

One behavior change with no compile error to warn you: contract-less
providers' non-success completions and all three `VerifyClient::verify`
failure branches now surface
`http_client::Error::InvalidStatusCodeWithDetails` where they previously
surfaced `InvalidStatusCodeWithMessage`. Both are `HttpError`, both `Display`
identically, and every `provider_response_*` helper reads both — but a `match`
arm naming the old variant silently stops firing. Match on the accessors
rather than the variant.

### Anthropic's streamed terminal record grows five fields

`anthropic::streaming::StreamingCompletionResponse` — the provider-native
terminal record `GenericCompletionModel::raw_stream` yields — carried nothing
but `usage` on 0.41. It now carries the metadata the blocking twin already had,
plus the transport request id #2313 added to both:

```rust
pub struct StreamingCompletionResponse {
    pub usage: PartialUsage,
    pub stop_reason: Option<String>,          // #2257
    pub stop_sequence: Option<String>,        // #2329
    pub message_id: Option<String>,           // #2257
    pub model: Option<String>,                // #2257
    pub provider_request_id: Option<String>,  // #2265, via #2313
}
```

`stop_sequence` is the one that changes what a caller can learn: Anthropic's
terminal `message_delta` names *which* of the caller's `stop_sequences`
matched, the adapter parsed that field and then dropped it, and Anthropic
strips the matched sequence from the text — so the frame is its only source.
A streamed turn could report only that *a* sequence fired while the blocking
`CompletionResponse::stop_sequence` named it.

The type is not `#[non_exhaustive]`, so reading the record is unchanged and
only a full struct literal has to move. It derives `Default`, so
`..Default::default()` covers this growth and the next one. All five new fields
are `#[serde(default, skip_serializing_if = "Option::is_none")]`, so terminal
records persisted by 0.41 still load. Reach the record through `raw_stream`:
`CompletionModel::stream` normalizes into `StreamFinal`, which carries the
finish reason but not the sequence itself. Every Anthropic-compatible gateway
sharing this adapter — minimax, moonshot, xiaomimimo, zai — gets the fields
too.

### Anthropic `Usage` and `PartialUsage` gain two usage-breakdown fields (#2312, #2334)

Anthropic reports a per-TTL breakdown of its cache writes and the tokens Claude
spent thinking, and neither wire type modeled either, so serde dropped both.
`anthropic::completion::Usage` and `anthropic::streaming::PartialUsage` each
gain `cache_creation: Option<CacheCreation>` (#2312) and
`output_tokens_details: Option<OutputTokensDetails>` (#2334), where
`anthropic::completion::CacheCreation { pub ephemeral_5m_input_tokens: u64,
pub ephemeral_1h_input_tokens: u64 }` and
`anthropic::completion::OutputTokensDetails { pub thinking_tokens: u64 }` are
new public types.

Neither usage type is `#[non_exhaustive]` — neither ever was — so a full struct
literal that compiled on 0.41 no longer does:

```rust
// Before
let usage = anthropic::completion::Usage {
    input_tokens: 12,
    cache_read_input_tokens: None,
    cache_creation_input_tokens: None,
    output_tokens: 34,
};

// After: `None` on both reproduces the old value exactly.
let usage = anthropic::completion::Usage {
    input_tokens: 12,
    cache_read_input_tokens: None,
    cache_creation_input_tokens: None,
    cache_creation: None,
    output_tokens: 34,
    output_tokens_details: None,
};
```

`PartialUsage` derives `Default`, so `..Default::default()` absorbs both there;
`Usage` does not derive it. Code that only reads these types is unaffected, and
both fields are `#[serde(default, skip_serializing_if = "Option::is_none")]`,
so usage JSON persisted by 0.41 still deserializes and an absent breakdown is
not serialized.

One behavior change with no compile error to warn you:
`completion::Usage::reasoning_tokens` was hard-`0` for every Anthropic turn and
now reports the real thinking-token count, on both the blocking and streaming
transports. It is a breakdown of `output_tokens`, already counted there, so it
does not enter `total_tokens` — cost accounting that adds `reasoning_tokens` on
top of the total will double-count.

### `anthropic::completion::ToolDefinition` gains a `strict` field (#2296)

`with_strict_tools()` on the Anthropic completion model marks every
Rig-generated tool `strict: true`, so the wire type had to grow the flag:

```rust
// Before
let tool = anthropic::completion::ToolDefinition {
    name: "get_weather".to_string(),
    description: Some("Look up the weather".to_string()),
    input_schema: schema,
    cache_control: None,
};

// After
let tool = anthropic::completion::ToolDefinition {
    name: "get_weather".to_string(),
    description: Some("Look up the weather".to_string()),
    input_schema: schema,
    strict: false,
    cache_control: None,
};
```

`strict` is `#[serde(default, skip_serializing_if = "is_false")]`, so the
serialized shape is unchanged while the flag is off and tool definitions
persisted by 0.41 still deserialize. Adding `strict: false` to full struct
literals is the whole migration — the type has all-public fields, no `Default`
derive, and since #2335 removed `#[non_exhaustive]` workspace-wide there is no
constructor the compiler was steering you toward.

### `cohere::completion::Usage` gains a `cached_tokens` field (#2263)

The move from `billed_units` to `tokens` (see "Cohere token counts come from
`tokens`, not `billed_units`" under Silent behavior changes) also gave the
public wire type the counter that `completion::Usage::cached_input_tokens` is
now read from:

```rust
// Before
let usage = cohere::completion::Usage { billed_units: None, tokens: Some(tokens) };

// After
let usage = cohere::completion::Usage {
    billed_units: None,
    tokens: Some(tokens),
    cached_tokens: None,
};
```

The field is `#[serde(default)]`, so Cohere usage persisted by 0.41 still
deserializes and no stored record needs rewriting. The only breakage is the
struct literal: the type has all-public fields, no `Default` derive, and no
`#[non_exhaustive]` pointing you at a constructor.

### `openai::TranscriptionResponse` gains a `usage` field (#2332)

The transcription endpoint reports what a transcription cost — `whisper-1` by
audio duration, the `gpt-4o-transcribe` family by token — and the response type
modeled only `{ text }`, so the accounting was dropped even from the raw
provider response. It is now:

```rust
pub struct TranscriptionResponse {
    pub text: String,
    pub usage: Option<TranscriptionUsage>,
}
```

`openai::TranscriptionUsage` (with `DurationTag`, `TokensTag` and
`TranscriptionInputTokenDetails`) is the new public type behind it. The
response type has no constructor and is not `#[non_exhaustive]`, so a full
struct literal — a test double, a hand-made response — must add `usage: None`.
Decoding is unaffected: the field is `#[serde(default)]`, so a payload that
omits `usage` still deserializes. The same type is the transcription response
for Groq, Azure OpenAI and Venice (they share the OpenAI transcription model)
and HuggingFace re-exports it, so one edit covers all five providers.

### `model::Model` gains `max_output_tokens` (#2324)

`rig_core::model::Model` (re-exported as `rig::model::Model`) gains
`max_output_tokens: Option<u32>` — the provider-reported output ceiling,
distinct from `context_length`, which is the input window. It is `None` when a
listing does not report one, never a rig-invented default, and rig deliberately
does not send it on requests.

`Model` is not `#[non_exhaustive]`, and its own rustdoc example builds one with
a struct literal, so a full literal needs the new field:

```rust
// Was
Model { id, name, description, r#type, created_at, owned_by, context_length }
// Now
Model { id, name, description, r#type, created_at, owned_by, context_length,
        max_output_tokens: None }
```

`Model::from_id` and `Model::new` are unchanged and set the field to `None`, so
construction through them needs no edit. The field is
`skip_serializing_if = "Option::is_none"`, so serialized listings that predate
it still deserialize and gain nothing on the wire.

### A truncated OpenAI-compatible turn now succeeds with an empty choice

A turn the provider cut short can carry no content at all — the usual case is a
reasoning model whose `max_tokens` was consumed entirely by hidden reasoning,
which answers with an empty message and `finish_reason: "length"`. Normalization
used to reject that as a malformed response; it now returns `Ok` with an empty
`CompletionResponse::choice` and the finish reason attached, so the caller can
tell "you hit the cap" from "the provider misbehaved". A turn that ran to
completion (`stop`, `tool_calls`) with nothing in it is still an error.

This affects OpenAI and every OpenAI-compatible provider. Agent and extractor
users need to change nothing — the agent already recognizes an answerless turn
and reports it with the remedy for the reason. Code calling
`CompletionModel::completion` directly and indexing the choice does need a
guard:

```rust
let response = model.completion(request).await?;

// Before: unreachable, because a truncated turn arrived as `Err`.
// Now: reachable, and indexing would panic.
if response.choice.is_empty() {
    // `response.finish_reason()` is `Length` or `ContentFilter` here.
    return Err(/* … raise max_tokens, or relax the prompt … */);
}
```

Matching on `finish_reason()` is the more direct form:
`FinishReason::truncated_output()` is the predicate normalization itself uses.

### A length-truncated Chat Completions tool call no longer loses the turn

OpenAI-compatible providers can end a turn with `finish_reason: "length"`
while a tool's JSON argument string is still empty or cut off. Blocking
deserialization used to reject the entire response, losing its text, valid
sibling calls, usage, identity, model, and finish reason. It now discards only
the incomplete call and preserves the rest of the turn; raw and normalized
streaming apply the same rule, so an empty truncated argument slot cannot be
dispatched as a zero-argument side-effect tool.

The tolerance is intentionally keyed to the outer finish reason. An ordinary
`finish_reason: "tool_calls"` turn whose model generated invalid JSON still
fails loudly—as the OpenAI Chat reference warns callers to expect and validate—
and a real parameterless call remains `{}`. If code previously treated every
blocking decode failure as a retry signal, check `finish_reason() == Length`
and the surviving choice instead; the response now carries enough information
to make that decision directly.

### xAI uses the shared Responses wire response

`providers::xai::completion::CompletionResponse` is now an alias for
`providers::openai::responses_api::CompletionResponse`. The xAI model still
sends xAI's request shape to `/v1/responses`, preserves xAI error envelopes and
request ids, and emits completed streamed tool calls at the same boundary; only
the duplicated response and streaming implementation is gone.

Code inspecting `raw_completion()` results should use the shared field names
and types: `created_at` replaces `created`, `status` is a `ResponseStatus`
instead of `Option<String>`, `object` is a `ResponseObject` instead of a
`String`, and the complete Responses metadata surface is
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
| `one_or_many::string_or_option_one_or_many` | none — deserialize the field as a `Vec<T>` with `json_utils::string_or_vec` and apply `message::non_empty` where the `Option` carried the "absent" case |

`json_utils` is `#[doc(hidden)]` and documented in-tree as "Not part of
rig-core's stable public API" — it was already so at 0.41, and it is where both
replacements live, so treat those two rows as the shortest path off the deleted
helpers rather than as a supported destination.

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
  `ReasoningEnd { id, reasoning, signature, wire_sent }` and
  `TextEnd { id }`. `wire_sent` records whether the wire itself sent the end
  frame or an adapter synthesized it at a boundary the wire never announces —
  the flag the consumer bullet below turns on. `RawStreamingChoice` is not
  `#[non_exhaustive]` and the variant's fields are all public, so an
  out-of-tree adapter that builds or exhaustively matches `ReasoningEnd` must
  account for the fourth field. The
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
- `rig_agent::agent::run::StreamedTurnEvent::EmitToolCallDelta` loses its `id`
  field with them — it carried the provider-supplied tool-call id and now
  holds only `internal_call_id` and `content`. A hand-written `AgentRun`
  driver destructuring `EmitToolCallDelta { id, internal_call_id, content }`
  stops compiling; drop the `id` binding and correlate on
  `internal_call_id`, which is what the completed `ToolCall` restates.
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
  `UserContent::tool_result_from_wire(wire_id, name, content)`
  (single-identifier wire echo),
  `UserContent::tool_result_with_call_id(item_id, call_id, name, content)`
  (dual-identifier), or
  `UserContent::tool_result_for(call, provider, name, content)` — the
  agent-driver form; `tool_result_named` is gone. All four are `UserContent`
  constructors taking `Vec<ToolResultContent>`. The whole-message shortcuts
  moved with them: `Message::tool_result` grew a third parameter and is now
  `(call, name, content)` where 0.41 took `(id, content)`, and
  `Message::tool_result_with_call_id(id, Some(call_id), content)` is deleted
  outright — build the content with `UserContent::tool_result_with_call_id`
  and wrap it with `Message::from(..)`.
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

> **Superseded in this release**: the identity type shipped as the opaque
> `StreamPartId`, not `PartId` — read every `PartId` below as `StreamPartId`,
> whose representation is private (there are no `Wire`/`Minted` variants to
> match on) and which is constructed with `StreamPartId::wire` /
> `StreamPartId::minted`. The public `StreamedAssistantContent::ReasoningDelta`
> id is not a rendering of that key either: it is a rig-generated correlator,
> with the provider's item id alongside it on `provider_id: Option<String>`.
> The reserved spelling below goes with it: the Responses `item_id`-less
> fallback is an opaque `MintKind::Output` key, minted from the adapter's
> per-stream counter and then held for that `output_index` slot — not an
> `output-{output_index}` string. A minted key has no string form at all,
> and its index is the counter's, not the wire's `output_index`.
> See "Stream keys are opaque; durable ids and correlators are separate
> values" above for the contract that actually ships.

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

> **Superseded in this release**: `TextStart` carries `id: StreamPartId`, not
> `id: PartId`, and a wire that never announces text boundaries opens its
> block under a key minted from `MintKind::Text` rather than a `text-{n}`
> string — see "Stream keys are opaque; durable ids and correlators are
> separate values" above. The other reserved spelling below is superseded the
> same way: Anthropic's content-block index reaches the adapter as
> `MintKind::Block.for_wire_index(index)`, not as a `block-{index}` string —
> a minted key has no string form at all.

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

### Streaming part identity carries provenance (`StreamPartId`)

> **Superseded in this release**: this section described `PartId` — an enum
> with public `Wire(String)` / `Minted { kind, index }` variants, a
> `render()`, and reserved string namespaces — which never reached a release.
> The shipped type is the opaque `rig_core::streaming::StreamPartId`:
> `Eq + Hash + Clone + Debug` and nothing else, private representation, no
> rendering, no `Serialize`. Read "Stream keys are opaque; durable ids and
> correlators are separate values" above for the contract. Only the
> id-less-wire behavior change below still applies.

Behavior change on wires that issue no tool-call id (gemini REST/interactions,
ollama, chat-compat gateways): rig no longer fabricates a *provider* tool-call
id — not from an index, and not from the tool name (two calls to the same tool
in one turn no longer collide). Such a call still has a durable correlation
handle: `ToolCall::id` is a `ToolCallId`, always present and non-empty, minted
when the wire issued none, with `provider: None`. Whether that handle reaches
the wire is per-wire — gemini REST and ollama omit the id entirely, while
wires whose `tool_call_id` slot is required replay the minted handle
self-consistently. Gemini's `functionResponse.name` is written from
`ToolResult::name`, which is required data; only an *empty* name is filled in,
by `providers::internal::resolve_empty_tool_result_names`, which matches a
result to its call by `ToolCallId` first and then by `provider.call_id` /
`provider.item_id` — nothing pairs by position or by name.

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
  `ToolCallDelta`, and `TextStart` event carries a `StreamPartId` — the
  wire's own identifier via `StreamPartId::wire` when it exists, else a key
  minted via `SyntheticIds` / `MintKind::for_wire_index`. There are no
  reserved string shapes to steer clear of: a minted key is a `MintKind` plus
  an index, with no rendering and no `Serialize`, so a wire id cannot collide
  with a minted one by spelling (the `identity_leak` compile-fail suite pins
  that boundary). Aggregation treats minted keys as per-stream constants
  (other output closes the open block). Upstream, provenance is structural
  rather than gated: the durable handle travels separately as `WireId`
  (`provider_id` on reasoning events, `tool_id` on tool calls), so a part
  keyed by a minted `StreamPartId` aggregates with no durable id and the
  Responses serializer simply skips it — there is nothing to parse and no
  gate to keep in sync. Wires where identity is structurally required — the
  chat `tool_call_id` pair — replay rig's minted `ToolCallId`
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
    pub provider_request_id: Option<String>,
    // private — read with `response.finish_reason()`, write with the `with_*` setters
    finish_reason: Option<FinishReason>,
    pub provider: String,
    pub model: Option<String>,
}
```

| Before | After |
| --- | --- |
| `response.raw_response.model` | `response.model` |
| provider stop/finish reason off `raw_response` | `response.finish_reason()` |
| provider/message identity off `raw_response` | `response.provider`, `response.message_id` |
| response-scoped ID (`chatcmpl-*`, `responseId`, …) off `raw_response` | `response.response_id` |
| a genuinely provider-specific field | `model.raw_completion(request).await?` on a concrete model, or read `response.raw` (the same value, serialized — populated on every call, and the route from an agent, whose model type is erased) — see [Raw provider responses are reachable from an agent (#2366)](#raw-provider-responses-are-reachable-from-an-agent-2366) |

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

`provider_request_id` is the third, transport-level axis, added later in this
cycle by the response-identity work (#2265): the id the provider's HTTP
response headers carried (Anthropic `request-id`, OpenAI/xAI/Groq
`x-request-id`, Bedrock's SDK response metadata) — the one provider support
asks for when investigating a request. It is never a body id, and `None` means
the provider reported no such header: a documented outcome, never an error.

`CompletionResponse::finish_reason` is private, so an external struct literal
was never possible and still is not — #2335's workspace-wide
`#[non_exhaustive]` removal does not change that. Build it with
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

The trade: typed raw access now requires the concrete provider model rather
than any `CompletionResponse`. Code that was generic over `CompletionModel`
could never touch `raw_response` without a bound anyway, so in practice this
affects code that had already committed to a provider — including agent users,
whose model type is erased at `AgentBuilder::new`. For them the same value is
available, serialized, on every call as `CompletionResponse::raw` /
`StreamFinal::raw` — see [Raw provider responses are reachable from an agent
(#2366)](#raw-provider-responses-are-reachable-from-an-agent-2366).
On the OpenAI-compatible family and Copilot's chat route the transport request
id is not on the wire type: use `raw_completion_with_request_id` when the typed
route must reproduce everything `completion()` returns.

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

`StreamingPrompt<M, R>` and `StreamingChat<M, R>` lost **both** parameters,
not just `R`: the model type went with the rest of the agent surface (see
"Agents erase the model type at construction" below), so the traits are now
the bare `StreamingPrompt` and `StreamingChat`, their bounds
(`M: CompletionModel + 'static`, `M::StreamingResponse: WasmCompatSend`,
`R: Clone + Unpin + GetTokenUsage`) are gone, and both methods return a bare
`StreamingPromptRequest` rather than `StreamingPromptRequest<M>`. Delete the
angle brackets from your impls and type annotations — an
`impl StreamingPrompt<M> for MyAgent` now fails with "trait takes 0 generic
arguments but 1 generic argument was supplied". Call sites of `stream_prompt` /
`stream_chat` are unchanged.

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

### Borrow-shaped signatures: filter builders, rmcp registration, streaming prompt construction

An ownership audit converted signatures that took owned values they never
consumed into borrows, removing the caller-side clones they forced. Call
sites passing string literals compile unchanged; sites that built an owned
`String`/`Value`/`Vec` just to hand it over now pass a reference (and can
usually stop building the owned value at all).

- Vector-store filter builders take `&str` keys (and, where the value was
  only serialized, borrowed values): `rig_qdrant::QdrantFilter`
  (`exists`/`is_null`/`is_empty`/the four `range_*`, values now
  `&serde_json::Value`), `rig_milvus::Filter` (`gte`/`lte`/`in_values`/
  `not_in`/`like`/`array_*`), `rig_surrealdb::SurrealSearchFilter`
  (`contains`/`all`/`any`/`member`/geometry ops, values now `&Value`),
  `rig_lancedb::LanceDBFilter`, `rig_neo4j::Neo4jSearchFilter`,
  `rig_postgres::PgSearchFilter`, and `rig_vectorize::VectorizeFilter`
  (`ne`/`gte`/`lte` take `&Value`; `in_values`/`nin` take `&[Value]`).
- `MilvusVectorStore::auth(&str, &str)` (was `String, String`).
- `SqliteVectorStore::add_rows_with_txn` takes `&[(T, Vec<Embedding>)]`.
- `StreamingPromptRequest::new(&Agent, …)` (was `Arc<Agent>`) — it only ever
  read through the `Arc`; callers holding an `Arc` pass `&arc`.
- `ToolServer::rmcp_tools` / `rmcp_tools_with_timeout` and the corresponding
  `AgentBuilder` wrappers take `client: &ServerSink` (each registered tool
  clones it anyway); the single-tool `rmcp_tool_with_timeout` still consumes
  the sink it stores. Callers drop their `peer.clone()`.
- `rig_vertexai::types::completion_response::map_finish_reason(&FinishReason)`.
- `VectorStoreIndex::top_n`'s bound is spelled `DeserializeOwned` instead of
  `for<'a> Deserialize<'a>` — the same bound, so implementations and callers
  are unaffected; only the spelling changed.

### Transcription, image-generation and audio-generation responses are concrete and normalized

The same argument #2257 applied to completions now applies to the three
remaining response-bearing model traits. `TranscriptionModel`,
`ImageGenerationModel` and `AudioGenerationModel` lose `type Response`, and
`TranscriptionResponse<T>`, `ImageGenerationResponse<T>` and
`AudioGenerationResponse<T>` lose their parameter. Each is now a concrete
struct carrying the payload plus the metadata every provider can report:

```rust
pub struct TranscriptionResponse {
    pub text: String,                    // ImageGenerationResponse: image: Vec<u8>
    pub usage: Usage,                    // AudioGenerationResponse: audio: Vec<u8>
    pub provider: String,                // stable descriptor name, always set
    pub model: Option<String>,           // provider-reported model, when any
    pub response_id: Option<String>,
    pub provider_request_id: Option<String>,
    pub raw: serde_json::Value,          // the provider's own payload, serialized
}
```

plus `identity() -> ResponseIdentity`. Delete the type argument from your
annotations — `TranscriptionResponse<openai::TranscriptionResponse>` becomes
`TranscriptionResponse` — and the parameter from
`TranscriptionRequestBuilder<M, D>` / `ImageGenerationRequestBuilder<M, P>` /
`AudioGenerationRequestBuilder<M, T, V>` callers, whose `send()` now returns
the concrete type.

**The `response` field is gone.** Everything a caller used to read off it has
a named home or lives behind the raw route:

| 0.42 read                                    | now                                                          |
| -------------------------------------------- | ------------------------------------------------------------ |
| `r.response.text` (OpenAI / OpenRouter)      | `r.text`                                                     |
| `r.response.usage` token counts              | `r.usage` (`Usage`, zero when the provider reports none)     |
| `r.response.usage` duration-billed seconds   | `raw_transcription(..)` / `r.raw["usage"]["seconds"]`        |
| `r.response.model` (Mistral) / `model_version` (Gemini) | `r.model`                                         |
| `r.response.response_id` (Gemini)            | `r.response_id`                                              |
| `r.response.id` (Venice image)               | `r.raw["id"]`, or `raw_image_generation(..).id`              |
| anything else provider-specific              | `r.raw` (deserialize it into the provider type), or `raw_*`  |

Every concrete provider model gains inherent `raw_transcription`,
`raw_image_generation` or `raw_audio_generation` returning the provider's
native type from the same request, transport, parser and error path as the
normalized call — the same escape hatch `raw_completion` is for completions,
designed in up front this time rather than restored by a follow-up. For
providers whose endpoint answers with bytes and no JSON envelope (Hugging Face
images, every OpenAI-style text-to-speech endpoint) the native type is the
bytes and `raw` on the normalized response stays `Null`; `raw_*` is the typed
route.

Normalization is a trait, implementable out of tree: replace
`impl TryFrom<MyPayload> for TranscriptionResponse<MyPayload>` with

```rust
impl NormalizeTranscriptionResponse for MyPayload {
    fn normalize(self, provider: &str) -> Result<TranscriptionResponse, TranscriptionError> {
        Ok(TranscriptionResponse::new(self.text, provider).with_usage(..))
    }
}
```

(`NormalizeImageGenerationResponse` / `NormalizeAudioGenerationResponse`
likewise). The provider name is an *input*: several providers share one wire
shape, and a conversion that hardcoded a name would mislabel every provider
but one. Custom models implement only the operation:

```rust
impl TranscriptionModel for MyModel {
    async fn transcription(&self, req: TranscriptionRequest)
        -> Result<TranscriptionResponse, TranscriptionError> { /* ... */ }
}
```

The three traits also drop `Clone` from their supertraits (and
`AudioGenerationModel` drops `Sized`), exactly as `CompletionModel` did:
`transcription_request()` / `image_generation_request()` /
`audio_generation_request()` gate on `where Self: Sized + Clone`, and each
trait is implemented for `Arc<M>` by forwarding, so wrapping a non-`Clone`
model in an `Arc` works through every generic API. Generic code that cloned a
model through one of these traits must bound `M: …Model + Clone` explicitly.

The shared OpenAI-style drivers now read the provider's transport request-id
header onto `provider_request_id` where the provider has one (OpenAI, Groq:
`x-request-id`; Mistral: `mistral-correlation-id`; Bedrock: the SDK's
`x-amzn-RequestId`). Gemini, Hugging Face, Azure, Venice, xAI and OpenRouter
report none on these endpoints; `None` is the documented outcome.

### Construction moved off every model trait

`EmbeddingModel`, `RerankModel`, `TranscriptionModel`, `ImageGenerationModel`
and `AudioGenerationModel` lose `type Client` and `fn make`, matching
`CompletionModel`. Delete both from your impls. `EmbeddingModel::MAX_DOCUMENTS`
is now `fn max_documents(&self) -> usize` (a constant cannot survive type
erasure — see the next section). `RerankModel::MAX_DOCUMENTS` and
`ImageEmbeddingModel::MAX_DOCUMENTS` follow in the "Embedding and rerank
responses" section below.

The capability client traits construct models themselves:
`EmbeddingsClient::embedding_model` / `embedding_model_with_ndims`,
`RerankingClient::rerank_model`, `TranscriptionClient::transcription_model`,
`ImageGenerationClient::image_generation_model` and
`AudioGenerationClient::audio_generation_model` (which loses its default body)
are required methods that call your model's own constructor. Call sites —
`client.embedding_model(..)`, `client.transcription_model(..)` — are
unchanged.

A provider extension built on the generic `rig::client::Client<Ext, H>`
implements the new public hooks instead: `ConstructEmbeddingModel<C>`
(`construct(client, model, ndims: Option<usize>)`), `ConstructRerankModel<C>`,
`ConstructTranscriptionModel<C>`, `ConstructImageGenerationModel<C>`,
`ConstructAudioGenerationModel<C>` (`construct(client, model)`), all beside
`ConstructCompletionModel`. The blanket capability-client impls over
`Client<Ext, H>` bound on them, so an out-of-tree extension reaches
`embedding_model`/`transcription_model`/… through public API only — the
orphan rule that forced `ConstructCompletionModel` to be public applies to
every modality, and this closes the hole. `rig-core` ships a compile probe of
exactly that (`client::external_modality_extension_probe`).

Tests that called `Model::make(&client, ..)` directly should go through the
client: `client.embedding_model(..)`.

`ImageEmbeddingModel` also drops `Clone` from its supertraits.

### Vector stores erase the embedding model at construction

Every vector store and index lost its embedding-model type parameter — the
structural twin of `Agent<M>` losing its model:

```rust
// 0.42
let index: QdrantVectorStore<openai::EmbeddingModel> = QdrantVectorStore::new(client, model, ..);
let index: InMemoryVectorIndex<openai::EmbeddingModel, Doc> = store.index(model);
// now
let index: QdrantVectorStore = QdrantVectorStore::new(client, model, ..);
let index: InMemoryVectorIndex<Doc> = store.index(model);
```

Constructors take `impl EmbeddingModel + 'static` and erase it once into
`rig::embeddings::EmbeddingModelHandle`, a cloneable handle that itself
implements `EmbeddingModel`; `ndims()` and `max_documents()` are captured by
value at erasure. Affected: `InMemoryVectorIndex<M, D>` → `<D>`,
`QdrantVectorStore<M>`, `LanceDbVectorIndex<M>`, `ScyllaDbVectorStore<M>`,
`MongoDbVectorIndex<C, M>` → `<C>`, `MilvusVectorStore<M>`,
`VectorizeVectorStore<M>`, `Neo4jVectorIndex<M>`, `S3VectorsVectorStore<M>`,
`PostgresVectorStore<M>`, `SqliteVectorStore<E, T>` / `SqliteVectorIndex<E, T>`
→ `<T>`, `SurrealVectorStore<C, M>` → `<C>`, `HelixDBVectorStore<C, E>` →
`<C>`. Construction call sites are unchanged; delete the parameter from type
annotations. Heterogeneous collections of indexes (`Vec<Box<dyn
VectorStoreIndex<Filter = _>>>`) no longer need a provider name per element.

Unlike `Agent`, this is **not** a swapping mechanism and there is no
`set_model`: an index populated under one model is only meaningful under that
model, so the handle a store holds is fixed for its lifetime. The payoff is
type ergonomics and dyn-storability, nothing more. `EmbeddingsBuilder<M, T>`
keeps its parameter: it is transient and dropped at `build()`, like
`CompletionRequestBuilder`.

### Embedding and rerank responses are concrete and normalized; `embed_texts_with_usage` is `embed_texts_response`

The two response types #2385 left alone get the same treatment as every
other modality. `EmbeddingResponse` and `RerankResponse` now carry the full
normalized metadata:

```rust
pub struct EmbeddingResponse {           // RerankResponse: results: Vec<RerankResult>
    pub embeddings: Vec<Embedding>,
    pub usage: Usage,                    // zero when the provider reports none
    pub provider: String,                // stable descriptor name, always set
    pub model: Option<String>,           // provider-reported model, when any
    pub response_id: Option<String>,
    pub provider_request_id: Option<String>,
    pub raw: serde_json::Value,          // the provider's own payload, serialized
}
```

plus `identity() -> ResponseIdentity`, and a new `ImageEmbeddingResponse` of
the same shape for `ImageEmbeddingModel`. Both types derive `Serialize` /
`Deserialize`; construct with `EmbeddingResponse::new(embeddings, provider)`
and the `with_*` setters.

**`EmbeddingModel::embed_texts_with_usage` is renamed `embed_texts_response`**
(and `embed_text_with_usage` → `embed_text_response`): it now returns identity
and the raw payload, not only usage, so the old name would lie. It is also now
the **required** method — `embed_texts`, `embed_text` and
`embed_text_response` are defaults derived from it. Custom models flip what
they implement:

```rust
// 0.42
impl EmbeddingModel for MyModel {
    const MAX_DOCUMENTS: usize = 100;
    fn ndims(&self) -> usize { 768 }
    async fn embed_texts(&self, texts: impl IntoIterator<Item = String> + Send)
        -> Result<Vec<Embedding>, EmbeddingError> { /* ... */ }
}
// now
impl EmbeddingModel for MyModel {
    fn max_documents(&self) -> usize { 100 }
    fn ndims(&self) -> usize { 768 }
    async fn embed_texts_response(&self, texts: impl IntoIterator<Item = String> + Send)
        -> Result<EmbeddingResponse, EmbeddingError> {
        Ok(EmbeddingResponse::new(/* embeddings */ vec![], "my-provider"))
    }
}
```

(The other direction is impossible: a defaulted full method forwarding to
`embed_texts` would have to invent the provider name.) `ImageEmbeddingModel`
likewise: `embed_images_response` is required, `embed_images` / `embed_image`
derive from it, and `MAX_DOCUMENTS` is `fn max_documents()`.

**`RerankModel::MAX_DOCUMENTS` is `fn max_documents(&self) -> usize`** (a
constant cannot survive erasure), and **`RerankResponse::model` is now
`Option<String>`** — a server that omits `model` still produced a ranking,
and `None` is the honest report where the shared Jina-shaped driver used to
substitute the requested name. Replace `response.model == "x"` with
`response.model.as_deref() == Some("x")`.

Normalization is a trait, implementable out of tree: `NormalizeEmbeddingResponse`
(`normalize(self, provider: &str, documents: Vec<String>)` — the request's
inputs, in order, for `Embedding::document`), `NormalizeImageEmbeddingResponse`,
`NormalizeRerankResponse` (`normalize(self, provider: &str)`). Every concrete
provider model gains inherent `raw_embed_texts` / `raw_embed_images` /
`raw_rerank` returning the provider's native type from the same request,
transport, parser and error path, and the normalized response carries that
value serialized in `raw`. Where the provider answers one request per input
(Cohere images, Bedrock embeddings, Gemini gRPC) the raw route returns a
`Vec` of answers; where there is no JSON payload at all (FastEmbed in-process,
Gemini gRPC's prost messages) `raw` stays `Null` and `raw_*` is the typed
route. Provider wire types that were private are now public where they are
the raw route's return type (`openai::CompatibleEmbeddingResponse`,
`copilot::CopilotEmbeddingResponse`, `cohere::{ImageEmbeddingResponse,
FloatEmbeddings}`, the Jina rerank types in `providers::internal::rerank`,
`gemini::embedding::gemini_api_types`), and gain `Serialize`.

The shared OpenAI-compatible embeddings driver reads `x-request-id` onto
`provider_request_id` for OpenAI (and Copilot does for its route); the other
embedding and rerank endpoints report no transport id, and `None` is the
documented outcome.

`EmbeddingsBuilder::build`'s result is **unchanged**: the builder aggregates
many model calls, so there is no single response identity. Identity is
per-call — call `embed_texts_response` on the model directly when you need
it.

### `RerankModelHandle` and `ImageEmbeddingModelHandle`

`rig::rerank::RerankModelHandle` and `rig::embeddings::ImageEmbeddingModelHandle`
erase any `RerankModel + 'static` / `ImageEmbeddingModel + 'static`, exactly
as `EmbeddingModelHandle` does for `EmbeddingModel`: cloneable, themselves
implementing the trait, `max_documents` / `ndims` captured by value at
erasure, the model never cloned, no `set_model`. No in-tree type held a
`RerankModel` or `ImageEmbeddingModel` generically, so nothing lost a
parameter; the handles exist for dyn-storability and parity.

Image embedding deliberately gets **no** capability client (`ImageEmbeddingsClient`)
or `Construct*` hook: one provider (Cohere) offers it, through the inherent
`cohere::Client::image_embedding_model()`, which is unchanged.

### `ModelLister` construction moved to `ConstructModelLister`

`ModelLister<H>` loses `type Client` and `fn new` — the last construction
associated type on any trait in `rig-core`. Delete both from your impls and
add the public hook beside them:

```rust
impl<H> ConstructModelLister<Client<MyExt, H>> for MyLister<H>
where
    H: Clone,
{
    fn construct(client: &Client<MyExt, H>) -> Self {
        Self { client: client.clone() }
    }
}
```

The blanket `ModelListingClient` impl over `Client<Ext, H>` bounds on it, so
an out-of-tree extension reaches `list_models` through public API only (the
`client::external_modality_extension_probe` test asserts it). `construct`
takes `&C`, like every other `Construct*` hook. `ModelLister` keeps its `H`
parameter — the transport, not a provider leak. Call sites
(`client.list_models()`) are unchanged; code that called `MyLister::new(client)`
directly calls `MyLister::construct(&client)`.

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
data). `AdditionalParams` is `#[serde(transparent)]`, so the newtype adds no
nesting of its own — the JSON under the key is exactly what a plain `Value`
would have written. What moved is the *placement*: 0.41's flatten wrote these
keys as siblings of `text`, and they now sit under the named key:

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
  and provider extras are a named field" above.
- Pre-provider-split `ToolCall` JSON is no longer migrated on load — see the
  "Persisted histories" bullet in the tool-call identity section above for
  what a legacy `call_id` key now means and how to migrate the JSON by hand.

### Errors carry the transport request id; two error-shape changes (#2314)

Failed calls now preserve the provider's transport request id. Two breaks:

- **`http_client::Error` gains the `InvalidStatusCodeWithDetails { status, body, headers }` variant** (the reqwest transport now reports non-success through it, preserving the failed response's headers). Exhaustive matches on `http_client::Error` need a new arm; its Display is identical to `InvalidStatusCodeWithMessage`, and `provider_response_status()`/`provider_response_body()` read both.
- **`ProviderResponseError` is `#[non_exhaustive]`** with a new `provider_request_id` field. Construct via `ProviderResponseError::new(status, body)` / `::without_status(body)` instead of a struct literal; read the id via the `provider_request_id()` accessor on the error enums (also forwarded through `PromptError`).

Behavior: providers with a request-id contract classify non-success HTTP responses as `CompletionError::ProviderResponse` instead of `HttpError`. The contract is a defaulted trait constant, so the affected set is wider than the providers that spell it out: anthropic and every Anthropic-dialect gateway client (`minimax::AnthropicClient`, `moonshot::AnthropicClient`, `xiaomimimo::AnthropicClient`, `zai::AnthropicClient`), which inherit `AnthropicCompatibleProvider`'s `request-id` default; openai on both APIs, plus xai and chatgpt, which inherit `ResponsesProviderExt`'s `x-request-id` default; groq and copilot; and mistral, which joined the set in #2331 with `mistral-correlation-id`. Matchers on `CompletionError::HttpError(_)` for those providers' 4xx/5xx need updating; the `provider_response_*` accessors are shape-independent and keep working. Providers that leave the constant `None` (gemini, cohere, ollama, the OpenAI-compatible default) are unchanged.

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

### Model-turn termination metadata reaches hooks (#2184)

`ModelTurnFinished` now reports why the provider stopped and what output-token
cap the attempt ran under, so a hook can retry a truncated turn without
inspecting provider-typed responses. One source-level break:

- **`ModelTurnFinished` gains `finish_reason: Option<&FinishReason>` and
  `max_tokens: Option<u64>`.** Hooks that only *read* the event are
  unaffected. Code constructing it by hand (test harnesses) must supply both;
  `None` for each preserves the old behavior:

  ```rust
  // Was
  ModelTurnFinished { turn, content, usage, identity }
  // Now
  ModelTurnFinished { turn, content, usage, identity, finish_reason: None, max_tokens: None }
  ```

Both fields describe the attempt the event fires for, not the run: on a retry
they are the retried attempt's own, and `max_tokens` reflects the merged
completion-call `RequestPatch`, so a hook that raises the cap for a retry sees
its new value rather than the agent's configured baseline. `finish_reason` is
`None` when the provider reported no reason — that is not normalized into
`FinishReason::Stop`, so match on it explicitly if you treat the two alike.

`cargo run -p rig-agent --example retry_on_truncation` is a working
retry-on-truncation policy built on the two fields, on both surfaces.

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

- **`ModelTurn` and `StreamedTurn` gain `finish_reason`.** `ModelTurn::new(..)`
  is unchanged; attach the reason with
  `.with_finish_reason(resp.finish_reason())`. Neither type is
  `#[non_exhaustive]` any more — #2335 removed the attribute workspace-wide —
  and every field on both is public, so a struct literal of *either* type needs
  the new field. It is serde-defaulted, so persisted run JSON still loads.

- **A turn that delivered no answer and reports `Length` or `ContentFilter`
  now fails** with a `CompletionError::ResponseError` instead of finalizing as
  `""`. "No answer" means no tool call, no image, and no non-empty text —
  **reasoning does not count**, which is the case most likely to affect you:
  providers bill thinking tokens against the output limit, so a thinking
  model that exhausts its budget mid-thought produces reasoning and no text,
  and that shape used to report success with an empty string. An image *is*
  an answer: a truncated image-generation turn that delivered its image still
  succeeds. This matches what the blocking Gemini path already did for a
  content-less candidate.

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

### `ToolSetBuilder` is gone, and public API goes with the consolidation pass (#2320)

**Tool sets are populated in place.** `ToolSet::builder()` and the whole
`ToolSetBuilder` type (`static_tool`, `dynamic_tool`, `portable_dynamic_tool`,
`retrieved_tool`, `build`) are removed, and the `rig` facade no longer
re-exports `ToolSetBuilder` from `rig::tool`. Register on the set itself:

```rust
// Was
let toolset = ToolSet::builder()
    .static_tool(Adder)
    .retrieved_tool(Subtract)
    .build();
// Now
let mut toolset = ToolSet::default();
toolset.add_tool(Adder);
toolset.add_retrieved_tool(Subtract);
```

`add_retrieved_tool` is new in this pass and is the builder's `retrieved_tool`;
`add_tool`, `add_dynamic_tool`, `add_portable_dynamic_tool`, `from_tools` and
`from_dynamic_tools` were already there.

**Superseded guidance.** The 0.40 → 0.41 registration table below, and its row
in the appendix, name `ToolSetBuilder::retrieved_tool` and
`ToolSetBuilder::dynamic_tool(DynamicTool)` as the destination of the 0.41
renames. Those rows stay accurate as history for 0.41; on `next` the
destination is the matching `ToolSet::add_*` method.

**`ProviderResponseExt` loses two items.** `type OutputMessage` and
`get_output_messages` are gone — nothing read them
(`SpanCombinator::record_response_metadata` records only the response id and
model name). An out-of-tree implementation that still defines them fails with
E0437/E0407; delete both. `get_text_response` is unchanged.

**Four smaller removals.** `json_utils::null_or_vec` folds into
`null_or_default` — switch `deserialize_with = "null_or_vec"` to
`deserialize_with = "null_or_default"`, which behaves the same for a `Vec<T>`
field. `anthropic::completion::apply_cache_control` is deleted with no public
successor (`apply_prompt_cache_control` is `pub(super)`); the provider applies
the breakpoints on the way out. `rig-s3vectors`' exported `document!` macro is
deleted; build `aws_smithy_types::Document` values directly. And
`TryFrom<message::AssistantContent> for anthropic::completion::Content` is
deleted with no public successor: one rig content block no longer maps to one
Anthropic block (an empty text block with nothing this wire can carry now
produces none, a multi-block `Reasoning` produces several), so the conversion
became a private one-to-many function. Convert a whole message with the
surviving `TryFrom<message::Message> for anthropic::completion::Message`
instead. The inbound direction, `TryFrom<Content> for message::AssistantContent`,
is unchanged.

**Anthropic citations carry a payload struct.** Every locator variant of
`anthropic::completion::Citation` is now a newtype wrapping a same-named
struct — `CharLocationCitation`, `PageLocationCitation`,
`ContentBlockLocationCitation`, `SearchResultLocationCitation` and
`WebSearchResultLocationCitation`, all newly public with public fields — so
each variant's fields move one level down:

```rust
// Before
match citation {
    Citation::CharLocation { cited_text, document_index, .. } => { /* ... */ }
    Citation::WebSearchResultLocation { url, encrypted_index, .. } => { /* ... */ }
    Citation::Unknown(raw) => { /* ... */ }
    _ => {}
}

// Now
match citation {
    Citation::CharLocation(
        CharLocationCitation { cited_text, document_index, .. },
    ) => { /* ... */ }
    Citation::WebSearchResultLocation(
        WebSearchResultLocationCitation { url, encrypted_index, .. },
    ) => { /* ... */ }
    Citation::Unknown(raw) => { /* ... */ }
    _ => {}
}
```

Construction moves the same way — wrap the braces in the payload type. Field
names and types are unchanged, `Citation::Unknown(serde_json::Value)` is
untouched, and the hand-written `Serialize`/`Deserialize` still reads and
writes the same `type`-tagged shapes (`char_location`, `page_location`,
`content_block_location`, `search_result_location`,
`web_search_result_location`), so persisted citations still load and the
serialized JSON is unchanged. This is a `match`-and-literal break only. You
reach these values through the provider-native surfaces — `Content::Text`'s
`citations` on `raw_completion`, `streaming::ContentDelta::CitationsDelta` on
`raw_stream` — or off a normalized text part with the unchanged
`anthropic::completion::anthropic_citations` helper.

**Gemini Interactions deltas reuse the content types.**
`interactions_api_types::ContentDelta`'s variants now carry `ImageContent`,
`AudioContent`, `DocumentContent`, `VideoContent`, `FunctionCallContent`,
`FunctionResultContent`, `CodeExecutionCallContent`,
`CodeExecutionResultContent`, `UrlContextCallContent`,
`UrlContextResultContent`, `GoogleSearchCallContent`,
`GoogleSearchResultContent`, `McpServerToolCallContent`,
`McpServerToolResultContent` and `FileSearchResultContent`, and the fifteen
identically-shaped `*Delta` structs they replace are deleted. `TextDelta`,
`ThoughtSummaryDelta` and `ThoughtSignatureDelta` stay, joined by the
`ArgumentsDelta` that #2262 added earlier in this cycle — these are the deltas
whose payload really differs from its content counterpart. The JSON is
unchanged; only names in a `match` or a type annotation are.

**rig-bedrock converts from the mirror, not the SDK.** `RigAssistantContent`
and `RigUserContent` convert from `types::converse_output::ContentBlock`,
`RigMessage` from `converse_output::Message`, `RigImage` from `ImageBlock`,
`RigDocument` from `DocumentBlock`, and `RigToolResultContent` from
`ToolResultContentBlock` — each took the `aws_bedrock::` type before. All 38
mirror→SDK `TryFrom` impls are deleted outright — the 17 written out by hand
and the 21 the `mirror_enum!` / `mirror_union!` / `mirror_location!` macros
generated — so there is no supported way back from a mirror value to its
`aws_sdk_bedrockruntime` counterpart.

### Dead public API removed (#2301)

A sweep over public items with no in-tree caller. Nothing here changes
behavior; if you called one, either drop the call or use the replacement.

| Removed | Replacement |
| --- | --- |
| `ModelListingError::{RateLimitError, ServiceUnavailable, UnknownError}` | `ApiError` — the shared listing driver classified every real failure as `ApiError`, `RequestError` or `ParseError` |
| `ModelListingError::{auth_error, rate_limit_error, service_unavailable, unknown_error}` | build the remaining variants directly; `api_error`, `request_error` and `parse_error` stay |
| `message::Reasoning::optional_id` | assign the public `id` field, or `with_id` for the `Some` case |
| `message::Image::try_into_url` | none |
| `message::DocumentSourceKind::{raw, unknown}` | `DocumentSourceKind::Raw(..)` / `DocumentSourceKind::Unknown` — the variants stay, only the constructors went |
| `message::Message::assistant_with_id` | none |
| `CompletionRequest::{with_provider_tool, with_provider_tools}` | none |
| `streaming::RawStreamingToolCall::with_internal_call_id` | none |
| `vector_store::in_memory_store::InMemoryVectorStore::get_document` | `iter()` |
| `azure::{EmbeddingResponse, EmbeddingData, Usage}` | `openai::embedding::{EmbeddingResponse, EmbeddingData}` plus `openai::completion::Usage` — `openai::embedding` publishes no `Usage` path of its own. Azure's embeddings run through the shared path now, which reads `EmbeddingData` directly and wraps it in an internal response type rather than the public `EmbeddingResponse` |
| `ollama::{UserContent, ImageUrl}`, and ollama's `SystemContent`/`SystemContentType` re-export | the identically-named `openai::completion` types |
| `ollama::AssistantContent` | none |
| `deepseek::Message::{System, User, ToolResult}` | none — `Message` is the response-side shape and only `Assistant` ever appeared there |
| `doubleword::client::doubleword_api_types` (`ApiErrorResponse`, `ApiError`, `ApiResponse<T>`) | none |
| `openai::responses_api::OutputReasoning` | none |
| `TryFrom<message::Message> for Vec<responses_api::Message>` | the live `TryFrom<completion::Message> for Vec<InputItem>` converter |
| `rig-agent`'s `MultiTurnStreamItem::final_response_with_history` | `final_response` |

One of these reaches persisted data. `ModelListingError` derives `Serialize`
and `Deserialize`, so a stored value tagged `RateLimitError`,
`ServiceUnavailable` or `UnknownError` no longer loads; re-encode such records
as `ApiError` or `RequestError` before upgrading.

### Four public items removed, and `azure::EmbeddingModel` takes `ndims: usize` (#2305)

- **`http_client::with_bearer_auth(req, auth)` is gone.** It only wrapped
  `bearer_auth_header`, which remains: take the builder's header map and call
  `http_client::bearer_auth_header(req.headers_mut().ok_or(Error::NoHeaders)?, auth)?`.
- **`InMemoryVectorStoreBuilder::documents_with_id_f` is gone.** Either
  precompute the ids and use `documents_with_ids`, or build the store directly
  with `InMemoryVectorStore::from_documents_with_id_f` /
  `add_documents_with_id_f`, which are different methods on the store and were
  not removed. Their element type follows the workspace-wide
  `OneOrMany<Embedding>` → `Vec<Embedding>` change, and
  `add_documents_with_id_f` now takes `impl IntoIterator` where it took a
  `Vec`.
- **`mira::MiraError` and mira's inherent `Client::list_models` are gone.** Mira
  now uses the shared lister every other provider uses. Bring
  `ModelListingClient` into scope and call `list_models()` there: it returns
  `Result<ModelList, ModelListingError>` rather than
  `Result<Vec<String>, MiraError>`, so a `match` on
  `MiraError::{InvalidApiKey, ApiError, RequestError, Utf8Error, JsonError}`
  becomes a match on `ModelListingError`, and the model ids come off the entries
  rather than being the items themselves.
- **`azure::EmbeddingModel` is a type alias.** It is now
  `openai::embedding::GenericEmbeddingModel<AzureExt, T>`, so the inherent
  `new(client, model, ndims: Option<usize>)` and
  `with_model(client, model, ndims: Option<usize>)` are replaced by the generic
  ones taking `ndims: usize`:

  ```rust
  // Before
  let model = azure::EmbeddingModel::new(client, TEXT_EMBEDDING_3_SMALL, Some(1536));
  let inferred = azure::EmbeddingModel::new(client, TEXT_EMBEDDING_3_SMALL, None);

  // Now
  let model = azure::EmbeddingModel::new(client, TEXT_EMBEDDING_3_SMALL, 1536);
  // `None` meant "infer from the model identifier" — that lives on the trait:
  use rig::embeddings::EmbeddingModel as _;
  let inferred = azure::EmbeddingModel::make(&client, TEXT_EMBEDDING_3_SMALL, None);
  ```

  The client helpers `embedding_model` / `embedding_model_with_ndims` are
  unchanged and remain the shortest path.

### Copilot's chat response is the shared OpenAI type (#2308)

`providers::copilot::CopilotCompletionResponse::Chat` held a
`Box<copilot::ChatCompletionResponse>`; it now holds a
`Box<openai::completion::CompletionResponse>`, and both
`copilot::ChatCompletionResponse` and `copilot::ChatChoice` are gone — they were
a field-for-field copy of the shared OpenAI chat wire types.

`CompletionModel::raw_completion` returns `CopilotCompletionResponse`, so this is
the surface anyone inspecting Copilot's provider-native response sees:

```rust
// Before — 0.41 had no `raw_completion`; the native value arrived on
// `CompletionResponse::raw_response`.
use rig::providers::copilot::CopilotCompletionResponse;
let response = model.completion(req).await?;
if let CopilotCompletionResponse::Chat(chat) = response.raw_response {
    let reason: Option<String> = chat.choices[0].finish_reason.clone();
}

// Now
use rig::providers::copilot::CopilotCompletionResponse;
if let CopilotCompletionResponse::Chat(chat) = model.raw_completion(req).await? {
    let reason: String = chat.choices[0].finish_reason.clone(); // "" means absent
}
```

The optionality moves with the type: `object`, `created` and the choice's
`finish_reason` are `String`, `u64` and `String` instead of `Option`, with
"absent" spelled as the empty string or `0`. Persisted `CopilotCompletionResponse`
JSON still loads — the shared type accepts a missing key or an explicit `null`
for all three, as described under Silent behavior changes — but re-serializing
writes `""` and `0` where it used to write `null`.

### Doubleword embeddings ride the shared OpenAI-compatible path (#2286)

`doubleword::EmbeddingModel` is a type alias for
`openai::embedding::GenericEmbeddingModel<DoublewordExt, T>` instead of its own
struct, and the three response types the hand-rolled path decoded —
`doubleword::{EmbeddingResponse, EmbeddingData, Usage}`, re-exported from
`providers::doubleword` by `pub use embedding::*` — are deleted.

`EmbeddingModel::new(client, model, ndims)` keeps its signature and the
`EmbeddingModel` trait impl is unchanged, so building and using the model needs
no edits. Only code that *named* those response types has to move: decode the
data entries with `openai::embedding::EmbeddingData`, but wrap them in a
response type of your own whose `usage` key is optional.
`openai::embedding::EmbeddingResponse` is not a drop-in — it declares a
required `usage`, which the deleted `doubleword::EmbeddingResponse` did not
have at all and which Doubleword does not always send. That is also why rig's
own shared path decodes through an internal `usage`-optional type and marks
Doubleword `REQUIRES_USAGE = false`.

The behavioral half: `embed_texts_with_usage` no longer falls through to the
zero-usage default, so Doubleword embeddings now report the `usage` the API
already returned. The request bytes are unchanged — Doubleword still never
receives a `dimensions` field.

### rig-candle drops seven public items (#2310)

Each was an alias over an API that is still there, so every call site has a
one-line replacement:

| Removed | Use instead |
| --- | --- |
| `LlamaModelBuilder<'a>` (also re-exported from the crate root) | `CandleModelBuilder<'a>` |
| `CandleModel::from_artifacts(artifacts)` | `CandleModel::builder_from_artifacts(artifacts).build()` |
| `CandleModel::from_artifacts_async(artifacts)` | `CandleModel::builder_from_artifacts(artifacts).build_async().await` |
| `CandleModel::from_gguf_async(data)` | `CandleModel::builder_from_artifacts(ModelArtifacts::Gguf(data)).build_async().await` |
| `CandleModel::from_gguf_bytes_async(data)` | `CandleModel::builder_from_gguf_bytes(data).build_async().await` |
| `CandleModel::model_family()` | `CandleModel::conversation_protocol()` |
| `CandleModelBuilder::model_family(family)` | `CandleModelBuilder::conversation_protocol(protocol)` |

Nothing loses a capability. `ModelFamily` itself is untouched — it is still
exported from the crate root as an alias for `ConversationProtocol`, so only the
two method names change and the argument and return types are the same. The
synchronous `CandleModel::from_gguf` and `from_gguf_bytes` also stay; only their
`_async` twins are gone, and `build_async` is available on
`CandleModelBuilder<'static>`, which is what `builder_from_artifacts` and a
`'static` buffer passed to `builder_from_gguf_bytes` both give you.

### Bedrock model identifiers (#2309)

`rig-bedrock` shipped 72 `pub const` model ids in `crate::completion`; 40
remain.

**38 constants are gone and will not compile.** Every `ANTHROPIC_CLAUDE*`
constant the crate had — `ANTHROPIC_CLAUDE`, `_2`, `_2_1`, `_INSTANT`,
`_INSTANT_V1_2`, `_3_HAIKU`, `_3_OPUS`, `_3_SONNET`, `_3_5_HAIKU`,
`_3_5_SONNET`, `_3_5_SONNET_V2`, `_3_7_SONNET`, `_OPUS_4`, `_SONNET_4` — plus
`AMAZON_TITAN_TEXT_EXPRESS_V1`, `AMAZON_TITAN_TEXT_LITE_V1`,
`AMAZON_TITAN_TEXT_PREMIER_V1_0`, `AMAZON_TITAN_IMAGE_GENERATOR_G1`,
`AMAZON_TITAN_IMAGE_GENERATOR_G1_V2`, `AMAZON_NOVA_PREMIER`,
`AI21_JAMBA_INSTRUCT`, `AI21_JAMBA_1_5_LARGE`, `AI21_JAMBA_1_5_MINI`,
`COHERE_COMMAND`, `COHERE_COMMAND_LIGHT_TEXT`, `COHERE_COMMAND_R`,
`COHERE_COMMAND_R_PLUS`, `LLAMA_3_1_405B_INSTRUCT`, `LLAMA_3_2_1B_INSTRUCT`,
`LLAMA_3_2_3B_INSTRUCT`, `LLAMA_3_2_11B_INSTRUCT`, `LLAMA_3_2_90B_INSTRUCT`,
`MISTRAL_LARGE_24_07`, `STABILITY_SD3_LARGE_1_0`, `STABILITY_SDXL_1_0`,
`STABILITY_STABLE_IMAGE_CORE_1_0_V1_0`,
`STABILITY_STABLE_IMAGE_ULTRA_1_0_V1_0` and `TWELVELABS_MARENGO_EMBED_V2_7`.
`crate::image` additionally drops its aliases
`AMAZON_TITAN_IMAGE_GENERATOR_V1` and `AMAZON_TITAN_IMAGE_GENERATOR_V2_0`, and
now re-exports `STABILITY_SD3_5_LARGE`, `STABILITY_STABLE_IMAGE_CORE_1_0` and
`STABILITY_STABLE_IMAGE_ULTRA_1_0` beside `AMAZON_NOVA_CANVAS`.

None of them could be invoked. Checked against `ListFoundationModels` in
us-east-1, us-west-2, eu-central-1 and ap-northeast-1, each is either absent
from every region — Bedrock answers `ResourceNotFoundException` ("This model
version has reached the end of its life") — or servable only through a
cross-region inference profile that is itself retired. For Claude, move to one
of the six replacements: `ANTHROPIC_CLAUDE_HAIKU_4_5`,
`ANTHROPIC_CLAUDE_SONNET_4_5`, `ANTHROPIC_CLAUDE_OPUS_4_5`,
`ANTHROPIC_CLAUDE_SONNET_4_6`, `ANTHROPIC_CLAUDE_SONNET_5`,
`ANTHROPIC_CLAUDE_OPUS_5`. For anything else, pass the identifier your account
can actually reach as a plain string — `CompletionModel::new(client, model)`
takes any `impl Into<String>`, so the constants are a convenience, not a gate.

**Seven surviving constants silently changed value.** These still compile, and
now name a US cross-region inference profile:

| Constant | 0.41 | next |
| --- | --- | --- |
| `DEEPSEEK_R1` | `deepseek.r1-v1:0` | `us.deepseek.r1-v1:0` |
| `META_LLAMA_3_3_70B_INSTRUCT` | `meta.llama3-3-70b-instruct-v1:0` | `us.meta.llama3-3-70b-instruct-v1:0` |
| `META_LLAMA_4_MAVERICK_17B_INSTRUCT` | `meta.llama4-maverick-17b-instruct-v1:0` | `us.meta.llama4-maverick-17b-instruct-v1:0` |
| `META_LLAMA_4_SCOUT_17B_INSTRUCT` | `meta.llama4-scout-17b-instruct-v1:0` | `us.meta.llama4-scout-17b-instruct-v1:0` |
| `MISTRAL_PIXTRAL_LARGE_2502` | `mistral.pixtral-large-2502-v1:0` | `us.mistral.pixtral-large-2502-v1:0` |
| `WRITER_PALMYRA_X4` | `writer.palmyra-x4-v1:0` | `us.writer.palmyra-x4-v1:0` |
| `WRITER_PALMYRA_X5` | `writer.palmyra-x5-v1:0` | `us.writer.palmyra-x5-v1:0` |

The bare identifiers they replaced answer `ValidationException` ("Invocation of
model ID … with on-demand throughput isn't supported"), so the new values are
the working ones — but a `us.` prefix names a *region family*, and the six
`ANTHROPIC_CLAUDE_*` replacements above carry it too. If your client runs in
Europe or Asia-Pacific, substitute `eu.` or `apac.` (some Anthropic models also
offer `global.`) instead of using the constant verbatim; otherwise code that
compiled and ran before will now name a profile your region cannot invoke.

### `#[non_exhaustive]` is gone from the whole workspace (#2335)

All 53 `#[non_exhaustive]` attributes were removed from `rig-core`, `rig-agent`,
`rig-bedrock` and `rig-candle`. Every public enum in these crates can now be
matched exhaustively without a wildcard, and every public struct whose fields
are all `pub` can be built with a struct literal or functional update from any
crate.

The attribute was not always the only thing in the way. Types that keep private
or `pub(crate)` fields stay constructor-only — `Agent` and `AgentRunner` (every
field `pub(crate)`), `EmbeddingsBuilder` (private `model` and `documents`),
`AudioGenerationRequestBuilder` / `ImageGenerationRequestBuilder` (private
builder state), and `completion::CompletionResponse` (private `finish_reason`).
Keep reaching those through `AgentBuilder`, `AgentRunner::from_agent`,
`EmbeddingsBuilder::new`, the request builders and `CompletionResponse::new`;
nothing about their construction changed.

**Nothing breaks.** This is a permissive change — `cargo semver-checks` reports
"no semver update required". You do not have to change any code.

**But you may get a new warning.** A `match` arm that existed only to satisfy a
previously non-exhaustive enum can now be unreachable:

```rust
// Was required; now warns `unreachable_patterns`
match content {
    ReasoningContent::Summary(t) => …,
    ReasoningContent::Text { .. } => …,
    ReasoningContent::Encrypted(d) => …,
    ReasoningContent::Redacted { .. } => …,
    _ => …,   // <- delete this
}
```

Delete the wildcard once you cover every variant. If you build with
`-D warnings` this is a hard failure rather than a warning; two in-tree matches
over `ReasoningContent` needed exactly this fix.

**Keep the wildcard if you target both native and wasm.** `CandleError` has
three `#[cfg(not(target_family = "wasm"))]` variants, so a single exhaustive
`match` can no longer compile for both targets: the arms are missing variants on
wasm, and adding them plus a wildcard trips `unreachable_patterns` on wasm. A
wildcard-only match remains the portable form.

**What this costs going forward.** The bargain is now reversed: adding a field
to any of these structs, or a variant to any of these enums, is a breaking
change and has to wait for a breaking release. `#[non_exhaustive]` also cannot
be added back outside a breaking window, so this is not a decision that can be
revisited cheaply.

**One invariant is widened.** `StreamFinal` and `completion::CompletionResponse`
normalize an empty identifier string to `None` — the rule lives in the
`with_*_id` setters ("an empty string is treated as absent … so the invariant
lives here rather than at every provider call site"), and both types route
`Deserialize` through those same setters. The attribute was the last thing
forcing external construction down that path. A hand-written literal can now
produce `Some("")` where every rig-built value has `None`, and
`StreamedAssistantContent::final_response` accepts a `StreamFinal` by value. If
you build either type yourself, prefer `::new(..)` plus the `with_*` setters
over a literal.

**Superseded guidance.** Earlier sections of this file and the changelog
describe types as `#[non_exhaustive]` and tell you to construct them through
constructors for that reason. Those passages remain accurate as history for the
releases they document, but the attribute claim no longer holds on `next`:
"Core errors are `#[non_exhaustive]`" (0.40→0.41), `ProviderResponseError`
(#2314), `CompletionResponse`, `ProviderCapabilities`, `ModelTurn`/`StreamedTurn`
(#2322), `MultiTurnStreamItem`, `DocumentSourceKind`, `RerankError`,
`MemoryError`, and openai's `ToolChoice`. The constructors those passages point
at all still exist and are still the recommended way to build these types — only
the compiler no longer insists.

### Raw provider responses are reachable from an agent (#2366)

Rig kept the provider's exact body on every *failed* call
(`ProviderResponseError::body`) and nothing on a successful one — and after
#2257 erased the model type at agent construction, `raw_completion` /
`raw_stream` became unreachable from an agent run altogether (`ModelHandle`
holds a two-method `ErasedModel` with no downcast). Both are addressed the way
identity (#2313) and finish reason (#2324) were, threaded through both
surfaces:

```rust
let response = agent.prompt("…").extended_details().await?;
for call in response.completion_calls() {
    let stop_sequence = &call.raw["stop_sequence"];   // the provider's own response, serialized
}
```

`CompletionResponse::raw` and `StreamFinal::raw` (`serde_json::Value`) carry
**the value the model's inherent `raw_completion` / `raw_stream` would have
returned, serialized** — the response as rig's wire type parsed it, not the
literal bytes. Every provider seam populates it unconditionally, per call —
the same parity pre-#2257 `raw_response: T` had. The payload is exposed on
the `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished` hook
events (`raw: &serde_json::Value`), on each `CompletionCall` in
`PromptResponse::completion_calls`, and on the streamed
`StreamedAssistantContent::Final` terminal record. It is per attempt: on a
retried turn it is the retried attempt's own. Typed access is recoverable
(`anthropic::CompletionResponse::deserialize(&raw)?`), and
`NormalizeCompletionResponse` converts forward. The field is a plain `Value`,
not an `Option`: `Value::Null` means the value was built without a provider
behind it — a hand-constructed response, a test double, or one persisted
before the field existed — never that a provider sent nothing (no seam
produces `Null`) and never that capture was switched off; there is nothing to
switch. Six source-level breaks:

- **`CompletionResponse` and `StreamFinal` gain `raw`.** Both are built with
  `new(..)` plus setters (`with_raw` joins the shared metadata setters), so
  only a struct literal or exhaustive destructure needs the field
  (`raw: serde_json::Value::Null` reproduces the old value). Serde-defaulted
  and omitted when `Null`: persisted responses and terminal records load
  unchanged.

- **`normalize_stream` requires `R: Serialize`.** It now serializes the
  provider-native terminal onto `StreamFinal::raw` before mapping it. An
  out-of-tree provider whose terminal type is not `Serialize` must derive it
  (every in-tree terminal already is). The signature is otherwise unchanged.

- **The `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished`
  hook events gain `raw: &serde_json::Value`.** Hooks that only read events
  are unaffected; hand-constructed events (test harnesses) add
  `raw: &serde_json::Value::Null`, exactly like `finish_reason` /
  `max_tokens` on `ModelTurnFinished` (#2184).

- **`ModelTurn` gains `raw`.** `ModelTurn::new(..)` is unchanged; attach with
  `.with_raw(resp.raw.clone())`. All fields are public and the type is not
  `#[non_exhaustive]`, so a struct literal needs the field. Serde-defaulted;
  persisted run state loads.

- **`AgentRun::record_streamed_completion_call` takes the attempt's raw payload
  as a fourth argument.** Read it off the same terminal record you read
  identity and finish reason from; pass `serde_json::Value::Null` when no
  terminal arrived:

  ```rust
  // Was
  run.record_streamed_completion_call(usage, identity, finish_reason)?;
  // Now
  run.record_streamed_completion_call(usage, identity, finish_reason,
      terminal.map_or(serde_json::Value::Null, |t| t.raw.clone()))?;
  ```

- **`CompletionCall` is no longer `Eq`.** `serde_json::Value` is `PartialEq`
  but not `Eq` (floats), so a `CompletionCall` cannot be a set or map key any
  more. `PartialEq`, `Clone`, and serde are unchanged; `raw` is
  serde-defaulted and omitted when `Null`.

One additive change makes the *typed* escape hatch complete:
`openai::GenericCompletionModel::raw_completion_with_request_id` is now public
(it was `pub(crate)`), and `copilot::CompletionModel` gains the same method,
returning `(raw, Option<String>)`. On the OpenAI-compatible family and
Copilot's chat route the transport request id has no slot on the shared wire
type, so plain `raw_completion(..).normalize(..)` silently lacked the
`provider_request_id` that `completion()` reports; reassemble with
`.with_optional_provider_request_id(id)`. `CopilotCompletionResponse` also
implements `NormalizeCompletionResponse`, so both of Copilot's routes have a
public forward conversion. For every provider,
`raw_completion[_with_request_id]` + `normalize` now reproduces `completion()`'s
`identity()`, `finish_reason()`, `model`, and `usage` — pinned by the
`raw_completion_parity_matrix` cassettes.

The cost is one `serde_json::to_value` of the provider's parsed response per
call, plus a clone of the value wherever the runner clones the response or a
`CompletionCall` (a few per call); the payload travels with the response, so
a large body — an image-generation response carrying base64 bytes, say — now
lives on `PromptResponse::completion_calls`, in persisted run state, and in
transcripts that serialize either. If that is unwanted, drop `raw` at your own
boundary (`call.raw = serde_json::Value::Null`) before persisting.

Deliberately not in this change: a raw *frame* channel (rig never exposed
per-SSE-event payloads on any surface — a different mechanism), literal-body
capture (fields rig's wire type never modeled; not possible for SDK/gRPC/local
providers), and `ModelHandle::downcast_ref` (recovers the model, not the
turn's response). `cargo run -p rig-agent --example raw_response_hook` reads
a provider-specific field off `raw` on both surfaces.

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
| `rig_agent::agent::model::ModelHandle` | `rig_core::completion::ModelHandle` (re-exported at `rig_agent::ModelHandle` / `rig_agent::agent::ModelHandle`) | next |
| `rig_agent::tool::{Tool, ToolEmbedding, ErasedTool, DynamicTool, ToolSet, tool_definition}` | `rig_core::tool::{..}` (module `rig_core::tool::contextual`; re-exported at the old paths and at `rig::tool::*` without the `agent` feature) | next |
| `rig_agent::tool::server::ToolRegistrySnapshot` (struct) | `pub type ToolRegistrySnapshot = rig_core::tool::ToolCatalog` | next |
| `rig_agent::agent::hook::RequestPatch` | `rig_run::policy::RequestPatch` (re-exported at the old path) | next |
| `#[rig_tool]` contextual-tool expansion target `rig_agent::tool::Tool` | `rig_core::tool::Tool` (the macro's "contextual tools require `rig`/`rig-agent`" error is gone) | next |
| `rig_core::OneOrMany<T>` (and the `one_or_many` module, both prelude re-exports) | `Vec<T>` — no replacement type; see the conversion table in "0.41 → next" | next |
| `rig_core::EmptyListError` | none — use `message::require_non_empty` where you relied on the rejection | next |
| `one_or_many::string_or_option_one_or_many` | none — `json_utils::string_or_vec` into a `Vec<T>`, then `message::non_empty` where the `Option` carried "absent" | next |
| `ToolSet::builder()` / `ToolSetBuilder` (whole type) | `ToolSet::default()` plus `add_tool` / `add_dynamic_tool` / `add_portable_dynamic_tool` / `add_retrieved_tool` | next |
| `ModelListingError::{RateLimitError, ServiceUnavailable, UnknownError}` and the `auth_error` / `rate_limit_error` / `service_unavailable` / `unknown_error` constructors | `ApiError` (build the surviving variants directly) | next |
| `telemetry::ProviderResponseExt::{OutputMessage, get_output_messages}` | none — `get_text_response` is kept | next |
| `json_utils::null_or_vec` | `json_utils::null_or_default` | next |
| `http_client::with_bearer_auth` | `http_client::bearer_auth_header` on the builder's header map | next |
| `InMemoryVectorStoreBuilder::documents_with_id_f` | `documents_with_ids`, or the store's own `from_documents_with_id_f` / `add_documents_with_id_f` | next |
| `InMemoryVectorStore::get_document` | `iter()` | next |
| `message::Reasoning::optional_id` | the public `id` field, or `with_id` | next |
| `message::DocumentSourceKind::{raw, unknown}` | `DocumentSourceKind::Raw(..)` / `::Unknown` (the variants stay) | next |
| `message::Message::tool_result_with_call_id` | `UserContent::tool_result_with_call_id(item_id, call_id, name, content)` wrapped with `Message::from(..)`; `Message::tool_result` itself now takes `(call, name, content)` | next |
| `message::Image::try_into_url`, `message::Message::assistant_with_id`, `CompletionRequest::{with_provider_tool, with_provider_tools}`, `streaming::RawStreamingToolCall::with_internal_call_id` | none | next |
| `StreamedTurnEvent::EmitToolCallDelta::id` | none — `internal_call_id` is the correlator, and the provider's id arrives on the completed `ToolCall` | next |
| `MultiTurnStreamItem::final_response_with_history` | `final_response` | next |
| `anthropic::completion::apply_cache_control` | none — `apply_prompt_cache_control` is `pub(super)` | next |
| `TryFrom<message::AssistantContent> for anthropic::completion::Content` | none — convert the whole message with the surviving `TryFrom<message::Message> for anthropic::completion::Message` | next |
| `anthropic::completion::Citation`'s struct variants (`CharLocation { .. }` … `WebSearchResultLocation { .. }`) | the same five variants as newtypes over `CharLocationCitation`, `PageLocationCitation`, `ContentBlockLocationCitation`, `SearchResultLocationCitation`, `WebSearchResultLocationCitation` — field names and the wire shape unchanged | next |
| `azure::EmbeddingModel::{new, with_model}(.., Option<usize>)` | the `GenericEmbeddingModel` pair taking `ndims: usize`; `EmbeddingModel::make(.., None)` still infers | next |
| `azure::{EmbeddingResponse, EmbeddingData, Usage}`, `doubleword::{EmbeddingResponse, EmbeddingData, Usage}` | `openai::embedding::{EmbeddingResponse, EmbeddingData}` plus `openai::completion::Usage` (`openai::embedding` exposes no `Usage` path); for Doubleword, keep a `usage`-optional response type of your own | next |
| `doubleword::client::doubleword_api_types`, `openai::responses_api::OutputReasoning`, `deepseek::Message::{System, User, ToolResult}`, `ollama::AssistantContent` | none | next |
| `ollama::{UserContent, ImageUrl, SystemContent, SystemContentType}` | the `openai::completion` types of the same names | next |
| `copilot::ChatCompletionResponse` / `copilot::ChatChoice` | `openai::completion::CompletionResponse` / `openai::completion::Choice` | next |
| `mira::MiraError`, mira's inherent `Client::list_models` | `ModelListingClient::list_models` → `Result<ModelList, ModelListingError>` | next |
| gemini `interactions_api_types::{ImageDelta, AudioDelta, DocumentDelta, VideoDelta, FunctionCallDelta, FunctionResultDelta, CodeExecutionCallDelta, CodeExecutionResultDelta, UrlContextCallDelta, UrlContextResultDelta, GoogleSearchCallDelta, GoogleSearchResultDelta, McpServerToolCallDelta, McpServerToolResultDelta, FileSearchResultDelta}` | the matching `*Content` types, carried directly by `ContentDelta` | next |
| `rig_bedrock::streaming::BedrockUsage` | `rig_bedrock::types::converse_output::TokenUsage` | next |
| rig-bedrock's mirror→AWS-SDK `TryFrom` impls (`types::converse_output`) | none — the conversion is one-way now | next |
| `rig_candle::LlamaModelBuilder<'a>` | `rig_candle::CandleModelBuilder<'a>` | next |
| `CandleModel::{from_artifacts, from_artifacts_async, from_gguf_async, from_gguf_bytes_async}` | `builder_from_artifacts` / `builder_from_gguf_bytes` plus `build()` / `build_async()` | next |
| `CandleModel::model_family()` / `CandleModelBuilder::model_family(..)` | `conversation_protocol()` / `conversation_protocol(..)` | next |
| `rig_s3vectors::document!` | none — build `aws_smithy_types::Document` values directly | next |
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
| `ToolSetBuilder::dynamic_tool(ToolEmbedding)` | `retrieved_tool` (0.41), then `ToolSet::add_retrieved_tool` (next) | 0.41 |
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
