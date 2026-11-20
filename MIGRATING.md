# Migrating Rig

<!-- MIGRATING-GUIDE-INSTRUCTIONS:START -->

## Maintainers: generate this guide for every release

> [!IMPORTANT]
> This section is the immutable preamble. Keep it at the top of this file,
> unchanged and between these markers, when updating the migration guide. Edit
> the release material below the end marker. CI pins both its placement and its
> contents.

This file is edited only on the release PR (a `release-plz-*` branch or a PR
labeled `release`). An ordinary PR puts its migration notes in the PR
description under `## Migration`; CI rejects any other edit to this file.

`MIGRATING.md` is an editorial synthesis, not the output of one generator. Use
the public API diff as its exhaustive spine, the PR notes for release context,
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
3. Run `scripts/release-notes.sh PREVIOUS_TAG` and read the `## Changelog` and
   `## Migration` sections of every pull request merged since the previous tag
   (`prs.md` in its output), together with the changelog section release-plz
   generated for the release. Use their breaking-change entries to explain
   intent and identify the replacement API, but do not assume they are
   exhaustive.
4. Inspect the pull request, commits, tests, and documentation for each change
   found by the API diff or the PR notes. Record the old form, the new form, and
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

The public API diff finds compiler-visible changes; the PR notes and targeted
history explain why and how to migrate; the final behavior review catches the
changes that still compile. All three inputs are required for each release.

<!-- MIGRATING-GUIDE-INSTRUCTIONS:END -->

This guide covers migration from 0.30 through 0.47.
Releases 0.36, 0.37, 0.40, 0.41, 0.43, 0.44, 0.45, 0.46 and 0.47 were
disruptive; 0.40 alone carried 31 breaking changes, and 0.37 renamed
`rig-core`'s library target.

## Which sections apply to you

Sections run newest-first. Find the version you are on and read every section
above it, in order. Each one is self-contained.

| You are on | Start at |
| --- | --- |
| 0.46 | [0.46 → 0.47](#046--047) |
| 0.45 | [0.45 → 0.46](#045--046) |
| 0.44 | [0.44 → 0.45](#044--045) |
| 0.43 | [0.43 → 0.44](#043--044) |
| 0.42 | [0.42 → 0.43](#042--043) |
| 0.41 | [0.41 → 0.42](#041--042) |
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

### 0.42

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

### 0.43

The 0.43 changes that compile unchanged are listed under
[Behavior Changes](#behavior-changes-1) in the 0.42 → 0.43 section.

### 0.44

The 0.44 changes that compile unchanged are listed under
[Behavior Changes](#behavior-changes) in the 0.43 → 0.44 section. Check
three first: a text answer with an unknown finish reason now fails the run,
`Ollama::completion` now targets `/v1` (the native route is
`native_completion`), and `Cohere::completion` now defaults to the
Compatibility API.

---

### 0.45

#### Hook outcomes no longer mean accepted turns

Every dispatched completion now gets exactly one `on_outcome`, including
rejected, retried, recovered and abandoned attempts. An `Ok` outcome only
means a whole response arrived, not that the agent accepted it. Move logic
that persists or bills accepted assistant content to `on_model_turn_finished`.
Pairing dispatch/outcome ids needs no workaround. Replacement outcomes are
honored only in the accepted turn's outcome slot; other closes are
observe-only. A hook stop mid-response produces a cancelled error outcome.
Denied effects also clear any earlier hook's salvaged patch.

#### Usage observations are normalized and emitted once

Usage comes from the decoder's final normalized `Usage`, once per completed
attempt, immediately before terminal `Finished`, not once per frame. Failed,
truncated or dropped attempts emit no usage observation. An end containing
only cost or cache-creation counters also emits no mapped usage event.
`AdapterUsage::total_tokens` is always `None`; use the response's
`Usage::total_tokens`. Anthropic input includes cache reads and writes, Gemini
output includes thoughts, and Gemini input includes hosted-tool prompt tokens.
Do not derive uncached Anthropic input as `input - cached`: cache-creation
input has no separate adapter counter. Ollama/Cohere no longer invent a total;
Interactions, per-choice usage and cache fallbacks are observed consistently.

#### In-band errors can now be retryable

Statusless stream/2xx error envelopes use the same code classification as HTTP
rejections. Transient Anthropic/OpenAI-compatible errors and numeric 429/503
codes can now trigger retries after partial output. Known codes outrank
`with_transient`; that hint applies only to absent/unrecognized codes.
`with_status`/`with_provider_status` supplies an explicit status verdict.
Numeric nested codes populate `http_status` and add a `status N:` Display
prefix. Coded gRPC/Vertex/Bedrock errors may serialize without `transient`;
loading a stored record re-derives its verdict from the body.

#### Old string documents change meaning

Legacy document JSON with `"type":"string"` loads as plain text, even if its
media type is PDF. If that value really holds base64, migrate it to
`"type":"base64"` before replay. `DocumentData::Text` is sent as text on every
wire. Legacy string image/audio/video sources instead load as Base64 and are
recorded as blobs, not telemetry text. Gemini Interactions no longer
base64-encodes an already encoded image/PDF a second time.
New text documents serialize with `"type":"text"`, which 0.44 cannot read.
Malformed document data now reports an untagged-enum error rather than the old
source-specific location detail.

#### Tool streams and histories include recovery results

Committed-history projection emits calls and results for Skip, Retry,
preresolved recovery and output-tool reprompts, not only executed tools. An
output-tool name need not be registered. A call id can occur both in a provider
End part and a ToolCall stream item; deduplicate by the projection semantics,
not by assuming every id appears once. `ToolResult` does not prove execution;
use `ToolExecutionCommitted` for that. Skip results arrive with step
projection; if usage recording fails first, they remain in the error history.

`AgentRun::full_history()` now closes pending calls with their preresolved
result or `No result provided`. Cancellation and mid-batch failure keep
settled sibling results and close the rest. Use `messages()` for the raw
committed view. `next_step()` while awaiting a model re-emits the same request
without consuming another turn; cancel/drain the earlier call before
reissuing it. A whole-turn recovery skip goes straight to the next model call.

Pairing accepts results split over adjacent user messages/system boundaries
and owes no results for failed or aborted turns. Repeated call ids need one
result per occurrence. Repair old partially answered repeats with
`conversation::pairing::repair` before replay.

#### Catalog facts now reach every encoder

Overrides apply to encoding, validation and pricing, including gateway models
whose facts come from their origin vendor. Wires that previously ignored
unsupported reasoning/sampling/cache settings now apply the request's
`OnUnsupported` policy. Unknown reasoning controls no longer imply that every
effort is forbidden. An unlisted cache price leaves that cost component
`None`, not priced at the input rate; `Cost::is_complete()` tells whether the
total covers every component.

MiniMax references default to the Anthropic endpoint; explicitly select
`Format::OpenAi` to keep its OpenAI endpoint. Bedrock now rejects a Specific
tool choice with zero or several names instead of dropping it or using its
first name. Name one tool, or advertise the allowed set and use Required.

#### Provider wire and credential behavior

- Transcription array parameters are repeated multipart `name[]` fields;
  `timestamp_granularities` now takes effect.
- MCP tools with an output schema return structured content alone, not both
  structured data and content blocks. Registered tools keep their connection
  alive even if the original service handle is dropped.
- MiniMax media HTTP-200 failure envelopes fail the call. TypeSafe accepts
  numeric OpenRouter `usage.cost` and ignores unknown usage fields.
- Gemini non-square image aspect ratios are width-first, fixing inverted
  `4:3`/`16:9` requests. Interactions streams may include `Item::Unknown`
  cursor items; event-only consumers ignore them. gRPC sends reasoning effort
  as a thinking level, matches REST embedding widths and accepts replies up
  to 256 MiB; it uses system `protoc` at build time.
- ChatGPT sends `session-id`, not `session_id`, as its header. Supported wires
  forward the new cache key as `prompt_cache_key`; ChatGPT also uses it for
  its session identity.
- Copilot's `access-token` and `api-key.json` are no longer read. Sign in again
  to create the single `copilot.json` record. Nonstandard ChatGPT filenames
  must be moved to `chatgpt.json`. Unix credential files/directories are
  owner-only, writes are atomic, and shared refreshes can reuse another
  client's saved token. Logout revokes ChatGPT credentials.
- Tool arguments are no longer written to the DEBUG dispatch log. Credential
  patterns are redacted in agent session/effect logs, spill files and replies.

#### Stored effect streams must be complete

`StreamRecord` (formerly `streaming::Transcript`) deserialization now validates
with `from_items`, not the old prefix parser. An incomplete saved stream that
previously deserialized as a prefix can now fail. Use `StreamRecord::parse_prefix`
explicitly when inspecting intentionally incomplete data; do not treat it as a
completed replay. Old logs containing the removed memory effect family also
need conversion or an old reader, not just an updated type import.

#### Storage, compaction and the coding-agent fronts

Overlapping conversation saves now fail rather than corrupting a log.
`DirStore` uses cross-process locking and creates `.rig-memory.lock` files.
Save checks stamps and the expected log length, not whether every change
applies; reads still validate changes. Earlier session formats are refused;
see [persistent storage](#13-persistent-storage-and-replay).

For the new coding agent, the system prompt is fixed rather than including
date, directory, instruction-file contents and changing plugin lists. The
model reads project instructions itself; `CLAUDE.md` applies only where
`AGENTS.md` is absent. A changing call cannot start if its recovery mark
cannot be saved, and untrusted remote/MCP calls are not automatically rerun
after a crash.

Compaction recovery is clear, then compact, then fail, once each until a reply
advances the run. Configure it in `[compaction]`, not `/compaction` or per-agent
knobs. `per_turn` is refused. Clearing keeps 20k tokens by default (formerly
40k), keeps small outputs under 500 tokens, and no longer applies the old
near-window tenth-of-room cap. A held summary is cancelled on stop rather
than applied. Forks are agents in the same session, not session branches.
Unknown startup `--model` values now fail with suggestions; `/model` leaves
the old model selected. Providers must register by Startup.

Shell output is line-interleaved, strips terminal escapes, and keeps head,
error lines and tail when cut. Commands stop with SIGTERM, then SIGKILL after
the grace interval. The default timeout also stops calls quiet for 60 seconds;
choose an explicit timeout to avoid that default quiet limit. Killing a remote
transport does not prove its remote process stopped. Remote file calls are
limited to two per host; read-only exit-255 failures retry once. Failed writes
report whether they happened or might have happened. Edits apply in order,
atomically as one call, and replies show changed lines and context. Search
paths preserve relative versus absolute input; `files: true` lists files.
A diff exceeding its time budget remains exact but may be less minimal.

A subagent's message does not complete its request; only its final report
does. Messages/reports are delivered on the next model call and a wait also
ends when the awaited turn ends without a message. Subagents always retain
message/wait access and inherit task environment. `-p` with a prompt only
waits briefly for piped input; JSON mode ends at EOF. Terminal state and
styling changed, including nonzero-exit calls marked failed, NO_COLOR-aware
selections and status in the input row; do not parse terminal rendering as a
machine protocol.

### 0.46

#### JSON key order and float parsing follow your own serde_json features

rig-core no longer turns on serde_json's `preserve_order` and
`float_roundtrip`. Unless something else in your build enables them, schema
properties are sorted, JSON text in prompts, embeddings and tool results can
change order, and parsed floats can differ in their low bits. Generated text,
embedding vectors and exact comparisons can change. See
[serde_json features](#1-serde_json-features-are-yours-to-choose).

#### Streams that go quiet fail after five minutes

A streamed reply that sends nothing for 300 s now fails as the retryable
`ProviderError::Stalled` instead of waiting forever. Any byte, `: ping`
comments included, restarts the timer. Opt out per model:

```rust
let model = Model::new(wire, http).with_idle_timeout(None);
```

#### Provider requests that changed without an API change

- On OpenAI Responses, `CacheRetention::Long` on GPT-5.6 and later asks for
  the 12 hour prompt cache, whose cache writes OpenAI bills at 2.5x the
  input rate.
  Use `Short` for the 30 minute default.
- On OpenAI's own Responses API, requests that set `cache_points` against
  GPT-5.6 or later carry `prompt_cache_breakpoint` markers.
- On MiniMax, Moonshot, Groq and Doubleword Responses, typed options that
  were refused are now sent, and a `store: false` conversation sends earlier
  reasoning items back.
- MiniMax China callers reach `api.minimax.cn`; pin the old
  `api.minimaxi.com` with `with_base_url` only if your network needs it.
- On the ChatGPT backend, `Reasoning::Effort` also sends
  `"summary": "auto"`. Set `additional_params` to
  `{"reasoning":{"summary":"detailed"}}` for another summary.
- Tools on Chat Completions are refused for GPT-6.1 Terra at every effort.
- Catalog facts for the models listed in `crates/rig-core/TYPED_OPTIONS.md`,
  section 12.0 (Catalog refresh), changed their reasoning and image support. Models models.dev retired
  follow each provider's rule for an unlisted model.
- `ProviderId::resolve("ollama")` is `SelectionError::Ambiguous`.
- DeepSeek requests with strict tools go to its `/beta` endpoint, with
  schemas rewritten into what strict mode takes.
- A catalog override file that sets `rig.chat_tools_need_reasoning_off` no
  longer loads; use `chat_tools` (see
  [section 3](#3-provider-selection-dialects-and-the-catalog)). Override files
  may now also set `limit.input`, `provider`, `experimental`, `interleaved`,
  `cost.tiers` and `cost.context_over_200k`, and a misspelled key inside them
  is reported.

#### Costs, telemetry and error bodies

- Native Anthropic completion cost estimates use the response's cache TTL
  split. Nonzero writes with missing details give an incomplete estimate
  (`Cost::is_complete()` is false); provider-reported costs still win.
- Agents with `record_content_telemetry(true)` also export each completion's
  tool names, descriptions and parameter schemas on
  `gen_ai.tool.definitions`. Use `record_content_telemetry(false)` to keep
  them out.
- rig-reqwest keeps at most 64 KiB of a non-success response body in the
  error, with a truncation notice. Use a custom transport if you need more.
- `Observe::event` is called for every streamed item, and `Recorder::event`
  also for a stream's first, even when `keep_events()` is false. An existing
  observer that stores everything it is given now stores those items too; check
  `keep_events()` in it (see [agent runs](#4-agent-runs-observers-and-effect-records)).

#### Stored data

- Sessions saved before 0.46 load, but their reasoning blocks record no blob
  slot: `Native::reasoning_blob()` returns `None` for them.
- A saved session's stop note with the 0.45 wording of
  `conversation::STOPPED` is no longer recognized and draws as a plain rig
  message.
- `MemoryStore::read_log` on an invalid log name returns `InvalidInput`, not
  `NotFound`.
- `SummaryState::track` counts only calls whose paired result is not an
  error. Calls with an error result or no result no longer count, so a summary
  lists only files whose read, edit or write succeeded.
- Two exhaustive enums gained variants in the middle, so their numeric values
  moved: `rig_basics::compaction::Recovering::Compacted` is 2 (was 1), and
  `rig_ecs::stage::PrepareSystems::Fit` is 5 (was 3). Code that stores or
  compares `as` casts of them maps the variants explicitly instead.

#### The coding agent

- The shell tool has no default timeout and no quiet limit: a command runs
  until it ends, unless the call sets `timeout_secs` or the user interrupts.
  A remote command stops when its transport's stdin closes, so a transport
  must forward stdin (`ssh`, `koh exec`); `ssh -n` stops every command at once.
- `read` returns PNG, JPEG, GIF and WebP files as an image (up to 3840 KB)
  and refuses a file that is not UTF-8. A third call of one missing tool name
  in a turn fails that turn. One Ctrl+D on an empty input quits.
- A tool path whose first component is `~` is the home directory. A session
  whose history wrote into a project folder named `~` reaches the home
  directory when the same call runs again.
- Compaction measures against the model's input limit, not its context
  window, so it starts earlier (for the GPT-6 models, about 906K tokens
  instead of about 1,034K). After an overflow the older messages are summarized instead of
  asking for notes. A request with no output limit asks for at most 64,000
  tokens plus any thinking budget and 1,024, never above the catalog's output
  limit, instead of the whole window left; a model with no known window or
  output limit keeps the server's default.
- A reply stopped with Esc is sent back to the model as the text it had
  written. Quitting mid-turn stops the turn as Esc does, and resuming after a
  quit or a crash runs nothing until you send a message.
- The prompt history moved to the project's
  `$RIG_HOME/projects/<name>-<key>/tui/history.jsonl`; earlier prompts in
  `$RIG_HOME/data/tui/history.jsonl` are not carried over.
- A session that only changed settings is not saved and not listed in
  `/resume`.
- SIGTERM always stops the agent cleanly; SIGINT and SIGHUP do unless rig was
  started with them ignored. The launcher kills tool process groups that a
  dead agent left behind.
- The default system prompt no longer forces a short status line at the end
  of each turn, and an explicit request in the conversation beats a
  conflicting `AGENTS.md` rule.
- Headless exit codes and closing lines changed; see
  [headless drivers](#9-headless-drivers-rig--p-rig--p---json-and-rig---json).

### 0.47

#### Library calls that behave differently

- A run with an output schema now fails with
  `PromptError::OutputSchemaMismatch` when the final answer does not fit the
  schema, after one more try, where 0.46 returned unchecked text. Code that
  parsed the text leniently afterwards now sees an error instead. See [agent
  runs](#3-agent-runs-rig-agent).
- A summed `Usage` remembers when a call reported no input, output or total
  figure. `Usage::tokens_complete()` reads the mark, such a sum serializes
  `"incomplete_tokens": true`, and it no longer equals the same counters
  unmarked. A sum of calls that reported nothing serialized as `{}` and now
  serializes as `{"incomplete_tokens":true}`. Where you compared `sum ==
  Usage::default()` to ask whether anything was reported, call
  `!sum.is_reported()`. Fold from `Usage::default()`; adding an empty `Usage`
  on the right marks the sum.
- A streamed run yields the new `MultiTurnStreamItem::ToolExecutionStarted`
  and `ToolResultItem` items. A consumer that prints or counts every item it
  does not know sees more items. See [agent runs](#3-agent-runs-rig-agent).
- A tool call that a reply cut at the output limit is never run and is
  answered "your reply reached the output limit, send less". It is not offered
  to `on_invalid_tool_call`, and a cut call to the output tool is never taken
  as the run's answer. `CompletionResponse::unfinished_calls()` lists more
  calls for cut replies on OpenAI Chat Completions, OpenAI Responses and
  Anthropic; 0.46 could run such a call with `{}`.
- A tool call denied through an `ErrorKind::Denied` report (a hook's
  `DispatchAction::Deny`, a layer's `BeforeServe::deny`) is a refusal:
  `is_refused()` is true and its status is `denied`. 0.46 recorded a skip or a
  failure. `DispatchAction::skip` still skips.
- A tool that panics fails its own call with an internal error reading
  `panicked: <message>`, and the run goes on with that as the tool's failed
  result. 0.46 unwound through the run. If you relied on a panic to abort,
  return an error from the tool and stop the run in a hook.
- An `on_dispatch` patch of a completion that changes the tool choice, the
  tool names, the provider-hosted tools or the output schema fails the turn
  before the provider is called. Make that change in `on_completion_call` with
  `CompletionCallAction::patch(RequestPatch::new() .tool_choice(..))`.
- `Reply::into_stream` on a finished reply, and rig-agent's tool-call deltas
  for an answer that did not stream, carry a malformed call's arguments as the
  model sent them, which may not be JSON. 0.46 restated the repaired object.
  Read the arguments from the call the `End` event carries, or parse the text
  with `streaming::parse_partial_arguments`. Calls that share an id stay
  distinct through a restated reply.
- `pairing::validate` and `Conversation::validate` accept two assistant turns
  in a row when the two cannot merge (different origins or outcomes), and
  `repair` and `normalize` leave them apart. Code that assumed a repaired
  history never has two assistant turns in a row reads each turn on its own.
- Tool argument validation and vector-store JSON filters compare an integer
  with a float exactly, and enforce `minLength`, `maxLength`, `minItems` and
  `maxItems` written as a whole float such as `2.0`. `args_schema` keeps
  `null` in the schemas of collection items, map values and tuple elements.
  Arguments that passed by rounding can now fail.
- `ChatBot::run` ends at end of input (Ctrl+D or a closed pipe) instead of
  sending empty prompts, and returns `StreamToStdoutError::Incomplete` when a
  stream ends with no final response. A `HookStack` member whose `observes`
  returns false for run-start, run-settled or a delta event is no longer
  called for it. `prepare_request` fails when provider tools are set and
  `additional_params` is a string, an array or a number.
- `EmbeddingsBuilder::build` rejects a document with no text before it sends
  any request.
- `BeforeServe::deny` marks its report as a refusal, so a tool call a layer
  denies ends `Refused`, not failed, and denied completions and uploads are
  marked the same way.
- rig-memory's `estimate_tokens` keeps its name and signature but counts
  differently. It was the default `HeuristicTokenCounter`'s count (four bytes
  a token plus four per message). It is now four characters a token, rounded
  up, with each tool result rounded apart (`estimate_message_tokens`).
  Thresholds tuned against the old numbers shift. See [memory and
  compaction](#4-memory-and-compaction-rig-memory).
- `HeuristicTokenCounter::anthropic()` (3.5 bytes a token) is gone, so code
  moved to `HeuristicTokenCounter::default()` counts at four bytes a token and
  reports fewer tokens for the same text.
- A summary request built by rig-memory (`summary_request`,
  `summarization_request`) is a separate request with prompt-cache writes off.
  It no longer shares the conversation's cached prefix. `summary_text` also
  refuses a reply that was cut at the output limit or called a tool.
- `apply_summary` returns `None` and changes nothing when the conversation no
  longer starts with the very messages the summary read, or when what stays
  would start with a tool result whose call was summarized. Check its result.
- A crate that depends on `rig` with default features now compiles the coding
  agent's launcher (`rig-cli`) through the new default `cli` feature. Nothing
  in your program changes; the build gets longer. See [cargo
  features](#5-vector-stores-cassettes-mcp-and-the-facades-features).
- rig-lancedb: a row whose id column is missing, or neither text nor a number,
  makes `top_n` and `top_n_ids` fail with `MissingIdError`. 0.46 invented an
  id (such as `unknown3`, or an empty string). A number id reads as its
  decimal text. A dictionary column decodes to its plain value per row, and a
  null cell of any type, at any depth, decodes to JSON `null` where it used to
  come back as `0`, `""`, `[]` or `{}`. A struct field that receives a null
  must be an `Option`, or the row fails to deserialize. `top_n_ids` returns
  the distance instead of `0.0`.
- rig-lancedb: `insert_documents` refuses an embedding whose squared length is
  not finite, and `top_n` and `top_n_ids` return an error naming the row when
  its distance is not finite, instead of scoring it `0.0`.
- rig-postgres: `top_n` returns the deserialization error for a row it cannot
  read. 0.46 dropped the row and returned the others.
- rig-mongodb: a filter value BSON cannot represent is
  `FilterError::Serialization`. 0.46 turned it into `null` and ran the query.
- rig-vectorize: `VectorizeFilter::and` keeps every operator on a shared
  field, and a repeated operator with another value fails `validate()`. 0.46
  kept one side silently.
- rig-helixdb: `top_n_ids` applies the metadata filter, so it can return fewer
  ids. With an API key, `HelixDB::new`'s client follows redirects only within
  the endpoint's origin; a redirect to another origin fails the query.
- rig-sqlite: a filter on an indexed column that declares a collation (such as
  `TEXT COLLATE NOCASE`) follows that collation, and an insert removes the
  embeddings of every document a unique-key replacement deletes.
- rig-rmcp: a server that serves tools or resources without declaring the
  capability yields none from `McpConnection::tools`, `resources` and
  `McpClientHandler::connect`. See
  [rig-rmcp](#5-vector-stores-cassettes-mcp-and-the-facades-features).
- rig-typesafeai: a stored `JevConfig` with an unknown top-level key fails to
  load instead of loading without it.

#### Provider requests and errors in 0.47

- Only an `http_client::Error::Transient` failure is retried. A transport that
  reports every failure as `Permanent` is never retried, and one that reports
  every failure as `Transient` always is, so a custom `HttpClientExt` must
  pick the class with care. reqwest's builder and redirect errors now fail at
  once. See [HTTP
  errors](#2-providers-sign-in-the-catalog-and-http-rig-core-rig-http-rig-reqwest).
- The default client (rig-reqwest) forwards only `accept`, `content-encoding`,
  `content-type`, `range` and `user-agent` on a redirect to another scheme,
  host or port. Provider key headers such as `x-api-key`, `api-key` and
  `x-goog-api-key` no longer follow. A `base_url` that redirects to another
  origin and needs the key there stops working: point `base_url` at the final
  origin, or pass your own client.
- Gemini sends the API key in the `x-goog-api-key` header on every wire and
  never as `?key=` in the URL. A gateway that accepts only the query parameter
  must accept the header.
- A streamed reply that sends no frame for 5 minutes fails as
  `ProviderError::Stalled` even while keep-alive bytes such as SSE comments
  arrive. In 0.46 any byte restarted the timer. Use `Model::with_idle_timeout`
  for a longer wait. The OpenAI Responses WebSocket session times each event
  wait the same way by default and fails as `Stalled`, not
  `ProviderError::Provider`; `.without_event_timeout()` waits indefinitely.
- A streamed reply with any 2xx status is read as a stream (before, only 200).
  A provider error code listed as final, such as `insufficient_quota`, is not
  retried, whatever the HTTP status.
- `chatgpt::DEFAULT_MODEL` is `gpt-6.2-sol` (was `gpt-6.1-sol`). Name
  `openai::GPT_6_1_SOL` to stay on the old model. The catalog is refreshed
  from models.dev: 35 retired rows are removed and follow each provider's rule
  for an unlisted model, and prices and limits changed.
  `mistral/mistral-large-latest` now reasons with an effort and refuses
  `Reasoning::Off`. Gemini `Effort::XHigh` is sent as `thinkingLevel: "xhigh"`
  to the models the catalog lists with it, where it was refused. The gRPC
  route (rig-gemini-grpc) still refuses it.
- `voyageai::VOYAGE_3_5_LITE` is `"voyage-3.5-lite"`, the id Voyage AI serves.
  It was `"voyage.3-5.lite"`. Code that wrote the old literal uses the
  constant.
- `ollama/*` catalog models read `OLLAMA_API_BASE_URL` as the daemon's address
  and append `/v1` themselves. Drop a trailing `/v1` from the variable.
- DeepSeek Chat Completions sends user images as `image_url` parts to the
  models the catalog says read them, where it sent a text placeholder, so
  those requests are billed for the image. Azure OpenAI transcription posts to
  `/audio/transcriptions` and keeps the spoken language, where it posted to
  `/audio/translations` and returned English. A Bedrock Converse tool call
  typed `"tool_use"` is a client call again. Copilot routes by the model
  catalogue before its `codex` name check, so `gpt-5.5`, `gpt-5.4` and the
  GPT-6.1 family go to `/responses`. The rule holds for any dialect that sets
  both `routes_by_catalog` and a `model_route` hook: the catalogue row wins,
  and the hook routes only the models the catalogue does not. A route set on
  the provider itself still beats both. An embedding width the caller names is
  sent even when it equals the model's native width.
- On the Hugging Face router a sub-route is selected by sending the model as
  `model:slug`, and the slugs changed (`SubRoute::HFInference` is
  `hf-inference`, `SubRoute::Fireworks` is `fireworks`).
  `SubRoute::model_identifier` returns `model:slug` for every sub-route. Pass
  a Hub model id, not a sub-provider's own id. A model that already selects
  another provider is refused.
- Cohere's 200-status error body on the Compatibility wire fails the turn as a
  truncated reply, without Cohere's message text. `CohereExtras` and
  `GeminiExtras` read an empty list as absent. A Gemini reply without
  `responseId` reports `response_id: None`, not `Some("")`. `Gemini::verify`
  accepts a 2xx reply whose body is not JSON.
- The Anthropic cache-marker checks run once, on the body that is sent, and
  report the first problem in walk order (tools, system, messages, top level),
  so a request with several problems can report a different one. A
  `Reasoning::Budget` on an Anthropic Messages request for a model with no
  dialect default `max_tokens` is checked against the wire's
  `default_max_tokens` and refused locally when it is at or above it. Set
  `max_tokens` above the budget.
- `MultipartForm::encode` escapes `"`, CR and LF in part names and filenames
  as `%22`, `%0D` and `%0A`. A third-party transport that renders forms with
  it now sends the escaped names.
- A stored `providers::registry::Format` with an unknown family reports
  serde's "unknown variant" error. Provider reply types no longer read a short
  JSON array by filling missing trailing `Option` fields with `None`.

#### Stored data and files

- A serialized `AgentRun` from 0.46 is refused (`RUN_FORMAT` 5 → 7). See
  [agent runs](#3-agent-runs-rig-agent).
- A saved `TurnEnd::Stopped` writes `by` and `cut` where 0.46 wrote `why` and
  `calls`. A record saved by 0.46 still reads; its cause is `None` and its cut
  calls are empty.
- Conversation entries saved by 0.46 carry no `ResultFacts`. They load, with
  empty facts.
- A conversation's change log records a push after a halt as one change. 0.46
  followed the push with a separate `Change::Halt { halt: None }`. Code that
  counts or matches recorded changes sees one record fewer.
- `FileTokenStore` refuses a credential file that is a symlink, not a regular
  file or over 64 KiB, and on Unix narrows one that group or others can read
  to 0600. A Copilot credential saved by an older build is exchanged once more
  on first use, because the bootstrap-token fingerprint changed. A
  checkpointed Gemini cache lease whose name is not `cachedContents/<id>`
  comes back from `CacheBook::restore` as lost. See [sign-in
  types](#2-providers-sign-in-the-catalog-and-http-rig-core-rig-http-rig-reqwest).
- A compaction's only record is its summary entry. Sessions saved by 0.46 load
  unchanged; their `rig.compaction` asides stay in the log and nothing reads
  them. `summarized_in_log` forgets what was replaced once the log's
  conversation is reset.
- `Session::save` refuses, with `SessionError::Changed`, a session read before
  another writer saved or rewrote one of its logs. In 0.46 only a log holding
  changes was checked. Two writers on one store now need a
  read-again-and-retry loop. `Session`'s `PartialEq` also compares where its
  logs ended. See [sessions](#4-memory-and-compaction-rig-memory).
- Sessions save a new key, `rig::current_agent` (`rig_memory::CURRENT`), and
  `Session::main_agent()` returns the top-level agent it names, else the first
  by id. A 0.46 session has no such key.
- Tool results are saved with the facts their call published (`ResultFacts`).
  A `Rewrite` that does not implement the new `facts` hook leaves them as they
  are, so a redacting rewrite should cover them.
- Removing a saved resource is saved, so a restart no longer brings its old
  value back.
- Files rig creates through `fs::replace` (a first `settings.toml`, a
  session's name and working-directory files, a run's end record) are private
  to the user (0600 on unix). An existing file keeps its mode and group.
- A session whose saved working-directory file cannot be read is refused. 0.46
  opened it in the current directory and overwrote the file.
- On unix, `sessions/`, `projects/`, `data/` and `run/` under `RIG_HOME` and
  each new session directory are tightened to mode 0700 when a session starts.

#### The coding agent in use

- `write` replaces an existing file only if the agent read or wrote it in this
  session and it has not changed on disk since. Otherwise it writes nothing
  and tells the model the file `has not been read by you since rig started` or
  `changed since you last read or wrote it`.
- A confined file tool follows a dangling symlink before it checks its base,
  so a write through a link that leads outside is refused.
- `RIG_HOME/plugins.toml` and `RIG_HOME/defaults.json` are no longer read.
  Until their contents are moved into `settings.toml`, the agent runs rig's
  default plugins and picks a model on its own, with a warning at each start
  that prints the steps. See [upgrading a 0.46
  home](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- Deleting `[compaction] keep_recent` or `limits.reserve`, as the start's
  refusal advises, drops your value for the default (20000 and 16384 tokens).
  Rename them to `keep_recent_tokens` and `reserve_tokens`. See [settings
  tables](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- `[tui] mark_nonce_env` no longer applies, so OSC 133 prompt marks carry no
  nonce until the key is moved to `[title]`. A leftover `[tui]` or
  `[checkpoints]` table only gets a "no enabled plugin reads" notice.
- Compaction summary calls no longer reuse the prompt cache. 0.47 sends a
  separate summary request with cache writes off; 0.46 asked for the summary
  in place, over the cached conversation. Measured on matched runs with 3
  compactions in 27 model calls, uncached input was 1.91 times 0.46's on one
  model (81k to 155k tokens; 0.47's summary calls cached 0 of 62k input
  tokens, 0.46's 65k of 66k) and 2.30 times on another. Ordinary sessions
  compact far less often than those runs did, so the cost over a whole session
  is smaller.
- Compaction keeps about `[compaction] keep_recent_tokens` (20000) of the
  newest messages and summarizes the rest; it no longer clears old tool
  outputs first, and `/compact` keeps the same amount as an automatic
  compaction. After one, the status line shows `ctx ?` until the next reply.
  See [commands and behaviour](#10-the-coding-agents-commands-and-behaviour).
- Forks, rewind (Alt+Up, then Enter) and `/agents` are off by default; `rig
  plugin enable rig_basics::sessions::forks::ForksPlugin` turns them on. A
  session saved while a fork was shown reopens on its first agent, and
  `/agents` picks the fork. See [section
  9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- Nothing is redacted unless `rig_basics::redact::RedactPlugin` is enabled;
  0.46 always redacted. With it on, a secret is a placeholder such as
  `‹SECRET_1›` (it was `[REDACTED]`), and a tool call that names the
  placeholder runs with the real value. See [section
  9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- With the MCP plugin on, the servers of a project's `.rig/mcp.json` start
  with no question and replace a user server of the same name. Asking first is
  the opt-in `rig_mcp_client::approval::McpApprovalPlugin`. See [section
  9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- A project's `.rig/settings.toml` is not read, and no notice says so.
  Settings live only in `RIG_HOME/settings.toml`. See [section
  9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- rig finds one project root, the nearest directory at or above the working
  directory that holds `.git` or `.rig`, below your home. Started in a
  subfolder, it reads the root's `.rig/mcp.json`, `AGENTS.md` files and
  skills, and keeps per-project data such as the prompt history under the
  root, so a subfolder's 0.46 prompt history is not carried over. A folder
  with its own `.rig` directory is its own project. Started in the home
  directory, rig reads no project `mcp.json`.
- `--tools` limits only the run it is given to. A later run of the session
  without it has every tool again.
- Only `/model` or `/effort`, once that model has answered, changes the
  default for new sessions. `/model` lists only models usable now, plus the
  current one with why it cannot be used, and a ChatGPT plan's models while
  the plan is still being checked. With nothing saved or named, the terminal
  view starts on an automatic pick, as `rig -p` and ACP do, and that pick
  takes the ChatGPT plan's default model first when you are signed in.
  `chatgpt`'s default model is `gpt-6.2-sol`.
- A reasoning effort the model's route refuses, such as a token budget on an
  Ollama model, is reset at start with a warning instead of failing every
  turn. `[model] effort = "high"` is read by the model the session starts on,
  so it works on a model that takes a budget and no levels.
- The mouse wheel scrolls the transcript instead of browsing prompt history,
  and a drag copies text in rig. The terminal's own selection and middle-click
  paste need Shift.
- Ctrl+V pastes the clipboard's text when it holds no image, pastes a JPEG or
  WebP image when no PNG is offered, and names the clipboard program that is
  missing (`wl-paste`, `xclip`).
- In `/plugins`, Enter ticks a row instead of applying at once, and Esc asks
  when to apply. A command refused because a turn is running comes back in the
  input with the reason.
- A shell command's background job that still holds the call's output open is
  ended with the call, and the result says so and how to keep one running. A
  stopped call stops the processes it started. `shell` no longer kills a
  command after 10 MB of output. `timeout_secs: 0`, `read`'s `offset: 0` and
  `list`'s `depth: 0` are refused.
- Search, list, watch and the `@` picker leave out only directories named
  `target`, so a file named `target` is listed.
- Text that starts with a path, `//`, or `/` and a space is a message for the
  model, not a slash command.
- A `settings.toml` table that does not read, an unknown plugin name in
  `[plugins]`, or a `[[plugins.extra]]` path that holds no crate refuses the
  start; 0.46 started anyway or ran the previous build. A table no enabled
  plugin reads gets a notice.
- A saved `/login` key file that cannot be read or parsed refuses its
  provider's models, naming the file, instead of falling back to the
  environment's key. A credential file that is a symlink is refused, and one
  that group or others can read is narrowed to 0600.
- `cargo install`, `rig build`, cargo and the other processes the launcher
  starts no longer inherit exported `RIG_SECRET_*` variables or the launcher's
  `RIG_*` variables. A build script or cargo wrapper that read them during a
  build no longer sees them. The browser opened for a sign-in does not get the
  secrets either.
- A stopped turn (Esc, `/compact`, an exit) is not an error: the title reads
  `rig · error` and the OSC 133 mark is `D;1` only after a failed turn. A stop
  writes `D;0`.
- The first agent build after upgrading recompiles every dependency once,
  because dependencies now build without debug info.
- Steel's `spawn-agent`, `send` and `reply` work only with
  `rig_basics::subagents::SubagentsPlugin` on. Subagents have no depth limit.

#### Headless runs

- `rig --json` speaks protocol version 3 and refuses a version 1 line with an
  error reply, so a 0.46 driver that does not read replies sees its requests
  do nothing. `tool_call` and `tool_result` ids are places such as `m4/1`, not
  provider call ids. See [headless
  drivers](#11-headless-drivers-rig--p-rig---json-and-rig-acp).
- `exit`'s `usage` is the session's total and `turn_end`'s includes its
  helpers' spend. A driver that summed children's costs itself now counts them
  twice.
- `rig -p` waits for piped stdin to close, refuses input that is not UTF-8,
  exits 2 for a command that opens a list (it exited 0 and printed nothing),
  exits 1 when the agent build fails instead of answering with the previous
  build, and writes tool progress on stderr without the `rig: ` prefix. See
  [headless drivers](#11-headless-drivers-rig--p-rig---json-and-rig-acp).
- `rig acp` refuses a word after `acp`; an editor command that put options
  there had them ignored and now fails to start.

#### Plugins built against the agent

- `StreamDelta` now carries the text of every model call's reply. A reply that
  did not stream, such as one a plugin answered in place or a reviewer edited,
  is written once just before the call settles. A reader that also rebuilt
  such replies from `Outcome` on `Add<Settled>` now shows them twice: read the
  deltas alone. Tell an in-place answer by `Decision::by()`.
- A system in `Last` that is `.in_set(WriteLogs)` or ordered
  `.after(WriteLogs)` still compiles, and no longer runs with the save. Move
  it to `Persist` (see [the kernel](#6-the-kernel-rig-ecs)).
- Once the session can no longer be saved, the kernel refuses every effect
  whose `Footprint` says it may change something before it starts, not only
  tool calls. Give a plugin `EffectRequest` that changes files or anything
  outside the session a `Footprint`.
- `ToolAccess::allows` answers for this run only, within the run's `RunTools`.
  Code that derives a child's tools or a saved list from it now loses the
  tools a `--tools` run left out; use `keeps` (see [the
  kernel](#6-the-kernel-rig-ecs)).
- A model call that was stopped, superseded or failed mid-stream counts the
  usage its provider reported before the end (`ReportedUsage`), else an
  estimate. Spend meters see calls they did not see before.
- `Incoming::command` returns `None` for text that starts with a path
  (`/etc/hosts: what is this?`), with `//`, or with `/` and a space. Such text
  is a message for the model in every front.
- `NoticeLevel`'s variants are reordered to `Warn`, `Error`, `Info`, so `as`
  casts of them changed value. It now serializes as `info`, `warn` or `error`.
- A settings table registered in a plugin's `finish` compiles and refuses
  every start. Register it in `build`. See [plugin host and
  tools](#7-plugin-host-and-tools-rig-harness-rig-harness-protocol-rig-tools).
- A front that shows pickers and does not say so with `Front::shows_pickers()`
  has every picker command refused in its runs.
- A per-call shell starts its commands as the app's `Processes` say and cuts
  its output as the app's `ShellOutput` says. The `launcher` and `cut` of the
  `Shell` you registered are ignored.
- A wrapper around a live per-call tool that does not forward `const LIVE`
  gets no live output for its calls.
- Each agent has its own record of the files its tools saw (`FilesSeen`). A
  `Seen` you shared between the file tools is no longer shared between agents,
  and a fork or a subagent starts with none.
- An agent that exits with code 75 and wrote no `AgentEnd::Reload` record is a
  failed run, not a reload.
- A slash command registered with `In<NoArgs>` refuses text after its name.
  One registered with `In<CommandArgs>` accepts it as before.
- A file watch tells a failed listing as `WatchEvent::Unavailable` and the
  next good one as a new `Snapshot`. In 0.46 a failed listing looked like
  every file removed and then added again.
- Commands started through rig-tools find no editor and no pager
  (`process::NON_INTERACTIVE`): `git commit` without `-m` fails at once
  instead of waiting. A variable the command sets itself is kept.
- The `status` item's text is no longer exactly `thinking…` and the like while
  the agent works: a wave of bars can come first and the elapsed time after 5
  seconds. Read `Activity::status` instead of matching the text.
- A remote `@HOST:PATH` attachment arrives after `AttachSystems`, under a hold
  on its message, and later user-kind messages to that agent wait behind it.
  See [plugin crates](#8-plugin-crates-rig-basics-rig-tui-and-the-others).
- `rig_tui::TuiPlugin` alone no longer draws the status line, the key hints or
  the terminal title and OSC 133 marks. An app or test that adds the view
  itself adds `rig_tui::TuiPlugins` to keep them. See [plugin
  crates](#8-plugin-crates-rig-basics-rig-tui-and-the-others).
- `ToolRender::result_lines()` and `result_text()` include a result's JSON
  blocks and a line counting its images. A look that appends JSON itself now
  shows it twice; drop the JSON content first, as [plugin
  crates](#8-plugin-crates-rig-basics-rig-tui-and-the-others) shows.
- rig-basics no longer redacts anything unless
  `rig_basics::redact::RedactPlugin` is added. A plugin that relied on the
  kernel or the logs redacting for it redacts what it writes with
  `rig_core::redact`.
- An app that adds `rig_basics::activity::ActivityPlugin` without
  `WorkingAnimationPlugin` has no working animation, and one that adds
  `SteelPlugin` without `SubagentsPlugin` has no `spawn-agent`, `send` or
  `reply` in its Steel programs.

---

## 0.46 → 0.47

0.47 is the largest release in this guide: 280 pull requests, 95 of them
breaking. Most of it is the coding agent. Its settings live in one file,
`RIG_HOME/settings.toml`, which replaces `plugins.toml` and `defaults.json`.
Compaction is rewritten, forks, redaction and MCP approval are optional
plugins that are off by default, and `rig --json` speaks protocol version 3.
A home that 0.46 wrote does not start as it is:
[section 9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates) is the upgrade procedure. Three crates are new: `rig-cli`
(the launcher, as a library), `rig-markdown` and `rig-fold`.

The library changes are mostly removals of public items nothing used, plus a
few renames. The ones most code meets: the shared sign-in types have one
path, `providers::auth`; `message::Document` is `message::DocumentContent`;
`RunId` lost its raw conversions; rig-agent's `ChatBotBuilder` is
`ChatBot::new`; rig-memory's compaction policy and `ClearToolOutputs` are
replaced by functions; and the `audio`, `image` and (in rig-core and
rig-http) `websocket` cargo features are gone because the code always
compiles. Runs persisted by rig-agent 0.46 are refused on resume
(`RUN_FORMAT` 5 → 7). Rust **1.97.1** is unchanged. The agent is built on
Bevy **0.20.1**, and a plugin writes its Bevy requirements as `"0.20"`.

Who reads what:

- Library users: sections [1](#1-core-messages-conversations-tools-and-streaming-rig-core) to [5](#5-vector-stores-cassettes-mcp-and-the-facades-features).
- Plugin authors: sections [6](#6-the-kernel-rig-ecs) to [8](#8-plugin-crates-rig-basics-rig-tui-and-the-others), then the plugin crate step
  of [section 9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).
- Users of the `rig` coding agent: sections [9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates) and [10](#10-the-coding-agents-commands-and-behaviour).
- Scripts and editors that drive rig: [section 11](#11-headless-drivers-rig--p-rig---json-and-rig-acp).

The Rust examples in sections 1 to 5 describe the final 0.47 API, and most of
their `after` forms were compiled against the release tree. The examples in
sections 6 to 11 are **source-checked migration sketches, not compiled
standalone programs**, as are the `before` forms everywhere.

### 1. Core messages, conversations, tools and streaming (rig-core)

#### `message::Document` is `message::DocumentContent`

The document block of a user message no longer shares a name with the RAG
`completion::Document`. Its fields and the `UserContent::Document` variant are
unchanged, and so is its JSON.

```rust
// before
use rig::message::{Document, DocumentData, DocumentMediaType, UserContent};
let doc = UserContent::Document(Document {
    data: DocumentData::Text("notes".into()),
    media_type: Some(DocumentMediaType::TXT),
    additional_params: None,
});
// after
use rig::message::{DocumentContent, DocumentData, DocumentMediaType, UserContent};
let doc = UserContent::Document(DocumentContent {
    data: DocumentData::Text("notes".into()),
    media_type: Some(DocumentMediaType::TXT),
    additional_params: None,
});
```

`history::Media::Document` borrows a `DocumentContent` too.

Three message items are removed. `message::MediaType`, the enum over the four
media kinds, has no replacement: ask the kind you mean, or chain them.
`message::MessageError` is removed; nothing returned it.

```rust
// before
let kind = MediaType::from_mime_type("image/png");
// after
use rig::message::{ImageMediaType, MimeType};
let kind = ImageMediaType::from_mime_type("image/png");
```

New readers save a `match` on content: `AssistantContent::as_text`,
`as_tool_call` and `concat_text`, `UserContent::as_text` and
`as_tool_result`, `Message::results`, and `ToolResultContent::to_text`.

#### `RunId` is a plain newtype

`RunId::from_raw`, `RunId::to_raw`, `impl FromStr for RunId`, both
`NonZeroU64` conversions and `id::ParseIdError` are removed. `RunId::new`,
`Display`, `Debug`, ordering and serde stay, and a `RunId` still serializes
as its number. Go through serde, or through the printed number:

```rust
// before
let id = RunId::from_raw(7);   // Option<RunId>
let id: RunId = "7".parse()?;
let raw = id.to_raw();         // u64
// after
let id: RunId = serde_json::from_value(serde_json::json!(7))?;
let id: RunId = serde_json::from_str("7")?;
let raw = serde_json::to_value(id)?.as_u64();   // Option<u64>
let text = id.to_string();
```

`conversation::ParseIdError` is another type and stays.

`effect::family::Custom<E>` is only a type marker now. `Custom::new()` and
its `Default`, `Debug`, `PartialEq`, `Eq` and `Hash` impls are removed;
`Clone` and `Copy` stay. Name it as a type, as in `Key<family::Custom<E>>`,
and drop any bound that asked for those traits.

#### Conversations: stops, call places and entry facts

A stopped turn records who stopped it, and names the calls it cut by their
place in the reply, not by provider id. `TurnEnd::Stopped { why, at, calls }`
becomes `TurnEnd::Stopped { by, at, calls }`: `by` is an
`Option<StopCause>` (`User`, `Exit(String)`, `Restarted`,
`Plugin { by, why }`) and `calls` is a `Vec<CallRef>`. A `CallRef` is the
reply's `MessageId` and the call's `index` in it, so two calls that share a
provider id stay apart.

```rust
// before
if let TurnEnd::Stopped { why, calls, .. } = &end {
    let reason = why.as_deref().unwrap_or("the user stopped the turn");
    let ids: &Vec<CallId> = calls;
}
// after
if let TurnEnd::Stopped { by, calls, .. } = &end {
    let reason = by.as_ref().map(ToString::to_string);
    let places: &Vec<CallRef> = calls;
}
```

`Conversation::pending_calls` returns each owed call with its place:

```rust
// before
let owed: Vec<ToolCall> = conversation.pending_calls();
// after
let owed: Vec<(CallRef, ToolCall)> = conversation.pending_calls();
```

The free function `pairing::pending_calls` still returns `Vec<ToolCall>`.

`conversation::Entry` and `EntryRef` have a new public field, `facts`
(`ResultFacts`): what a tool call published about its result, kept with the
result's entry. A struct literal names it, or use `Entry::new`:

```rust
// before
Entry { message, author, id, made }
// after
Entry { message, author, id, made, facts: ResultFacts::default() }
```

Removed, with what to write instead:

- `conversation::pairing::answers` and `AnswerError`: no replacement. Match
  each result's `call` against the pending ids yourself;
  `pairing::answered_by` and `pairing::placed` do the lookups.
- `Conversation::reread(log)`: assign the conversation read from the log,
  `*conversation = from_log;`, and put back the unsaved asides you want.
- `Conversation::aside_position(aside)`: read `aside.after`, then
  `conversation.index_of(id)` or `conversation.start_of(id)`.

#### Tools

`ToolEmbedding` loses `InitError`, `State` and `init`. Nothing called `init`.
`Context` needs only `Serialize`:

```rust
// before
impl ToolEmbedding for MyTool {
    type InitError = MyInitError;
    type Context = MyContext;
    type State = ();
    fn init(_: (), context: MyContext) -> Result<Self, MyInitError> {
        Ok(Self { context })
    }
    fn embedding_docs(&self) -> Vec<String> { vec!["Adds numbers".into()] }
    fn context(&self) -> MyContext { self.context.clone() }
}
// after
impl ToolEmbedding for MyTool {
    type Context = MyContext;
    fn embedding_docs(&self) -> Vec<String> { vec!["Adds numbers".into()] }
    fn context(&self) -> MyContext { self.context.clone() }
}
```

`tool::feedback::tool_result_output` is removed; `ToolCall::answer` builds the
same result from the call:

```rust
// before
let content = tool_result_output(call.id.clone(), call.function.name.clone(), &result);
// after
let content = UserContent::ToolResult(call.answer(&result));
```

`tool::feedback::invalid_call_feedback` takes the invalid call's position in
the content, not its id, because a peer call may share the id:

```rust
// before
let results = invalid_call_feedback(&content, &invalid_call.id, "bad arguments");
// after
let results = invalid_call_feedback(&content, invalid_at, "bad arguments");
```

`structured_output::text_satisfies_schema` is replaced by `validate_text`,
which says what is wrong:

```rust
// before
if !text_satisfies_schema(Some(&schema), text) { /* refuse */ }
// after
if let Err(problems) = validate_text(Some(&schema), text) {
    for problem in &problems {
        eprintln!("{problem}"); // `at: what`
    }
}
```

`rig_core::redact::json_text` and `redact::tool_result` are removed, and
`redact::json` returns whether it changed anything. Parse a JSON line, or
serialize a result, and pass the value to `redact::json`:

```rust
// before
let line = redact::json_text(&line);
// after
let mut value: serde_json::Value = serde_json::from_str(&line)?;
let changed = redact::json(&mut value);
```

#### Streaming, usage and loaders

`streaming::Relayed` has a new variant, `Usage(Usage)`: what the provider
reported so far, sent before the reply ends. The enum is exhaustive, so a
`match` needs an arm:

```rust
// before
origin @ Relayed::Origin(_) => out.push(origin),
// after
passed @ (Relayed::Origin(_) | Relayed::Usage(_)) => out.push(passed),
```

`completion::CacheCost`, `completion::CacheRates` and the
`completion::cache_cost` module are removed; see
[cache cost helpers](#2-providers-sign-in-the-catalog-and-http-rig-core-rig-http-rig-reqwest).

`loaders::file::IntoIter`, `loaders::pdf::IntoIter` and
`loaders::epub::IntoIter` are removed. The `IntoIterator::IntoIter` of
`FileLoader`, `PdfFileLoader` and `EpubFileLoader` is
`Box<dyn Iterator<Item = T> + 'a>`; a `for` loop is unchanged, and only code
that names the iterator type changes.

On wasm only, `EmbedError` implements `From<Box<dyn Error>>` instead of
`From<Box<dyn Error + Send + Sync>>`, and `EmbedError::new` accepts an error
that is not `Send` or `Sync`. Cast an existing `Send + Sync` box first.
Native targets are unchanged.

```rust
// before (wasm)
let error: EmbedError = boxed.into();
// after (wasm)
let error = EmbedError::from(boxed as Box<dyn std::error::Error>);
```

`json_utils` is doc-hidden, but three of its helpers had callers:

| Removed | Write instead |
| --- | --- |
| `json_utils::merge_inplace(a, b)` | `*a = json_utils::merge(std::mem::take(a), b)` |
| `json_utils::string_or_vec` | none; deserialize through an untagged enum or a visitor of your own |
| `json_utils::Lenient::i64(key)` | `value.get(key).and_then(Value::as_i64)`, or parse the string form yourself |

`ReplayTarget::route` is removed with Cohere's automatic chat route; see
[providers](#2-providers-sign-in-the-catalog-and-http-rig-core-rig-http-rig-reqwest).

rig-core's `audio` and `image` cargo features are removed. They added no
dependency, and `audio_generation`, `image_generation` and the provider audio
and image models now always compile. Delete the features from your manifest;
[the facade's features](#5-vector-stores-cassettes-mcp-and-the-facades-features) shows the `rig` form.

The `test-utils` feature lost a few helpers; see
[provider test utilities](#2-providers-sign-in-the-catalog-and-http-rig-core-rig-http-rig-reqwest).

### 2. Providers, sign-in, the catalog and HTTP (rig-core, rig-http, rig-reqwest)

#### The `websocket` feature is gone from rig-core and rig-http

`rig-core` and `rig-http` no longer have a `websocket` cargo feature. It
added no dependency: `ws_client` and the OpenAI Responses WebSocket session
now always compile. cargo refuses a manifest that still names the feature.
The facade `rig`'s `websocket` feature is unchanged. rig-core's `audio` and
`image` features went the same way; see [the facade's features](#5-vector-stores-cassettes-mcp-and-the-facades-features).

```toml
# before
rig-core = { version = "0.46", features = ["websocket", "derive"] }
# after
rig-core = { version = "0.47", features = ["derive"] }
```

The workspace now requires `serde >= 1.0.181`. Run `cargo update -p serde`
if your lockfile pins an older one.

#### HTTP errors say whether a retry can work

`http_client::Error::Instance` and `Unbuilt` are replaced by `Transient` and
`Permanent`, built with `Error::transient` and `Error::permanent`. Only
`Transient` failures are retried. A transport picks the class where it makes
the error:

```rust
// before
Err(http_client::Error::instance(e))
// after: a reset connection, a timeout, a body cut short
Err(http_client::Error::transient(e))
// after: a request that cannot be sent, missing credentials, a client that
// could not be built
Err(http_client::Error::permanent("no credentials"))
```

The old `From<Box<dyn Error + Send + Sync>>` conversion is removed with
`Instance`, so a bare `?` or `.into()` from a boxed error no longer compiles.
Choose a class instead. `Error::NoHeaders` and `Error::InvalidHeaderValue`
(with its `From<http::header::InvalidHeaderValue>`) are removed too. The enum
is `#[non_exhaustive]`, so only code that named those variants changes.

`rig_core::error::transient_transport(&error)` is now the method
`error.is_transient()` on `http_client::Error`.

```rust
// before
if rig_core::error::transient_transport(&error) { retry(); }
// after
if error.is_transient() { retry(); }
```

#### `HttpClientExt` sends and returns `Bytes`

`HttpClientExt::{send, send_multipart, send_streaming}` lose their type
parameters. A request body is `Bytes`, and `LazyBody` is no longer generic:
it resolves to `Bytes`. `http_client::text` takes `Response<LazyBody>`.
`DynHttpClient` and `rig_reqwest::ReqwestClient` change the same way.

```rust
// before
impl HttpClientExt for MyTransport {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
           + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.inner.send(req)
    }
    // send_multipart<U>, send_streaming<T> ...
}

// after
impl HttpClientExt for MyTransport {
    fn send(
        &self,
        req: Request<Bytes>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody>>>
           + WasmCompatSend + 'static {
        self.inner.send(req)
    }
    // send_multipart and send_streaming likewise
}
```

Callers drop the turbofish (`send::<_, Bytes>(req)` becomes `send(req)`),
build the body as `Bytes` (`Bytes::new()` for an empty one) and read the
reply as `Bytes`. A type written `LazyBody<Bytes>` is now `LazyBody`.

The `rig_http::wasm_compat::WasmCompatSendStream` trait is removed.
`http_client::BoxedStream` is now
`WasmBoxedStream<'static, http_client::Result<Bytes>>`. As a bound, write
`Stream<Item = http_client::Result<Bytes>> + WasmCompatSend`.

#### Framers refuse a frame over 64 MiB

`SseFramer::push`, `NdjsonFramer::push` and `Framer::push` yield
`Result<_, TooLarge>`. A frame or line over `framing::FRAME_LIMIT` (64 MiB)
is refused, naming the limit. `SseFramer::with_limit` and
`NdjsonFramer::with_limit` set another limit.

```rust
// before
for event in framer.push(&chunk) {
    handle(event);
}
// after
for event in framer.push(&chunk) {
    handle(event?);
}
```

Your error type needs a `From<TooLarge>` for the `?`. Do not `.flatten()`
the iterator: that drops the refusal.

`gemini::interactions_api::streaming::usage_of` now returns
`Result<Usage, ProviderError>`. The doc-hidden
`GenerateContentResponse::chunk` and `::document` are removed: call
`absorb(&GenerateContentChunk(map))` and `finish()` with
`rig_core::wire::document::Reassemble` in scope.

#### The shared sign-in types have one path

`TokenStore`, `FileTokenStore`, `MemoryTokenStore`, `StoreLock`,
`DeviceCodeHandler`, `DeviceCodePrompt`, `AuthContext` and `AuthError` are
imported from `providers::auth` only. They were re-exported from both
`providers::chatgpt::auth` and `providers::copilot::auth`. Those two modules
keep their own `AuthSource` and `Authenticator`.

```rust
// before
use rig::providers::copilot::auth::{
    AuthSource, Authenticator, DeviceCodeHandler, MemoryTokenStore,
};
// after
use rig::providers::auth::{DeviceCodeHandler, MemoryTokenStore};
use rig::providers::copilot::auth::{AuthSource, Authenticator};

let auth = Authenticator::new(
    AuthSource::OAuth,
    MemoryTokenStore::default(),
    DeviceCodeHandler::default(),
    true,
);
```

`AuthError` gains the variant `Reply { message, source }`, for a sign-in
server that answered with a failure. An exhaustive match needs the arm.

`FileTokenStore::load` now refuses a credential file that is a symlink, is
not a regular file, or is over 64 KiB, with the new
`providers::auth::FileStoreError` (`NotAFile`, `TooLarge`). On Unix it
narrows a file that group or others can access to 0600 before reading it,
and refuses the file with `FileStoreError::NotNarrowed` when that fails.
Replace a symlinked credential file with a copy, or sign in again. A
symlinked directory above the file still works.

New in the same module: `providers::auth::loopback::Loopback`, a native-only
loopback listener for a browser OAuth sign-in.
`chatgpt::auth::Authenticator::with_browser_opener` and `with_app_name`, and
`chatgpt::auth::browser_command`, are new too.

#### Provider observations are removed

The `rig_core::observe` module and everything that produced or carried a
provider observation are removed. Nothing in Rig produced or read them.
There is no replacement for observation traces.

| Removed | Use |
| --- | --- |
| `Model::call_observed(request, context)` and `DynModel`'s twin | `Model::call(request)` |
| `Model::stream_observed(request, context)` and `DynModel`'s twin | `Model::stream(request)` |
| `Dispatch::with_adapter_context`, `Dispatch::replaced_by` | nothing; drop the call |
| `Encoded::with_route`, `with_projection`, `with_analysis_only` and the `route`, `project`, `analysis_only` fields | nothing; drop the call |
| `wire::Projector`, `wire::ObservationSink`, `wire::AdapterEvent`, `AdapterUsage`, `AdapterVerdict`, `AdapterErrorEnvelope` | nothing |
| `Operation::observed_usage` | nothing; drop the method from your `Operation` impl |
| `serve::Observe`, `Dispatch::with_observer` | `serve::Recorder`, `Dispatch::recorded_by` |

```rust
// before
let reply = model.call_observed(request, context).await?;
let dispatch = dispatch.with_observer(Box::new(observer));
// after
let reply = model.call(request).await?;
let dispatch = dispatch.recorded_by(Arc::new(recorder));
```

A `serve::Recorder` takes `&self` in every method, and its `discard` and
`patch` get the effect's id, not the layer's name. Add the dispatch's scopes
before `recorded_by`, as before.

`EffectRecord` has a new public field, `call: Option<CallRef>`. A struct
literal adds `call: None`. The `rig_core::wire::Event<W>` alias is removed;
write `<<W as Wire>::Op as Operation>::Event`.
`ErrorReport::context_overflow(..)` is removed, and
`ErrorReport::is_context_overflow()` stays.

#### Cohere chat is two explicit wires

`Cohere::completion` returns the Compatibility API's wire,
`Model<openai::wire::Chat>`, and the new `Cohere::native_completion` returns
the native `Model<NativeChat>`. `CohereChat`, `ChatRoute`, `CohereDecoder`
and `CohereEvent` are removed. The Compatibility API was `ChatRoute`'s
default, so `client.completion(model)` sends what it sent before.

```rust
// before
let mut model = client.completion(cohere::COMMAND_A_03_2025);
model.wire = model.wire.with_route(ChatRoute::Native);
// after
let model = client.native_completion(cohere::COMMAND_A_03_2025);
```

For `ChatRoute::Auto`, build both models from one client and send each
request that carries documents to the `native_completion` model.
`ReplayTarget::route` is removed with it; a custom `ReplayTarget` drops its
override.

#### OpenAI and its dialects

- `OpenAIConfig::with_alternate_key` is removed. Say which header carries
  the key:

  ```rust
  // before
  let config = OpenAIConfig::with_alternate_key(&AZURE, "entra-token");
  // after
  let config = OpenAIConfig::with_key(&AZURE, "entra-token").with_auth(Auth::Bearer);
  ```

- `OpenAIConfig::with_audio_api_version` is removed. Set the public field:

  ```rust
  // before
  let config = config.with_audio_api_version("2025-04-01-preview");
  // after
  let mut config = config;
  config.audio_api_version = Some("2025-04-01-preview".into());
  ```

- `openai::wire::DialectHooks` has a new field,
  `model_listing: Option<ModelCatalogue>`. A literal adds
  `model_listing: None`, which keeps the shared `{ "data": [...] }` reading.

  ```rust
  // after
  static HOOKS: DialectHooks = DialectHooks {
      default_endpoint: None,
      model_route: None,
      completion_envelope: None,
      modality_envelope: None,
      model_listing: None,
  };
  ```

- `openai::wire::Responses` has a new public field, `store: Option<bool>`.
  A literal adds `store: None`. `Responses::with_store(false)` makes every
  request from the wire stateless.
- `<openai::wire::Models as Wire>::Event` is `model::ModelList`, decoded by
  the new `openai::wire::ModelsDecoder`. It was `ModelsReply`.
- The embeddings knobs are removed: `openai::embedding::EncodingFormat`,
  `Embeddings::with_encoding_format`, `Embeddings::with_user`, the
  `encoding_format` and `user` fields, and
  `EmbeddingQuirks::supports_encoding_format` and `supports_user`. Float is
  the provider's default encoding. There is no replacement for `user`.

  ```rust
  // before
  let wire = Embeddings::new(config, "text-embedding-3-small", None)
      .with_encoding_format(EncodingFormat::Float)
      .with_user("end-user-1");
  // after
  let wire = Embeddings::new(config, "text-embedding-3-small", None);
  ```

- `openai::Usage`, `openai::PromptTokensDetails`,
  `openai::CompletionTokensDetails` and the strict
  `openai::EmbeddingResponse` are removed.
  `CompatibleEmbeddingResponse::usage` is `Option<EmbeddingUsage>`, whose
  counters are `u64`:

  ```rust
  // before
  let usage: Option<openai::Usage> = response.usage;
  let prompt = usage.map(|u| u.prompt_tokens as u64);
  // after
  let usage: Option<openai::EmbeddingUsage> = response.usage;
  let prompt = usage.map(|u| u.prompt_tokens);
  ```

  Code that read Chat usage through `openai::Usage` reads
  `completion::Usage` from the completion response.
- `transcription::DurationTag`, `TokensTag` and the `r#type` fields of
  `TranscriptionUsage::{Duration, Tokens}` are removed. The JSON is the
  same. `TranscriptionUsage::Tokens::total_tokens` is `Option<u64>`.

  ```rust
  // before
  TranscriptionUsage::Duration { r#type: DurationTag::Duration, seconds: 6.0 }
  // after
  TranscriptionUsage::Duration { seconds: 6.0 }

  // before
  if let TranscriptionUsage::Tokens { total_tokens, .. } = usage {
      record(total_tokens);
  }
  // after
  if let TranscriptionUsage::Tokens { total_tokens: Some(total), .. } = usage {
      record(total);
  }
  ```

- `providers::copilot::model_listing` is removed. List Copilot's models
  through the dialect: `config.connect(http).list_models().await?`, where
  `config` is the Copilot `OpenAIConfig`. Each `ModelInfo` it returns now
  carries `supports_tool_calls`, `type` and `owned_by`.
- The ChatGPT constants `chatgpt::{GPT_5_4, GPT_5_4_PRO, GPT_5_3_CODEX,
  GPT_5_3_CODEX_SPARK, GPT_5_3_INSTANT}` are removed with their catalog
  rows. Use `chatgpt::DEFAULT_MODEL` or a row the plan still serves, such as
  `chatgpt/gpt-6.1-sol`.
- `azure::GPT_4_TURBO` and `azure::GPT_4_32K_0613` are removed. They were
  aliases: use `azure::GPT_4` and `azure::GPT_4_32K`, which are the same
  strings.

#### Anthropic, Gemini and Voyage AI

- Anthropic: `anthropic::completion::ANTHROPIC_VERSION_2023_01_01`,
  `Default` for `anthropic::streaming::MessagesDecoder` and
  `anthropic::Messages::with_default_max_tokens` are removed.

  ```rust
  // before
  let decoder = MessagesDecoder::default();
  let wire = wire.with_default_max_tokens(4096);
  // after
  let decoder = MessagesDecoder::new(anthropic::wire::Quirks::anthropic());
  let mut wire = wire;
  wire.default_max_tokens = Some(4096);
  ```

- Gemini: `gemini::InteractionsModels` is removed; use `gemini::Models`.
  `gemini::VerifyKeyDecoder` is removed; `gemini::VerifyKey` decodes with
  `rig_core::operation::VerifyDecoder`. With the audio API,
  `gemini::audio_generation::SpeechDecoder` is removed; take the wire's own
  decoder (`Wire::decoder`).
- Gemini: the free functions and constants
  `gemini::completion::{accepts, encodes, normalize_tool_call_id,
  declares_tools, CALL_ID_SLOT, BLOB_SLOTS}` are no longer public. Call the
  `ReplayTarget` methods of the same names on a Gemini wire
  (`wire.declares_tools(&request)`, `wire.call_id_slot()`). Every
  GenerateContent wire, `rig-vertexai`'s and `rig-gemini-grpc`'s included,
  now gets its `ReplayTarget` impl from one blanket impl, so bring
  `ReplayTarget` into scope to call them.
- Gemini: `HarmCategory::as_str` and `HarmBlockThreshold::as_str` are
  removed. Both enums serialize to the same strings:

  ```rust
  // before
  let threshold = HarmBlockThreshold::BlockOnlyHigh.as_str();
  // after: Value::String("BLOCK_ONLY_HIGH")
  let threshold = serde_json::to_value(HarmBlockThreshold::BlockOnlyHigh)?;
  ```

- Voyage AI: `Rerank::with_top_k` and `Rerank::with_return_documents` are
  removed. Set the public fields:

  ```rust
  // before
  let wire = wire.with_top_k(1).with_return_documents(true);
  // after
  let mut wire = wire;
  wire.top_k = Some(1);
  wire.return_documents = true;
  ```

#### The catalog and model listings

- `model::ModelInfo` has a new public field,
  `supports_tool_calls: Option<bool>`. A literal adds it, or starts from a
  constructor:

  ```rust
  // before
  let info = ModelInfo { id, name, description, r#type, created_at, owned_by,
      context_length, max_output_tokens };
  // after
  let info = ModelInfo { name, description, ..ModelInfo::from_id(id) };
  ```

- `catalog::ModelStatus` gains `Checking`, `NotOffered(String)` and
  `Unconfigured`. An exhaustive match adds the arms.
- `catalog::UnknownReasoning` has a new public field,
  `budget: Option<RangeInclusive<u32>>`. A literal adds
  `budget: spec.reasoning.budget().cloned()`, or `None`. New beside it:
  `ReasoningSupport::parse`, `name_of` and `reads`, one reader and printer
  for a reasoning setting written as text (`high`, `off`, `4096 tokens`).
- `ModelSpec::output_limit()` is new: a model's output limit held to its
  context window. Use it where you clamped `max_output_tokens` by hand.

#### Cache cost helpers are removed

`completion::{CacheCost, CacheRates}`, the `completion::cache_cost` module
and `impl From<&gemini::CacheReport> for CacheCost` are removed. Price a
usage from `Usage`'s counters with your own rates:

```rust
// before
use rig_core::completion::{CacheCost, CacheRates};
let cost = CacheCost::from_usage(&usage);
let usd = cost.usd(&CacheRates {
    input: 3.0, cached_read: 0.3, cache_write: 3.75, storage_per_hour: 0.0,
});
// after: USD per million tokens
let input = usage.input_tokens.unwrap_or(0);
let reads = usage.cached_input_tokens.unwrap_or(0);
let writes = usage.cache_creation_input_tokens.unwrap_or(0);
let uncached = input.saturating_sub(reads + writes);
let usd = (uncached as f64 * 3.0 + reads as f64 * 0.3 + writes as f64 * 3.75) / 1e6;
```

For Gemini explicit caches, take the written tokens and the stored
token-hours from `CacheBook::report()` and price them the same way.

#### Provider test utilities

`test_utils::RecordingHttpClient::with_error_response` is removed; use
`with_error_response_headers`. `test_utils::raw_parity::assert_raw_parity`
is removed; compare the two halves of `raw_pair` through `comparable`. In
`test_utils::streaming_conformance`, `ScenarioReport` and the xfail
arguments are removed: `ScenarioOutcome::Ran` is a unit variant, and
`check_gated_outcome` and `check_ungated_outcome` take the scenario, the
capability (gated only) and the outcome.

### 3. Agent runs (rig-agent)

#### One `ChatBot`

`integrations::cli_chatbot` is one `ChatBot::new(agent)`. `ChatBotBuilder`,
`Chat`, `ChatImpl` and `AgentImpl` are removed, and `ChatBot` is no longer
generic. `run` returns `StreamToStdoutError`, not `PromptError`, so a function
that returns its result changes its return type.

```rust
// before
use rig::integrations::cli_chatbot::ChatBotBuilder;
async fn chat(agent: Agent) -> Result<(), PromptError> {
    ChatBotBuilder::new().agent(agent).max_turns(10).build().run().await
}
// after
use rig::agent::StreamToStdoutError;
use rig::integrations::cli_chatbot::ChatBot;
async fn chat(agent: Agent) -> Result<(), StreamToStdoutError> {
    ChatBot::new(agent).max_turns(10).run().await
}
```

`show_usage()` stays. A custom `impl Chat` passed to `ChatBotBuilder::chat`
has no replacement: write the read loop yourself over `agent.prompt(line).stream()`.
`StreamToStdoutError` has `Run(PromptError)`, `Io` and `Incomplete`.

#### Persisted runs: `RUN_FORMAT` 5 → 7

`RUN_FORMAT` is 7; 0.46 wrote 5. A serialized `AgentRun` written by 0.46 does
not deserialize: the error reads `resume refused: the run is format 5, this
rig reads format 7`. There is no converter. Finish or discard the runs you
parked before you upgrade, or start them again from their history on 0.47.
If you tried a build from `main` that wrote format 6, the same holds.

#### `prepare_request` and `PreparedRequest`

`prepare_request` takes the prompt and builds the whole `CompletionRequest`.
`PreparedRequest` is `{ request, output_mode, policy }`: the fields that
mirrored the request and `PreparedRequest::apply` are removed. The error is
`ProviderError` (`ProviderError::Request`), and `run::PrepareError` is gone.

```rust
// before
let prepared = prepare_request(&spec, &caps, &history, tools, pinned, patch)?;
run.advertise_tools(turn, prepared.tools.clone());
let policy = prepared.policy.clone();
let request = prepared.apply(CompletionRequest::new(prompt));
// after
let PreparedRequest { request, policy, .. } =
    prepare_request(&spec, &caps, prompt, &history, tools, pinned, patch)?;
run.advertise_tools(turn, request.tools.clone());
```

Read `prepared.request.<field>` where you read `prepared.<field>`. The
provider-hosted tools that were `prepared.provider_tools` are in
`prepared.request.additional_params` under `tools`. Where you matched
`PrepareError::Request(reason)`, match `ProviderError::Request(reason)`:
`reason` is a shared error (`Arc<dyn Error>`), not a `String`, so print it
with `to_string()`,
and `ProviderError` is `#[non_exhaustive]`, so the match needs a `_` arm.
`TurnPolicy::new` returns `ProviderError` too, and `TurnPolicy::allowed` is
removed.

`ToolServerError` is removed: `ToolServerHandle::tool_defs` and
`Agent::tool_definitions` return `ProviderError`.

#### Response identity and turn state

`ModelTurn` and `CompletionCall` carry one `identity: ResponseIdentity`
instead of `response_id` and `provider_request_id`. Their JSON keys are
unchanged. The `with_identity` and `with_finish_reason` builders and
`CompletionCall::identity()` are removed:

```rust
// before
let call = CompletionCall::new(0, usage, raw).with_identity(identity).with_finish_reason(reason);
let id = call.response_id.clone();
// after
let call = CompletionCall { identity, finish_reason: reason, ..CompletionCall::new(0, usage, raw) };
let id = call.identity.response_id.clone();
```

`StreamedTurn` has a new public field, `cut_off: Vec<usize>`: the positions
in `choice` of the calls the output limit cut. A struct literal adds
`cut_off: Vec::new()`. Its `finish_reason` field is removed; read
`CompletionCall::finish_reason`. `PartialStreamedTurn::rollback_messages`
returns `[Message; 2]`, not `Option<(Message, Message)>`.

#### Removed conveniences

Four `AgentRunner` overrides are removed with no replacement:
`without_max_tokens`, `replace_additional_params`,
`without_additional_params` and `without_tool_choice`. Build the agent
without the setting instead:

```rust
// before
let agent = AgentBuilder::new(model).max_tokens(10).build();
let run = agent.prompt("go").without_max_tokens();
// after
let agent = AgentBuilder::new(model).build();
let run = agent.prompt("go");
```

`AgentRunner::without_memory` is removed too: build the agent without
`AgentBuilder::memory`.

The rest were used only by tests. Each line is the old call, then what to
write:

| Removed | Write instead |
| --- | --- |
| `Agent::with_model(m)`, `Agent::with_model_label(l)` | `agent.set_model(..)`, `agent.set_model_label(l)`, or `using_model(l)` on the run |
| `Agent::model_descriptor()` | `agent.handler_descriptor(agent.model_key().raw())` |
| `Agent::model_label()` | the `model` of that descriptor's `family`, when it is `FamilyDescriptor::Completion { model, .. }` |
| `Extractor::with_model`, `with_model_label`, `agent()` | pick the model on the builder (`ExtractorBuilder::from_agent_builder`) or per run with `using_model` |
| `AgentBuilder::retrieved_tools_handler(..)` | `builder.tool_server_handle(ToolServer::new().retrieved_tools_handler(..).run())` |
| `HookStack::with(hook)`, `HookStack::len()` | `let mut stack = HookStack::new(); stack.push(hook);` |
| `HookContext::tool(&key)`, `index(&key)`, `model(&key)` | `ctx.bind(&Key::<F>::new_unchecked(key.clone()))`, with `F` the handler's family: `family::Tool`, `family::Retrieve`, `family::Completion`. Another family is refused at the bind |
| `HookContext::agent_name()` | none |
| `Scratchpad::contains::<T>()` | `scratchpad.get::<T>().is_some()` |
| `DispatchEvent::completion_request()` | match `event.kind` on `EffectKind::Completion { request, .. }` |
| `ModelSelection::new(..)` | the struct literal; its fields are public |
| `MultiTurnStreamItem::final_response(content, usage)` | `MultiTurnStreamItem::FinalResponse(PromptResponse::new(text, usage))` |
| `TypedPromptResponse::new(..).with_completion_calls(..)` | the struct literal |
| `PromptResponse::empty`, `with_messages`, `with_completion_calls` | `PromptResponse::new` and its public fields |
| `ToolCatalog::key(name)` | `catalog.get(name).map(RegisteredTool::key)` |
| `TurnTools::contains(name)` | search its definitions |
| `AgentRun::entries_of(kind)`, `last_entry_of(kind)` | filter `run.entries()` on `kind` |
| `AgentRun::with_validated_history(h)` | `pairing::validate(&h)?;` then `run.with_history(h)` |
| `AgentRun::response()` | keep the `PromptResponse` the run returned |
| `ModelTurn::from_response(&resp, &prepared)` | `ModelTurn::from_policy(&resp, prepared.policy.clone())` |
| `StepEventKind::{CompletionCall, ModelTurnFinished, InvalidToolCall}` | none; the other kinds stay |
| `test_utils::HookTestHarness` | run the hook on an agent with a mock model |

`ExtractorBuilder<T>` no longer requires `T: Serialize`.

The facade path `rig::agent::tool` is removed. Import from `rig::tool`, and
write a context parameter of a `#[rig_tool]` function as
`rig::tool::ToolContext`:

```rust
// before
use rig::agent::tool::{ToolContext, ToolSet};
// after
use rig::tool::{ToolContext, ToolSet};
```

rig-agent's `audio` and `image` cargo features are removed, as in rig-core.

#### A run with an output schema checks its answer

A run with an output schema fails when its final answer does not fit the
schema, in every output mode. 0.46 returned the text unchecked in the native
and prompted modes. The model is asked again while the output-retry budget
lasts (`RunSpec::DEFAULT_OUTPUT_RETRIES` is 1), then the run ends with the new
`PromptError::OutputSchemaMismatch`:

```rust
match agent.prompt("summarize").await {
    Ok(response) => { /* the output fits the schema */ }
    Err(PromptError::OutputSchemaMismatch { output, problems, .. }) => {
        for problem in &problems {
            eprintln!("{}: {}", problem.at, problem.what);
        }
    }
    Err(other) => return Err(other.into()),
}
```

The extractor and `prompt_typed` return it wrapped, as
`StructuredOutputError::Prompt(PromptError::OutputSchemaMismatch { .. })`.
An extractor whose model answered in prose used to return
`StructuredOutputError::EmptyResponse`.

#### New stream items

`MultiTurnStreamItem` has two new variants: `ToolExecutionStarted
{ tool_call }`, once for each call a run dispatches, and `ToolResultItem
{ tool_result, index, count }`, each tool's accepted output as soon as it is
accepted. The enum is `#[non_exhaustive]`, so a `match` already has a `_`
arm and still compiles. A consumer that counts or prints every item sees
more of them: handle the two variants or let them fall to `_`. Committed
`ToolResult` items and the history are unchanged.

### 4. Memory and compaction (rig-memory)

rig-memory's compaction was rewritten. The policy, summarizer and
output-clearing types are gone. In their place are plain functions that plan
a compaction, build its summary request, read the reply and put the summary
in the conversation. Sessions and the token counter changed in smaller ways.
The coding agent's `[compaction]` settings table changed with it; see
[upgrading a 0.46 home](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).

#### Compaction is a set of functions, not a policy

These 0.46 items are removed from `rig_memory`:

| Removed | Use instead |
| --- | --- |
| `CompactionPolicy` and its `cut`, `keep`, `over_threshold`, `plan`, `cut_summary`, `plan_summary` | `CompactionSettings` (`enabled`, `reserve_tokens`, `keep_recent_tokens`, with `should_compact`), `prepare_compaction`, `prepare_compaction_at`, `find_cut_point` |
| `Summarizer` (`request`, `instruction`, `notes_instruction`, `max_tokens`, `summary_text`, `DEFAULT`) | `summary_request`, `turn_prefix_request`, `summarization_request`, `summary_text` |
| `SummaryPrompts` | the built-in prompts (`SUMMARIZATION_SYSTEM_PROMPT` is public), or your own text through `summarization_request` |
| `SummaryLimits` | `clamp_max_tokens`, or the `max_output` argument of `summarization_request` |
| `SummaryPlan`, `SummaryCut` | `CompactionPreparation` |
| `SummaryError` | `SummaryFailure` |
| `Compacted`, `COMPACTED`, `Summarized` | the summary's own entry: `summary_state`, `SummaryState::read`, `summarized_in_log` |
| `known_limits` | `spec.max_output_tokens.is_some() \|\| spec.context_window.is_some()` |

These have no replacement: `ClearToolOutputs`, `ClearPlan`, `Cleared`
(old tool outputs are no longer cleared), `NearWindow`, `TrackArgument`,
`TrackedSet` (a summary lists the files its messages read and changed, and
nothing else), `CompactReason` (the coding agent has its own,
`rig_basics::compaction::CompactReason`), `AFTER_OVERFLOW`, and the undo:
`undo_summary`, `Uncompacted` and `UndoError`.

A compaction keeps about `keep_recent_tokens` of the newest messages and
never splits a tool exchange. Everything before that is summarized in a
request of its own. When the cut falls inside a turn, that turn's earlier
part comes back in `CompactionPreparation::turn_prefix`, to be summarized
apart (`turn_prefix_request`) and joined with `merge_split_turn`.

```rust
// before
let policy = CompactionPolicy::default();
let Some(plan) = policy.plan_summary(&conversation, &state, &CompactReason::Threshold, &spec)?
else {
    return Ok(());
};
// send plan.request; `reply` is the answer
let summary = Summarizer::summary_text(&reply)?;
let state = SummaryState { summary, ..plan.state };
apply_summary(&mut conversation, plan.keep_from, &state);

// after
use rig_memory::{SummaryState, apply_summary, prepare_compaction, summary_request, summary_text};

let settings = rig_memory::CompactionSettings::default();
let Some(prepared) = prepare_compaction(conversation, settings.keep_recent_tokens, false)
else {
    return Ok(()); // nothing to summarize
};
let previous = (prepared.previous.as_ref(), "");
let request = summary_request(&prepared.to_summarize, previous, spec, reasoning)?;
// send `request` with your model; `reply` is its answer
let summary = summary_text(reply, "Summarization")?;
let state = SummaryState {
    summary,
    read_files: prepared.files.read_files,
    modified_files: prepared.files.modified_files,
};
// `None` when the conversation no longer starts with the messages the summary read
let _spliced = apply_summary(conversation, &prepared.replaces, &state);
```

The third argument of `prepare_compaction` is `true` after the provider
refused a request as too long. The second element of `summary_request`'s
tuple is the user's focus text, empty for none. `summary_text` takes a label
that its `SummaryFailure` names (`Failed`, `Truncated`, `CalledTool`,
`Empty`).

`apply_summary` changed its second argument. It took the id of the first
message to keep. It now takes the messages the summary read
(`CompactionPreparation::replaces`) and replaces exactly those, only while
the conversation still starts with them:

```rust
// before
apply_summary(&mut conversation, keep_from, &state);   // Option<MessageId>
// after
apply_summary(&mut conversation, &prepared.replaces, &state);   // &[Arc<Message>]
```

For your own prompts, build the request with `summarization_request`. It
sets no prompt-cache writes. Its last argument is the output limit: `None`
fits the model's limit to the input (`clamp_max_tokens`), and `Some(n)` is
sent as it is:

```rust
// before
let summarizer = Summarizer { prompts: MY_PROMPTS, ..Summarizer::DEFAULT };
let request = summarizer.request(&older, previous, &CompactReason::Threshold, &spec)?;

// after
let ask = format!(
    "<conversation>\n{}\n</conversation>\n\n{MY_ASK}",
    rig_memory::serialize_conversation(messages)
);
rig_memory::summarization_request((MY_SYSTEM, ask), spec, None, None)
```

#### `SummaryState` and a compaction's record

`SummaryState` holds file lists instead of tracked sets, and no longer
implements `Serialize` or `Deserialize`. The summary's entry in the
conversation is the compaction's only record. There is no `Compacted` aside
any more.

```rust
// before
let state = SummaryState { summary, tracked };
let state = Compacted::of(&conversation).last().map(|(_, record)| record.state);
let Summarized { replaced, records } = summarized_in_log(store, log)?;

// after
let state = SummaryState { summary, read_files, modified_files };
let state = rig_memory::summary_state(conversation);
let replaced = rig_memory::summarized_in_log(store, log)?;
```

`summarized_in_log` now returns the map itself,
`BTreeMap<MessageId, Vec<Entry>>`: the entries each summary replaced, by the
summary's id. A reset of the log's conversation drops them all.
`SummaryState::read(text)` reads a state back from a summary entry's text,
and `SummaryState::track` is removed.

There is no undo. Code that called `undo_summary` can still read what a
summary replaced with `summarized_in_log`, but nothing puts it back.

#### Token counting

The `TokenCounter` trait is removed, with its impls for closures, `Arc`,
`Box` and `&dyn`. `HeuristicTokenCounter::count` is an inherent method, and
`HeuristicTokenCounter::new` and `::anthropic` are removed: only
`HeuristicTokenCounter::default()` remains (four bytes a token).
`recent::TokenBudget` lost its type parameter and its `counter` field.

```rust
// before
use rig_memory::{HeuristicTokenCounter, TokenCounter, recent::TokenBudget};
let recent = TokenBudget { max_tokens: 8_000, counter: HeuristicTokenCounter::anthropic() };

// after
use rig_memory::recent::{TokenBudget, view};

let recent = TokenBudget::new(8_000);
let kept = view(messages, &recent);
let tokens = rig_memory::HeuristicTokenCounter::default().count(message);
```

No rig-memory API takes a custom counter now. To window by your own count,
implement `recent::Recent`:

```rust
struct MyBudget {
    max_tokens: usize,
}

impl rig_memory::recent::Recent for MyBudget {
    fn start(&self, messages: &[Arc<Message>]) -> usize {
        let mut used = 0usize;
        let mut start = messages.len();
        for (at, message) in messages.iter().enumerate().rev() {
            used = used.saturating_add(my_count(message));
            if used > self.max_tokens {
                break;
            }
            start = at;
        }
        start
    }
}
```

The request-size helpers are removed too: `Prepared`, `RequestStart`,
`next_request_tokens`, `OUTPUT_MARGIN` and `TURN_OUTPUT`. Estimate a request
with `request_estimate(&CompletionRequest)` or `request_tokens`, a message
with `estimate_message_tokens`, and fit an output limit beside an input with
`clamp_max_tokens`:

```rust
// before
let room = spec.context_window.map_or(asked, |window| {
    asked.min(u64::from(window).saturating_sub(input_tokens + rig_memory::OUTPUT_MARGIN))
});
// after
rig_memory::clamp_max_tokens(spec, input_tokens, asked)
```

`clamp_max_tokens` keeps a larger margin than `OUTPUT_MARGIN` did: 4,096
tokens or 15% of the input, whichever is more, while that leaves 32,768
tokens of output.

#### Sessions

`Session` has a private field, so a struct literal no longer compiles
outside rig-memory, and a pattern that names every field needs `..`. Build
one from `Session::default()` or `Session::read`:

```rust
// before
let mut session = Session { agents: vec![agent], ..Session::default() };

// after
let mut session = Session::default();
session.agents = vec![agent];
session.save(store)?;
```

`Session::save` now fails with `SessionError::Changed { log }`, before
writing anything, when a log was started, written, rewritten or removed
since the session was read or last saved. Read the session again and redo
the edit. A save to an empty store still works, so a session can be copied.

Smaller removals:

- `main_order` is removed. Use `Session::main_agent()`, which also follows
  the new saved key `CURRENT`.
- The `migration` module (`Chain`, `KEPT`, `SHIPPED`, `Verify`) is removed
  with no replacement. A log in another format is still refused with
  `SessionError::Format`.
- `store_images`, `load_images` and `rewrite_value` are removed with no
  replacement. `save` and `load` handle image blobs themselves.

`Rewrite` has a new hook, `facts`, for what a tool call published beside its
result (an edit's diff, a read's line numbers). It has a default body, so
existing impls compile. A `Rewrite` that redacts, or one that forwards every
hook to an inner rewrite by hand, should implement it too.

### 5. Vector stores, cassettes, MCP and the facade's features

#### Cargo features of `rig`, `rig-core`, `rig-agent` and `rig-http`

The `audio` and `image` features are removed from `rig`, `rig-core` and
`rig-agent`, and the `websocket` feature from `rig-core` and `rig-http`. They
added no dependency. The code they gated always compiles now. Cargo refuses a
manifest that still names one, so delete the names. The facade's `websocket`
feature, which adds the bundled backend, is unchanged.

```toml
# before
rig = { version = "0.46", features = ["audio", "image", "derive"] }
rig-core = { version = "0.46", features = ["websocket", "derive"] }
# after
rig = { version = "0.47", features = ["derive"] }
rig-core = { version = "0.47", features = ["derive"] }
```

The facade has a new default feature, `cli`. It builds the `rig` binary, the
coding agent's launcher, from the new `rig-cli` crate. `cargo install rig`
works as before. A library that depends on `rig` with default features now
also compiles `rig-cli` on native targets. To leave it out, turn the defaults
off and name the rest of the default set:

```toml
# before: the default features
rig = "0.46"
# after: the same library features, without the launcher
rig = { version = "0.47", default-features = false, features = ["reqwest", "agent", "derive", "rustls"] }
```

The facade's internal `facade-build-tests` feature is removed.

For contributors to the repository: nextest 0.9.130 or newer is required
(`cargo install cargo-nextest --locked`). Each crate's integration tests are
one binary, `integration`, and `cargo xtask verify` lost `--changed`, `--pr`
and `--lanes`:

| Before | After |
| --- | --- |
| `cargo xtask verify --changed`, `--pr --base X` | `cargo xtask verify --affected --base X` (`--plan --base X` prints the plan) |
| `cargo nextest run -p rig-cassette --test anthropic` | `cargo nextest run -p rig-cassette --test integration -E 'test(/^anthropic::/)'` |
| `cargo test -p rig --features facade-build-tests --test tool_facade_features` | `cargo xtask verify --check guard-facade-features` |

A plugin test that leaned on built-in catalog rows through
`rig-harness-test-support`'s `scripted_models` now gets only the scripted
models and a client that refuses every request. Insert the row with
`connector.catalog_mut().insert(..)`, give its vendor a key with
`connector.provide(..)`, and install a fake client with `wrap_http`.

#### `rig::agent::tool` is `rig::tool`

The facade no longer re-exports rig-agent's tool module under `rig::agent`.
The same items are in `rig::tool`. `#[rig_tool]` recognizes a context
parameter written `rig::tool::ToolContext`, and no longer one written
`rig::agent::tool::ToolContext`.

```rust
// before
use rig::agent::tool::{ToolContext, ToolSet};
// after
use rig::tool::{ToolContext, ToolSet};
```

#### Vector store behavior

No vector-store crate changed its public signatures, apart from one
addition: `LanceDbVectorIndex` implements `InsertDocuments`. Several return
errors where they used to guess; see
[Silent behavior changes](#silent-behavior-changes). In short:

- rig-lancedb: `top_n` and `top_n_ids` return
  `VectorStoreError::MissingIdError` for a row whose id is missing or is
  neither text nor a number. A null cell decodes to JSON `null`, so a field
  that can be null must be an `Option`.
- rig-postgres: `top_n` returns the deserialization error for a row it
  cannot read.
- rig-mongodb: a filter value BSON cannot represent is
  `FilterError::Serialization`.
- rig-vectorize: `VectorizeFilter::and` with one operator repeated on a field
  with two values fails `validate()`.

#### rig-cassette

`effect_log::EffectFilter` is removed. Filter the records yourself; this is
the rule it applied:

```rust
// before
let filter = EffectFilter::default().agent(agent).kind("tool_call").turn(1);
let records: Vec<_> = records.into_iter().filter(|record| filter.matches(record)).collect();

// after
records
    .into_iter()
    .filter(|record| {
        record.scope.as_deref() == Some(agent)
            && record.kind.name() == "tool_call"
            && record.turn == Some(1)
    })
    .collect()
```

rig-basics' `EffectLog::read_matching`, which took the filter, is removed
with it (see [plugin crates](#8-plugin-crates-rig-basics-rig-tui-and-the-others)).

Also removed, with no replacement:

- `EffectLogReplayer::check`. Keep the `RequestCheck` you pass to
  `checking(..)`.
- `RecordedAnswers::scopes`. Collect the distinct `record.scope` values of
  the log.
- `AccountFailure::ALL` (feature `http`). Name the four variants: `Auth`,
  `RateLimit`, `Quota`, `Credit`.
- `http::owned_headers` (feature `http`).

#### rig-rmcp

`From<RunningService<RoleClient, S>> for McpConnection` is removed; call
`McpConnection::new`. `McpTool::timeout()` is removed; keep the value you
gave `with_timeout`.

```rust
// before
let connection: McpConnection = ().serve(transport).await?.into();
// after
let connection = McpConnection::new(rig_rmcp::client_config("my-app", "1.0").serve(transport).await?);
```

`client_config(name, version)` is new. It offers the newest released MCP
protocol version instead of rmcp's draft default, which some servers accept
and then refuse the session for. rig-rmcp now builds on rmcp 3.9.

A server whose `initialize` answer declares no `tools` capability is no
longer asked for tools: `McpConnection::tools`, the new
`McpConnection::list_tools(timeout)` and `McpClientHandler::connect` give
none. `McpConnection::resources` does the same for a server that declares no
resources. To ask such a server anyway, call rmcp through
`connection.peer()`.

#### rig-typesafeai

A stored `JevConfig` with an unknown top-level key no longer loads. The
error names the key. In 0.46 the key was ignored, so a misspelled
`provider` loaded without the restriction it meant to set. `JevConfig` gains
`with_token`, which sets the credential on a config loaded from storage, and
`provider`, which sends OpenRouter routing preferences.

### 6. The kernel (rig-ecs)

This section is for plugin authors and for apps built on `rig-ecs`
directly. The kernel now owns three things a plugin used to do by hand:
making agents, stopping turns and saving the frame. Types that only the
kernel may build have private fields, so the compiler points at each place.

#### Agents are made through `rig_ecs::spawn`

`Agent` and `SpawnedBy` cannot be built outside rig-ecs. `Agent` no longer
implements `Default`, and `SpawnedBy`'s field is private. A root agent comes
from `spawn_root(world, bundle)`, which inserts the bundle and `Agent` in one
insert. Every other agent comes from an `Origin` and an `AgentSpec`, through
the `Spawner` system parameter or the free function `spawn`. What the spec
leaves out is the origin's.

```rust
// before
let root = world.spawn((Agent, AgentId("main".into()))).id();
let child = commands.spawn((Agent, SpawnedBy(parent), Name::new("reviewer"))).id();
let parent = spawned_by.0;

// after
use rig_ecs::spawn::{AgentSpec, Origin, Spawner, spawn_root};

let root = spawn_root(world, AgentId("main".into())).id();

fn add_reviewer(parent: Entity, mut spawner: Spawner) {
    let spec = AgentSpec { name: "reviewer".into(), ..AgentSpec::default() };
    spawner.spawn(Origin::ChildOf(parent), spec).ok();
}

let parent = spawned_by.parent();
```

`spawn_root(world, ())` makes a root with only the defaults. With exclusive
world access, `rig_ecs::spawn::spawn(world, origin, spec)` returns the new
`Entity`. `Origin` is `Root`, `ChildOf(agent)` or `Like(agent)`, which puts
the new agent beside `agent`, under its parent if any. `AgentSpec` has
`name`, `id`, `model`, `effort`, `tools`, `system_prompt` and `under`.

A spawn can be refused. `SpawnRefused` is `NotAnAgent`, `IdTaken`,
`NotALogName`, `Tool(NotYours)` or `Model(String)`. An explicit id is
refused when it is empty, holds `/` or `\`, is `.` or `..`, is longer than
`rig_memory::MAX_NEW_LOG_NAME` bytes, is the session log's own name, or
differs only in letter case from an id already taken. Agents saved under
such an id are still restored.

`On<Add<Agent>>` observers run before anything inserted on the returned
entity afterwards. Put what they must see, such as the id or the model, in
the spec or in `spawn_root`'s bundle.

The kernel sets no limit on how deep agents nest. A plugin that wants one
counts `SpawnedBy` ancestors itself. The removed `[subagents] max_depth`
setting is covered in [plugin crates](#8-plugin-crates-rig-basics-rig-tui-and-the-others).

#### `ToolAccess` is a struct

`ToolAccess` was the enum `All` / `Only(Vec<String>)`. It is now a struct
with private fields. Its saved form is unchanged.

```rust
// before
ToolAccess::All
ToolAccess::Only(vec!["read".to_owned()])
if let ToolAccess::Only(names) = &access { /* ... */ }

// after
ToolAccess::all()
ToolAccess::only(["read"])
if let Some(names) = access.names() { /* ... */ }
```

| Method | What it gives |
| --- | --- |
| `ToolAccess::all()` | every registered tool |
| `ToolAccess::only(names)` | an allow-list built from no other access |
| `access.narrow(names)` | this access kept to `names`, never more than it was |
| `access.within(&other)` | the tools both allow |
| `access.grant(&registered, asked)` | a child's access, or `NotYours` for the first tool this access does not keep |
| `access.allows(name)` | whether the tool may be called in this run |
| `access.keeps(name)` | whether the agent keeps the tool, whatever this run's tools |
| `access.names()` | the names it keeps, `None` for every tool |

The split between `allows` and `keeps` is new. The `RunTools` resource holds
the tools one run leaves every agent, such as the ones `--tools` names. The
kernel keeps each agent's access within `RunTools` and within the access of
every agent above it, and the session never saves `RunTools`. Code that
builds a child's tool list from its parent's access filters with `keeps`, not
`allows`, so the child's tools outlive the run:

```rust
// before
let tools: Vec<_> = registered.iter().filter(|name| access.allows(name)).collect();
// after
let tools: Vec<_> = registered.iter().filter(|name| access.keeps(name)).collect();
```

#### Stopping a turn: `StopTurn`, `StopReached`, `Stop` and `StopCause`

`StopTurn`'s fields are private, and the free-text `reason` is gone. Build
one with a constructor. It is still an entity event that you trigger.

```rust
// before
commands.trigger(StopTurn { entity: agent, reason: None });
commands.trigger(StopTurn { entity: agent, reason: Some("over the limit".into()) });

// after
commands.trigger(StopTurn::new(agent));                                  // the user's stop
commands.trigger(StopTurn::plugin(agent, "my-plugin", "over the limit"));
```

| Constructor | What it stops |
| --- | --- |
| `StopTurn::new(agent)` | the user's stop: the agent's turn and those below it |
| `StopTurn::plugin(agent, by, why)` | the same, with the plugin as the cause; the kernel shows no notice for it |
| `StopTurn::plugin_alone(agent, by, why)` | only the agent's own turn; the agents below go on |
| `StopTurn::plugin_done(agent, by, why)` | a turn the plugin ran for work of its own, such as a summary; messages the agent accepted meanwhile carry the turn on |

`StopTurn` is only a request. To react to an agent being stopped, observe
the new `StopReached`, which the kernel sends once on each agent a stop
reached, idle or not, after its turn is gone. Read the agent with
`event_target()`, the cause with `cause()` and whether it had no running
turn with `idle()`. `plugin_alone` and `plugin_done` send none, and neither
do the stops of the app's exit and of a restart.

`TurnOutcome::Stopped` holds a `Stop` instead of an `Option<String>`. Only
the kernel makes a `Stop`, so a plugin can no longer insert a stopped
outcome by hand.

```rust
// before
match outcome {
    TurnOutcome::Stopped(None) => { /* the user */ }
    TurnOutcome::Stopped(Some(why)) => { /* why */ }
    _ => {}
}

// after
match outcome {
    TurnOutcome::Stopped(stop) => match stop.cause() {
        StopCause::User => {}
        StopCause::Exit(why) => {}
        StopCause::Restarted => {}
        StopCause::Plugin { by, why } => {}
    },
    _ => {}
}
```

`stop.alone()` says whether a plugin ended only that turn. `StopCause` is
rig-core's and is in the rig-ecs prelude. It implements `Display` with the
text a stopped call is told. The constant `rig_ecs::restore::RESTARTED` is
removed: match `StopCause::Restarted`. The saved form of a stop,
`TurnEnd::Stopped`, changed with it; see [core conversations](#1-core-messages-conversations-tools-and-streaming-rig-core).

#### A call is known by its reply and its place

Two tool calls that share a provider's call id are now told apart. Each
tool call a reply asked for carries the new component `InReply`, and
`InReply::at()` gives its `CallRef` (rig-core's `{ reply, index }`). Key
your own per-call state by `CallRef`, not by the provider's `CallId`.
`Conversation::pending_calls()` returns `(CallRef, ToolCall)` pairs; see
[core conversations](#1-core-messages-conversations-tools-and-streaming-rig-core).

`Partial`, the reply a model call has streamed so far, has a new public
field `calls: Vec<PartialCall>`. A struct literal names it:

```rust
// before
Partial { text, reasoning }
// after
Partial { text, reasoning, calls: Vec::new() }
```

`Partial::follow(&event)` folds a stream event into it, and
`Partial::stopped()` gives the content a stop keeps: the text, and each
call the stop cut while it streamed, which is never run.

#### Forking is a request entity

`ForkAgent` was an entity event with public `agent` and `through` fields.
It is now a bundle that you spawn. The kernel inserts a `ForkResult` on the
request and then despawns it. A fork that cannot be made is a `ForkError`
in that result, no longer a notice.

```rust
// before
commands.trigger(ForkAgent { agent, through: None });

// after
let keep = |made: On<Add<ForkResult>>, results: Query<&ForkResult>| {
    if let Ok(ForkResult(Ok(fork))) = results.get(made.entity) {
        // `fork` is the new agent
    }
};
commands.spawn(ForkAgent::new(agent, None)).observe(keep);
```

`AgentForked` is still announced. A fork now carries `ForkedFrom`, the list
of agents its conversation came from.

A saved component says what a fork gets of it. A plain
`#[reflect(Component, Saved)]` is copied, as before. Write `Saved(Reset)`
for a value the fork starts afresh at its `Default`, such as a meter of the
agent's own calls, and `Saved(Never)` for one the fork must not get, such as
a duty the source owes.

```rust
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved(Reset), Default)]
struct CallsMade(u32);
```

A pending request of a plugin is an entity with the new relationship
`rig_ecs::requests::RequestOf(owner)`. It is despawned with its owner, and
`Requests` on the owner lists them.

#### Model choice and connections

`Models::connect` returns `rig_core::providers::registry::ConnectError`
instead of a `String`. `Models::cannot_use` words the refusal, and an app
may set the `ConnectWording` resource to word refused models itself.

```rust
// before
let connection = models.connect(reference)?;   // Err(String)
// after
let connection = models
    .connect(reference)
    .map_err(|why| Models::cannot_use(reference, why))?;
```

`SetModel` and `SetReasoning` have a new field, `kind: ChoiceKind`. It says
who chose: `ByUser` for the user's own command, which a plugin may keep for
later runs, or `ForThisRun` for a flag, an editor, an automatic pick or a
plugin's own choice.

```rust
// before
commands.trigger(SetModel { entity, model });
// after
commands.trigger(SetModel { entity, model, kind: ChoiceKind::ForThisRun });
```

A choice the agent took triggers the new `ChoiceTaken`. A refused one
triggers nothing.

`ReasoningSetting` no longer implements `Display`, because a setting's name
depends on the model. Print it through the model's spec:

```rust
// before
let text = setting.to_string();
// after
let text = spec.reasoning.name_of(setting.0);
```

#### Saving runs in the `Persist` schedule

`SaveSystems` is removed. The kernel saves what a frame changed after every
plugin's `Last` system ran, on the exit frame too, so there is nothing to
order before. Log writers moved from `Last` to the new `Persist` schedule,
where `WriteLogs` is now a set.

```rust
// before
app.add_systems(Last, count.after(StopTurnsOnExit).before(SaveSystems));
app.add_systems(Last, write_mine.in_set(WriteLogs));
app.add_systems(Last, report.after(WriteLogs));

// after
app.add_systems(Last, count.after(StopTurnsOnExit));
app.add_systems(Persist, write_mine.in_set(WriteLogs));
app.add_systems(Persist, report.after(WriteLogs));
```

A system in `Persist` that changes saved state is saved with the next
frame, and on the exit frame never. `restore::check_saved` is removed with
rig-memory's `migration` module ([memory](#4-memory-and-compaction-rig-memory)) and has no replacement.

#### Smaller changes

- `turn::RETRY_POLICY` is replaced by the `RetryPolicy` resource
  (`Res<rig_ecs::retry::RetryPolicy>`). The coding agent fills it from
  `[retry]` in `settings.toml`. `rig_ecs::tools::UnavailableCalls` holds
  `[tools] max_unavailable_calls` the same way.
- `DroppedOptions` is removed. Read the options a model call leaves out
  with `Adapted::dropped(&model_request)` in a `PrepareSystems::Fit` system.
  To use the list after Start, store it on the call yourself.

  ```rust
  // before
  fn warn(calls: Query<&DroppedOptions, Added<DroppedOptions>>) { /* ... */ }
  // after, in PrepareSystems::Fit
  fn warn(calls: Query<(&Adapted, &ModelRequest)>) {
      for (adapted, request) in &calls {
          let dropped = adapted.dropped(request);
      }
  }
  ```

- `SavedKeys::iter` is removed. Look one key up with `SavedKeys::path(key)`
  or `SavedKeys::key(type_path)`.
- New schedules and sets, none of which breaks existing code:
  `BetweenStages` runs between each two stages of an agent pass, so a system
  there can rewrite what earlier stages wrote, such as a `Notice`, before a
  later stage, an observer or a front reads it. `StartupSystems::FirstAgent`
  and `StartupSystems::Reconcile` name the kernel's `Startup` steps, so a
  system can run after the first agent exists and before restored turns
  start.
- `ContextStale` is unchanged. It is now also set on a fork cut before its
  source's end, until the fork's first reply.
- `stage::follow_successor`'s query also reads `Option<&RequestOf>`. Adding
  it as an observer is unchanged. A wrapper that forwards its arguments adds
  the third element to the query tuple.
- Types that stay and lost an impl or a method:

  | Type | Lost | Do instead |
  | --- | --- | --- |
  | `agent::StopTurn` | `bevy_reflect::Struct` (its fields are private) | build it with its constructors; read nothing by field name |
  | `fork::ForkAgent` | `Clone`, `Debug`, and being an event | build one per request with `ForkAgent::new` |
  | `agent::ToolAccess` | `map_entities` | nothing; it holds no entity |
  | `tasks::Running` | `Unpin` | nothing, unless a bound of yours asked for `Unpin` |
- The session's current agent is rig-harness's `CurrentAgent`; see
  [plugin host](#7-plugin-host-and-tools-rig-harness-rig-harness-protocol-rig-tools).

### 7. Plugin host and tools (rig-harness, rig-harness-protocol, rig-tools)

This section is for plugin authors and for code that builds an agent on
rig-harness by hand. rig-harness re-exports rig-ecs (`rig_harness::ecs`, the
prelude), so every kernel change in [the kernel section](#6-the-kernel-rig-ecs) reaches you
through it too. What changed for the plugin crates built on the host is in
[plugin crates](#8-plugin-crates-rig-basics-rig-tui-and-the-others).

#### Settings: no project settings, and tables are registered in `build`

A project has no settings file. `AppSettingsExt::add_project_setting`,
`ProjectSetting`, `ProjectSettings` and `ProjectSettingsSystems` are removed.
Register the table with `add_setting`, which reads `RIG_HOME/settings.toml`
only:

```rust
// before
app.add_project_setting::<TestRunner>("test_runner");
// after
app.add_setting::<TestRunner>("test_runner");
```

`add_setting` must be called in the plugin's `build`. A table read once the
run's front is chosen, which in practice is any read in a `finish`, refuses
the run with `a plugin read [section] once the run's front was chosen; it must
read it in its `build`, so a build describing itself checks it`. Keep the
front check for the systems and move the registration:

```rust
// before
fn finish(&self, app: &mut App) {
    if !app.runs_front(FRONT) {
        return;
    }
    app.add_setting::<MySettings>("mine").add_systems(Update, work);
}
// after
fn build(&self, app: &mut App) {
    app.add_setting::<MySettings>("mine");
}
fn finish(&self, app: &mut App) {
    if !app.runs_front(FRONT) {
        return;
    }
    app.add_systems(Update, work);
}
```

`rig plugin check --build` now checks every table a start reads, so a table
your plugin refuses fails there too.

#### The project and the working directory

`ProjectRoot` is still the directory the session works in. The project's own
files are found through the new `rig_harness::services::Project` resource:
`Project::root` is the `ProjectRoot` or the nearest directory above it that
holds `.git` or `.rig`, and `Project::config` is the path of its `.rig`
directory, which may not exist. It is `None` only when the root is your
home directory or the `.rig` is the `RIG_HOME` itself. A plugin that read `ProjectRoot/.rig` or keyed its data by
`ProjectRoot` reads `Res<Project>` instead:

```rust
// before
fn read(root: Res<ProjectRoot>) {
    let file = root.0.join(".rig").join("mine.json");
}
// after
fn read(project: Res<Project>) {
    let Some(config) = &project.config else { return };
    let file = config.join("mine.json");
}
```

Relative paths of the user and the model still resolve from `ProjectRoot`.

#### Per-call tools

`AppPerCallExt::add_live_tool_per_call` is removed. A tool whose calls report
their output as they run says so on its `PerCallTool` impl, and
`add_tool_per_call` reads it:

```rust
// before
app.add_live_tool_per_call(tool, ToolOptions::default());
// after
impl PerCallTool for MyTool {
    const LIVE: bool = true;
    fn at(&self, place: CallPlace) -> Self { /* ... */ }
}
app.add_tool_per_call(tool, ToolOptions::default());
```

A wrapper around a live tool forwards the flag (`const LIVE: bool =
Inner::LIVE;`), or its calls get no live output.

`rig_tools::Shell` no longer implements `PerCallTool`. Register a shell per
call through `PerCallShell`:

```rust
// before
app.add_live_tool_per_call(Shell::default(), ToolOptions::default());
// after
let shell = PerCallShell::new(Shell::default(), app.world_mut());
app.add_tool_per_call(shell, ToolOptions::default());
```

A `PerCallShell` ignores two fields of the `Shell` it holds. Its commands
start as the app's `Processes` say, not as `Shell::launcher` says; change how
they start with `Processes::set`. Its output is cut as the app's `ShellOutput`
resource says, not as `Shell::cut` says. `ShellOutput` moved from
`rig_basics::coding_tools` to `rig_harness::per_call` (and the prelude):

```rust
// before
use rig_basics::coding_tools::ShellOutput;
// after
use rig_harness::prelude::ShellOutput;

// a plugin that cuts or summarizes output itself
app.world_mut().get_resource_or_init::<ShellOutput>().set(None);
```

`CallPlace` has three new public fields, so a struct literal names them:
`filter: Option<LineFilter>` (what each output line goes through first, such
as a redaction plugin's filter), `seen: Option<Seen>` (the calling agent's
`FilesSeen`) and `overlaid: bool` (whether the workspace keeps the call's
writes from the files, as a preview overlay does). Prefer
`PerCallTools::place`, which fills them. A plugin that builds its own `Shell`
for a call carries `place.filter` into the new `Shell::filter` field.

What the file tools saw is each agent's own `FilesSeen` component in
`rig_harness::per_call`. It replaces rig-basics' app-wide `SeenFiles`
resource; read it with `Query<&FilesSeen>` on the calling agent.

`PerCall::build` is removed. Build a call's handler by the tool's name:

```rust
// before
let handler = per_call.build(place);
// after: `tools` is a `PerCallTools` system parameter
let Some(handler) = tools.build("read", place) else { return };
```

The `move` tool is gone (see below), and with it `impl PerCallTool for Move`.

#### Slash commands, pickers and fronts

`SlashCommand` has a new field, `args: Args`, set from the input type of the
command's system. A command that takes text keeps `In<CommandArgs>` and
compiles unchanged. A command that takes none says so with `In<NoArgs>`, and
text after its name is then refused before the system runs:

```rust
// before
fn clear(In(args): In<CommandArgs>, /* ... */) { let agent = args.agent; }
// after
fn clear(In(args): In<NoArgs>, /* ... */) { let agent = args.agent; }
app.add_command("clear", "Clear the notes", clear);
```

A front that shows `Picker`s and answers them must say so. Without it, every
picker a command opens in that front's runs is refused with the commands to
type instead:

```rust
// after
app.add_front(Front::new("mine").shows_pickers());
```

A plugin that shows pickers beside the front, such as a client attached to
the session, spawns a `ShowsPickers` component instead. `Front` has a new
public field `pickers` for the same thing.

`Picker::checked` is removed, and `PickItem` has a new public field,
`parent: Option<usize>`, the row a tree picker draws it under. Build items
with the new `PickItem::new` and set the fields you need:

```rust
// before
PickItem { label, command, detail: None, more: None, checked: false }
// after
PickItem { checked: true, ..PickItem::new(label, command) }
```

The types of the run contract moved from rig-basics to rig-harness, so a
front no longer depends on rig-basics for them:

```rust
// before
use rig_basics::print::{KeepRunning, Starting, Work, TOOLS};
// after
use rig_harness::run::{KeepRunning, Starting, Work, TOOLS};
```

`Starting`, `KeepRunning` and `Work` are in the prelude too.

#### How a run ends

`ExitReason::Failed` holds a `FailCode` instead of a `u8`, and a refusal is
its own reason, `ExitReason::Refused(String)`. A `FailCode` is never 0, 75 or
130. An exhaustive `match` on `ExitReason` adds the new variant.

```rust
// before
commands.trigger(Exit(ExitReason::Failed(1)));
commands.trigger(Exit(ExitReason::Failed(status))); // a child's code
commands.trigger(Exit(ExitReason::Failed(2)));      // a refusal
// after (FailCode is in rig_harness::prelude)
commands.trigger(Exit(ExitReason::Failed(FailCode::FAILURE)));
commands.trigger(Exit(ExitReason::Failed(FailCode::or_failure(status))));
commands.trigger(Exit(ExitReason::Refused(why)));
```

The exit codes are constants of rig-harness-protocol:
`rig_harness::host::exit::SIGNAL_EXIT_CODE` is now
`rig_harness_protocol::SIGNAL_EXIT_CODE` (130), beside the new
`REFUSED_EXIT_CODE` (2) and the existing `RELOAD_EXIT_CODE` (75).

An exit code of 75 alone no longer means a reload. The launcher reloads or
switches sessions only when the agent wrote that end (the `AgentEnd` record
described under rig-harness-protocol below), and `ExitReason::of` reads a plain `AppExit` of 75 as a failure. A
plugin front that exited 75 to restart triggers
`Exit(ExitReason::Reload)` instead.

Two launcher helpers changed shape. `host::launcher::Refused` carries the
refusal's text (`Refused(pub String)`, was a unit struct), and
`build_failure_message` takes the home whose build log it names:

```rust
// before
let text = launcher::build_failure_message(what, summary);
// after
let text = launcher::build_failure_message(&RigHome::of(world), what, summary);
```

#### Sessions and the main agent

`MainAgentQuery` is a `SystemParam` struct, not an alias of a `Query`. A
system that passes it to `main_agent(&query)` compiles unchanged. Code that
iterated it as a query takes its own
`Query<(Entity, &AgentId), (With<Agent>, Without<SpawnedBy>)>`.

The session's current agent is the new saved resource
`rig_harness::main_agent::CurrentAgent(pub String)`, which `main_agent`
reads. It replaces rig-basics' `ShownId`.

`edit_stopped` is removed. Rewrite another, stopped session with
`rewrite_stopped`, which v0.46.0 already had:

```rust
// before
edit_stopped(&stores, &paths.0, &dir, |session| { /* edit */ Ok(()) })?;
// after: `redact` implements rig-memory's `Rewrite`
rewrite_stopped(&stores, &paths.0, &dir, Author::plugin("redact"), &redact)?;
```

The `SessionDeleteFailed` message is removed. A failed delete is still
reported as a `Notice`; a `DeleteSession(id)` that wrote no
`SessionDeleted(id)` failed.

Logs are written in the kernel's new `Persist` schedule (see
[the kernel section](#6-the-kernel-rig-ecs)). A `Persist` system that reads the session's saved
marker orders itself after the new `host::session::MarkSaved` set.

#### File watches

`FileChanges::paths` is replaced by `FileChanges::event`, a
`rig_tools::WatchEvent`: `Snapshot(paths)` when the watch first lists the
files and when it can list them again after it could not, `Changed(paths)`
for each batch of changes, `Unavailable(error)` when listing failed, and
`Closed`.

```rust
// before
for change in changes.read() {
    for path in &change.paths { /* ... */ }
}
// after
for change in changes.read() {
    for path in change.event.paths() { /* ... */ }
}

// before: skipping the first batch, which was the listing
if !std::mem::replace(&mut armed, true) { continue; }
// after
let WatchEvent::Changed(paths) = &change.event else { continue };
```

A plugin that reads its files again on news waits for
`change.event.names_files()`.

`FileChanges` no longer implements `PartialEq`, `Eq`, `UnwindSafe` or
`RefUnwindSafe`, because an `Unavailable` event carries an error; match on
`change.event` where you compared two messages. `rig_tools::remote::Remote`
lost `UnwindSafe` and `RefUnwindSafe` too: wrap it in `AssertUnwindSafe` if
you pass it through `catch_unwind`.

#### Timings and other removals

- `timings::Timing`, `SystemTimings::systems`, `SystemTimings::clear` and
  `PluginTimings::totals` are removed. Sum `PluginTimings::last_frame()`
  yourself:

  ```rust
  // before
  let totals = timings.totals();
  // after
  let mut totals = BTreeMap::<_, Duration>::new();
  for run in timings.last_frame() {
      *totals.entry((run.plugin, run.schedule)).or_default() += run.took;
  }
  ```

- `PluginSystems::iter` is removed. Ask for one system with
  `PluginSystems::plugin(system, schedule)`.
- `rig_harness::PLUGIN_GUIDE` is removed. The guide is
  `crates/rig-harness/PLUGINS.md`.

#### rig-harness-protocol: launcher paths and the end record

These matter to code that starts an agent or reads a home itself.

`Home::plugins_toml`, `Home::defaults`, `ESSENTIAL_PLUGINS` and the top-level
`essential` are removed, because `plugins.toml` and `defaults.json` are no
longer read (see [upgrading a 0.46 home](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates)). The plugin list is read from
`settings.toml` through the new `config` module:

```rust
// before
let list = std::fs::read_to_string(home.plugins_toml())?;
let why = rig_harness_protocol::essential(plugin);
// after
let plugins = rig_harness_protocol::config::chosen(&home)?; // from Home::settings()
let why = rig_harness_protocol::config::essential(plugin);
```

`Home::trial_for` is removed with no replacement: a launcher keeps its trial
and rollback builds in its own run directory.

The agent tells its launcher how it ended with one record,
`end::AgentEnd` (`Quit`, `Reload`, `Switch(Option<SessionId>)`,
`Refused(String)`, `Signal`, `Failed(FailCode)`), and the public file paths
are gone: `RunDir::ready`, `RunDir::refusal`, `RunDir::screen` and
`SessionDir::switch`.

```rust
// before
let reload = status.code() == Some(i32::from(RELOAD_EXIT_CODE));
let switch = fs::read_to_string(home.session(&id).switch());
let refusal = fs::read_to_string(run.refusal());
let started = run.ready().exists();
// after
let end = AgentEnd::read(&run, status); // Reload, Switch(target), Refused(why), ...
let started = run.heard_ready();
run.clear_messages()?;                  // before the next start
```

An agent that is not built on rig-harness writes `end.write(&run)` before it
exits with `end.code()`, calls `run.say_ready()` once it started and
`run.say_screen()` once its front took the screen (`unsay_screen` gives it
back; the launcher asks `took_screen`).

`SessionDir::name` and `SessionDir::working_directory` return
`io::Result<Option<_>>`. `None` is a missing file only; any other failure is
an error naming the file:

```rust
// before
let name: Option<String> = dir.name();
// after
let name: Option<String> = dir.name()?;
// to keep "any error is nothing" where the name is only shown
let name = dir.name().ok().flatten();
```

`update` joined `RESERVED_COMMANDS`, so a plugin front command named `update`
is refused. Rename it.

New and breaking nothing: the `launch` module (`Launch`, `Carry`, `Agent`,
`Purpose`, `AbsPath`, `LaunchError`), which builds every process a launcher
starts; the `fs` module, whose functions name the step and the path in their
errors, with `fs::optional` (only a missing file is `None`) and `fs::replace`
(a whole-file replace that creates a new file private); `Stamp`,
`long_version`, `without_secrets` and `Home::keep_private`.

#### rig-tools: file tools and the workspace

The `move` tool (`Move`, `MoveArgs`) and the `checkpoint` module
(`Checkpointed`, `Snapshot`, `Taking`, `snapshot`, `forget_kept`) are removed
with no replacement. Drop the tool from your registrations.

`Read`, `Write`, `Edit`, `Search`, `List` and `Delete` hold one
`root: workspace::Root` instead of the `workspace`, `base` and `confine`
fields:

```rust
// before
let read = Read { workspace: Arc::new(hosts), base: Some(dir), confine: true, ..Read::default() };
// after
use rig_tools::workspace::Root;
let read = Read {
    root: Root { workspace: Arc::new(hosts), base: Some(dir), confine: true },
    ..Read::default()
};
```

`Root` is `Default` and `From<Arc<dyn Workspace>>`, and has the path helpers
that were free functions. `fs::resolve`, `fs::place` and `fs::read_text` are
removed:

```rust
// before
let path = rig_tools::fs::resolve(base, "src/lib.rs");
let text = rig_tools::fs::read_text("a.txt")?;
// after
let path = <dyn Workspace>::resolve(&Local, base, "src/lib.rs"); // or root.resolve("src/lib.rs")
let text = <dyn Workspace>::read_text(&Local, Path::new("a.txt"))?;
```

The `Workspace` trait changed in four places. An implementation updates each;
a caller updates the ones it uses.

| Method | 0.46 | 0.47 |
| --- | --- | --- |
| `read_files` | `Vec<Option<Vec<u8>>>` | `Vec<Observation<Vec<u8>>>` (`Found`, `Missing`, `Excluded`, `Failed`) |
| `files` | `io::Result<Vec<PathBuf>>` | `io::Result<Listed>` (`files`, `failed`) |
| `write_if` | `held: &ContentHash` | `held: Option<&ContentHash>`; `None` writes only while there is no file |
| `replace` | present | removed: `read`, then `write_if` |

```rust
// before
let files = workspace.files(root, None, listing)?;
workspace.write_if(path, bytes, &hash)?;
// after
let files = workspace.files(root, None, listing)?.files;
workspace.write_if(path, bytes, Some(&hash))?;
```

The trait also gained a provided method, `host_for(path)`, which defaults to
`host()`. A workspace that routes some paths to another machine, or wraps one
that does, implements it.

Smaller changes:

- `Listing::keeps` takes whether the path is a directory:
  `listing.keeps(below, is_dir)`. `Listing::for_glob` is removed with no
  public replacement.
- `clip(line, n)` is replaced by `excerpt(line, None, n).kept()`. Give
  `excerpt` the byte range the line was picked for and it keeps that part in
  view.
- `ReadArgs::shown`, which returned the text and a `LinesShown`, is removed.
  `ReadArgs::show(text)` returns the text.
- Struct literals: `workspace::Excerpt` gains `endings`, and `search::Found`
  gains `unread` and `skipped`. Write `..Found::default()`.

#### rig-tools: processes, attachments, the clipboard and watches

`OutputLine` holds bytes. `OutputLine::Out` and `Err` carry a `Vec<u8>`, and
`text()` returns a `Cow<str>` with bytes that are not UTF-8 replaced:

```rust
// before
OutputLine::Out(text) => buffer.push_str(&text),
// after
OutputLine::Out(_) => buffer.push_str(&line.text()), // or line.bytes()
```

Every process has one owner. `list_group` and `unlist_group` are removed:
start the process with `Process::start(command, Spawn::default())`, or
`Process::start_with_stdio(command)` when you take its pipes yourself, and it
is listed until its group is empty. `record_groups` takes the groups to adopt
(`record_groups(path, [])`), and `Remote::new` takes the launcher as a
closure, read as each command starts:

```rust
// before
let remote = Remote::new("box", transport, launcher)?;
// after
let remote = Remote::new("box", transport, move || launcher.clone())?;
```

`Spawn` gains the field `grace` (write `..Spawn::default()`), `Shell` gains
`filter: Option<LineFilter>`, and `process::Ended` gains the variant
`EndedJobs(ExitStatus)`, for a command that left a background job writing to
its output. An exhaustive `match` adds it:

```rust
// after
Ended::Exited(status) | Ended::EndedJobs(status) => { /* ... */ }
```

`attach::Attached::missing` is replaced by `warnings`, whole lines for the
user, and `Attached` gains `unattached`, the same news as notes for the
model. References are read by `attach::Reference::parse`, and
`Reference::format` writes one, quoting a name with spaces as
`@"my file.txt"`.

`clipboard::read_text` returns `io::Result<Option<String>>`. `Ok(None)` is a
clipboard with no text. A `NotFound` error now means no clipboard program is
installed, and a read gives up after 5 s:

```rust
// before
match clipboard::read_text(&launcher) {
    Ok(text) => use_text(text),
    Err(error) if error.kind() == ErrorKind::NotFound => {} // empty
    Err(error) => return Err(error),
}
// after
if let Some(text) = clipboard::read_text(&launcher)? {
    use_text(text);
}
```

`Watch` yields `WatchEvent` instead of `Vec<PathBuf>`, and `Watch::new` takes
an `UnboundedReceiver<WatchEvent>`. The events are described under "File
watches" above.

### 8. Plugin crates (rig-basics, rig-tui and the others)

This section is for code that depends on the plugin crates as libraries: a
plugin that reads rig-basics components, draws in rig-tui's transcript, or an
app that adds the plugins itself. The settings and the default plugin list are
in [upgrading a 0.46 home](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).

#### New crates

- `rig-fold` is the optional `rig_fold::FoldPlugin`, off by default. It folds
  a finished tool call in the terminal view to its header, first and last
  lines. Only the view's rows change.
- `rig-markdown` holds the markdown rendering (`markdown`), code helpers
  (`code`) and word wrapping (`wrap`) that rig-tui and print mode share.
  `rig_tui::wrap` and `rig_tui::RowWrap` are now re-exports of
  `rig_markdown::wrap`, so `rig_tui::wrap::wrap` and `wrap_all` keep their
  paths. `DrawnRow::continues` is an `Option<rig_markdown::wrap::RowWrap>`.
- `rig-cli` is the launcher as a library. Its one public item is
  `rig_cli::main() -> ExitCode`, which the `rig` package's binary calls.

`rig-acp`, `rig-claude-cloak`, `rig-inspect` and `rig-traffic` remove and
change no public item.

#### Forks, the agent picker and the current agent (rig-basics)

`rig_basics::sessions::agents` is gone. Its plugin is split in two, and both
halves are off in the default plugin list:

| 0.46 | 0.47 |
| --- | --- |
| `sessions::agents::AgentsPlugin` (`/fork`, rewind, `/agents`) | `sessions::forks::ForksPlugin` (`/fork`, rewind), which adds the picker |
| `/agents` | `agent_picker::AgentPickerPlugin`, also added by `SubagentsPlugin` |
| `sessions::agents::{AgentColumn, ColumnOf, AgentColumns}` | `agent_picker::{AgentColumn, ColumnOf, AgentColumns}` |
| `sessions::agents::ShownId` | `rig_harness::main_agent::CurrentAgent` |

```rust
// before
use rig_basics::sessions::agents::{AgentColumn, AgentsPlugin, ColumnOf, ShownId};
app.add_plugins(AgentsPlugin);
// after
use rig_basics::agent_picker::{AgentColumn, ColumnOf};
use rig_basics::sessions::forks::ForksPlugin;
use rig_harness::main_agent::CurrentAgent;
app.add_plugins(ForksPlugin);
```

`CurrentAgent(pub String)` is a saved resource of rig-harness: the id of the
top-level agent shown last, which a resume continues.

#### Subagents and pending requests (rig-basics)

- `subagents::SubagentPolicy` and its `[subagents] max_depth` are removed.
  Subagents have no depth limit. A plugin that wants one counts the
  `SpawnedBy` ancestors itself.
- `subagents::SubagentSystems` is removed. Delete
  `.after(SubagentSystems::Gate)`.
- `subagents::OpenRequests` is removed and `subagents::Request::asker` is now
  `Request::to`. The agent that waits holds `subagents::Awaiting(pub
  Vec<Request>)`; a plugin's own pending request is an entity with
  `rig_ecs::requests::RequestOf` (see [the kernel](#6-the-kernel-rig-ecs)).
- `login::PendingLogin::agent` is removed. `PendingLogin` has no public field;
  the request entity's `RequestOf` names the agent.
- `helpers::AT_ONCE` is removed. The limit is the `[helpers] at_once` setting
  (4 by default), beside `[helpers] timeout_secs` (600).
- New: `subagents::SubagentDefaults` (`[subagents] model` and `tools`).

```rust
// before
let asker = &request.asker;
// after
let asker = &request.to;
```

#### Compaction (rig-basics)

The compaction plugin was rewritten on rig-memory's new functions
([memory and compaction](#4-memory-and-compaction-rig-memory)). In `rig_basics::compaction`:

- `CompactionDefault` wraps `rig_memory::CompactionSettings` (`enabled`,
  `reserve_tokens`, `keep_recent_tokens`), not `CompactionPolicy`.
- `CompactReason` is rig-basics' own enum, `Asked { focus }`, `Threshold` or
  `Overflow`, with `focus()`. rig-memory no longer has one.
- The free function `plan` is removed. Plan a summary call with
  `Summarizing::plan(&conversation, keep_recent_tokens, reason)`, or
  `Summarizing::plan_at(&conversation, first_kept, reason)` to cut at your
  own message index. Both return `Option<Summarizing>`. Build its request
  with `Summarizing::request(&spec, reasoning)`.
- `Summarizing::keep_from` is replaced by `replaces`, the messages the
  summary replaces. `Summarizing` also gains `previous`, `messages`,
  `turn_prefix` and `history`, so a struct literal no longer builds: use
  `plan` or `plan_at`.
- `Uncompact`, `Recovering` and `Noting` are removed with no replacement. A
  compaction cannot be undone, and old tool outputs are no longer cleared.

```rust
// before
let planned = rig_basics::compaction::plan(&conversation, &policy, reason, &spec)?;
let Some((summarizing, request)) = planned else { return Ok(()) };
// after
let keep = settings.0.keep_recent_tokens; // settings: Res<CompactionDefault>
let Some(summarizing) = Summarizing::plan(&conversation, keep, reason) else {
    return Ok(());
};
let request = summarizing.request(&spec, reasoning)?;
```

#### Usage and spend (rig-basics)

`telemetry::usage::AgentUsage::record` is no longer public. Record spend by
triggering `Spent`, which also adds it to every agent above:

```rust
// before
usage.record(Some(model), Some("my-purpose"), &spent);
// after
commands.trigger(Spent {
    agent: Some(agent),
    turn: None,
    model: Some(model.to_owned()),
    purpose: Some("my-purpose".to_owned()),
    usage: spent,
});
```

`AgentUsage` gains the public field `below`, what the agents under this one
spent, so match it with `..`. `StoppedUsage` and `UnknownSpend` are new saved
components. `telemetry::effect_log::EffectLog::read_matching` is removed:
filter what `EffectLog::read()` returns.

#### Moved to rig-harness

| 0.46 (rig-basics) | 0.47 |
| --- | --- |
| `print::{Starting, KeepRunning, Work, TOOLS}` | `rig_harness::run::{Starting, KeepRunning, Work, TOOLS}` |
| `coding_tools::ShellOutput` | `rig_harness::per_call::ShellOutput` (also in the prelude) |
| `coding_tools::SeenFiles` (a resource) | `rig_harness::per_call::FilesSeen` (a component of each agent) |

```rust
// before
use rig_basics::coding_tools::{SeenFiles, ShellOutput};
use rig_basics::print::KeepRunning;
fn seen(files: Res<SeenFiles>) { /* ... */ }
// after
use rig_harness::per_call::FilesSeen;
use rig_harness::prelude::{KeepRunning, ShellOutput};
fn seen(files: Query<&FilesSeen>) { /* one per agent */ }
```

What the file tools saw is no longer shared between agents.

#### Other rig-basics changes

- `checkpoint` is removed whole: `CheckpointPlugin`, `Checkpoints`,
  `CheckpointAlso`, `CallCheckpoint` and `TurnCheckpoint`, with no
  replacement.
- `activity::{MessageFeed, FedMessage, FEED_LEN, FEED_CHARS}` are removed
  with no replacement.
- `reload::ReloadStatus::ask` takes what to rebuild, and
  `ReloadStatus::Queued` carries it:

  ```rust
  // before
  status.ask(agent)?;
  if let ReloadStatus::Queued { agent } = *status { /* ... */ }
  // after
  status.ask(agent, Rebuild::Build)?;
  if let ReloadStatus::Queued { agent, .. } = *status { /* ... */ }
  ```

  `/update` is `reload::update::UpdatePlugin`; `ReloadPlugin` adds only
  `/reload`.
- `agents_md::instruction_files` and `skills::roots` take the
  `rig_harness::services::Project` first:
  `instruction_files(&project, dir, home)`.
- `coding_tools::remote::RemoteSettings::remotes` takes
  `&rig_harness::services::Processes` instead of a `Launcher`.
- `AttachPlugin` reads a file on another machine (`@HOST:PATH`) off the main
  thread, under a hold on the message, so that attachment arrives after
  `AttachSystems`. A system ordered `.after(AttachSystems)` sees local
  attachments only; read the rest once the message's hold is released.
  While a message waits, later user-kind messages to the same agent wait
  behind it.
- The `status` item's text changed while the agent works: it starts with a
  six-bar wave and a space when `WorkingAnimationPlugin` is on, and gains
  the time after 5 seconds (`thinking… 1m12s`). A plugin that matched the
  text exactly reads `Activity::status` instead.
- `print::VERSION` is 3 (was 1); see [headless drivers](#11-headless-drivers-rig--p-rig---json-and-rig-acp).
  `print::PIPED_LIMIT` and `REQUEST_LIMIT` are new.
- New plugins, each off by default: `redact::RedactPlugin` (redaction, which
  0.46 did in the kernel) and `budget::BudgetPlugin`. New default plugin:
  `activity::animation::WorkingAnimationPlugin`; an app that adds
  `ActivityPlugin` itself and wants the moving bars adds it too.
  `models::FirstPickPlugin`, which `ModelsPlugin` adds, picks the first
  model when nothing is saved or named.

#### The terminal view's parts are plugins (rig-tui)

`rig_tui::TuiPlugin` alone no longer draws the status line, the key hints, or
the terminal title with its OSC 133 marks. Each is its own plugin:
`status::StatusLinePlugin`, `hints::KeyHintsPlugin` and
`title::TerminalTitlePlugin`. Code that adds the view itself adds the new
`rig_tui::TuiPlugins`, which is `TuiPlugin` plus those three, each unless the
app already has it:

```rust
// before
app.add_plugins(rig_tui::TuiPlugin);
// after
app.add_plugins(rig_tui::TuiPlugins);
```

This compiles unchanged and loses the three parts, so check it by eye. The
OSC 133 nonce setting moved with the title: `[tui] mark_nonce_env` is
`[title] mark_nonce_env`, read by `TerminalTitlePlugin`. A plugin that draws a
status-line part declares it in `build` with the new
`AppStatusPartsExt::add_status_part(&item, "Label")`, so `/plugins` and
`[status] hide` can name it.

#### Render inputs replace the borrowed views (rig-tui)

A look takes the batch's owned input. The borrowed `ToolCallView`,
`EntryView`, `ContentView`, `AsideView` and `ContentBlock`, and the `view()`
and `block()` accessors that made them, are removed. Their methods are on
`ToolRender`, `EntryRender` and `ContentRender`:

```rust
// before
fn look(view: &ToolCallView<'_>) -> Vec<LookLine> {
    let mut lines = vec![view.header("grep", view.argument("pattern").unwrap_or("")).into()];
    lines.extend(view.result_lines());
    lines
}
look(&call.view());
let view = ToolCallView::new(&call, Some(&result), state);
let kind = content.block().kind();

// after
fn look(call: &ToolRender) -> Vec<LookLine> {
    let mut lines = vec![call.header("grep", call.argument("pattern").unwrap_or("")).into()];
    lines.extend(call.result_lines());
    lines
}
look(call);
let render = ToolRender::new(call, Some(result), state);
let kind = content.content.kind();
```

Fields are owned, so borrow where the view held a reference:
`call.facts.as_ref()`, `call.result.as_ref()`, `call.output.as_deref()`,
`&call.children`, `&entry.text`, `&aside.aside`. A content block is the
`content` field of `ContentRender`, a `RenderContent`, and
`RenderContent::Document` holds a `DocumentContent` (see [core
messages](#1-core-messages-conversations-tools-and-streaming-rig-core)).

`ToolRender` has a new public field, `item: Option<CallItem>`, so a struct
literal adds `item: None`; `ToolRender::new` sets it. `ChildCall::call` is an
`Option<ToolFunction>`, not an `Option<ToolCall>`.

`ToolRender::result_lines()` and `result_text()` now include a result's JSON
blocks, each as a line of JSON text, and end with a line counting its images.
In 0.46 they held the text blocks only. A look that draws JSON itself drops
the JSON content first, or it shows twice:

```rust
// after: a look with its own JSON drawing
let mut text_only = call.clone();
if let Some(result) = &mut text_only.result {
    result
        .content
        .retain(|content| !matches!(content, ToolResultContent::Json { .. }));
}
lines.extend(my_json_lines(call));
lines.extend(text_only.result_lines());
```

#### Calls are named by place; message decorators are gone (rig-tui)

`TranscriptItem::Call` and `TranscriptItem::Decoration` take a `CallItem`
instead of a `CallId`: `CallItem::Reply(CallRef)` for a call of a reply in
the conversation, `CallItem::Streaming(index)` for one still streaming. Two
calls that share a provider id are now two items.

```rust
// before
TranscriptItem::Call(call_id)
// after
TranscriptItem::Call(CallItem::Reply(call_ref))
```

The message decoration path is removed with no replacement:
`DecoratedMessage`, `MessageRender`, `MessageView`, `Role`,
`RenderInput::Message`, `RenderKey::Message`, `RenderBatch::messages_mut` and
`TranscriptItem::MessageDecoration`. A plugin can no longer add lines under a
message; draw an aside (`AsideRender`) or a note (`EntryRender`) instead.

#### Smaller rig-tui removals

| 0.46 | 0.47 |
| --- | --- |
| `DrawnTranscript::item(&item)` | `DrawnTranscript::rows_of(&item)`, a range into `rows()` |
| `KeyPress::shift(code)` | `KeyPress::new(code, KeyModifiers::SHIFT)` |
| `PromptHistory::KEPT` | none: history keeps every distinct prompt |
| `Notices`, `Notices::all`, `ShownNotice` | none: read the `Notice` messages yourself |
| `TranscriptSlot` | none: no longer public |
| `PickerInput::filter`, `PickerInput::ticked` | none; a ticked choice is `rig_harness::pick::Ticked` on the picker |

```rust
// before
let call = drawn.item(&item).unwrap_or_default();
let lit = call.rows().iter().position(|row| row.lit_total() > 0);
// after
let rows = drawn.rows_of(&item).unwrap_or_default();
let lit = drawn
    .rows()
    .get(rows)
    .unwrap_or_default()
    .iter()
    .position(|row| row.lit_total() > 0);
```

Pickers themselves are rig-harness types. `PickItem` gains `parent`, and
`Picker::checked` is removed in favor of `PickItem::checked`; build items with
`PickItem::new(label, command)` and `.detail(text)`. See [plugin host and
tools](#7-plugin-host-and-tools-rig-harness-rig-harness-protocol-rig-tools).

#### rig-mcp-client and rig-steel

- `McpState::NotApproved` and `McpState::Denied` are one variant,
  `McpState::Held`. Why a server is held is its `McpHold(pub String)`
  component. Asking before a project's server starts is the new opt-in
  `rig_mcp_client::approval::McpApprovalPlugin`; `McpPlugin` alone starts
  them.

  ```rust
  // before
  matches!(state, McpState::NotApproved | McpState::Denied)
  // after
  matches!(state, McpState::Held)
  ```

- `McpSettings::connect_timeout_secs` and `first_retry_secs` are
  `NonZeroU64`. A `0` in `[mcp]` refuses the start.
- `rig_steel::AgentSpec` is removed. `Harness::spawn_agent` takes
  `rig_ecs::spawn::AgentSpec`, which has the same `name`, `model`, `tools` and
  `system_prompt` plus `id` and `effort`; build it with
  `..Default::default()`.

### 9. Upgrading a 0.46 home: settings, plugins and scaffolded plugin crates

This section is for users of the `rig` coding agent. A `RIG_HOME` (default
`~/.rig`) that 0.46 wrote does not start on 0.47 as it is. rig rewrites none
of your files: it refuses, or warns, and you make the edits. Do the steps in
order. Each `rig` start names one unknown key of a settings table, so fixing
the whole file from the tables below saves several starts.

#### The steps

1. Install 0.47. 0.46 has no `rig update`, so reinstall by hand. From 0.47
   on, `rig update` does this.

   ```console
   cargo install rig --force --locked
   ```

2. Fix `RIG_HOME/settings.toml` (details under
   [Settings tables](#settings-tables-that-changed)):

   - `[compaction]`: rename `keep_recent` to `keep_recent_tokens` and
     `limits.reserve` to `reserve_tokens`. Delete every other old key.
   - `[subagents]`: delete `max_depth`.
   - `[tui] mark_nonce_env`: move it to `[title] mark_nonce_env`.
   - `[checkpoints]`: delete the table.

3. Run `rig plugin list`. While `RIG_HOME/plugins.toml` or
   `RIG_HOME/defaults.json` exists and `settings.toml` has no `[plugins]` or
   `[model]` table, every `rig plugin` command and every start prints a
   warning with the commands and lines that rebuild your old setup. Run the
   commands and add the lines (details under
   [The plugin list and the saved model](#the-plugin-list-and-the-saved-model)).

4. Turn on what 0.46 had on by default and 0.47 does not, if you want it
   (details under [Defaults that changed](#defaults-that-changed)):

   ```console
   rig plugin enable rig_basics::sessions::forks::ForksPlugin
   rig plugin enable rig_basics::redact::RedactPlugin
   rig plugin enable rig_mcp_client::approval::McpApprovalPlugin
   ```

   The warning of step 3 already prints the first line when your
   `plugins.toml` had `AgentsPlugin` on, and the third when it had
   `McpPlugin` on. It never prints the second: 0.46 redacted without a
   plugin.

5. In each plugin crate of your own, such as one `rig plugin new` made, raise
   the rig crates to `0.47.0` and write the Bevy crates as `"0.20"` (details
   under [Plugin crates](#plugin-crates-made-by-046s-rig-plugin-new)). Then
   port its code with sections [6](#6-the-kernel-rig-ecs), [7](#7-plugin-host-and-tools-rig-harness-rig-harness-protocol-rig-tools) and [8](#8-plugin-crates-rig-basics-rig-tui-and-the-others).

6. Move anything a project's `.rig/settings.toml` set into
   `RIG_HOME/settings.toml`. rig no longer reads the project's file.

7. Start `rig`. `rig plugin check --build` builds the agent with your plugins
   first, without starting it, and fails on a settings table the start would
   refuse.

Keep `plugins.toml` and `defaults.json` if you may go back to 0.46: 0.47
never changes them, and 0.46 reads only them.

#### The plugin list and the saved model

0.46 kept the plugin list in `RIG_HOME/plugins.toml` and the model a new
session starts with in `RIG_HOME/defaults.json`. 0.47 reads neither and
converts neither. Both live in `RIG_HOME/settings.toml`: `[plugins]` says
which of rig's plugins run, `[[plugins.extra]]` adds crates that are not
rig's, and `[model]` holds the model and reasoning. A new default plugin now
reaches every home, because the file names only your changes.

```toml
# before: RIG_HOME/plugins.toml (an entry behind `#` was off)
[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::sessions::agents::AgentsPlugin"

[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::subagents::SubagentsPlugin"

# [[plugin]]
# crate = "rig-basics"
# plugin = "rig_basics::welcome::WelcomePlugin"

[[plugin]]
crate = "hello"
path = "plugins/hello"
plugin = "hello::HelloPlugin"
```

```json
{"model": "openai/gpt-5.5", "effort": {"effort": "high"}}
```

```toml
# after: RIG_HOME/settings.toml
[model]
default = "openai/gpt-5.5"
effort = "high"

[plugins]
enable = [
    "rig_basics::sessions::forks::ForksPlugin",   # was sessions::agents::AgentsPlugin
    "rig_basics::subagents::SubagentsPlugin",
]
disable = ["rig_basics::welcome::WelcomePlugin"]

[[plugins.extra]]
crate = "hello"
plugin = "hello::HelloPlugin"
path = "plugins/hello"        # relative to RIG_HOME
```

`enable` and `disable` hold only the plugins whose choice differs from rig's
default. `[plugins] off = ["rig-tui"]` turns off a whole branch, a crate or
one of its areas such as `rig-basics/sessions`.

You do not write these by hand. The warning reads your old files and prints
the steps, and the commands edit `settings.toml`, keeping its comments:

```text
warning: rig no longer reads ~/.rig/plugins.toml. Move its plugin list to [plugins] in ~/.rig/settings.toml (see MIGRATING.md):
  rig plugin add hello::HelloPlugin --path /home/you/.rig/plugins/hello
  rig plugin enable rig_basics::sessions::forks::ForksPlugin
  rig plugin enable rig_basics::subagents::SubagentsPlugin
  rig plugin disable rig_basics::welcome::WelcomePlugin
The file was not changed.
warning: rig no longer reads ~/.rig/defaults.json. Move its model to [model] in ~/.rig/settings.toml (see MIGRATING.md):
  Add to ~/.rig/settings.toml:
  [model]
  default = "openai/gpt-5.5"
  effort = "high"
The file was not changed.
```

rig prints the full paths. The lines it prints depend on your files:

- A `rig plugin add` for each crate that is not rig's, then a
  `rig plugin enable` or `disable` for each of rig's plugins whose state in
  your list differs from today's default.
- `rig_basics::sessions::agents::AgentsPlugin` is printed under its new
  name, `rig_basics::sessions::forks::ForksPlugin`.
- With `rig_mcp_client::McpPlugin` on, it also prints
  `rig plugin enable rig_mcp_client::approval::McpApprovalPlugin`, because
  0.46's MCP plugin asked before a project's server ran.
- `<type> is no longer one of rig's plugins.` for a plugin that is gone and
  was on, such as `rig_basics::checkpoint::CheckpointPlugin`.
- `Add an empty [plugins] table to <settings.toml>: the list was rig's
  defaults.` when nothing differs, and `Add an empty [model] table` when
  `defaults.json` names no model.

Each warning stops once `settings.toml` has the table it names, `[plugins]`
or `[model]`, even an empty one. A terminal start whose `settings.toml` is
refused shows only the refusal, so fix step 2 first.

The commands, as 0.47 takes them:

```console
rig plugin list
rig plugin enable <type or branch>
rig plugin disable <type or branch>
rig plugin add <type> (--path <dir> | --git <url> [--branch <b> | --rev <r>] | --version <req>) [--crate <name>]
rig plugin remove <type>
rig plugin new <name>
rig plugin check [--build]
```

`rig plugin add` and `remove` existed in 0.46 and wrote `plugins.toml`. They
now write `[[plugins.extra]]`. `rig plugin list` prints a tree of crates,
areas and plugins, rig's own first and then yours. A script that read the
0.46 list line by line should stop parsing it.

A plugin name in `[plugins] enable` or `disable` that nothing provides
refuses the start. 0.46 started the previous build instead. The refusal
names the fix, and for the renamed plugin it reads:

```text
unknown plugin `rig_basics::sessions::agents::AgentsPlugin` in `enable`, now `rig_basics::sessions::forks::ForksPlugin`: `rig plugin disable rig_basics::sessions::agents::AgentsPlugin` takes it out, then `rig plugin enable rig_basics::sessions::forks::ForksPlugin` sets the new one
```

#### Settings tables that changed

A table of `RIG_HOME/settings.toml` that an enabled plugin cannot read
refuses the start with exit 2, naming the file, the table and the key. 0.46
started with a misspelled key. A table that no enabled plugin reads only
gets a notice at each start:

```text
<settings.toml>: no enabled plugin reads [checkpoints].
```

| 0.46 | 0.47 | What rig does with the old form |
| --- | --- | --- |
| `[compaction] keep_recent` | `[compaction] keep_recent_tokens` | refuses the start |
| `[compaction] limits = { reserve = N }` | `[compaction] reserve_tokens = N` | refuses the start |
| `[compaction] keep_asked`, `clearing`, `near_window`, `prompts`, `tracked`, and `summary_tokens`, `snippet_chars`, `unknown_window_chars` in `limits` | removed | refuses the start |
| `[subagents] max_depth` | removed: subagents have no depth limit | refuses the start when Subagents is on |
| `[tui] mark_nonce_env` | `[title] mark_nonce_env` | notice; the prompt marks carry no nonce |
| `[checkpoints]` | removed with the checkpoint plugin | notice |
| `[mcp] connect_timeout_secs = 0`, `first_retry_secs = 0` | a value of 1 or more | refuses the start when MCP is on |
| `<project>/.rig/settings.toml` | not read; use `RIG_HOME/settings.toml` | nothing, silently |

The old `[compaction]` table needs care, because rig's advice is wrong for
two of its keys. The refusal reads:

```text
rig: ~/.rig/settings.toml [compaction]: unknown field `keep_recent`, expected one of `enabled`, `reserve_tokens`, `keep_recent_tokens`; delete `keep_recent` from [compaction]; fix the file and start again.
```

It says "delete" for every unknown key. For `keep_recent` and
`limits.reserve` the right step is to rename. Both still count tokens, with
the same defaults (20000 and 16384), so the value carries over as it is. If
you delete them instead, rig starts and uses the default without a word. The
other old keys have no replacement and are deleted. rig names one key of a table
per start, so a table with three old keys takes three starts if you follow
the messages one at a time.

```toml
# before (0.46)
[compaction]
keep_recent = 30000
keep_asked = 0
near_window = "clear"
limits = { reserve = 8192, summary_tokens = 32000 }

# after (0.47)
[compaction]
keep_recent_tokens = 30000   # was keep_recent
reserve_tokens = 8192        # was limits.reserve
# enabled = false            # new: turns automatic compaction off; /compact still works
```

`keep_recent_tokens` is about how many tokens of the newest messages a
compaction leaves as they are. `/compact` now keeps the same amount, where
`keep_asked` set its own. 0.46 also held `keep_recent` to a quarter of the model's input
limit, and 0.47 does not. `reserve_tokens` is how much of the model's input
limit stays free for the reply. Section [10](#10-the-coding-agents-commands-and-behaviour) says how compaction
itself changed.

The tables 0.47 adds (`[model]`, `[plugins]`, `[retry]`, `[tools]`,
`[helpers]`, `[steel]`, `[login]`, `[title]`, `[status]`, `[activity]`,
`[fold]`, and `[subagents] model` and `tools`) need no migration.
`docs/agent.md` describes each.

#### Defaults that changed

These worked in a default 0.46 home and are off, gone or different in 0.47.

- **Forks, rewind and `/agents` are off.** `/fork` and rewind (Alt+Up to an
  earlier message of yours, then Enter) are
  `rig_basics::sessions::forks::ForksPlugin`, renamed from
  `rig_basics::sessions::agents::AgentsPlugin` and off by default. `/agents`
  is its own plugin, `rig_basics::agent_picker::AgentPickerPlugin`, also off.
  Forks and Subagents each add it, so enabling Forks brings all three back:

  ```console
  rig plugin enable rig_basics::sessions::forks::ForksPlugin
  ```

- **A session saved while a fork was shown reopens on its first agent.**
  0.46 saved which agent was shown under a key 0.47 does not read, so
  `rig --resume` and `/resume` open such a session on its first top-level
  agent. Nothing is lost. With Forks on, `/agents` picks the fork, and 0.47
  then remembers it.
- **Redaction is off.** 0.46 always redacted secrets. 0.47 redacts nothing
  unless `rig_basics::redact::RedactPlugin` is on. With it, each secret
  becomes a placeholder such as `‹SECRET_1›` (it was `[REDACTED]`), and a
  tool call that names a placeholder runs with the real value.

  ```console
  rig plugin enable rig_basics::redact::RedactPlugin
  ```

- **A project's MCP servers start without asking.** With
  `rig_mcp_client::McpPlugin` on, the servers of `<project>/.rig/mcp.json`
  start with no question, and one replaces a server of yours with the same
  name. 0.46 held each until `/mcp approve`. To be asked first again:

  ```console
  rig plugin enable rig_mcp_client::approval::McpApprovalPlugin
  ```

  With it on, `/mcp approve <name>` and `/mcp deny <name>` work as before,
  and the answers 0.46 saved in
  `RIG_HOME/projects/<project>/mcp/approvals.json` still count. Both
  commands belong to the approval plugin. `McpPlugin` itself is off by
  default in both versions.
- **An `mcp.json` may not name a `RIG_SECRET_*` variable.** A server whose
  `mcp.json` entry expands one, in any letter case, is refused with
  `` `<field>`: ${RIG_SECRET_X} is a secret of rig's, which no server is
  given ``. 0.46 expanded it and sent the value. Pass the value through a
  variable of its own:

  ```json
  { "env": { "API_TOKEN": "${RIG_SECRET_API_TOKEN}" } }
  { "env": { "API_TOKEN": "${MY_SERVER_API_TOKEN}" } }
  ```

  The first line is the 0.46 form and the second the 0.47 form.
- **Checkpoints and the `move` tool are gone**, with no replacement:
  `rig_basics::checkpoint::CheckpointPlugin`, its `[checkpoints]` table and
  the `move` tool.
- **Subagents have no depth limit.** `[subagents] max_depth` is removed.
- **The model a new session starts with changes only when you pick one.**
  Only `/model` or `/effort` sets `[model]`, and only once that model has
  answered. `-m`, a helper's model, an editor's choice and a resume no
  longer change it.

#### Plugin crates made by 0.46's `rig plugin new`

A crate that 0.46 scaffolded names rig-harness at `0.46.0` and pins Bevy
exactly. The 0.47 agent is built on Bevy 0.20.1 and on rig's 0.47.0 crates,
so both must change:

```toml
# before: RIG_HOME/plugins/<name>/Cargo.toml, as 0.46 wrote it
rig-harness = "0.46.0"
bevy_ecs = { version = "=0.20.0", default-features = false }
bevy_reflect = { version = "=0.20.0", default-features = false }

# after
rig-harness = "0.47.0"
bevy_ecs = { version = "0.20", default-features = false }
bevy_reflect = { version = "0.20", default-features = false }
```

Raise every other rig crate the plugin names, such as `rig-basics` or
`rig-tui`, to `0.47.0` too. Any other Bevy crate takes `"0.20"`.

Change both at once. The build error names only the Bevy lines (shortened
here; cargo's own conflict lines come between and after):

```text
error: a plugin must write its Bevy requirements as "0.20", such as `bevy_ecs = { version = "0.20", default-features = false }`, or be taken out with `rig plugin remove <type>`. These ask for a Bevy that is not this agent's 0.20.1:
  <plugin>/Cargo.toml:22: [dependencies] bevy_ecs = { version = "=0.20.0", default-features = false }
  <plugin>/Cargo.toml:23: [dependencies] bevy_reflect = { version = "=0.20.0", default-features = false }
```

If you fix only those, the next build fails with the same advice and one
different line, `bevy =0.20.0, required by rig-harness v0.46.0`: the old
rig-harness requirement pulls 0.46.0 from crates.io, and that pins the old
Bevy. `rig-harness = "0.47.0"` is the fix. The manifest then resolves; the
plugin's code still needs the API changes of sections [6](#6-the-kernel-rig-ecs), [7](#7-plugin-host-and-tools-rig-harness-rig-harness-protocol-rig-tools) and
[8](#8-plugin-crates-rig-basics-rig-tui-and-the-others). `rig plugin new` in 0.47 writes `"0.20"` and the launcher's own
version.

A plugin that registered a per-project table with
`app.add_project_setting::<T>("section")` no longer compiles. The compiler
points at `app.add_setting::<T>("section")`, which builds, but the table is
then read from `RIG_HOME/settings.toml` only. See section [7](#7-plugin-host-and-tools-rig-harness-rig-harness-protocol-rig-tools).

#### Sessions, files and going back

- Sessions saved by 0.46 resume and continue. In the release test, plain,
  compacted, forked and subagent sessions from a home built with 0.46 all
  resumed through `rig --resume` and the `/resume` picker, before and after
  the settings were moved.
- `cargo install rig` is unchanged. The launcher's code moved to the new
  library crate `rig-cli`, which the `rig` package's default `cli` feature
  builds into the `rig` binary.
- The project is now the working directory or the nearest directory above it
  that holds `.git` or `.rig`, below your home directory. rig started
  anywhere inside it reads the same `.rig/mcp.json`, `AGENTS.md` files,
  skills and per-project data. Per-project data such as the prompt history is
  keyed by that root, so a subfolder you used to start rig in begins with an
  empty history. A folder below the repository root that holds its own `.rig`
  is its own project and no longer reads the repository root's `AGENTS.md`.
  Started in your home directory, rig reads no project `mcp.json`.
- On unix, `sessions/`, `projects/`, `data/` and `run/` under `RIG_HOME` and
  each new session directory are made private to you (mode 0700) when a
  session starts. A tool of yours that read them as another user loses access.
- A new home gets a `writing-rig-plugins` skill in `RIG_HOME/skills`. rig
  writes it once, with a home's first agent project, so an existing home is
  not given one.
- With `rig_claude_cloak::ClaudeCloakPlugin` on, do not run a 0.46 rig and a
  0.47 rig on the same home at the same time, a rollback trial included.
  The credentials file is the same, but 0.46 locks `lock` beside it and 0.47
  locks `credentials.lock`, so the two no longer keep each other out while a
  token is refreshed.
- Going back to 0.46 works. In the release test it built and resumed every
  session, including one compacted by 0.47, after each plugin crate's
  `rig-harness` line was put back to `0.46.0`. 0.46 reads the untouched
  `plugins.toml` and `defaults.json`, so plugin and model changes made on
  0.47 do not apply there, and it skips the values only 0.47 saves with
  `Skipped saved resource` lines.

### 10. The coding agent's commands and behaviour

Nothing here is a Rust API. It is what someone who types into `rig` sees
change. The settings that go with it, and the steps for a 0.46 home, are in
[upgrading a 0.46 home](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates).

#### Forks, rewind and `/agents` are off by default

0.46's default plugin list had `AgentsPlugin` on, which gave `/fork`, rewind
(Alt+Up to an earlier message of yours, then Enter) and `/agents`. The plugin
is now `rig_basics::sessions::forks::ForksPlugin`, and it is off by default.
`/agents` is its own plugin, `rig_basics::agent_picker::AgentPickerPlugin`,
off by default and added by forks and by subagents.

```console
# before (0.46): /fork, Alt+Up then Enter, and /agents worked on a new home
# after
rig plugin enable rig_basics::sessions::forks::ForksPlugin
```

`/plugins` turns it on too. A session saved while a fork was shown reopens on
its first agent while forks are off; with forks on, `/agents` picks the fork.

#### Compaction: three settings, no undo, and `ctx ?`

Compaction now works as in the pi coding agent. It keeps about
`keep_recent_tokens` (20000) of the newest messages as they are, never cut
between a tool call and its result, and has the model summarize what lies
before them in a separate call. A later compaction updates the previous
summary. It runs before a request is sent, when that request would leave
less than `reserve_tokens` (16384) of the model's input limit free, and once
more after a provider refuses a request as too long or the window cuts a
reply short. Old tool outputs are no longer cleared first.

`[compaction]` takes `enabled`, `reserve_tokens` and `keep_recent_tokens`
and nothing else. A 0.46 table refuses the start; [section 9](#9-upgrading-a-046-home-settings-plugins-and-scaffolded-plugin-crates) has the
renames.

- `/compact` keeps the same amount as an automatic compaction, and
  `/compact <instructions>` says what the summary should focus on.
- With nothing to summarize, `/compact` says ``Nothing to compact yet: the session
  fits in what `[compaction] keep_recent_tokens` keeps`` as a plain
  notice, or `Compaction failed: Already compacted` right after a compaction.
- After a compaction the status line shows the context as `ctx ?` until the
  next reply reports it.
- A summary that fails, is cut short, calls a tool or is blank replaces
  nothing, and a notice says why. A failed summary call is retried as
  `[retry]` says.
- Compaction has no undo. The `Uncompact` event and rig-memory's
  `undo_summary` are removed, so a plugin command built on them, such as the
  plugin corpus's `/uncompact`, is gone. The default 0.46 agent had no such
  command. The summarized messages stay in the transcript under the line
  that shows the summary. To go back, rewind or fork (with `ForksPlugin` on).

#### `/plugins` and `rig plugin`

In `/plugins`, Enter now ticks or unticks the highlighted row, where it
applied the change at once. Changed rows show `+ on` or `- off`. Esc with
changes asks whether to reload now, apply at the next start, or keep
editing. The list is a tree of crates, areas and plugins, and Enter on a
branch switches the whole branch.

`rig plugin list` prints the same tree under the headers `rig's plugins:` and
`your plugins (extras):`. A script that read one plugin per line should not
parse it. `rig plugin enable` and `rig plugin disable` take a plugin's type
path or a branch name. `rig plugin add` and `rig plugin remove` now edit
`[[plugins.extra]]` tables in `RIG_HOME/settings.toml`, and `add` says where
the crate comes from:

```console
# before (0.46): wrote a [[plugin]] table into RIG_HOME/plugins.toml
# after
rig plugin add my_plugin::MyPlugin --path ~/code/my-plugin
rig plugin add my_plugin::MyPlugin --git https://example.com/my-plugin --rev 1a2b3c
rig plugin add my_plugin::MyPlugin --version 0.3 --crate my-plugin
```

`update` is now a launcher command (`rig update`), so a plugin's front
command may not take that name.

#### `/model`, the default model and local servers

`/model`, its completion and an editor's model list offer only the models
you can use now: a provider whose key is exported or saved with `/login`, a
sign-in that is logged in, or a local Ollama or llama.cpp server that an
environment variable names. Two exceptions: the current model is listed even
when it cannot be used, with the reason, and a ChatGPT plan's models show
as `checking your plan` until the plan is known. `ollama/*` models read `OLLAMA_API_BASE_URL` as
the daemon's address and append `/v1` themselves:

```console
# before
export OLLAMA_API_BASE_URL=http://localhost:11434/v1
# after
export OLLAMA_API_BASE_URL=http://localhost:11434
export LLAMACPP_API_BASE_URL=http://localhost:8080/v1
```

Only a model or reasoning setting picked with `/model` or `/effort` becomes
the default for new sessions, and only once that model has answered. `-m`, a
helper's model, an editor's choice, `/login`'s switch, a resume and `/fork`
no longer change it. The default lives in `[model]` of
`RIG_HOME/settings.toml`. A `/model vendor/model` that does not connect is
refused and the current model stays. A fresh ChatGPT sign-in starts on
`gpt-6.2-sol`; pick `/model chatgpt/gpt-6.1-sol` to stay on the old one.

#### `--tools` limits one run

```console
rig --tools read,shell -r <id>   # this run has only read and shell
rig -r <id>                      # 0.46: still only read and shell; 0.47: every tool again
```

#### `@` references with spaces

An `@` reference attaches exactly the file that was picked. A name with
spaces is quoted. An unquoted reference always ends before its trailing
punctuation: `@notes.` attaches `notes`, and is no longer also tried as a
file named `notes.`. Quote a name that ends in punctuation:

```console
# before
look at @my file.txt
look at @notes.
# after
look at @"my file.txt"
look at @"notes."
```

`@HOST:PATH` attaches a file from a remote host. A path that cannot be read
is reported as `could not read <path>: <error>`, and the model's message
says which `@` paths were not attached.

#### The mouse

rig now asks the terminal for mouse reports. The wheel scrolls the
transcript, where it browsed the prompt history, and moves the choice in a
picker. A drag selects and copies text in rig itself. For the terminal's own
selection hold Shift while dragging, and middle-click paste needs Shift too.

#### Removed: the `move` tool, checkpoints and the subagent depth limit

- rig-tools' `move` tool and `rig_basics::checkpoint::CheckpointPlugin`
  (listed disabled in 0.46) are removed with no replacement. A 0.46
  `plugins.toml` that lists `CheckpointPlugin` needs no edit: the upgrade
  warning says the plugin is gone and prints no command for it. Do not copy
  the name into `[plugins]` of `settings.toml`: an unknown plugin there
  refuses the start, and `rig plugin disable <name>` removes the name. A leftover `[checkpoints]` table only gets a notice that no
  enabled plugin reads it.
- Subagents have no depth limit, and `[subagents] max_depth` is no longer a
  key. `[subagents]` now takes `model` and `tools`, which a `task` call that
  names neither gives its subagent:

```toml
# before
[subagents]
max_depth = 2
# after: delete max_depth; optionally
[subagents]
model = "openai/gpt-5.5"
tools = ["read", "search"]
```

#### New commands and keys

None of these needs a migration step. They are listed because they replace
things people did by hand in 0.46.

| New | What it does |
| --- | --- |
| `rig update`, `/update` | reinstall the launcher where it is installed, then rebuild the agent |
| `rig --safe` | start a build of rig's default plugins alone, when a plugin keeps rig from building or starting |
| `/budget 2.00`, `/budget 500000 tokens` | a spending limit, with `rig_basics::budget::BudgetPlugin` enabled |
| Ctrl+O | show a folded tool call whole, with `rig_fold::FoldPlugin` enabled |
| `[plugins] off = ["rig-tui/hints"]` | turn off a crate or an area of the plugin tree |

### 11. Headless drivers: `rig -p`, `rig --json` and `rig acp`

`rig help-agents` prints the whole protocol as this version speaks it, with
a Python driver. This section is what changed since 0.46.

#### `rig --json` and `rig -p --json`: protocol version 3

0.46 spoke version 1. 0.47 speaks version 3, and there is no compatibility
mode: a request line with `"v":1`, `"v":2` or no `v` is refused, and the
request is not carried out. The four requests (`send`, `stop`, `answer`,
`state`) and their fields are otherwise unchanged. A driver written for
0.46 changes these things:

1. **Send `"v":3` on every line.**

   ```json
   {"v":1,"type":"send","text":"Fix the tests."}
   {"v":3,"type":"send","text":"Fix the tests."}
   ```

   The first line is the 0.46 form and the second the 0.47 form. A version 1
   line gets this reply, and rig goes on reading:

   ```json
   {"v":3,"type":"reply","ms":1795199253486,"id":"old-driver","ok":false,"error":"this line is version 1 and this rig speaks version 3: update the driver, whose every line carries \"v\":3"}
   ```

   A driver that does not read `reply` lines sees nothing happen. Check
   `ok` on the reply to your first request. Every line rig writes carries
   `"v":3` too, so a driver that refuses an unknown version must accept 3.

2. **Pair tool lines by `agent` and `id`.** On `tool_call` and
   `tool_result`, `id` is the call's place in its reply, such as `"m4/1"`
   (the reply's message and the call's place in it). It was the provider's
   call id. Ids are an agent's own, so two subagents can both have `"m1/0"`.
   A plugin's own call has `"id":null` and pairs with nothing. A call run
   again, such as a retry, writes another `tool_call` with the same id
   before its one `tool_result`.

   ```json
   {"v":1,"type":"tool_call","ms":0,"agent":"…","id":"call_9f2c","name":"read","args":{}}
   {"v":3,"type":"tool_call","ms":0,"agent":"…","id":"m4/1","name":"read","args":{}}
   ```

3. **Expect `usage` lines.** A new event, one per recorded model call:
   `usage {model, purpose, usage}`, with the agent it counts for. `purpose`
   is `turn`, `compaction` and the like for the agent's own calls, and
   `helper` for a call one of its helpers reported. A driver that fails on
   an unknown event type must let it pass.

4. **Do not add helpers' spend again.** `turn_end`'s `usage` now includes
   what the turn's helpers reported while it ran. `exit`'s `usage` is the
   session's total, earlier runs of a resumed session included, where it
   counted only the session's own model calls. `exit` gains
   `usage_complete`: `false` when a helper's spend was not all reported, so
   `usage` is a lower bound, or when no usage is kept.

   ```json
   {"v":1,"type":"exit","ms":0,"code":0,"answer":"…","error":null,"session":"…","usage":{}}
   {"v":3,"type":"exit","ms":0,"code":0,"answer":"…","error":null,"session":"…","usage":{},"usage_complete":true}
   ```

5. **Send well-formed answers.** An `answer` is checked by one rule in every
   front. Several choices for a question that takes one are refused; 0.46
   took one of them, so `"choices":["deny","allow"]` could run the gated
   call. An empty list is accepted for a question that takes several. A
   second answer to a question gets `"ok":false` with `the question was
   already answered`, where it got `"ok":true`.

6. **Keep lines under 64 MiB.** A longer request line, or one that is not
   UTF-8, is answered with an error reply and the next line is read.

What a correct driver gets without changing anything:

- `ready` is always the first line, even when a request already waits on
  stdin or a resumed session carries on a turn.
- A reply that did not stream, such as one a plugin answered in place,
  arrives as a `text` line just before its turn ends. 0.46 wrote no `text`
  for it.
- The `Reasoning reset to default` notice has `"level":"warn"`, where it had
  `"info"`.
- With `RedactPlugin` on, a secret reads `‹SECRET_1›`, where 0.46 wrote
  `[REDACTED]`. With it off, which is the default, nothing is redacted.
- When rig exits mid-turn, it still writes a `tool_call` and a `tool_result`
  (`Not run: …`) for each call of the cut reply that had not started.

#### `rig -p`: exit codes and refusals

| Case | 0.46 | 0.47 |
| --- | --- | --- |
| A command that opens a list: `rig -p /model`, `/effort`, `/resume`, `/login` | exit 0, prints nothing | exit 2, with the commands to type instead (`/model chatgpt/gpt-6.2-sol`) |
| Text after a command that takes none: `rig -p "/new hello"` | exit 0 | exit 2 |
| A prompt that starts with a path, `//`, or `/` and a space | read as a slash command | sent to the model |
| Piped input that is not UTF-8, over 64 MiB, or fails to read | read lossily | exit 2 |
| An open stdin that stays silent | ignored after a second | waited for until it closes |
| The agent build fails to compile | answers with the previous build | exit 1, the compiler's errors on stderr |
| `-m` names a model with no key | exit 1 | exit 2 |
| A plugin, such as the budget, stops the run's agent | not applicable | exit 1, even after an answer |
| A `settings.toml` table that does not read | starts | exit 2 (exit 1 when the file itself or `[plugins]` does not read) |

A script that runs `rig -p` with no piped input gives it an empty stdin:

```console
# before
rig -p "summarize the repo"
# after, from a script whose stdin stays open
rig -p "summarize the repo" </dev/null
```

Output changed only where a person reads it:

- Tool progress lines on stderr no longer start with `rig: `. Errors and
  warnings still do. A script that matched `^rig: shell: ` matches
  `^shell: `.
- When stdout is a terminal, the answer's markdown is drawn and
  word-wrapped to the terminal's width (`COLUMNS`, else its size). When
  stderr is a terminal, each tool call shows there as a dim line. Piped or
  redirected, stdout gets the same bytes as before.
- A reply that did not stream is printed as narration, as a streamed one is.

#### `rig acp`

- Options go before `acp`. A word after it, such as `rig acp --tools=read`,
  was ignored and is now refused. Change the editor's command:

  ```console
  # before
  rig acp --tools=read
  # after
  rig --tools=read acp
  ```

- `session/load` and `session/resume` of a session the client already has
  open re-attach to it, where they failed with `session <id> is already
  open`. A load replays the history first. It is refused while a prompt
  runs, and a request naming other MCP servers is refused; the same servers
  in another order are accepted.
- The model list an editor shows holds the models usable now, as
  `/model` does, and a command that opens a list is refused with the
  commands to type instead.
- A reply a plugin answers without streaming reaches the editor. A saved
  call of a stopped reply replays as failed, not pending, and a tool result
  of several parts shows the same live and after `session/load`.
- A line over 64 MiB (`rig_acp::LINE_LIMIT`) is refused.
- The launcher's warnings, such as the one for a 0.46 home, show once, in
  the first session `rig acp` opens.

---

## 0.45 → 0.46

0.46 is mostly a coding-agent release. Its plugin extension points become
ordinary Bevy data and systems, the TUI draws through render systems, and the
headless JSON protocol, the shell and file tools, compaction and session
restore all change behavior. Four crates are new: `rig-mcp-client`,
`rig-acp`, `rig-traffic` and `rig-claude-cloak`. The library changes are
smaller: rig-core stops forcing two serde_json features, a few message,
provider-dialect and catalog types gain or lose fields, and some constants
move. Rust **1.97.1** and Bevy **`=0.20.0`** are unchanged.

The examples below are **source-checked migration sketches, not compiled
standalone programs**. Most are transcribed from the pull requests that made
the change. They describe the final 0.46 API. If you tried `0.46.0-rc.1`,
everything here still applies; the one rc-only change is that the
unpublished `claude-cloak` path crate became the published
`rig-claude-cloak` (see [section 8](#8-the-coding-agents-pluginstoml-and-settings)).

### 1. serde_json features are yours to choose

rig-core no longer forces serde_json's `preserve_order` and `float_roundtrip`
features. Without them, schema properties use sorted keys, JSON text inside
prompts, embeddings and tool results can change order, and parsed floats can
differ in their low bits. To keep the 0.45 behavior, add a direct dependency:

```toml
[dependencies]
serde_json = { version = "1", features = ["preserve_order", "float_roundtrip"] }
```

Some rig builds still enable them: `rig-cassette` with `traffic`, `http` or
`bedrock` (and so `rig-traffic`) enables both, `rig-claude-cloak`,
`rig-mcp-client` and so `rig-acp` enable `preserve_order`, and `rig-mongodb`
enables it through BSON.

### 2. Message data: `Native`, reasoning blobs and images

`Native` has three new public fields: `blob_slot`, `baseline` and `edited`.
Prefer `AssistantContent::with_native(item)` over a struct literal. It records
a tool call's `baseline` and leaves `edited` false and `blob_slot` unset:

```rust
// before
Native { item, fingerprint }
// after
Native { item, fingerprint, baseline: None, edited: false, blob_slot: None }
```

`Native::reasoning_blob` takes no `Api`; it reads `blob_slot`, which the
reply fold records from the wire's `ReplayTarget::blob_slots`. `Blob::key` is
now `&'a str`. A custom completion wire whose items carry a reasoning blob
declares where:

```rust
// before
let blob = native.reasoning_blob(&origin.api);
// after
let blob = native.reasoning_blob();

impl ReplayTarget for MyWire {
    // ...
    fn blob_slots(&self) -> &'static [&'static str] {
        &["/signature"]
    }
}
```

`Native` equality includes `blob_slot`, so a test comparing a decoded block
with one built by `with_native` must set the wire's pointer.

`history::IMAGE_OMITTED` replaces `USER_IMAGE_OMITTED` and
`ASSISTANT_IMAGE_OMITTED`. `ImageDetail` no longer implements `FromStr`:

```rust
// before
let detail: ImageDetail = "low".parse()?;
// after
let detail: ImageDetail = serde_json::from_value(serde_json::json!("low"))?;
```

`rig_tools::attach::attached_file` moved to
`rig_core::conversation::attached_file`, where the `FILE_OPEN` and
`FILE_CLOSE` markers are now public. `Conversation::resume` is removed.

### 3. Provider selection, dialects and the catalog

Ollama is registered in two families, so the bare selection `ollama` is
ambiguous. Model references such as `ollama/qwen3:4b` and serialized ids are
unchanged.

```rust
// before
ProviderId::resolve("ollama")?          // ollama/openai
// after
ProviderId::resolve("ollama/openai")?   // "ollama" is SelectionError::Ambiguous
```

A custom OpenAI-compatible dialect that set `Identity::session_ids` names the
header on its quirks instead. `openai::wire::Quirks` also gains
`catalog_priced` and `short_lived_key`; it is `#[non_exhaustive]`, so start
from a shipped constant:

```rust
let mut quirks = Quirks::openai();
// before
quirks.identity = Some(Identity { originator, originator_env, user_agent_env, session_ids: true });
// after
quirks.identity = Some(Identity { originator, originator_env, user_agent_env });
quirks.session_header = Some("session-id");
```

`MessagesDecoder::new` takes the dialect's quirks:

```rust
// before
MessagesDecoder::new(false)
MessagesDecoder::new(true) // a dialect that takes unsigned thinking back
// after
MessagesDecoder::new(anthropic::wire::Quirks::anthropic())
MessagesDecoder::new(anthropic::wire::MOONSHOT.quirks) // or your dialect's quirks
```

`Quirks::anthropic()` also prices cache writes by their TTL split.

Catalog pricing is decided per wire through `ReplayTarget::catalog_priced`
and the dialects' `Quirks::catalog_priced`, not a list of provider names.

`Compat::chat_tools_need_reasoning_off: bool` becomes
`Compat::chat_tools: Option<ChatTools>` (`Any`, `Never` or `ReasoningOff`).
In a JSON override file's `rig` object, `"chat_tools_need_reasoning_off": true`
becomes `"chat_tools": "reasoning_off"` (the same rule) or `"never"` (stricter:
no tools on Chat Completions at any effort), and `false` becomes `"any"`. The
old key is rejected. `ContextUse::window` is renamed `limit` and holds
the model's input limit:

```rust
// before
ContextUse { tokens, window: spec.context_window }
// after
ContextUse { tokens, limit: spec.input_limit() }
```

`xiaomimimo::MIMO_V2_FLASH`, `MIMO_V2_OMNI` and `MIMO_V2_PRO` are removed
because Xiaomi retired them; use `MIMO_V2_6_FLASH`, `MIMO_V2_6_PRO` or
`MIMO_V2_6_PRO_ULTRASPEED`.
`ModelSpec` and `ProviderError` are `#[non_exhaustive]`, so their new
`max_input_tokens`, `details` and `Stalled` need no code change. The
`test-utils` scripted error is exhaustive, though:
`rig_core::test_utils::MockError` gains `Stalled(Duration)`, so a `match` on
it adds an arm:

```rust
match error {
    MockError::Provider(_) | MockError::Request(_) | MockError::ProviderResponse(_) => {}
    MockError::Stalled(_) => {} // new in 0.46
}
```

### 4. Agent runs, observers and effect records

`RunSpec` and `PreparedRequest` have a new `provider_tools` field, the
provider-hosted tools `AgentBuilder::provider_tool` adds. A struct literal
adds it:

```rust
// unchanged: a literal that fills the rest from `RunSpec::new()`
RunSpec { preamble: Some(preamble), max_turns: Some(8), ..RunSpec::new() }
// before: a literal that lists every field
PreparedRequest { chat_history, documents, tools, temperature, max_tokens, additional_params, /* ... */ }
// after
PreparedRequest { chat_history, documents, tools, temperature, max_tokens, additional_params, provider_tools: Vec::new(), /* ... */ }
```

`Observe::event` is called for every streamed item and `Recorder::event` also
for a stream's first, even when `keep_events()` is false. An implementation
that stores what it is given checks `keep_events()` itself. `EffectRecord`
has a new `first_item_ms: Option<u64>`; build records with
`EffectRecord::new(..)`.

`TurnEnd::Stopped` has two new fields, `at: Option<MessageId>` and
`calls: Vec<CallId>`. Use `at: None, calls: Vec::new()` for a turn your code
ends itself, or `..` in patterns. `conversation::STOPPED` is reworded.

### 5. Memory and compaction

`Summarizer::request` takes the compaction's reason instead of a focus
string, and `Summarizer::instruction` takes the token budget:

```rust
// before
summarizer.request(older, previous, "keep the tests", &spec)?;
summarizer.request(older, previous, "", &spec)?;
let text = summarizer.instruction("keep the tests");
// after
let asked = CompactReason::Asked { focus: "keep the tests".to_owned() };
summarizer.request(older, previous, &asked, &spec)?;
summarizer.request(older, previous, &CompactReason::Threshold, &spec)?;
let text = summarizer.instruction("keep the tests", tokens);
```

`rig_memory::output_room` is removed; output fitting lives in rig-basics'
compaction. `SummaryLimits::DEFAULT.summary_tokens` is 32,000 (was 12,000).
`SummaryError` has a new `Failed(String)` variant and rig-basics'
`compaction::Recovering` a new `NotSummarized`; `Recovering` is exhaustive, so
a `match` on it adds an arm.

`MemoryStore::read_log` (and `read_log_async`) on a name that is no log name,
such as `"a/b"` or `"../x"`, returns `io::ErrorKind::InvalidInput` instead of
`NotFound`. Files under `rig/kept/` in a session's store are reserved for the
copies `rig_memory::migration` keeps.

New and optional: `SqliteSessionStore` and `PostgresSessionStore` behind each
crate's `memory` feature (the facade's `memory` feature enables them with
`sqlite` or `postgres`), and `rig_memory::conformance::check_store` behind
`test-utils` for testing your own store.

### 6. Cassette recording

```rust
// before
let http = DirectRecordingHttpClient::new(cassette.direct_recorder());
// after
let http = cassette.direct_http();
// a recorder for a transport that bypasses the proxy, such as Bedrock's
recorder.record_http_interaction(request, response); // was `.await`ed
```

Proxy recording now goes through the traffic recorder. A recording is refused
when a reply is never read, its body fails while it is read, a stream fails
midway or stops partway through a frame (a stop at a frame boundary is kept),
a request gets no reply, or an exchange is still open 5 s after `finish`.

### 7. Coding-agent plugin API

These matter if you write plugins for `rig-harness`.

**Model-call adapters are systems.** `Adapter`, `Adapters` and `ModelStarts`
are removed. Change the call's `Adapted(Connection)` in
`PrepareSystems::Adapt` (or `AdaptOutermost`):

```rust
// before
fn tools_as_text(mut calls: Query<&mut Adapters, (Added<Admitted>, With<TurnCall>)>) {
    for mut adapters in &mut calls {
        adapters.0.push(Adapter::new("text-tools", |mut connection: Connection| {
            connection.handler = Handler(connection.handler.0.map_stream("text-tools", pass_on));
            connection
        }));
    }
}
app.add_systems(AgentPass, tools_as_text.in_set(PrepareSystems::Edit));
// after
fn tools_as_text(mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>) {
    for mut adapted in &mut calls {
        let handler = adapted.0.handler.0.clone().map_stream("text-tools", pass_on);
        adapted.0.handler = Handler(handler);
    }
}
app.add_systems(AgentPass, tools_as_text.in_set(PrepareSystems::Adapt));
```

What `ModelStarts::sent` and `dropped` gave is `adapted.sent(model)` (the
request) and `adapted.dropped(model)` (the refused options), neither in a
`Result`; the connection is `adapted.0` and the prompt is `model.prompt`.
After Start, read `SentTo`. A call's connection is
chosen once, after `PrepareSystems::Edit`, so route in `Edit` or before.
`PrepareSystems` is exhaustive, so a `match` on it adds the `Adapt` and
`AdaptOutermost` arms.

**Per-call tools implement `PerCallTool`.** `add_tool_per_call` and
`add_live_tool_per_call` take the tool and its options, with no closure:

```rust
// before
app.add_tool_per_call(Read::default(), ToolOptions::default(), move |place| Read {
    workspace: place.workspace,
    base: Some(place.base),
    confine: place.confine,
    seen: seen.clone(),
});
// after (rig-tools' file tools and Shell implement PerCallTool)
app.add_tool_per_call(Read { seen, ..Read::default() }, ToolOptions::default());

// a tool of your own: PerCallTool needs Tool + Clone + 'static
#[derive(Clone)]
struct WordCount { workspace: Arc<dyn Workspace>, base: Option<PathBuf>, confine: bool }

impl PerCallTool for WordCount {
    fn at(&self, at: CallPlace) -> Self {
        WordCount { workspace: at.workspace, base: Some(at.base), confine: at.confine }
    }
}
```

**Retry, failure and restart.** The `Retry` event and
`Conversation::resume` are removed; after a failed turn the user's message is
back in the input. To send it again from code:

```rust
// before
commands.trigger(Retry { entity: agent });
// after
commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue));
```

The kernel stops every restored turn. A plugin that wants a restored turn to
go on, or a live turn to survive a quit, puts `rig_ecs::restore::CarryOn` on
the turn by `Stage::Gate`. `SubagentSystems::StopRestored` is removed; order
against `GateSystems::After`. `rig_ecs::turn::Exiting(String)` and
`rig_ecs::restore::RESTARTED` are public. `rig_ecs::agent::ContextStale`
marks an agent whose last context figure no longer applies.

**rig-tui looks are render systems.** `add_tool_renderer`,
`add_entry_renderer`, `add_content_renderer`, `add_aside_renderer`,
`add_call_decorator`, their renderer components (`ToolRenderer`,
`EntryRenderer`, `ContentRenderer`, `AsideRenderer`, `CallDecorator`,
`MessageDecorator`), their callback types (`RenderToolCall`, `RenderEntry`,
`RenderContent`, `RenderAside`, `DecorateMessage`) and `AppToolRenderersExt`
are removed. `RenderContent` is now an enum: the owned non-text content block a content look gets.
Write systems in the `TranscriptRender` schedule that fill `RenderBatch`:

```rust
// before
app.add_tool_renderer("probe", |call| vec![call.header("Probe", "")])
    .add_call_decorator("duration", |call| {
        call.took.map(|t| Line::from(format!("{t:?}"))).into_iter().collect()
    });
// after
use rig_tui::{RenderBatch, RenderSystems, TranscriptRender};
app.add_systems(TranscriptRender, draw.in_set(RenderSystems::Render))
    .add_systems(TranscriptRender, duration.in_set(RenderSystems::Decorate));
fn draw(mut batch: ResMut<RenderBatch>) {
    for (input, output) in batch.tools_mut("probe") {
        output.lines = vec![input.view().header("Probe", "").into()];
    }
}
fn duration(mut batch: ResMut<RenderBatch>) {
    for (input, output) in batch.calls_mut() {
        output.decorate("duration", input.took.map(|t| Line::from(format!("{t:?}"))).into_iter().collect());
    }
}
```

Entry, content and aside renderers become loops over `entries_mut(plugin)`,
`contents_mut(kind)` and `asides_mut(kind)`; message decorators loop over
`messages_mut()`. To replace a shipped look, order after its public set
(`CodingToolLooks`, `SummaryLooks`, `SubagentLooks`, `WelcomeLooks`,
`BuildFailedLooks`) instead of disabling its renderer entity.
`TranscriptLayout` is no longer a system parameter and has no lifetime
parameters; `lay_out` is an associated function that takes `&mut World`:

```rust
// before
let rows = layout.lay_out(view, agent, 80);
// after
let rows: Option<DrawnTranscript> = TranscriptLayout::lay_out(world, view, agent, 80);
```

Looks return `LookLine`s instead of ratatui `Line`s. `highlight::highlight`
and `ToolCallView::header_spans` are removed; `code_rows`,
`markdown::render`, `render_marked`, `diff::unified_lines`,
`ToolCallView::{result_lines, default_lines}` and
`MessageView::default_lines` return `LookLine`s:

```rust
// before
let colored = highlight(&text, Language::File(path)).unwrap_or_default();
let rows = colored.into_iter().map(|spans| Line::from([vec![Span::from("   ")], spans].concat()));
// after
let code = Code::new(&text, Language::File(path));
let rows = (0..code.len()).map(|index| code.line(index, vec![Span::from("   ")]));
```

A look that draws no code converts with `LookLine::from`; a panel outside the
transcript draws `LookLine::colored()`. For `header_spans`, use
`header(title, detail)` for a plain detail or `header_code(title, &code, index)`
for a line of code; a header of arbitrary styled spans has no replacement.

**Other plugin-facing changes:**

- `PickItem` gains `more: Option<String>` and `Picker` gains `secret: bool`;
  literals add them (`Picker` has `Default`).
- Exhaustive enums with new variants, so a `match` adds arms:
  `rig_tui::CallState::Stopped`, `rig_tui::InputAction::Exit`,
  `rig_tui::HistorySystems::Seed` and `rig_basics::activity::ActivityStatus::Writing`.
- `EntryView` gains `id: Option<MessageId>`.
- `AppCliExt` gains `refuse_run(why)`, with which a plugin refuses a run. It
  is a new required method only for a type of your own that implements the
  trait.
- `SessionPicker::show` and `SessionItem` are removed: put labels in a
  `ResumeLabel` or a listed saved value, or relabel the `Picker`'s items in a
  system after `SessionPickerSystems`. `SessionEntry::label()` no longer holds
  the directory: use `detail()` for the compact folder or `more()` for the
  full one.
- `rig_basics::reload::{SELF_EXTENSION_TAG, self_extension, ReloadAsked,
  MAX_RELOADS, ORIGIN}` are removed, with nothing replacing them.
- `rig_basics::coding_tools::ShellSettings` is removed. `rig_tools::Shell`
  loses `max_timeout` and `quiet`, `Process::run(deadline, quiet, sink)` is
  `run(deadline, sink)`, and `Ended::Quiet` and `Process::forced` are gone.
- `rig_tools::Read`'s output is `ToolOutput`, not `String`:

  ```rust
  // before
  let shown: String = read.call(&mut context, ReadArgs::new("a.txt")).await?;
  // after
  let shown = read.call(&mut context, ReadArgs::new("a.txt")).await?;
  let text = shown.as_text(); // None for an image read
  ```

- A custom `Workspace` that should expand `~` overrides `join`, as `Local`
  does; the trait's default `join` is unchanged.
- `RunInTerminal`: when a program exits with a failure code on a terminal,
  `TerminalExit` arrives after the user presses Enter, Ctrl+C or Ctrl+D. Use
  `exit.failure()` to describe it.
- The TUI's `Editor`: `clear()` keeps the text in `PromptHistory`, `take`
  keeps a draft hidden behind a recalled prompt, and `up` on the first line
  moves the cursor to its start before it browses. A repeated prompt moves to
  the end of `PromptHistory::entries` instead of being skipped. A filter of
  `PromptHistory::entries` that should also hide seeded prompts runs in
  `TuiKeys`:

  ```rust
  app.add_systems(TuiKeys, scrub.after(HistorySystems::Seed).before(InputSystems::Dispatch));
  ```

### 8. The coding agent's `plugins.toml` and settings

The launcher does not rewrite an existing `RIG_HOME/plugins.toml`. Until the
login entry is renamed the agent build fails, because
`rig_basics::login_chatgpt` no longer exists, and the previous build starts:

```toml
# before
plugin = "rig_basics::login_chatgpt::ChatgptLoginPlugin"
# after
plugin = "rig_basics::login::LoginPlugin"
```

Add the new default plugins, then `rig build` or `/reload`:

```sh
rig plugin add rig_basics::agents_md::AgentsMdPlugin
rig plugin add rig_basics::skills::SkillsPlugin
rig plugin add rig_traffic::TrafficPlugin
```

Without `AgentsMdPlugin` the model is told nothing about `AGENTS.md`, and
`CLAUDE.md` is no longer read: link `AGENTS.md` to it in a repository that
only has a `CLAUDE.md`. The MCP client, ACP and Claude Cloak plugins are
optional (`rig plugin add rig_mcp_client::McpPlugin`,
`rig plugin add rig_acp::AcpPlugin`,
`rig plugin add rig_claude_cloak::ClaudeCloakPlugin`; in a fresh
`plugins.toml` they are listed off, and `rig plugin enable` turns one on).
Code that reads the login flow's component imports it from the new module:

```rust
// before
use rig_basics::login_chatgpt::PendingLogin;
// after
use rig_basics::login::PendingLogin;
```

If you used the
`claude-cloak` path crate from a 0.46.0-rc.1 checkout, replace its entry with
`crate = "rig-claude-cloak"` and `plugin = "rig_claude_cloak::ClaudeCloakPlugin"`.
An agent built with corpus-base's `corpus_base::mcp_client::McpClientPlugin`
uses `rig_mcp_client::McpPlugin` instead.

In `settings.toml`, `[shell] max_timeout` is no longer read; remove it. The
OSC 133 mark nonce is opt-in:

```toml
[tui]
mark_nonce_env = "FUX_MARK_NONCE"
```

`/retry` is gone: after a failed turn, the message is back in the input;
press Enter to send it again. A key saved with `/login <provider>` is used
before the provider's environment variable, and then the provider's other
environment variables (such as a base URL) are not read for it. With
`McpPlugin` on, a project's `.rig/mcp.json` server waits for
`/mcp approve <name>` once; move a server you always want into
`$RIG_HOME/mcp.json` to skip that. The model-facing `reload` tool is gone, so
the model can no longer rebuild the agent, and a `--tools` list that names
`reload` is refused.

### 9. Headless drivers: `rig -p`, `rig -p --json` and `rig --json`

- Read to EOF. `exit` is the last line of `rig -p --json`, written after the
  session log, and names its error for every non-zero code. After a signal,
  the launcher's death or a closed stdin, every cut tool call gets a
  `tool_result` with `"cancelled": true` and every cut turn a `turn_end`
  `"stopped"` before it. Every `tool_result` has the `cancelled` and
  `refused` fields and every line has `ms`; `turn_end` has
  `first_token_ms`. A refused tool call writes its `tool_call` and a
  `tool_result` with `"refused": true`.
- Exit codes: an unknown `--tools` name exits 2 before any model call, also
  at a terminal. In `rig -p` and `--json`, an unknown `--model` also exits 2
  (it was 1); at a terminal it shows a notice and picks no model. A turn that
  ends with no answer text fails (`turn_end` `"outcome": "failed"`, exit 1;
  it was exit 0 with `"answer": null`). A run whose stdout nobody reads, such
  as after `| head -1`, stops with 130. A signal also exits 130.
- `--tools` can name a tool a plugin registers in the background, such as an
  MCP server's `<server>__<tool>`: the run waits for every `print::Starting`
  entity first. `--tools` now applies at a terminal too (0.45 ignored it
  there), and an empty `--tools` permits no tools (0.45 gave every tool).
- A failed `rig -p` run names the unanswered message on stderr
  (`rig: not answered: <text>`); `--json` writes it as a `recalled` event.
- A resumed run no longer replays or finishes a turn an exit cut off.
  `rig -r <id> -p "<prompt>"` answers the new prompt; with no prompt it exits
  2. After a crash, the crashed turn's lines come first, ending in `turn_end`
  stopped "the session restarted".
- `rig -p` and piped runs keep the agent build's output in
  `RIG_HOME/build.log`; stderr gets one line when cargo has work to do.

`HeadlessEnded::why` is `Some` for every non-zero code.

---

## 0.44 → 0.45

0.45 changes the classic agent's handler and tool APIs, shares messages behind
`Arc`, reorganizes conversation storage, and publishes a new Bevy-based coding
agent. Rust **1.97.1** is the minimum. The new kernel uses **Bevy `=0.20.0`**.
The `rig` facade still enables `agent` by default at the release tag. If you
use `default-features = false`, enable `agent` explicitly for the classic
runtime and `memory` for `rig::memory`.

The examples below are **source-checked migration sketches, not compiled
standalone programs**. Existing variables stand for your application's model,
transport, handlers and error conversions. They describe the final 0.45 API,
not intermediate APIs introduced and removed during development.

### 1. Conversation and transcript paths

`rig::core` and `rig_agent::core` are removed. Import portable contracts from
`rig` directly or from `rig_core`. `rig_agent::agent::run` becomes
`rig_agent::run`. `rig::tool::portable` is removed; use `rig::tool`.

The old `rig_core::memory` module is removed. Conversation data now lives at
`rig_core::conversation`; persistent storage and compaction live in
`rig_memory` (`rig::memory` with the `memory` feature).

| Before | 0.45 |
| --- | --- |
| Policy-backed memory collections | `rig_core::conversation::Conversation` plus a `rig_memory` store |
| `rig_core::transcript::validate_canonical` | `rig_core::conversation::pairing::validate` |
| `rig_core::transcript::TranscriptError` | `rig_core::conversation::pairing::PairingError` |
| `transcript::{tool_result_output, tool_result_message, invalid_arguments_feedback, invalid_call_feedback, not_executed}` | `rig_core::tool::feedback::{…}` |
| `transcript::is_empty_assistant_turn(content)` | `AssistantMessage::is_empty_turn()` on the turn |
| `transcript::assistant_text_from_choice(content)` | `AssistantMessage::text()` on the turn |

The old `ConversationMemory`, `InMemoryConversationMemory`, `MessageFilter`,
`DemotionHook`, `NoopDemotionHook`, `Compactor` and `MemoryError` contracts have
no compatibility layer. `rig-memory` also replaces `MemoryPolicy`, `IntoFilter`,
`BoxedFilter`, `NoopMemoryPolicy`, `SlidingWindowMemory`, `TokenWindowMemory`,
`PolicyMemory`, `DemotingPolicyMemory`, `CompactingMemory`, `TemplateCompactor`
and `TextSummary`. Use `Recent` for retained windows, `CompactionPolicy` and
`Summarizer` for compaction, and explicit conversation changes rather than
wrapping a `ConversationMemory` backend. `TokenCounter`'s
`count(&Message) -> usize` contract is unchanged. For persistence, use
`rig_memory::MemoryStore` or `DirStore`, or implement `AsyncStore`.
The classic agent accepts these stores:

```rust
let agent = AgentBuilder::new(model)
    .memory(store)
    .conversation("customer-42")
    .build();
agent.prompt("Remember this").await?;
// Override the log per call, or bypass memory with .without_memory().
agent.prompt("Next question").conversation("customer-43").await?;
```

`AgentBuilder::memory_handler` and the old effect-backed memory path are
removed. `conversation(ConversationId)` becomes `conversation(log_name)`;
`rig_core::id::ConversationId` is removed. Read and write failures arrive as
`PromptError::Memory { error, response }`, not the old tuple variant. A failed
save can retain the finished response in `response`; a load failure has none.
`PromptResponse::memory_append`, `with_memory_append` and `MemoryAppend` are
removed: a failed save makes the run fail instead of attaching an append
acknowledgment. Handle that error rather than repeating a tool-using run.

### 2. Shared messages, tool definitions, requests and answers

`Conversation::messages`, `Conversation::into_messages`, `Entry::message`,
`CompletionRequest::chat_history`, `AgentRun::messages` and the `CallModel`
step's prompt/history now carry `Arc<Message>`. `CompletionRequest::tools`
carries `Arc<ToolDefinition>`. `EffectKind::Completion::request` is an
`Arc<CompletionRequest>`. `effect::Outcome` is renamed `Answered`, and
`Answered::Completion` holds an `Arc<CompletionResponse>`.
`streaming::Relayed::Done` changes from `Box<CompletionResponse>` to the same
Arc; use `.into()` to wrap a response and `Arc::unwrap_or_clone` to own it.
Hook histories (`RunStart`, `CompletionCallEvent`, `ModelSelection`),
`RunSettled::messages`, `RequestPatch::history`, `PreparedRequest::chat_history`
and advertised tool definitions share their values too.

```rust
use std::sync::Arc;
// before: match request.chat_history.first() { Some(Message::System { .. }) => … }
match request.chat_history.first().map(Arc::as_ref) {
    Some(Message::System { content }) => { /* inspect content */ }
    _ => {}
}
// before: request.chat_history.push(Message::user("next"));
request.chat_history.push(Message::user("next").into());
request.tools.push(definition.into());
// before: &mut entry.message
if let Message::User { content } = Arc::make_mut(&mut entry.message) {
    /* edit content */
}
let request = Arc::unwrap_or_clone(shared_request);
```

Constructors such as `CompletionRequest::message`/`messages`,
`Conversation::from_messages`/`extend` and `Entry::new` accept
`Into<Arc<Message>>`; a plain message still works there. `From<Arc<Message>>
for Message` is available when you need an owned message. Serde writes the
contents, not an Arc wrapper, so this change alone does not change JSON.

Memory APIs that work on a window take shared-message slices, including
`Recent::{start, view}`, `CompactionPolicy::{cut, plan}`, `Summarizer::request`
and `ClearToolOutputs::clear`. Change custom slice annotations from
`&[Message]` to `&[Arc<Message>]` where required.

### 3. One contextual tool trait

`PortableTool`, `PortableToolEmbedding`, `ErasedTool` and the portable tool
module are removed. If skipping from before 0.41, skip its intermediate
`PortableTool` rename and implement the contextual `Tool` directly, including
the context parameter:

```rust
// before: impl PortableTool for Search { async fn call(&self, args: Self::Args) … }
impl Tool for Search {
    // Keep your existing associated types, name, description and schema.
    async fn call(&self, context: &mut ToolContext, args: Self::Args)
        -> Result<Self::Output, Self::Error>
    {
        /* your implementation */
    }
}
```

`#[rig_tool]` emits the current trait. Runtime-defined tools use
`DynamicTool::new_with_context` with a validated `ToolName`, not `ToolFn::new`.
`DynamicTool::run`/`run_with` and `ToolAdapter::run`/`run_with` execute them
without registering an agent. For registry execution:

```rust
// before: dynamic_tool.execute(arguments).await?
let result = RegisteredTool::from_dynamic(dynamic_tool)
    .execute(arguments.to_string(), &mut ToolContext::new())
    .await
    .into_result()?;
// before: handle.execute(name, &args, &mut context).await
let result = handle.toolset().execute(name, args, &mut context).await;
// before: catalog.execute(name, &args, &mut context).await
if let Some(tool) = catalog.get(name) {
    let result = tool.execute(args, &mut context).await;
}
```

`ToolDispatch`, `ToolLease`, `execute_tool`, `ToolSet::documents`,
`RegisteredTool::descriptor`, `ToolSet::from_dynamic_tools` and
`DispatchAction::try_rewrite_tool_args` are removed. Add dynamic tools with
`ToolSet::add_dynamic_tool`. Built-ins live at `rig_core::tool::builtin`, not
`rig_agent::tool::builtin`. `ToolContext::require_result` and `TypeMap` are
removed; use the typed context APIs and handle absence explicitly.
Full `ToolDefinition` literals add `annotations: Default::default()`; prefer
`ToolDefinition::new` and `.with_annotations`. Tool catalogs, `ToolSet` and
`ToolServerHandle` now return shared definitions where the old API returned
owned definitions. `ToolServerHandle::static_tool_defs` is removed; read a
`snapshot().definitions()` or the `toolset()`'s definitions instead.

### 4. Replace the effect bus with inline handlers

`rig::bus` / `rig_agent::bus` and `Bus::channel` are removed. There is no driver
task to spawn, dispatcher/registrar pair or bus backpressure. Register handlers
on `Handlers`; work runs in the calling future or stream:

```rust
// before
let (dispatcher, registrar, mut driver) = Bus::channel();
driver.register(model_key.clone(), model_handler)?;
tokio::spawn(driver);
let agent = AgentBuilder::over_bus(dispatcher, registrar, "guest", model_key).build();
// after
let handlers = rig::handlers::Handlers::default();
handlers.register(model_key.clone(), model_handler)?;
let agent = AgentBuilder::over_handlers(handlers, "guest", model_key).build();
```

Dropping a dispatch future cancels it. A handler needing a concurrency limit
or serialized execution must own its semaphore/mutex. Replacing or removing a
registration affects new dispatches; an in-flight dispatch keeps its handler.
Tool registrations are pinned for a model turn.

`AgentBuilder` is no longer generic over tool-server state. Write
`let builder: AgentBuilder = …`, not `AgentBuilder<WithToolServerHandle>`.
`NoToolConfig`, `WithToolServerHandle`, `WithBuilderTools`, `AgentParts`,
`Agent::into_parts`, `owns_bus`, `bus_config` and `configure_bus` are removed;
keep the `Handlers` clone you pass to the builder instead of extracting a bus.
`ErrorKind::{BusClosed, MemoryBackend, MemoryPolicy}` and the deferred-serving APIs
`serve::{ServingPolicy, deferred, Resolver}` are removed. So are
`effect::{Delivery, DeliveryKind}` and recorder delivery-tracking methods.
Use inline `Serve` handlers and the dispatch recorder instead.
The memory effect family, `MemoryOp`, `MemoryOutcome`, `memory_key`,
`serve::MemoryAdapter` and typed `MemoryHandle` are removed; persistence is a
store operation, not a memory effect dispatched through this table.
`Family::unwrap` now takes `Answered`.

Interception types `serve::Decision` and `serve::Verdict` become `BeforeServe`
and `AfterServe`. `serve::Origin` becomes `DispatchContext` (with an optional
`purpose`). Recorder implementations must use the new outcome type and
context instead of delivery-tracking callbacks. `streaming::Transcript`
becomes `StreamRecord`, with the same item-list role. `EffectRecord` literals
rename `tool_output` to `published` and add review, disposition, timing,
purpose and turn metadata; prefer `EffectRecord::new` and fill the metadata
that your recorder actually knows.

### 5. TurnPolicy owns a turn's tool rules

`PreparedRequest` replaces `tool_choice`, `executable_tool_names`,
`allowed_tool_names` and `output_tool_name` with `policy: TurnPolicy`.
`AgentRun::with_tool_choice`, `commit_output_tool_name`,
`ModelTurn::from_response_parts` and `prepare::allowed_tool_names_for_choice`
are removed.

```rust
// before: ModelTurn::from_response_parts(&response, executable, allowed)
let turn = ModelTurn::from_response(&response, &prepared);
// Or a hand-built turn:
let policy = TurnPolicy::new(executable, choice, output_tool)?;
let turn = ModelTurn::new(head, content, usage, policy.clone(), raw);
let assembler = StreamedTurnAssembler::new(policy);
```

Read `prepared.policy.executable()`, `.allowed()`, `.tool_choice()` and
`.output_tool()`. Policy construction rejects impossible Required/Specific
choices. Deserialization revalidates them. Skip, Repair, invalid-call hooks and
the output-tool intercept read the effective policy, including hook patches.
The run pins its first policy-named output tool itself. A call finalizes the
run only when that turn's policy names the output tool.

`AgentRun::with_output_tool_name` only seeds later preparation; it does not
make an arbitrary hand-built turn intercept the tool.

### 6. Sans-IO tool answers and restart-safe histories

`AgentRunStep::CallTools` now offers `PendingToolCall::Execute(ExecCall)` or
`PendingToolCall::Malformed(MalformedCall)`, not public `tool_call` and
`preresolved_result` fields. `AgentRun::tool_results(Vec<UserContent>)` is
removed. The run, not the host, constructs the result message:

```rust
for call in calls {
    match call {
        PendingToolCall::Execute(call) => {
            let result = tools.execute(call.name(), call.arguments().to_string(), &mut context).await;
            run.answer(call.answer(result))?;
        }
        PendingToolCall::Malformed(call) => run.answer(call.answer(None))?,
        _ => { /* handle future variants without executing them */ }
    }
}
```

Collect `ToolAnswer`s and use `run.answer_all(answers)` for atomic submission.
An empty answer list leaves the step open. Answers from the wrong turn, slot
or kind are refused. Already answered calls are not offered again after
resume. Use `run.malformed_context(&call, streaming)` instead of
`malformed_tool_call_context(&tool_call, streaming)`; the result is no longer
an `Option`. `None` on a malformed call asks for default feedback, not
execution. A skipped or denied executable call can answer with
`ToolResult::skipped(reason)`.

`PromptError::Cancelled::chat_history` remains `Vec<Message>` at the final
release tag, but the runtime closes unanswered calls before returning it.
Build cancellations with `run.cancel_error(reason)`.
`AgentRun::canonical_history()` returns the closed vector. For external
history, use `conversation::pairing::validate` to reject malformed input or
`pairing::repair(history).messages` to repair it. The intermediate
`CanonicalHistory` wrapper was removed before release; do not migrate to it.

`StreamedResolution::TurnAbandoned` is a unit variant; remove reads of
`skipped_tool_result`. Read committed results through `run::project`:

```rust
let cursor = run.projection_start();
// Advance the run, then project the new committed history.
for item in rig_agent::run::project(&run.messages()[cursor..]) {
    /* display CommittedItem::ToolCall or CommittedItem::ToolResult */
}
```

Persist the advanced cursor in your driver. Start at `projection_start()`, not
`messages().len()`, so a mid-batch resume announces pending calls.
`StreamedTurnAssembler::ingest` now returns one event rather than a vector;
remove the loop around its return value. Its `aggregated_text`,
`aggregated_reasoning` and `aggregated_arguments` helpers are removed; track
stream events in the host or read the completed turn.

`RunEntry` becomes `conversation::Aside`; append with
`Aside::new(kind, author, value)` rather than constructing `{ kind, turn, value }`.
An aside is anchored after a `MessageId` and has author/saved metadata, not a
model-call index. `AgentRun` and hook entry accessors return asides.
`RunSpec::output_schema` is `Option<schemars::Schema>`, not `Option<Value>`;
convert validated schema input rather than assigning arbitrary JSON.
`HookContext` binding methods now return the inline `handlers::Handle` types,
not lifetime-scoped `RunHandle`s; use the returned handle directly.

**Persisted runs use `RUN_FORMAT = 5`**, not 0.44's format. Finish parked runs,
including durable approvals, on the old version or discard them. There is no
old-run upgrade shim. If skipping development versions, skip formats 3 and 4
and migrate directly to 5.

### 7. Catalog lookup, references and connection

A connected model carries its catalog facts. Configure your catalog before
connecting; do not attach a separate spec to an agent.

```rust
// before: Catalog::from_json(&json)?; builtin.clone().merge(overrides)
let base = Catalog::builtin();
let overrides = Catalog::from_overrides(&json, base)?;
let catalog = base.with_overrides(&overrides);
// A models.dev document is a different, lenient input:
let (fetched, skipped) = Catalog::from_models_dev(&json)?;
let refreshed = catalog.with_models_dev(&fetched);
```

`from_overrides` is strict about unknown fields/vendors, values and incomplete
new rows. `from_models_dev` reports skipped rows. `Catalog::from_json`,
`Catalog::merge` and `CatalogError::Format` are gone. `Catalog` shares its rows;
`ModelSpec::new`, `Pricing::new`, `CacheSupport::new`, the supported setters and
`Catalog::insert` build custom rows. Do not copy a full `ModelSpec` literal.

```rust
// before: catalog.resolve(reference) -> Option<&ModelSpec>
let spec = catalog.resolve("anthropic/claude-sonnet-4-5")?.spec;
// before: catalog.find(provider, model)
let spec = catalog.get(provider, model).map(|resolved| resolved.spec);
let exact = catalog.get_exact(provider, model); // strict, no dated fallback
```

`get` and `resolve` recognize dated snapshots of listed ids and return
`Resolved`; `resolve` returns `Result<_, NotFound>` with suggestions.
`Catalog::find`, `find_vendor`, `lookup_snapshot`, `catalog::undated` and
Anthropic's `claude_spec` are removed.

References use **`vendor/model`**, not `vendor:model`. The old
`vendor/format:model` spelling still reads. Choose a non-row format with
`ConnectOptions::format`, rather than persisting an old `ProviderRef` value.
`ModelSelector::provider_ref` and `ProviderRef::parse` are removed; pass
reference text directly to catalog lookup/connection.

```rust
// before: connect(spec, key)?; connect_with(spec, key, http)?
let model = catalog.connect("deepseek/deepseek-chat")?; // environment credentials
let model = catalog.connect_with(
    "deepseek/deepseek-chat",
    ConnectOptions::new().api_key(key).http(http),
)?;
```

The old free `registry::connect`/`connect_with`, `ProviderRef`, `Provider` and
`RefError` are removed; use the catalog and `ModelSelector`.
`ProviderId::api_key_env()` becomes `api_key_envs()`, an ordered list.
`ConnectError` has `NotFound`, `MissingKey`, `Env` and `NoHttpClient` cases and
is no longer `Clone`/`PartialEq`; inspect variants instead of comparing values.

`ReasoningSupport` is an enum (`None`, `Unknown { default }`,
`Listed { levels, budget, can_disable, default }`). Replace field reads with
`.supported()`, `.levels()`, `.budget()`, `.can_disable()` and
`.default_effort()`. The latter accessors distinguish unknown from unsupported.

Remove `AgentBuilder::model_spec(spec)`. The model checks the request using its
own facts; `Model::check`/`DynModel::check` dry-run encoding and return
`CheckError`. A refusal arrives as `PromptError::Report` with kind `Request`,
not `PromptError::Provider(ProviderError::UnsupportedOption(..))`.

Provider wire literals need `facts: Default::default()` or their `with_facts`
builder. Anthropic `Dialect::default_max_tokens` takes facts;
`rig_bedrock::completion::spec(model)` becomes `converse.spec(model)`.
`gemini::AutoCache::for_model(id)` becomes `AutoCache::for_spec(spec)`.

### 8. Document text is not a media source

`DocumentSourceKind::String` and `.string()` are removed. Binary text buffers
holding base64 must use `DocumentSourceKind::Base64`. A document now holds
`DocumentData::{Text(String), File(DocumentSourceKind)}`:

```rust
// before: Document { data: DocumentSourceKind::String(text), … }
let content = UserContent::document_text(text, media_type);
// For a file source in a Document literal:
let document = Document { data: source.into(), media_type, /* other fields */ };
// before: match document.data { DocumentSourceKind::Base64(data) => … }
match document.data {
    DocumentData::Text(text) => { /* plain text */ }
    DocumentData::File(source) => { /* URL, base64, file id, bytes, unknown */ }
}
```

`DocumentData` deliberately has no `Display` or `try_into_inner`.
`DocumentSourceKind::try_into_inner` is also removed; match the variant.
The unused `ContentFormat` enum is removed without replacement.
`completion::history::NO_RESULT_PROVIDED` moves to
`conversation::pairing::NO_RESULT_PROVIDED`. See the
[0.45 silent changes](#045) before loading old JSON.

### 9. Wire, decoder and observation implementors

`Wire` has a new `Event` associated type. Its reassembler implements
`Reassemble<Self::Event>`, not `Reassemble<Self::Frame>`; `absorb` receives the
decoder's already-classified event. Whole-document wires can use
`wire::DocumentDecoder`. Prefer `<MyWire as Wire>::Decoder<'_>` to naming a
removed provider-specific decoder. `operation::Turn` is renamed `ReplyFold`;
`Completion::Fold` names it too. Anthropic's reassembler is now
`streaming::MessagesDocument`, not the old `streaming::document::Message`.
Provider extension implementors should also update the public, doc-hidden
helpers: `WireIds` now carries a history lifetime, and `classify_or_untagged`
takes a tag predicate and typed-event conversion instead of two classifiers.
Prefer the wire's associated types rather than copying internal helper types.

`CompletionRequest` gains defaulted `foreign_reasoning` and `cache_points`
settings; its non-exhaustive constructor API needs no migration for those
additions. Full `Accepts` literals add `foreign_reasoning: false`.
The `Operation::observed_usage` hook has a default, so existing operation
implementations need no extra method unless they expose
normalized usage.

`Observe::outcome` takes `Result<Answered, ErrorReport>` by value, and
`Observe::stream_item` takes `Option<Result<…>>` instead of borrowing it.
`ObservationSink::emit` is removed: use `sink.error(error)` for an error
envelope. Delete projector usage emission; the normalized decoder result owns
usage observation. The unused observation actions/stages are removed; only
`Action::Adapter` and `Action::Host` remain. Use
`Reason::with_detail(code, detail)` instead of `Reason::from_report`, `.code`
or `.unknown`. `Emitter::{unknown, is_unknown}` are removed; name the emitter
with `Emitter::named` or `versioned`. `observe::LARGEST_PAYLOAD_BYTES` is removed;
keep application-specific observation limits in your sink.

`wire::Descriptor` literals add `prepared: None`. `Model` has private state;
construct it with `Model::new(wire, transport)`, not a struct literal.
`ResponsesEvent::Frame::raw` is `Option<String>` and is retained only for
`response.failed`; handle `None` rather than assuming every frame has raw text.
`ReplyFold::snapshot`, `serve::OpenRecord` and the unused `json_utils`
serde helpers are removed. Use `serde_json::Value::to_string()` instead of
`json_utils::serialize_json_value`.

Custom option mappings must account for `GenerationOptions::cache_key`, the
new `OptionFields`/`OptionMap` field. Use `Mapping::unrouted(cache_key)` on a
wire that has no such option. `AnthropicConfig` literals add
`addressing: Addressing::Body`; `Addressing::Url` supports platforms that move
the model into their URL. `Messages::encode_with` exposes the shared body and
beta-header encoding for those platforms.

### 10. Authentication and provider construction

ChatGPT and Copilot `Authenticator::new` take a `TokenStore`, not credential
file paths:

```rust
// ChatGPT: before the second argument was Some(dir.join("chatgpt.json")).
let auth = chatgpt::auth::Authenticator::new(
    chatgpt::auth::AuthSource::OAuth,
    FileTokenStore::new(&dir),
    DeviceCodeHandler::default(),
    true,
);
// Copilot: the two optional path arguments become one store argument.
let auth = copilot::auth::Authenticator::new(
    copilot::auth::AuthSource::OAuth,
    FileTokenStore::new(&dir),
    DeviceCodeHandler::default(),
    true,
);
```

Use `MemoryTokenStore::default()` for nonpersistent credentials, or implement
`TokenStore::{load, save, delete}`. Store failures are `AuthError::Store`.
`TokenStore::lock` has a default no-op; implement it when clients share a
refreshable record. `StoreLock::new(guard)` releases your guard on drop;
`StoreLock::default()` is removed (`StoreLock::new(())` is an unlocked guard).
`copilot::auth::default_token_dir` is removed. ChatGPT's
`AuthContext::access_token` becomes `api_key`; its shared context adds
`api_base`, and Copilot's context is the shared ChatGPT context (with optional
`account_id`). Update full literals and credential field reads.

`CopilotConfig`, `CopilotWire`, `CopilotIntent` and its intent setters are
removed: use `OpenAIConfig::with_key(&copilot::DIALECT, token).connect(http)`.
The shared dialect supplies the editor envelope; there is no per-wire
panel/edits intent selector.
The old OpenAI client's `.authenticate(&auth)` path is replaced by
`client::signed_in::SignedInModel::new(spec, auth, http)?`; it refreshes
credentials per call. This type is generic over the credential source.
Config `.with_http(http)` becomes `.connect(http)`. TypeSafe's
`Jev::new(token).with_http(http)` becomes `JevConfig::new(token).connect(http)`.

### 11. Provider response and option changes

- **Embeddings:** OpenAI `EmbeddingData::embedding`, Cohere
  `EmbeddingResponse::embeddings` / `FloatEmbeddings::values` and Gemini
  `EmbeddingValues::values` are `Vec<f64>`, not JSON numbers. Remove
  `Number::as_f64` conversions; serialization may write `1.0` rather than `1`.
- **Transcription:** full literals of `TranscriptionResponse` and
  `openai::TranscriptionResponse` need the new language, duration, confidence,
  segments and words fields. Prefer their constructors. Provider-specific
  metadata is optional; do not assume every provider supplies it.
- **Image generation:** `ImageGenerationRequest` literals add
  `images: Vec::new()`. Wires without reference-image support refuse nonempty
  lists; the builder needs no change.
- **Gemini:** `AgentConfig::DeepResearch { thinking_summaries }` becomes
  `DeepResearchConfig::new().thinking_summaries(value).into()`.
  `InteractionsOptions::speech(InteractionSpeech::voice("Kore"))` becomes
  `.speech_config(InteractionSpeechConfig::voice(InteractionSpeech::voice("Kore")))`.
  Speech types live in `providers::gemini::interactions_api`.
  `InteractionsGenerationConfig::speech_config` is now an
  `Option<InteractionSpeechConfig>`, not `Vec<InteractionSpeech>`.
  `ImageConfig::image_size("2K")` becomes `.image_size(ImageSize::TwoK)`.
  `GEMINI_2_5_FLASH_IMAGE` moves from `gemini::completion` to
  `gemini::image_generation` and remains available at `gemini::`.
- **Venice images:** the provider-specific `ImageGenerationResponse` and
  `ImageGenerationTiming` structs are removed. Read the portable image response
  and its `raw` JSON for Venice's request echo and timing fields.
- **Bedrock images:** `TextToImageResponse::normalize` is an inherent method;
  remove the `NormalizeImageGenerationResponse` trait import.
- **gRPC Gemini:** `ApiKeyInterceptor` is private; use `GeminiGrpc`'s public
  `connect`/`with_channel` entry points. The `zstd` and `gzip` features are
  removed. The facade's `gemini-grpc` feature remains available.
  The old `rig_gemini_grpc::streaming::GrpcAdapter` module is removed;
  both it and Vertex AI now use Gemini's `GenerateContentDecoder` and
  `GenerateContentChunk`. Their transport `send` methods and wire `Frame`
  associated types return REST-shaped chunks rather than protobuf responses.
  `completion::rest_chunk` converts a protobuf response when you need it.
- **Vertex AI:** `VertexAiClientError` adds `Credentials` and `NoCredentials`;
  update exhaustive matches. Claude support uses `VertexAi::anthropic` and the
  new `anthropic::RawPredict` wire without changing the Gemini entry points.
- **Dialect enums:** `openai::wire::{ImageBody, SpeechBody}` add `MiniMax`.
  Update exhaustive matches in custom dialect plumbing. Gemini interaction
  status adds `Queued`; treat it as not yet complete.

Removed model constants have no aliases. Pick a current catalog model (for
example `minimax::MINIMAX_M3` instead of `MINIMAX_M2_1`) or use a literal only
if the provider still serves that exact id:

| Provider | Removed constants |
| --- | --- |
| OpenAI | `GPT_4_5_PREVIEW`, `GPT_4_5_PREVIEW_2025_02_27`, `GPT_4O_2024_05_13`, `GPT_4_TURBO_2024_04_09`, `GPT_4_TURBO_PREVIEW`, `GPT_4_0125_PREVIEW`, `GPT_4_1106_PREVIEW`, `GPT_4_VISION_PREVIEW`, `GPT_4_1106_VISION_PREVIEW`, `GPT_4_0613`, `O4_MINI_2025_04_16`, `O3_MINI_2025_01_31`, `O1_PRO`, `O1_2024_12_17`, `O1_PREVIEW_2024_09_12`, `O1_MINI_2024_09_12`, `GPT_IMAGE_1` |
| Azure | `GPT_35_TURBO`, `GPT_35_TURBO_INSTRUCT` |
| ChatGPT | `GPT_5_3_CHAT_LATEST` |
| Gemini | `GEMINI_2_5_PRO_PREVIEW_05_06`, `GEMINI_2_5_PRO_PREVIEW_03_25`, `GEMINI_2_5_FLASH_PREVIEW_04_17`, `GEMINI_2_0_FLASH_LITE` |
| MiniMax | `MINIMAX_M2`, `MINIMAX_M2_1`, `MINIMAX_M2_1_HIGHSPEED` |
| xAI | `GROK_2_1212`, `GROK_2_VISION_1212`, `GROK_3_FAST`, `GROK_3_MINI_FAST` |

### 12. MCP connections own their lifetime

`rig-rmcp` now uses **rmcp 3.5**. `McpConnection` replaces externally managed
`RunningService` ownership:

```rust
// before: tools_from_server(definitions, service.peer())
let connection = McpConnection::new(ClientConfig::default().serve(transport).await?);
let tools = connection.tools().await?;
let tool = McpTool::new(definition, &connection);
```

`McpTool::from_mcp_server` and `tools_from_server` are removed. Tools retain
the connection. `McpClientHandler::connect` returns it; use
`connection.close().await` instead of `service.cancel().await`.
`McpClientError::{ToolFetch, ToolFetchTimeout}` become `{Request, Timeout}`.

`McpMeta` wraps `rmcp::model::RequestMetaObject`; the `Meta` re-export is gone.
Read response metadata and structured content through
`context.result::<McpCallToolResult>()?`, not `McpResponseMeta` /
`McpStructuredContent`. Handle `Some(result)` before reading `result.0.meta`
or `result.0.structured_content`.
`McpTool::execute_mcp` and `preserve_mcp_result` are removed; call through
`DynamicTool`/`RegisteredTool` and inspect `McpCallToolResult` in the context
for the original MCP result. rmcp itself renames
`ClientInfo`/`ServerInfo` to `ClientConfig`/`ServerConfig`, and server handlers
return `CallToolResponse` / `ReadResourceResponse`.

### 13. Persistent storage and replay

`rig_memory::Store::append_log` and `replace_log` take an expected length and
return `LogWriteError`. Implement optimistic concurrency checks; a conflicting
save must not overwrite another writer. `AsyncStore` exposes
`read_log_async`, `append_log_async(name, expected_len, bytes)`,
`replace_log_async`, `read_file_async`, `write_file_async`; old `logs`/`files`
methods are absent. Local `Store` implementations also implement `AsyncStore`.
`Arc<dyn Store>` and `Box<dyn Store>` forward the store contract.

`Store` no longer has `remove_log`, `put_blob`, `blob`, `blobs` or `remove_blob`.
Image blobs are files under `blobs/`; use `rig_memory::{store_images,
load_images}`. `rig_memory::upgrade`, `Upgraded`, saved-value upgrades and
rename aliases are removed. **The session format is `rig-session/4`; older
sessions are refused without modification. Start a new session.**
`session.get::<T>(key, &upgrades)` becomes `session.value::<T>(key)`.
Live session rewrite is removed; `rewrite_stopped` remains.

Compaction configuration no longer takes named `CutRule`s or per-agent
`CompactionKnobs`/`Follows`. `CompactionPolicy::plan` loses the cuts argument;
`Summarizer::instruction(kept, focus)` becomes `.instruction(focus)`.
`SummaryCut::upto` is removed. Full literals need the new `near_window`,
`SummaryPrompts::{in_place, notes}` and `ClearToolOutputs::keep_under` fields;
prefer constructors/defaults.

Effect-log checkpoints now use **`CHECKPOINT_FORMAT = 7`** and add
`Checkpoint::last`; recreate old checkpoints. On a by-id resume,
`Handlers::resume_ids_after` advances dispatch identity past the checkpoint.
The old `rig-cassette` `ecs` feature and `rig_cassette::ecs` adapter are removed;
`effect_log` remains available for effect logs whose effects still exist.
`LogHeader::{bus, deliveries, delivery_limitations}` and recorder
`begin_delivery_tracking`, `delivery`, `unsupported_delivery` are removed;
stop populating them. `RequestCheck::Skip` is added for unchecked payload replay.
`jsonl::Writer::on(files, name)`
appends to a `Files` store, replacing the development path-based writer:

```rust
let files = Arc::new(rig_memory::DirFiles::new(dir));
let writer = jsonl::Writer::on(files.clone(), jsonl::SESSION_EFFECT_LOG);
let log = jsonl::read_session(&*files)?; // or jsonl::read_from(bytes)
```

With `rig-cassette`'s `agent` feature, replay `register_all` and
`register_all_checking` take the inline `Handlers` table instead of a bus
registrar/driver; pass the same table used by `AgentBuilder::over_handlers`.
`RequestCheck` is non-exhaustive. Replay by dispatch id, `RequestCheck::Hash`,
program identities, checkpoint helpers and refusal reporting are present in
0.45, despite intermediate PR notes announcing their removal.

### 14. HTTP and smaller removals

`rig_reqwest::runtime()` becomes `rig_http::runtime::handle()` with
`rig-http`'s `tokio` feature. SSE event names are `Cow<'static, str>` rather
than `String`; use `.into_owned()` if ownership is required.
`rig_http::http_client::Error` gains `Unbuilt` and is non-exhaustive. A client
that could not be built fails immediately; this error is not retryable.
`RecordingHttpClient::set_response` under `test-utils` is removed; construct a
new recording client instead.

`HttpMiddleware::{before_request_headers, before_request_body, after_response}`
replace their separate method/URI arguments with `Exchange<'a>`. Read
`exchange.method`, `.uri` and `.extensions`; the latter carries dispatch
metadata such as `serve::Serving`. For example, a signature fragment
`(&self, method: &Method, uri: &Uri, headers: &mut HeaderMap)` becomes
`(&self, exchange: Exchange<'_>, headers: &mut HeaderMap)` (preserving the
trait's named future lifetime).
`http_client::{make_auth_header, bearer_auth_header}` are removed; construct a
validated `HeaderValue` and insert `AUTHORIZATION` with the `Bearer ` prefix.
`LazyBytes` is removed; use `LazyBody<Bytes>`.

`GenAiOperation::{ChatStreaming, Interactions, InteractionsStreaming}` are
removed; use `Chat` and `SpanBuilder::streaming(true)` for streaming telemetry.
`ToolCall::error_result` is removed; use `call.answer(&tool_result)` to build
a message result that retains the tool result's status.
Also removed without direct replacement: `DocumentMediaType::is_code`,
`Text::clear_citations`, `KeyParts::to_key`,
`Native::without_reasoning_blob`, `ToolSchema::try_from`, unused `WireEvent`
constructors and obsolete test mocks. Use tool feedback/results, match media
types, or construct the surviving public types as appropriate. Full literals
of Cohere/Voyage response types must stop setting removed unused fields
(`texts`, `response_type`, `api_version`, `warnings`, `object`, `index`, where
formerly exposed); Cohere's standalone `ApiVersion` type is removed too.

Under `test-utils`, replace `history_conformance::decode` with
`wire::decode_reply`. Removed public fixtures include `MockTextDocument`,
`MockMultiTextDocument`, `CountingMemory`, `FailingMemory`, `AppendFailingMemory`,
`MockStringOutputTool`, `MockImageOutputTool`, `MockBarrierTool`,
`MockControlledTool` and `BarrierMockToolIndex`; define test-local fixtures for
the contracts you need. `MockTurn::with_raw`,
`MockStreamEvent::text_additional_params`, the mock model's `stream_script`,
`CapturedSpan::recorded_texts` and
`validate_result_redaction` are also removed; inspect the raw response,
recorded effects or captured telemetry directly.

The vector-store builder, LSH strategy, all `VectorDistance` metrics,
`ToolResultContent::deserialize_json`, TypeSafe `Response: Serialize`,
`TokenCounter` forwarding through Arc/Box and websocket timeout-disabling
builders remain in the final release. Do not remove them solely because an
intermediate PR's notes say they are gone. `InMemoryVectorStore::iter()` now
yields `(id, document, embeddings)` rather than `(id, (document, embeddings))`;
the index's `iter()` changes the same way. `LSHIndex::query` returns borrowed
`&str` ids rather than owned `String`s. The builder's explicit type path is now
`vector_store::in_memory_store::InMemoryVectorStoreBuilder`, not
`vector_store::builder`, and requires `D: Serialize`.

### 15. The new rig-ecs is a replacement, not an upgrade

`rig-ecs` 0.45 is a new Bevy kernel. It shares **no API** with the unrelated
0.43–0.44 implementation. Its agents, turns and tool calls are entities, its
hooks and slash commands are entities, and its session log saves reflected
components. There is no mechanical mapping from the old world/driver/plugin
surface to this one.

Either keep the old dependency set pinned, including
`rig-ecs = "0.44"` and `rig-cassette = { version = "0.44", features = ["ecs"] }`,
or port to the new kernel/harness. A non-Bevy host can instead drive
`rig_agent::run::AgentRun`; see [the stepper fixture](tests/fixtures/agent_run_stepper).
Do not try to restore an old ECS checkpoint into the new kernel. Its old
`assets`/`reflect` Cargo feature configuration also has no equivalent on the
new package; remove those feature requests when porting (not when pinning).

New published packages are `rig-harness`, `rig-tools`, `rig-harness-protocol`
and the plugins `rig-basics`, `rig-tui`, `rig-inspect`, `rig-steel`.
The launcher is installed with `cargo install rig`; the plugin guide is
[crates/rig-harness/PLUGINS.md](crates/rig-harness/PLUGINS.md).
`rig::harness_protocol` is not a facade export: depend on
`rig-harness-protocol` directly, or use `rig_harness::harness_protocol` inside
a harness plugin.

### 16. Plugins written against pre-release harness builds

These names were never a stable 0.44 harness API. They matter if you adopted
the coding agent before the tag. Ten plugin crates are now modules of
`rig-basics`. Update Cargo dependencies and `plugins.toml`, not only imports:

```toml
# before
[[plugin]]
crate = "rig-compaction"
plugin = "rig_compaction::CompactionPlugin"
# after
[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::compaction::CompactionPlugin"
```

| Old crate/path | Final 0.45 module in `rig_basics` |
| --- | --- |
| `rig_activity` | `activity` |
| `rig_coding_tools` | `coding_tools` (remote configuration in `coding_tools::remote`) |
| `rig_compaction` | `compaction` |
| `rig_login_chatgpt` | `login_chatgpt` |
| `rig_models::ModelsPlugin`, `DefaultsPlugin` | `models::ModelsPlugin`, `models::defaults::DefaultsPlugin` |
| `rig_print`, `rig_harness` headless APIs | `print` |
| `rig_reload::ReloadPlugin`, `PluginsPlugin` | `reload::ReloadPlugin`, `reload::plugins::PluginsPlugin` |
| `rig_sessions::SessionsPlugin`, `AgentsPlugin` | `sessions::SessionsPlugin`, `sessions::agents::AgentsPlugin` |
| `rig_subagents` | `subagents` |
| `rig_telemetry` | `telemetry::{usage, effect_log, diagnostics}` |
| harness `ApprovalPlugin`, `NeedsApproval`, `PreviewPlugin`, `CheckpointPlugin` | `approval::{ApprovalPlugin, NeedsApproval}`, `preview::PreviewPlugin`, `checkpoint::CheckpointPlugin` |
| harness session picker | `session_picker::{SessionPicker, SessionPickerPlugin}` |
| `rig_basics::DefaultPromptPlugin` | `prompt::DefaultPromptPlugin` |

The launcher no longer rewrites old `rig_<crate>::` paths. Run
`rig plugin check` and update each entry. There is no `rig-splits`; remove it
and use `/fork`, `/agents` and rewind. The session picker is opt-in in the final
default template. Approval, preview and checkpoint are opt-in too.

Other source changes for early plugins:

- `Home::auth(provider)` becomes `Home::auth().join(format!("{provider}.json"))`.
- `PerCall` and `per_call::{per_call, base}` are removed; use `PerCallTools`.
  `AppPerCallExt` and `CallPlace` remain in the harness prelude.
- `ToolOptions::rules` is removed. Put guidance in the tool description.
  `[context]`, `/context`, `ContextEntry`, `ContextLimits` and
  `RefreshProjectContext` are gone; use `PromptSection` or a note.
- `Writer::flush` / `SessionLog::flush` return `Result<(), SessionError>`;
  a changing call is refused if its recovery mark cannot be saved.
- Saved-value upgrade registries, `RewriteSession` and `SessionRewritten` are
  removed. `LastUsage` holds an `Anchor`; session UI columns live in
  `rig_basics::sessions::agents`.
- `ESSENTIAL_PLUGINS`, `RESERVED_COMMANDS`, `env::AGENT_ONLY` are slices.
  `StartReason` is non-exhaustive. Attribution names are `subagents`, `print`
  and `reload`, not their former crate names.
- `subagents::Request` loses `batch`; `SpawnSubagent` adds
  `environment: Option<String>`. A message to a requester is not a report;
  only the final answer completes the request.
- `AppHeadlessExt`, `KeepRunning`, `HeadlessEnded`, `HeadlessTurnEnded` live in
  `rig_basics::print`. `HeadlessEnded` has `code: u8`;
  `HeadlessTurnEnded` has no `failed` field. `print::Request`, `JsonIn` and
  `JsonOut` are private.

### 17. Tool, terminal and driver integrations

`rig-tools` is new in 0.45. For plugins built against its early versions:

- `Read`, `Edit` and `Write` literals add `seen: Seen` (share the same value
  to cache whole-file reads/writes); `Shell` adds `quiet: Duration`.
- `Process::run(deadline, sink)` becomes `run(deadline, quiet, sink)`; pass
  `None` for no quiet timeout. Handle `Ended::Quiet`. Process state also
  records `through`, `remote`, `forced`.
- `numbered` / `read_lines` become `show_lines`; tool text has no line-number
  prefix. Read `LinesShown` and `FilesChanged` for structured facts rather
  than parsing tool replies. `workspace::Excerpt` adds `whole`; use
  `Excerpt::of` or set `whole: None` when unavailable.
- `rig_core::tool::KeptEnd::End` becomes `KeptEnd::Ends`.
- Configure remote hosts with `[remote.hosts] NAME = ["ssh", "user@host"]`.
  The transport takes a shell line as its last argument; with koh use
  `["koh", "exec", "HOST", "--sh"]`, not `"--"`.

The old corpus JSON-stream protocol is replaced by built-in `rig --json`.
Every input line needs `"v":1`; use `send`, `stop`, `answer` (choice ids) and
`state`, not `prompt`, `interrupt`, `set_model` or index-based answers.
The output event vocabulary also changes. `rig -p --json` is one-shot;
`rig --json` ends on stdin EOF. Use `rig help-agents` for the versioned
protocol. `help-agents` is reserved; rename a conflicting plugin command.

For `rig-tui` integrations, `markdown::render(text)` becomes
`render(text, &theme, None)` (or `Some(width)`), and `render_marked` takes
`(text, marks, &theme, width)`. Hand-built `MessageView`s add `width` and
`theme`. Draw panels with `theme.panel(title)` / `theme.raised(title)`,
selections with `theme.selection()`, and tool output prefixes with
`OUTPUT_INDENT`. The default `StatusLine` now belongs to `tui:input`, not a
`tui:status` panel. `ToolCallView::failed` includes nonzero process exits.

---

## 0.43 → 0.44

0.44 had 68 merged PRs, not counting the release PR. Five themes account for most of the breaking changes:

- **Typed generation options.** Reasoning, caching, service tier and the other common knobs are now typed and portable. Every provider wire either maps each one or refuses it.
- **Assistant history keeps provenance.** Each assistant turn records where it came from and how it stopped.
- **Unknown finish reasons fail the run** instead of being silently dropped.
- **Tool-call arguments stream again.** They arrive in fragments, as the provider sends them.
- **The typed provider request and reply structs are gone**, replaced by typed `extension` options and extras.

Several items changed more than once between 0.43.0 and 0.44.0, for example the malformed-call limit and the setters for accepting unknown finish reasons. This section describes only the move from the 0.43.0 form to the 0.44.0 form. Unless marked as transcribed, every 0.44 code block was compiled against `v0.44.0` with the default `rig` features; inline forms for rig-bedrock, rig-gemini-grpc, rig-ecs and the vector-store crates were checked against the source.

### Upgrading at a Glance

| 0.43 | 0.44 |
|---|---|
| `agent.prompt(p).await?.output` | `agent.prompt(p).await?.output()` (an owned `String`) |
| `history.extend(response.messages.unwrap_or_default())` | `history.extend(response.messages)` |
| `Err(StreamingError::…)` from a streamed run | `Err(PromptError::…)`, the same error an awaited run returns |
| `PromptError::{CompletionError, MemoryError, MaxTurnsError, PromptCancelled}` | `PromptError::{Provider, Memory, MaxTurns, Cancelled}` (and add a `_` arm) |
| `MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { tool_result })` | `MultiTurnStreamItem::ToolResult { tool_result }` |
| one `StreamEvent::Arguments` holding a call's whole JSON | one `Arguments` fragment per provider delta; join them per part, or take the call from its `End` |
| `Message::Assistant { id, content }` | `Message::Assistant(AssistantMessage { content, origin, stop, .. })`; build with `Message::assistant(text)` |
| `CompletionRequest { model, chat_history, … }` | `CompletionRequest::new(prompt)` with its setters |
| `Usage { input_tokens: Some(3), ..Usage::default() }` | `Usage::new().input_tokens(3)` |
| Anthropic `with_automatic_caching()`; OpenAI Chat / Bedrock `with_prompt_caching()` | `.cache(CacheRetention::Short)` on the request or agent |
| typed provider structs (`openai::completion::Message`, `responses_api::*`, `gemini_api_types::*`, …) | `providers::<p>::extension::{…Options, …Extras}` with `provider_option(..)` and `extras_lossy::<P>()` |
| `streamed.raw["response_id"]` | `streamed.raw["id"]`: a streamed `raw` is now the unary document |
| `ToolDefinition { name: "add".to_string(), .. }` | `ToolDefinition { name: ToolName::new("add")?, .. }` |
| `agent.into_tool()` / `DynamicTool::from(mcp_tool)` | `agent.into_tool()?` / `DynamicTool::try_from(mcp_tool)?` |
| `UserContent::audio` / `video` / `document` | `UserContent::audio_base64` / `video_base64` / `document_text` (`document_base64` for PDFs) |
| `for (score, id, doc) in index.top_n(req).await?` | `for result in index.top_n(req).await?` (`result.score`, `.id`, `.document`) |
| `Ollama::completion(m)` (native `/api/chat`) | `Ollama::native_completion(m)`; `completion(m)` now targets `/v1` |
| `AnthropicConfig::with_dialect(key, &dialect)` | `AnthropicConfig::with_key(&dialect, key)` |
| `GeminiGrpc::from_env()` / `from_val(key)` | `GeminiGrpc::from_env().await?` / `GeminiGrpc::new(key).await?` |
| `response.provider` / `.model` / `.message_id` | `response.provider()` / `.model()` / `.response_id()` |
| `reasoning.display_text()` | `reasoning.text` |
| persisted `AgentRun` (format 1) | not loadable: `RUN_FORMAT` is 2 |

### Breaking Changes and Migration Guide

Ordered roughly by how many users each change affects.

#### Prompting an agent: results and errors

**`PromptResponse::output` is a method** ([#2690](https://github.com/0xPlaygrounds/rig/pull/2690)). It returns an owned `String` derived from `content`, so the text and the content can't disagree, and the serialized response no longer has an `output` key.

```rust
// 0.43
let text = agent.prompt("hi").await?.output;
history.extend(response.messages.unwrap_or_default());

// 0.44
let response = agent.prompt("hi").await?;
let text: String = response.output();
history.extend(response.messages);
```

If you also move other fields out of the response, such as `messages`, call `output()` first.

**`PromptResponse::messages` is a `Vec<Message>`** that every run fills, and `messages()` returns `&[Message]` ([#2686](https://github.com/0xPlaygrounds/rig/pull/2686)). A hand-built `PromptResponse` now has empty `messages` rather than `None`. A serialized response with `"messages": null` no longer deserializes; omit the field instead.

**`PromptResponse::with_content` is removed**, and `with_output_tool_calls` is no longer public. Set the public `content` field instead: `let mut r = PromptResponse::new("", usage); r.content = parts;`.

**`TypedPromptResponse` carries the accepted attempt's transcript** in `messages`, so typed and extractor runs can be continued.

**`stream_to_stdout` returns the run's error.** It now returns `StreamToStdoutError` instead of printing the error and returning an empty response.

**One error type for both run paths** ([#2644](https://github.com/0xPlaygrounds/rig/pull/2644)). `StreamingError` is removed: streamed runs (`StreamingResult`, `AgentRunner::stream`) now fail with `PromptError`. Its variants have dropped their stutter, and it is `#[non_exhaustive]` ([#2735](https://github.com/0xPlaygrounds/rig/pull/2735)):

```rust
// 0.43
match item {
    Err(StreamingError::Completion(e)) | Err(StreamingError::Prompt(PromptError::CompletionError(e))) => {}
    Err(StreamingError::Prompt(PromptError::MaxTurnsError { .. })) => {}
    // ...
}

// 0.44 (the same arms match a streamed item)
match agent.prompt("hi").await {
    Ok(_) => {}
    Err(PromptError::Provider(error)) => eprintln!("provider: {error}"),
    Err(PromptError::MaxTurns { .. }) => eprintln!("out of turns"),
    Err(PromptError::Cancelled { .. }) => eprintln!("cancelled"),
    Err(other) => eprintln!("{other}"),
}
```

- Renames: `CompletionError` → `Provider`, `MemoryError` → `Memory`, `MaxTurnsError` → `MaxTurns`, `PromptCancelled` → `Cancelled`, and `PromptError::prompt_cancelled(..)` → `cancelled(..)`.
- On an awaited run, a provider failure relayed over the effect bus is now `PromptError::Report(report)`, as it already was on streamed runs, instead of `CompletionError(ProviderError::Relayed(report))`.
- Structured output: `StructuredOutputError::PromptError` → `Prompt`, and `DeserializationError(error)` → `Deserialization { output, error }`, which keeps the model output that failed to parse.
- Messages no longer repeat themselves or carry type-name prefixes such as `CompletionError:`. A cancellation reads `the run was cancelled: <reason>`, and an exhausted budget reads `reached the max turns limit of <n>`. Match on variants, not on display text.

**Output-tool runs:** in output-tool mode, `content` (and so `output()`) holds only the serialized structured output; read the model's prose from `messages`.

**Typed runs and extractors** ([#2670](https://github.com/0xPlaygrounds/rig/pull/2670), [#2690](https://github.com/0xPlaygrounds/rig/pull/2690)):

- `TypedRun<T>` is now an alias for `AgentRunner<TypedOutput<T>>`, and `AgentRunner` is `AgentRunner<O = ()>`. Code that names either type keeps compiling. A trait implemented for both `AgentRunner` and `TypedRun<T>` must now be implemented for `AgentRunner<O>`.
- `ExtractorBuilder::tool_choice` is removed. Extraction always forces the `submit` tool.
- `TypedRun::retries` and `ExtractorBuilder::retries` take `usize` instead of `u64`.
- `From<serde_json::Error> for StructuredOutputError` is removed, so `?` on a `serde_json::Error` no longer converts. Map it: `.map_err(|error| StructuredOutputError::Deserialization { output: text.into(), error })?`.
- `TypedPromptResponse` gains the public `messages` field, which breaks struct literals in test fakes; add `messages: vec![]`.

#### A turn with an unknown finish reason fails the run

[#2726](https://github.com/0xPlaygrounds/rig/pull/2726), [#2733](https://github.com/0xPlaygrounds/rig/pull/2733), [#2734](https://github.com/0xPlaygrounds/rig/pull/2734).

In 0.43, a text answer with an unknown finish reason was returned as a success. The next turn then silently dropped it from history. In 0.44, a turn that ends in `StopReason::Error` fails the run with an error naming the raw reason, even when it holds text. That covers an unknown finish reason, text cut by a content filter, and a reported failure.

To accept unknown reasons as a normal stop, opt in at whichever level fits:

```rust
let agent = AgentBuilder::new(model).accept_unknown_finish_reasons(true).build();
// per run:      agent.prompt("..").accept_unknown_finish_reasons(true)
// per request:  CompletionRequest::new("..").accept_unknown_finish_reasons(true)
// rig-ecs:      world.entity_mut(agent).insert(AcceptUnknownFinishReasons(true));   (transcribed)
```

`CompletionRequest`, `RunSpec`, `PreparedRequest` and `rig_ecs::policy::RequestGraph` each gain a public `accept_unknown_finish_reasons` field. `CompletionResponse::accepts_unknown_finish_reasons()` reports the choice. An unknown reason is also logged with `tracing::warn!`.

#### Streaming: tool-call arguments and tool results

**Tool-call arguments stream as they arrive** ([#2721](https://github.com/0xPlaygrounds/rig/pull/2721)). This reverses 0.43's "tool calls stream whole".

- A call's `StreamEvent::Arguments` is now one fragment per provider delta. Joined together, a call's fragments are its argument JSON.
- `End` still carries the final call and its id.
- `StreamEvent::Start` gains `name: Option<ToolName>`: `Some(tool)` for `PartKind::ToolCall`, `None` for other kinds.

```rust
// 0.43: one Arguments per call, holding the whole JSON.
StreamEvent::Arguments { json, .. } => show(json),

// 0.44 (illustrative; `finish` is yours): fragments, joined per part; parse_partial_arguments renders the incomplete JSON.
StreamEvent::Start { part, kind: PartKind::ToolCall, name } => {
    calls.insert(part.index(), String::new());
}
StreamEvent::Arguments { part, json } => {
    let text = calls.entry(part.index()).or_default();
    text.push_str(&json);
    println!("so far: {:?}", parse_partial_arguments(text));
}
StreamEvent::End { content: AssistantContent::ToolCall(call), .. } => finish(call),
```

- **Hooks:** `on_tool_call_delta` fires once per fragment. `ToolCallDelta` drops `call_id` and gains `aggregated`, the argument text so far. Correlate calls by `part`, and take the id from the call's `End` or from `MultiTurnStreamItem::ToolCall`.
- **Exhaustive matches:** code that matches `StreamedTurnEvent` exhaustively must handle the new `EmitToolCallDelta`. `HoldToolCall` now applies only to calls naming a tool the turn doesn't allow.
- **Building events by hand:** code that builds `StreamEvent::Start` (test fixtures, relays) must add `name`. Serialized transcripts with a nameless tool-call start no longer parse.
- **Anthropic:** tool definitions now carry `eager_input_streaming: true`. For an Anthropic-compatible gateway that rejects that field, set `model.wire = model.wire.with_tool_input_streaming(ToolInputStreaming::BetaHeader)` (or `Off`). Struct literals of `anthropic::Messages` need the new `tool_input_streaming` field (`Quirks` was already `#[non_exhaustive]`).

**Streamed tool results** ([#2686](https://github.com/0xPlaygrounds/rig/pull/2686)):

```rust
// 0.43
MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { tool_result }) => { .. }
// 0.44
MultiTurnStreamItem::ToolResult { tool_result } => { .. }
```

`StreamedUserContent` is removed. The serialized item changes from `{"type":"streamUserItem",…}` to `{"type":"toolResult",…}`. `MultiTurnStreamItem` is `#[non_exhaustive]`, so matches need a `_` arm.

#### Assistant history and messages

[#2713](https://github.com/0xPlaygrounds/rig/pull/2713), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750). Assistant history is now item-shaped.

- **Assistant messages.** `Message::Assistant { id, content }` is now `Message::Assistant(AssistantMessage)`.
  - `AssistantMessage` holds `content`, the turn's `origin` (which provider and model wrote it) and its `stop`.
  - Each block keeps the provider item it came from. Content with no canonical slot is kept as `AssistantContent::Opaque` instead of failing the reply or vanishing.
- **Building and reading turns.**
  - Build a turn with `Message::assistant(text)`, or `AssistantMessage::new(content).with_origin(o).with_stop(s)`.
  - Read a reply's turn with `CompletionResponse::message()`. `message_id` is now `response_id()`.
- **Persisted history.** Assistant turns stored by 0.43 that hold tool calls or reasoning don't load; plain text turns still do (the old `id` is ignored).
- **Blocks.**
  - `Reasoning { text, redacted, native }` replaces `ReasoningContent`, and a block's provider data is `native_item()`.
  - `AdditionalParams` on assistant blocks is gone.
  - A provider call id serializes as `{"provider": "<id>"}` instead of `{"provider": {"call_id": …, "item_id": …}}`.
- **Tool-call arguments are always a JSON object.**
  - `ToolFunction::arguments` is a `serde_json::Map`. Use `arguments_value()`, `ToolFunction::new` or `ToolFunction::parse`.
  - Any other text is kept in `invalid_arguments`, and such a call is answered with an error result.
  - `ToolResult` gains `is_error`, so struct literals need the field.
- **Stops.** `FinishReason::Other` and `ContentFilter` end a turn in `StopReason::Error`. `CompletionResponse` gains `error` and `aborted` (and `origin`, see below).
- **Removed:** `IfMalformed`, `MalformedToolInput`, `ErrorDetail`, `DuplicateCallId` and the `require_non_empty*` helpers.
- **Rollbacks.** Runtimes build rollback turns with `AssistantMessage::rolled_back` or `::aborted`, and fold recorded events with `streaming::delivered`.
- **Request documents.** `Operation::prepare` (replacing `validate`) folds request documents into the first user message.

#### Core types are built with constructors

[#2735](https://github.com/0xPlaygrounds/rig/pull/2735), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750).

Outside rig-core, these types can no longer be built with a struct literal or `..base`:

| 0.43 | 0.44 |
|---|---|
| `CompletionRequest { model: None, chat_history, tools, max_tokens: Some(64), … }` | `CompletionRequest::from(chat_history).tools(tools).max_tokens(64)`; `model: Some(m)` → `.model(m)` |
| `CompletionRequest { tools: vec![], ..base }` | `let mut r = base; r.tools = vec![];` |
| `Usage { input_tokens: Some(3), output_tokens: Some(2), ..Usage::default() }` | `Usage::new().input_tokens(3).output_tokens(2)` |
| `Text { text, additional_params: None }` | `Text::new(text)` |

- Patterns on these types need `..`, for example `Text { text, .. }`.
- These enums are `#[non_exhaustive]`, so a `match` on any of them needs a `_` arm:
  - rig-core: `FinishReason`, `message::StopReason`, `AssistantContent`, `ProviderError` and `providers::openai::wire::BodyRewrite`.
  - rig-agent: `PromptError`, `MultiTurnStreamItem`, `StreamedTurnEvent` and `InvalidToolCallAction`.
- `ProviderError` also gains the `UnsupportedOption(_)` variant (see the next section).
- `Usage` (and `rig_ecs::agent::Usage`) no longer implements `Eq`, because `Usage::cost` holds `f64`. Drop `Eq` from any type that derived it through `Usage`.

```rust
let request = CompletionRequest::new("hi").max_tokens(64);
let usage = Usage::new().input_tokens(3).output_tokens(2);
let text = Text::new("hello");
let message = Message::assistant("hello");
```

#### Responses, reasoning and content blocks

[#2713](https://github.com/0xPlaygrounds/rig/pull/2713).

- **`CompletionResponse` fields.** `provider`, `model`, `response_id` and `message_id` are gone; the new `origin: Origin` field carries them. Use `response.provider()`, `response.model()` and `response.response_id()` (both `message_id` and `response_id` map to `response_id()`). `CompletionResponse::new` takes an `Origin` as its third argument instead of a provider string.
- **`Reasoning` helpers.** `Reasoning::{new_with_signature, sealed, with_id, multi, redacted, encrypted, summaries, display_text, first_text, first_signature, encrypted_content}` and the `Issuer` and `Sealed<T>` types are removed. `AssistantContent::Reasoning` holds a `Reasoning` directly. Read the text from `reasoning.text`; signatures and encrypted payloads live in the provider item (`AssistantContent::native_item()`). `AssistantContent::reasoning(issuer, text)` is now `AssistantContent::reasoning(text)`.
- **`additional_params` on content blocks.** The field is removed from `Image`, `Audio`, `Text` and `ToolCall`, which are user blocks as well as assistant blocks. On `Document` and `Video` it is now `Option<serde_json::Value>`. The `AdditionalParams` type is removed.
- **New fields on literal-built structs.** `Image::native`, `ToolCall::native` and `ToolFunction::invalid_arguments` break struct literals: add `native: None` / `invalid_arguments: None`, or use `ToolCall::new`, `ToolFunction::new` / `parse` and the `UserContent::image_*` constructors.
- **Tool-call identity.** `ToolCall::{signature, with_signature, with_additional_params, from_dual_wire}`, `AssistantContent::tool_call_with_call_id` and `CallId::from_dual_wire` are removed. Build calls with `ToolCall::new(call_id, function)`, and put provider data in `native`, for example `AssistantContent::ToolCall(ToolCall::new(call_id, function)).with_native(json)`. `ProviderCallId` is an opaque newtype: `ProviderCallId::new(id)` and `.as_str()` replace the `call_id` / `item_id` fields, and `with_item_id` is removed.
- **Smaller removals.** `ErrorReport::detail` and `with_detail`; `message::{EMPTY_RESPONSE_ERROR, canonical_streamed_choice, keys_lost_in_round_trip, non_empty, optional_additional_params, ordered_assistant_content}`; `Message::replays_to` and `CompletionRequest::replayable_to` (replay is `history::adapt`). `transcript::tool_result_output` takes `&rig_core::tool::ToolResult` (instead of `ToolOutput`) as its third argument. `streaming::PartKind` gains `Opaque`, so exhaustive matches need an arm.

#### Removed model constants

[#2713](https://github.com/0xPlaygrounds/rig/pull/2713). Retired model ids are gone; pass the id as a string literal if you still need one.

- **Together:** 91 of 99 constants are removed. The kept ones are `LLAMA_3_8B_CHAT_HF`, `LLAMA_3_8B_INSTRUCT`, `LLAMA_3_70B_INSTRUCT`, `LLAMA_2_70B_CHAT_TOGETHER`, `MIXTRAL_8X7B_INSTRUCT_V0_1`, `LLAMA_3_1_8B_INSTRUCT_TURBO`, `LLAMA_3_1_70B_INSTRUCT_TURBO` and `QWEN2_5_72B_INSTRUCT_TURBO`.
- **Vertex AI:** `completion::{GEMINI_1_5_PRO, GEMINI_1_5_FLASH, GEMINI_1_5_PRO_LATEST, GEMINI_1_5_FLASH_LATEST, GEMINI_2_0_FLASH_EXP, GEMINI_2_5_FLASH_LITE, GEMINI_2_5_PRO}` are removed; only `GEMINI_2_5_FLASH` remains. Use `"gemini-2.5-pro"` and so on.
- **Bedrock:** 16 non-completion constants leave `completion::`. Image models such as `AMAZON_NOVA_CANVAS` and `STABILITY_*` are in `image::`. Embedding models remain in `embedding::` (`AMAZON_TITAN_EMBED_TEXT_V1`, `AMAZON_TITAN_EMBED_IMAGE_V1`, `AMAZON_TITAN_EMBED_TEXT_V2_0`, `COHERE_EMBED_*_V3`), where 0.43 already re-exported them; only the `completion::` spellings are gone. `AMAZON_NOVA_REEL_V1_0`/`V1_1`, `AMAZON_NOVA_SONIC`, `AMAZON_RERANK_1_0`, `COHERE_RERANK_V3_5`, `LUMA_RAY_V2_0` and `TWELVELABS_PEGASUS_V1_2` have no replacement constant.

#### Generation options replace provider caching setters

[#2750](https://github.com/0xPlaygrounds/rig/pull/2750).

**New: typed, portable options.** `GenerationOptions` holds `reasoning`, `cache`, `service_tier`, `verbosity`, `parallel_tool_calls`, `top_p`, `seed`, `stop` and `on_unsupported`.

- `CompletionRequest`, `AgentBuilder` and `AgentRunner` have a setter for each of them, such as `.reasoning(Effort::High)`, `.cache(CacheRetention::Long)` or `.seed(1)`, beside `options(..)`.
- Every completion wire maps each option to its provider's JSON, or refuses it.
- Under the default policy (`OnUnsupported::Error`), a refusal is `ProviderError::UnsupportedOption` and nothing is sent. `.on_unsupported(OnUnsupported::Ignore)` skips the option with one warning instead.
- A request that sets no option at all is not checked: it is sent as built, and the provider decides.

**Removed caching setters:**

| Removed | Use |
|---|---|
| Anthropic `Messages::with_automatic_caching()` and the `automatic_caching` field | `.cache(CacheRetention::Short)` on the request, or `AgentBuilder::cache(..)` |
| Anthropic `with_automatic_caching_1h()` and `automatic_caching_ttl` | `.cache(CacheRetention::Long)` |
| `openai::wire::Chat::with_prompt_caching()`, its `prompt_caching` field, and `OpenAiWire::with_prompt_caching()` | `.cache(CacheRetention::Short)`; on a dialect that can't cache on request, where the setter did nothing, this is now `UnsupportedOption` |
| Bedrock `Converse::with_prompt_caching()` and its `prompt_caching` field | `.cache(CacheRetention::Short)`, or `Long` for a 1 h TTL; a checkpoint that can't follow a reasoning turn is now `UnsupportedOption` (add `.on_unsupported(OnUnsupported::Ignore)` to skip it, as the setter did silently) |

Anthropic's `with_prompt_caching()` and `with_static_prefix_cache_ttl(..)` are unchanged.

**Request bodies merge in one order:** the wire's encoding, then mapped options, then provider options, then `additional_params`.

- Objects merge key by key, other values replace, and a `null` is sent (except a top-level `generationConfig: null` on Gemini, which is still ignored).
- `additional_params.tools` is appended to Rig's tools.
- So `additional_params` now overrides typed fields on Gemini, Interactions and OpenAI Responses, and an object in it merges into the wire's object instead of replacing it. See [Behavior Changes](#behavior-changes) for the per-provider list.

```rust
use rig::completion::{CacheRetention, Effort, OnUnsupported};

let agent = AgentBuilder::new(model)
    .reasoning(Effort::High)
    .cache(CacheRetention::Long)
    .on_unsupported(OnUnsupported::Ignore)
    .build();
```

#### Typed provider types are replaced by extensions

[#2713](https://github.com/0xPlaygrounds/rig/pull/2713), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750).

Every wire now builds requests and reads replies as JSON, so the typed provider request and reply structs are removed. That includes:

- OpenAI Chat readback types and `ToolChoice`, and the Responses parameter types.
- Anthropic's typed replies and `CacheControl`.
- Gemini and Interactions API types, plus `gemini::completion::gemini_api_types::{tool_parameters_to_schema, flatten_schema}`.
- The OpenRouter and Venice parameter types.
- Bedrock's `ConverseRequest` fields.

What replaces them:

- **Sending provider fields:** typed `Options` types in `providers::<p>::extension` (and `rig_bedrock::extension`, `rig_vertexai::extension`, `rig_gemini_grpc::extension`, `rig_candle::extension`). Set them with `request.provider_option(options)`, `AgentBuilder::provider_option` or `AgentRunner::provider_option`. Each wire sends only its own provider's entry. Raw JSON in `additional_params` still works.
- **Reading provider reply fields:** use `response.extras::<P>()` (an `Option<Result<P::Extras, _>>`) or `extras_lossy::<P>()`, where `P` is the provider's marker (`OpenAiExt`, `AnthropicExt`, `OpenRouterExt`, …). Both read the same fields from unary and streamed replies, instead of indexing `raw`.
- **Coming from 0.43's typed builders:** the builders on OpenRouter, Anthropic, Responses, Venice, Ollama, Cohere and Bedrock come back as serialize-only `Options` types. For example, `ProviderPreferences` becomes `OpenRouterOptions::provider`, and `VeniceParameters` becomes `VeniceOptions::venice_parameters`. Its `disable_thinking` flag is now `Reasoning::Off`.
- **Provider keys** (`"gcp.gemini"`, `"aws_bedrock"`, `"azure.openai"`, …) are stable API from 0.44.

```rust
use rig::providers::openrouter::extension::{ModelFallbacks, OpenRouterExt, OpenRouterOptions};

let fallbacks = ModelFallbacks::new(["openai/gpt-5.5", "google/gemini-3.8-flash"])?;
let request = CompletionRequest::new("Name three Rust web frameworks.")
    .provider_option(OpenRouterOptions::new().models(fallbacks));
let response = model.call(request).await?;
let extras = response.extras_lossy::<OpenRouterExt>();
println!("served by {:?} for {:?} credits", extras.provider, extras.cost);
```

**Bedrock guardrail:** `Converse::with_guardrail` and the `Converse::guardrail` field are removed. The guardrail is now sent on streamed requests too, where it used to be dropped silently. `GuardrailTrace` is now rig's own `rig_bedrock::extension::GuardrailTrace`, not the AWS SDK type.

```rust
// transcribed from #2750
// 0.43
let wire = Converse::new(model).with_guardrail("g1", "DRAFT", GuardrailTrace::Enabled);
// 0.44
use rig_bedrock::extension::{BedrockOptions, Guardrail, GuardrailTrace};
let guardrail = Guardrail::new("g1", "DRAFT").trace(GuardrailTrace::Enabled);
let request = request.provider_option(BedrockOptions::default().guardrail(guardrail));
```

**Header options:** Anthropic betas and the Copilot intent have no typed channel yet; they stay on `AnthropicConfig::with_beta` and `CopilotWire::with_intent`. ChatGPT still sends a fresh `session_id` per request. On Bedrock, Anthropic betas are a typed body option, `BedrockOptions::anthropic_beta`.

**Typed reply types are removed.** 0.43's typed reply structs are gone, with no replacement type: `deepseek`, `mistral`, `openrouter`, `llamacpp` and `venice` `CompletionResponse`/`Usage`/`Choice`, the `openai::completion` and `responses_api` reply types, Anthropic's `CompletionResponse` and `Usage`, and Gemini's `GenerateContentResponse`. Code that deserialized `CompletionResponse::raw` into one of them should read the fields through `extras::<P>()` (for example `DeepSeekExt`, `MistralExt`, `OpenRouterExt`, `LlamaCppExt` for `Timings`, `VeniceExt` for `VeniceCost`), or index `raw` as JSON. ([#2661](https://github.com/0xPlaygrounds/rig/pull/2661) and [#2664](https://github.com/0xPlaygrounds/rig/pull/2664) reshaped the DeepSeek, Mistral and OpenRouter types after 0.43; [#2713](https://github.com/0xPlaygrounds/rig/pull/2713) then removed them, so skip those intermediate forms.)

#### A streamed reply's `raw` is the unary document

[#2750](https://github.com/0xPlaygrounds/rig/pull/2750).

In 0.43, each provider wrote its own summary as a streamed reply's `raw`. In 0.44, a streamed `raw` is the API's unary response document, rebuilt from the stream. A unary `raw` is unchanged, except that ChatGPT's `output` is now filled.

| API | 0.43 streamed `raw` | 0.44 streamed `raw` |
|---|---|---|
| OpenAI Chat and every dialect | `{usage, finish_reason, response_id, model, logprobs, additional_params}` | the `chat.completion` body: `raw["id"]`, `raw["choices"][0]["finish_reason"]`, `raw["choices"][0]["message"]`, `raw["usage"]` |
| Anthropic Messages and dialects | `{usage, stop_reason, stop_sequence, message_id, model}` | the `Message`: `raw["id"]` (was `message_id`), `raw["content"]`, `raw["stop_reason"]`, `raw["usage"]` |
| Gemini GenerateContent (REST, gRPC) | a snake_case summary (REST), the last chunk (gRPC) | the `GenerateContentResponse`: `raw["usageMetadata"]`, `raw["candidates"][0]["finishReason"]`, `raw["modelVersion"]`, `raw["responseId"]` |
| Gemini Interactions | `{usage, interaction, model_version}` | the interaction resource, with `steps` rebuilt |
| OpenAI Responses | the terminal `response` | the same, except that ChatGPT's `output` is filled and Copilot's `copilot_usage` sits at the top level |
| Cohere native | the `message-end` event | the chat response: `raw["id"]`, `raw["message"]`, `raw["finish_reason"]`, `raw["usage"]` |
| Ollama native | the final `done` record | the `/api/chat` body, with content, thinking and tool calls joined |
| Bedrock Converse | `{messageStart, messageStop, metadata}` | the `ConverseOutput`: `raw["output"]["message"]`, `raw["stopReason"]`, `raw["usage"]`, `raw["metrics"]` |

A stream that fails, or that runs out of frames before the provider ends it, now records the document rebuilt so far as `raw`, where 0.43 recorded `Null`. To know whether the provider ended the reply, check the finish reason, not whether `raw` is non-null.

#### Tools and MCP

**Tool names are `ToolName`** ([#2689](https://github.com/0xPlaygrounds/rig/pull/2689)). Declared tool names are now validated like tool-call names, and deserializing an empty name fails.

```rust
// 0.43
ToolDefinition { name: "add".to_string(), description, parameters }
ToolChoice::Specific { function_names: vec![Add::NAME.to_string()] }
DynamicTool::new("echo", "Echo", schema, callback)
builder.dynamic_tool(agent.into_tool())
let tool: DynamicTool = mcp_tool.into();

// 0.44
ToolDefinition { name: ToolName::new("add")?, description, parameters }
ToolChoice::Specific { function_names: vec![tool_name::<Add>()] }     // transcribed
DynamicTool::new(ToolName::new("echo")?, "Echo", schema, callback)   // transcribed
builder.dynamic_tool(agent.into_tool()?)                              // transcribed
let tool: DynamicTool = mcp_tool.try_into()?;                         // transcribed
```

`DynamicTool::new_with_context` takes a `ToolName` too. `ToolName` compares equal to `str` and `String`. To get a string, use `name.as_str()`, `name.to_string()` or `String::from(name)`. `McpClientHandler` skips a nameless server tool with a warning.

**Tool and MCP failures keep their structure** ([#2687](https://github.com/0xPlaygrounds/rig/pull/2687)):

- A tool call that fails at dispatch, rather than inside the tool, now reports a mapped `ToolErrorKind` (for example `Timeout`, `NotFound` or `Network`) instead of `Other`, and keeps the report's `retryable()`, `code()`, `http_status()` and `is_refusal()` values, which used to be lost. `ToolExecutionError` implements `From<ErrorReport>`.
- `ToolContextError::{Encode, Decode}` carry `source: serde_json::Error` instead of `message: String`. `ToolContextError` no longer implements `Clone`, `PartialEq` or `Eq`, so compare with `.unwrap()` instead of `assert_eq!(…, Ok(..))`.
- `McpClientError::ConnectionError(String)` → `McpClientError::Connection(rmcp::service::ClientInitializeError)`, and `ToolFetchError` → `ToolFetch`.

#### Invalid and malformed tool calls

[#2727](https://github.com/0xPlaygrounds/rig/pull/2727), [#2733](https://github.com/0xPlaygrounds/rig/pull/2733).

The invalid-call hook now also sees tool calls whose arguments aren't a JSON object, before they are answered.

- `InvalidToolCallReason` gains `DisallowedByToolChoice`, which takes the calls `UnknownTool` used to cover when the tool choice disallowed them. It is `#[non_exhaustive]` and no longer implements `Default` (write `InvalidToolCallReason::UnknownTool`).
- `InvalidToolCallContext` is `#[non_exhaustive]`, so it can no longer be built with a struct literal; contexts come from the run. `StreamedInvalidToolCall::reason` is removed: read `InvalidToolCallContext::reason`.
- A hook written for unknown names should check the reason before it acts. An unconditional `Repair` now fails a run that sends malformed arguments:

```rust
// transcribed from #2727
if matches!(context.reason, InvalidToolCallReason::MalformedArguments { .. }) {
    return None; // keep the default feedback
}
```

- **Optional limit:** `max_consecutive_malformed_tool_calls(n)` on `AgentRunner` or `AgentRun`, or `RunSpec::max_consecutive_malformed_tool_calls = Some(n)`. When set, a run fails after `n` consecutive turns with a malformed call; `None` clears the limit. Without it, malformed calls are answered until `max_turns`, as in 0.43.

#### Run specs and persisted runs

- **Persisted runs:** an `AgentRun` persisted by 0.43 is refused: "resume refused: the run is format 1, this rig reads format 2".
- **New fields:** `RunSpec` gains `accept_unknown_finish_reasons`, `max_consecutive_malformed_tool_calls`, `options` and `provider_options`, and `PreparedRequest` gains the same option fields. A literal that ends in `..RunSpec::new()` or `..Default::default()` is unaffected. A literal that lists every field needs `accept_unknown_finish_reasons: false`, `max_consecutive_malformed_tool_calls: None`, `options: GenerationOptions::new()` and `provider_options: ProviderOptions::new()`.
- **`RunSpec::default()` now equals `RunSpec::new()`**, including `augment_output_preamble: true` (0.43's default gave `false`). A spec deserialized without that key also gets `true`. To keep the old value, write `RunSpec { augment_output_preamble: false, ..RunSpec::default() }`.
- **Effect logs:** logs recorded under 0.43 are refused by `check_replayable`, because the stable hash of `RunSpec` changed. Record them again.

#### Media content constructors

[#2685](https://github.com/0xPlaygrounds/rig/pull/2685).

```rust
// 0.43
UserContent::audio(b64, None);
UserContent::document(text, Some(DocumentMediaType::TXT));
// 0.44
UserContent::audio_base64(b64, None);
UserContent::document_text(text, Some(DocumentMediaType::TXT));
UserContent::document_base64(b64, Some(DocumentMediaType::PDF)); // new: base64 documents
```

`UserContent::video` → `video_base64`. If you passed base64 to `UserContent::document`, for example a PDF, switch to `document_base64`: the old call built a text source, which providers handled inconsistently. `DocumentSourceKind::{url, base64, file_id, string}` now take `impl Into<String>`.

#### Vector stores

**Named search results** ([#2688](https://github.com/0xPlaygrounds/rig/pull/2688)):

```rust
// 0.43
for (score, id, doc) in index.top_n::<Doc>(req).await? { /* ... */ }
let first_id = &index.top_n_ids(req).await?[0].1;

// 0.44
for result in index.top_n::<serde_json::Value>(req.clone()).await? {
    println!("{} {} {}", result.score, result.id, result.document);
}
let first = &index.top_n_ids(req).await?[0].id;
```

- A custom `VectorStoreIndex` returns `VectorSearchResult { score, id, document }` and `VectorSearchIdResult { score, id }`.
- `VectorStoreOutput` becomes `VectorSearchResult<serde_json::Value>`, which has the same JSON shape.
- `SearchResult::into_result` (rig-postgres, rig-surrealdb) and agent retrievals return the new types.

**`VectorSearchRequest`** ([#2683](https://github.com/0xPlaygrounds/rig/pull/2683)):

- `additional_params` is removed. No backend ever read it, so delete the call and its `?`.
- `filter()` returns `Option<&F>`. Drop `.as_ref()`, and use `.cloned()` for an owned copy.
- `VectorStoreError::BuilderError(String)` → `SamplesOutOfRange { requested, max }`.

**Companion crates** ([#2662](https://github.com/0xPlaygrounds/rig/pull/2662), [#2698](https://github.com/0xPlaygrounds/rig/pull/2698)):

- **HelixDB:** `HelixDBVectorStore` is no longer generic, and the `HelixDBClient` trait is removed. Write `HelixDBVectorStore` instead of `HelixDBVectorStore<HelixDB>`, and customise transport with `HelixDB::with_client`.
- **MongoDB, SurrealDB, Milvus, ScyllaDB:** the `From`/`TryFrom<Filter<serde_json::Value>>` impls are removed. Use `DynamicSearchFilter::from_dynamic_filter(filter)?`.
- **Postgres:** `PgSearchFilter::like` and `similar_to` take a runtime pattern and bind it as a parameter. **Remove the SQL quotes you used to include.** A pattern that keeps them, like `"'glarb%'"`, still compiles but now matches the quote characters literally, so it finds nothing. `between` binds both bounds and accepts any `T: Into<serde_json::Value>`, floats included.

```rust
// 0.43
PgSearchFilter::like("document->>'name'", "'glarb%'");
// 0.44   (transcribed)
PgSearchFilter::like("document->>'name'", "glarb%");
PgSearchFilter::like("document->>'name'", format!("{prefix}%"));
```

#### Providers

- **Ollama** ([#2730](https://github.com/0xPlaygrounds/rig/pull/2730)). In 0.43, `Ollama::completion` used the daemon's native `/api/chat`. In 0.44 it returns `Model<openai::wire::Chat>` at `{base}/v1`, and the native route is `Ollama::native_completion`. If you send `think`, `keep_alive` or model `options` such as `num_ctx`, move to `native_completion`. On `/v1`, `num_ctx` and `options` are a request error instead of being silently ignored. Set reasoning with `.reasoning(..)` on either route.
- **Cohere** ([#2729](https://github.com/0xPlaygrounds/rig/pull/2729)). In 0.43, `Cohere::completion` returned the native `/v2/chat` wire. In 0.44 it returns `Model<CohereChat>`, which by default sends requests to Cohere's OpenAI Compatibility API. To keep documents, citations and `tool_plan`, opt back in with `model.wire = model.wire.with_route(cohere::ChatRoute::Native)`, or `ChatRoute::Auto` to go native only for requests that carry documents. Chat-only options live on `model.wire.compatibility_api`.
- **Anthropic dialects** ([#2691](https://github.com/0xPlaygrounds/rig/pull/2691), [#2738](https://github.com/0xPlaygrounds/rig/pull/2738)).
  - `AnthropicConfig::with_dialect(api_key, &dialect)` → `AnthropicConfig::with_key(&dialect, api_key)`, matching `OpenAIConfig::with_key`.
  - `xiaomimimo::ANTHROPIC_API_BASE_URL` and the `XIAOMIMIMO` dialect's `base_url` drop their trailing `/v1`. Code that joined paths onto the constant must add `/v1` itself.
  - `with_key` now trims a custom dialect's trailing `/`, `/v1`, `/messages` or `/v1/messages`.
- **Bedrock** ([#2691](https://github.com/0xPlaygrounds/rig/pull/2691), [#2713](https://github.com/0xPlaygrounds/rig/pull/2713)).
  - `client::Builder` is owned, accepts both `region` and `profile_name`, and `build()` is synchronous.
  - `BedrockRuntime::with_profile_name` and `DEFAULT_AWS_REGION` are removed. Use `BedrockRuntime::builder().profile_name("bedrock").region("eu-west-1").build()`.
  - **An unset region now follows the AWS SDK provider chain instead of defaulting to `us-east-1`.** Call `.region("us-east-1")` to keep the old behavior.
  - `types::assistant_content` (`RigAssistantContent`, `map_stop_reason`) and `types::converse_output` (about 45 mirror types) are removed with no replacement. Read Converse replies from `raw` (the `ConverseOutput` JSON) or `extras::<BedrockExt>()`. `streaming::BedrockStreamingResponse` is removed too.
- **Vertex AI** ([#2691](https://github.com/0xPlaygrounds/rig/pull/2691)). `VertexAiClientError` keeps the underlying SDK errors as typed `Arc` sources instead of strings. A client-initialization failure during a completion is now `ProviderError::Request` wrapping the `VertexAiClientError`, not `ProviderError::Provider(String)`.
- **Gemini gRPC** ([#2684](https://github.com/0xPlaygrounds/rig/pull/2684), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750)).
  - `GeminiGrpc::from_env()` is `async` and returns `GeminiGrpcError`. 0.43's version panicked on current-thread runtimes.
  - `GeminiGrpc::new` takes `impl Into<Secret>` and returns `GeminiGrpcError`, and `from_val` is removed.
  - The proto now uses `prost_types::{Struct, Value}` for JSON fields (`FunctionCall::args`, `FunctionResponse::response`, `Part::part_metadata`, …), so add `prost-types = "0.14"` if you build them; `proto::{Struct, Value, ListValue, NullValue}` are removed. About a dozen messages gained fields (`Candidate::grounding_metadata` among them), so end struct literals with `..Default::default()`. `CodeExecutionResult::outcome` and `ExecutableCode::language` are enum `i32`s, `GenerateContentRequest::cached_content` is an `Option<String>`, and `GenerationConfig::response_mime_type` is a `String` (empty when unset).
- **Gemini** ([#2663](https://github.com/0xPlaygrounds/rig/pull/2663), [#2713](https://github.com/0xPlaygrounds/rig/pull/2713)).
  - `operation::ContextCache` moved to `providers::gemini::cached_content::ContextCache`, and `operation::CachedContentFold` is removed. Calls through `Gemini::cached_contents()` are unchanged.
  - Tool schemas go to Gemini unchanged as `parametersJsonSchema`, so the schema converter is gone.
- **Candle** ([#2713](https://github.com/0xPlaygrounds/rig/pull/2713), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750)). The payload of `Generation` is `CandleRequest { request, params }`, and `top_p` and `seed` are now options. `Generation` itself is a struct with `model` and `protocol` fields instead of a unit struct, and loses `Default` and `Copy`; get it from `candle.completion()`.
- **Vertex AI requests.** `completion::VertexRequest` is removed: a custom `Transport<GenerateContent>` takes the SDK's `google_cloud_aiplatform_v1::model::GenerateContentRequest`. `VERTEX_TEXT_EXTRAS_KEY` is removed; read extras with `extras::<VertexExt>()`.
- **Moved wire types.** `ollama::wire::{Chat, Embeddings, Models}` are now `ollama::chat::Chat` (what `native_completion` returns), `ollama::embedding::Embeddings` and `ollama::model_listing::{Models, ListModelEntry}`. `cohere::wire::{Embeddings, ImageEmbeddings}` are `cohere::embeddings::*`, and the native chat wire is `cohere::chat::NativeChat`. `anthropic::wire::{Models, Verify}` are re-exported as `anthropic::{Models, Verify}`, and `copilot::wire::Models` is `copilot::model_listing::Models`.

#### Telemetry

[#2724](https://github.com/0xPlaygrounds/rig/pull/2724), thanks to @mateobelanger. This change compiles silently, so check anything keyed on span names or `gen_ai.operation.name`:

- `chat_streaming` → `chat` with `gen_ai.request.stream = true`. Gemini streams report `generate_content` with the same flag.
- `interactions` / `interactions_streaming` → `chat`.
- The agent's streaming `rig::agent_chat` span is named `chat`.
- `GenAiOperation::{ChatStreaming, Interactions, InteractionsStreaming}` are deprecated. Use `SpanBuilder::new(provider, model, GenAiOperation::Chat).streaming(true)`.

#### rig-ecs

- `rig_ecs::policy::text` and its duplicated helpers moved to `rig_core::structured_output` and `rig_core::transcript` ([#2660](https://github.com/0xPlaygrounds/rig/pull/2660)). For example, `policy::turn_is_empty` → `transcript::is_empty_assistant_turn`, `policy::answer_text` → `transcript::assistant_text_from_choice`, `policy::output_tool_name(&granted)` → `structured_output::output_tool_name(|name| granted.contains(&name))`, and `policy::invalid_peer_results` → `transcript::invalid_call_feedback`.
- Request assembly is one system, `systems::fold_turn`. `gather_turn` and `AssemblyInputs` are removed, so order against `fold_turn` or `RigSet::Assemble`. `recording.delivery(batch, id, kind)` → `recording.delivery(Delivery { batch, id, kind })`, `Recording::keep_events` as a path becomes `|r| r.keep_events()`, and `policy::user_text(text)` → `Message::user(text)` ([#2672](https://github.com/0xPlaygrounds/rig/pull/2672)).
- `save_world`, `load_world` and `Checkpoint::{validate, requirements, to_json, from_json}` return `checkpoint::CheckpointError` instead of `ErrorReport`. Match `DescriptorChanged { key }`, `MissingHandler { key }` and the other variants instead of message text. There is no `From<CheckpointError> for ErrorReport`, so change the error type of functions that `?` these calls. The tool-turn hold error `agent::checkpoint::CheckpointError` is renamed `TurnHoldError` ([#2692](https://github.com/0xPlaygrounds/rig/pull/2692)).
- `policy::RequestGraph` gains `accept_unknown_finish_reasons`, `options` and `provider_options` (borrowed: `&GenerationOptions::new()`). There are new `Options`, `ProviderOptions` and `AcceptUnknownFinishReasons` components ([#2726](https://github.com/0xPlaygrounds/rig/pull/2726), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750)).
- A tool dispatch report marked as a refusal is recorded as `ToolResultStatus::Refused` instead of `Error`.

#### Cargo features and dependencies

- rig-core now enables serde_json's `preserve_order` and `float_roundtrip` ([#2713](https://github.com/0xPlaygrounds/rig/pull/2713)). Cargo unifies features, so every crate in a binary with rig-core gets insertion-ordered JSON maps and exact float round trips.
- rig-gemini-grpc exposes `prost-types` 0.14 in its public API (see Providers).
- MSRV (1.95.0), edition and default features are unchanged.

#### Implementing a provider or a custom wire

[#2713](https://github.com/0xPlaygrounds/rig/pull/2713), [#2750](https://github.com/0xPlaygrounds/rig/pull/2750). These changes are transcribed from the PRs. `crates/rig-core/TYPED_OPTIONS.md` is the design contract.

- **`Operation::prepare`** replaces `validate`. Implement `ReplayTarget`: `api`, `provider`, `model`, `accepts` and `map_options` are required, and the rest have defaults. Pass the wire to `Descriptor::replay`.
- **Options mapping:** a `ReplayTarget` must implement `fn map_options(&self, request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap`. It answers each option with a `Mapping`, for example `top_p: Mapping::of(top_p, |v| Mapping::Send(json!({"top_p": v})))`.
- **Reassembler:** every `Wire` names `type Reassembler`. A non-completion wire uses `wire::document::Unreassembled`. A completion wire implements `Reassemble<Self::Frame>` and `Serves<Completion>` for its document, or reuses one of Rig's (for example `providers::openai::wire::OpenAiReassembler`).
- **Raw and citations:** `out.raw(..)` on `Out<'_, Completion>` no longer compiles; the driver records what the reassembler rebuilds. Set citations through `Out::cite` / `Out::set_citations`.
- **Writer:** 0.43's part methods on `Out<'_, Completion>` (`text`, `push_text`, `close_text`, `reasoning`, `call`, `push_arguments`, `tool_call`, `message_id`, `issued_by`, the `pending_*` family, …) are removed. Write parts with `open`, `push`, `edit`, `finish`, `close`, `whole`, `fragment`, `announce`, `fresh_index`, `finish_open`, `run`, `end_run`, `order_by_index`, `lead` and `restate`. For deltas, apply `operation::merge` inside an `edit`.
- **Serving replies:** `Reply::written` takes the reply's `Origin`, `StreamWriter::finish` takes only the `Finish`, and `EffectRecord` gains `stream_origin`. `serve::Recorder` and `serve::Observe` have a new required `origin` method. `operation::Finish` drops `message_id` and gains `error`.
- **Decoders and runtime turns:** decoder types lose their `'id` lifetime (`MessagesDecoder`, `GenerateContentDecoder`, `ResponsesDecoder`, `OpenAiDecoder`, …), and the Chat decoder types sit in a crate-private module (name them as `<Chat as Wire>::Decoder`). `BodyRewrite::{GroqCompoundTools, Hyperbolic}` are removed. In rig-agent, `ModelTurn::new` and `transcript::assistant_message` take an `AssistantMessage` (`response.head()`) instead of a message id; `ModelTurn`/`StreamedTurn` carry `head: AssistantMessage` instead of `message_id`, `PartialStreamedTurn` holds `{head, content}`, `CompletionCall::message_id` and `surface_malformed_input` are removed, and `StreamedTurnAssembler::finish` and `partial_turn` take only the response. In rig-core, `operation::Turn::origin()` replaces `message_id()`/`provider()`. In rig-ecs, `MessageParts::assistant` takes an `AssistantMessage`, and `Outputs`/`TurnRead` expose `head` instead of `message_id`.
- **Unwind safety:** `wire::Descriptor` and `rig_vertexai::client::VertexAiClientError` are no longer `UnwindSafe`; wrap them in `AssertUnwindSafe` where needed.
- **Registry:** `ProviderId::format()`, `config()` and `api_key_env()` return `Option<_>`, which is `None` only for the catalog-only vendors. `ProviderRef::provider()` returns `Provider` by value.

#### Smaller removals and renames

- `effect::RerankRequest` → `operation::RerankRequest` ([#2656](https://github.com/0xPlaygrounds/rig/pull/2656)).
- `embeddings::ImageEmbeddingResponse` → `EmbeddingResponse`. The Cohere and VoyageAI error-envelope types (`cohere::wire::EmbedReply`, `voyageai::wire::RerankReply`, `voyageai::RerankErrorEnvelope`) are removed ([#2666](https://github.com/0xPlaygrounds/rig/pull/2666)).
- `transcription::NormalizeTranscriptionResponse` is removed. Drop the import: OpenAI's `TranscriptionResponse::normalize()` is an inherent method. Gemini's typed `GenerateContentResponse`, and its transcript normalizer, are removed with no replacement ([#2680](https://github.com/0xPlaygrounds/rig/pull/2680), [#2713](https://github.com/0xPlaygrounds/rig/pull/2713)).
- Test utilities (`test-utils` feature): `MockStreamEvent::MessageId`, `MockStreamEvent::message_id` and `MockTurn::with_message_id` are removed, and `MockStreamEvent::Reasoning`'s `content` is `text`.
- `rig_cassette::http::DirectHeader` is removed. `record_http_interaction` takes any iterator of `(impl AsRef<str>, impl AsRef<str>)` pairs ([#2680](https://github.com/0xPlaygrounds/rig/pull/2680)).

### Other API Additions

- **Model catalog:** `rig_core::catalog`, with `Catalog::{builtin, get, find, resolve, iter, from_json, merge}`, `ModelSpec::validate` and `Pricing::cost`. It holds 1,492 models across 28 vendors.
- **Connecting from the catalog:** `providers::registry::{connect, connect_with}` connect from a catalog row or a `vendor/model` reference. `AgentBuilder::model_spec` checks every call's options against the catalog.
- **Citations:** `Text::citations()`, `Text::cited(..)` and `message::{Citation, Span, Source, SourceLocation, DocumentRange}`.
- **Cost:** `Usage::cost` and `Cost`, in USD.
- **Streaming arguments:** `rig::streaming::parse_partial_arguments` reads incomplete argument JSON into a best-effort object.
- **Provider configuration:** `ProviderConfig::{base_url, with_base_url}`.
- **Tools:** `tool_name::<T>()`.
- **Gemini caching:** `AutoCache::for_model(model)` takes the cached-read price ratio from the catalog.
- **Model constants:** `gemini::completion::GEMINI_3_8_FLASH` and `deepseek::DEEPSEEK_FLASH`.
- **OpenAI-compatible gateways:** `openai::wire::Quirks::done_without_finish_reason`.
- **Cassettes:** `ProviderCassette::{try_start_at, try_finish}`, shape-matched replay (`CassetteSpec::shape_matched()`) and request snapshots.
- **Structured output and transcripts:** `rig_core::structured_output`, and `transcript::{invalid_call_feedback, is_empty_assistant_turn, assistant_text_from_choice}`.
- **SQL filters:** `SqlCondition::{range, between, render_placeholders}`.
- **Test utilities:** `rig_core::test_utils::TraceCapture`.
- **In-memory vector stores** accept documents without `Eq` or `Default`, and closures as id generators.

### Behavior Changes

These compile as before but act differently:

- **Unknown finish reasons and filtered text fail the run** (see above).
- **Ollama `completion` targets `/v1`, and Cohere `completion` defaults to the Compatibility API** (see Providers).
- **Bedrock's builder resolves the region from the AWS SDK chain** instead of defaulting to `us-east-1`.
- **`RunSpec::default()`** sets `augment_output_preamble: true`.
- **Streamed `raw`** is the unary document, and a failed stream's `raw` is the partial document instead of `Null`.
- **`additional_params` overrides typed fields** on Gemini GenerateContent, Interactions and OpenAI Responses, and objects in it merge key by key into the wire's object on every wire.
  - Chat Completions: a raw `response_format` merges over the output schema's.
  - Responses: a raw `text` merges with the structured-output format.
  - Anthropic, Chat Completions, Responses and Cohere: a raw `tools: null` means no tools.
  - ChatGPT: the backend no longer strips `temperature`, `top_p`, `metadata`, `user`, `background` or `max_output_tokens` from `additional_params`.
  - Anthropic: `additional_params.cache_control` is sent as written.
  - A non-object `additional_params` is now an error on every wire (`` `additional_params` must be a JSON object ``). Bedrock merges `additional_params` into `additionalModelRequestFields`.
  - Vertex AI and Gemini gRPC let `additional_params.model` (and Vertex `.contents`) override the request's own.
  - Gemini Interactions: `additional_params.tools` must be an array, and `InteractionResume` refuses a request that sets options or raw tools.
  - Ollama native sends a raw `think` as written, without lowercasing or validating it.
- **Model facts come from the catalog** for models it lists:
  - OpenAI Chat decides between `max_tokens` and `max_completion_tokens` from the catalog's `reasoning` field.
  - Claude Opus 4 and 4.1 default to 32,000 output tokens.
  - A listed model without image input gets the text `(image omitted: model does not support images)` instead of the image. Fifteen rows that got the text now get the image.
- **`Usage::cost` is `Some` for most replies**, so a serialized `Usage` gains a `"cost"` object. A catalog-priced cost is a list-price estimate, not the bill.
- **A cited `Text` serializes a `citations` key**, and no longer equals `Text::new(same text)`. Compare `text.text` instead.
- **Telemetry span names** follow `gen_ai.operation.name` with `gen_ai.request.stream` (see Telemetry).
- **A dispatch-level tool failure** reports a specific `ToolErrorKind` instead of `Other`.
- **Anthropic sends `eager_input_streaming: true`** on every tool. Claude models that bind thinking to the request context get `thinking.block_binding` and the `thinking-binding-controls-2026-08-01` beta in adaptive thinking.
- **Postgres `like` / `similar_to` patterns are bound parameters**, so leftover SQL quotes now match literally.
- **`AgentRunner::stream()` on a resumed run** uses the run's persisted unhandled-tool-call policy, not the runner's.
- **A turn that ends in an error runs none of its tool calls.** A malformed output-tool call is reprompted or fails the run instead of becoming the output.
- **Groq built-in tools** such as `browser_search` go in `tools`, where Groq reads them. **Mistral** sends a forced tool choice as `required`, and **DeepSeek** honors a forced tool choice on `deepseek-chat`.
- **A broken replay fixture** (undecodable body, invalid header or status) now fails `ProviderCassette::start_at` (or `try_start_at` with `CassetteError::InvalidInteraction`) up front, not when that interaction is served.
- **Bedrock `InvokeModel` errors** carry the AWS request id. A mid-stream Bedrock exception with no message reads `Bedrock event stream failed (<ExceptionType>)`.

### Bug Fixes

- **Answers dropped from history.** A text answer with an unknown finish reason was returned and then silently dropped from history on the next turn ([#2726](https://github.com/0xPlaygrounds/rig/pull/2726)).
- **Provider replay bugs.** Item-shaped history closes a long tail of them ([#2713](https://github.com/0xPlaygrounds/rig/pull/2713)):
  - turns replayed without their partner item after a rollback ([#2647](https://github.com/0xPlaygrounds/rig/issues/2647));
  - colliding call ids ([#1508](https://github.com/0xPlaygrounds/rig/issues/1508), [#1962](https://github.com/0xPlaygrounds/rig/issues/1962));
  - replies that failed to decode on unexpected fields ([#2668](https://github.com/0xPlaygrounds/rig/issues/2668), [#1426](https://github.com/0xPlaygrounds/rig/issues/1426), [#2591](https://github.com/0xPlaygrounds/rig/issues/2591), [#1984](https://github.com/0xPlaygrounds/rig/issues/1984));
  - lost array-mode tool results on Chat ([#2201](https://github.com/0xPlaygrounds/rig/issues/2201));
  - a resumed run that appended to memory twice ([#2244](https://github.com/0xPlaygrounds/rig/issues/2244));
  - Claude thinking turns that could not replay after a tool or system change ([#2703](https://github.com/0xPlaygrounds/rig/issues/2703)).
- **Wrong Postgres parameters.** A `$` inside a key or pattern shifted every later placeholder onto the wrong value ([#2698](https://github.com/0xPlaygrounds/rig/pull/2698)).
- **Long Gemini streams lost their usage and output** after exhausting OpenTelemetry's per-span attribute limit ([#2732](https://github.com/0xPlaygrounds/rig/pull/2732)).
- **Xiaomi MiMo's Messages preset** posted to `/v1/v1/messages` ([#2738](https://github.com/0xPlaygrounds/rig/pull/2738)).
- **Streams could not be decoded item by item.** `StreamEvent` and `MultiTurnStreamItem` lost `Deserialize` before 0.44; they have it again ([#2723](https://github.com/0xPlaygrounds/rig/pull/2723)).
- **Gemini gRPC media.** gRPC rejected documents, audio, video and assistant images ([#2658](https://github.com/0xPlaygrounds/rig/pull/2658)).
- **Bedrock guardrails** were dropped on streamed requests ([#2750](https://github.com/0xPlaygrounds/rig/pull/2750)).
- **Cohere thinking.** `Reasoning::Off` on Command A Plus and North Mini Code, which think by default, was dropped silently ([#2750](https://github.com/0xPlaygrounds/rig/pull/2750)).
- **Errored 200s.** A Cohere 200 reply carrying an error envelope folded to an empty answer ([#2729](https://github.com/0xPlaygrounds/rig/pull/2729)).
- **Interleaved tool calls.** Streamed calls replayed or dropped each other's held items ([#2721](https://github.com/0xPlaygrounds/rig/pull/2721)).
- **Debug output.** Streaming printed a `PARTIAL ...` debug line to stderr when a streamed tool call was rejected ([#2680](https://github.com/0xPlaygrounds/rig/pull/2680)).

### Performance, Packaging, Docs and DX

- **Less duplicated code.** The provider layer lost its hand-kept typed mirrors of provider JSON, Bedrock dropped its `Rig*` newtype layer, and every reply now decodes and folds through one step and one EOF rule.
- **Docs and examples.** `crates/rig-core/TYPED_OPTIONS.md` documents typed options, the catalog, extensions, the `raw` shape, citations and cost. The new `agent_tool_call_streaming` example shows argument fragments streaming into a hook.
- **Tooling.** `cargo xtask catalog sync` / `check` keeps the catalog in step with models.dev. The cassette suite gained shape-matched replay, request snapshots and a coverage gate.

---

## 0.42 → 0.43

### Upgrading at a Glance

| 0.42 | 0.43 |
|---|---|
| `openai::Client::from_env()?` | `OpenAI::from_env()?` (`use rig::providers::openai::{self, OpenAI}`) |
| `client.agent(m)` / `client.extractor::<T>(m)` | `AgentBuilder::new(client.completion(m))` / `ExtractorBuilder::<T>::new(client.completion(m))` |
| `client.completion_model(m)` | `client.completion(m)` |
| `client.embedding_model(m)` / `embedding_model_with_ndims(m, n)` | `client.embedding(m, None)` / `client.embedding(m, Some(n))` |
| `deepseek::Client::from_env()?` (and other OpenAI-compatible vendors) | `deepseek::from_env()?` |
| `Client::builder().api_key(k).http_client(h).build()?` | `Anthropic::new(k).with_http(h)` (same on every client) |
| `agent.prompt(p).await?` → `String` | `agent.prompt(p).await?.output` |
| `agent.stream_prompt(p).await` / `agent.stream_chat(p, &h).await` | `agent.prompt(p).stream()` / `agent.prompt(p).history(&h).stream()` |
| `use rig::completion::{Prompt, Chat, TypedPrompt}` | delete the import: the methods are inherent on `Agent` |
| `extractor.extract(t).await?` → `T` | `extractor.extract(t).await?.output` |
| `extractor.extract_with_usage(t)` | `extractor.extract(t).await?` (a `TypedPromptResponse` with `output`, `usage`, `completion_calls`) |
| `model.completion_request(p)….send().await?` | `model.call(CompletionRequest::new(p)…).await?` |
| `CompletionError`, `EmbeddingError`, … | `ProviderError` |
| `usage.input_tokens` (`u64`) | `usage.input_tokens` (`Option<u64>`) |
| `StreamedAssistantContent::Text(t)` | `Item::Event(StreamEvent::Text { text, .. })` |
| `InMemoryVectorIndex<openai::EmbeddingModel, D>` | `InMemoryVectorIndex<D>` |
| `DynamicTool::new(.., \|_ctx, args\| ..)` | `DynamicTool::new(.., \|args\| ..)` (or `new_with_context`) |
| `#[rig::tool_macro]` | `#[rig::rig_tool]` |
| `rig-core` standalone (reqwest on by default) | `rig-core = { features = ["reqwest"] }` (the `rig` facade enables it by default) |

### Breaking Changes and Migration Guide

Ordered roughly by how many users each change affects. Every "0.42" snippet compiles against `v0.42.0` and every "0.43" snippet against `v0.43.0`.

#### Constructing providers and agents

The generic `Client<Ext, H>`, `ClientBuilder`, `ProviderClient`, the `*Client` capability traits (`CompletionClient`, `EmbeddingsClient`, …) and `AgentClientExt`/`AgentModelExt` are gone. A provider client is a concrete type that owns its transport, and an agent wraps a model.

```rust
// 0.42
use rig::prelude::*;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env()?;
    let comedian_agent = client
        .agent(openai::GPT_5_2)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response: String = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}
```

```rust
// 0.43
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // The client reads `OPENAI_API_KEY` and builds the models it serves.
    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let comedian_agent = AgentBuilder::new(model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response.output);
    Ok(())
}
```

The renames, row by row:

- `completion_model(m)` → `completion(m)`
- `embedding_model(m)` → `embedding(m, None)`
- `embedding_model_with_ndims(m, n)` → `embedding(m, Some(n))`
- `transcription_model` / `image_generation_model` / `audio_generation_model` → `transcription` / `image_generation` / `audio_generation`
- `list_models()` stays on the client; `verify()` is on the client.
- **OpenAI-compatible vendors:** `deepseek::Client::from_env()?` → `deepseek::from_env()?`. The same goes for Azure, ChatGPT, Doubleword, Groq, Hugging Face, Hyperbolic, llama.cpp, MiniMax, Mira, Mistral, Moonshot, OpenRouter, Perplexity, Together, Venice, xAI, Xiaomi MiMo and Z.ai, with `anthropic_from_env()`/`anthropic_new()` for the four Messages-format vendors.
- **Settings:** `openai::OpenAIConfig`, `AnthropicConfig`, `GeminiConfig`, … hold them (`from_env()`, `with_route(..)`, `connect(http)`, `client()`). The old stateful builders (`AnthropicBuilder`, `AzureExtBuilder`, …) are gone. Stored configs reject unknown fields.
- **Custom HTTP client:** `Client::builder().http_client(h).api_key(k).build()?` → `Anthropic::new(k).with_http(h)` (the same on every client), which takes any `HttpClientExt`.

```rust
// 0.42
use rig::prelude::*;
use rig::providers::anthropic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let http_client = rig::http_client::ReqwestClient::new();
    let client = anthropic::Client::builder()
        .http_client(http_client)
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .build()?;
    let agent = client.agent(anthropic::completion::CLAUDE_SONNET_4_6).build();
    println!("{}", agent.prompt("What is 2 + 2?").await?);
    Ok(())
}
```

```rust
// 0.43
use rig::http_client::ReqwestClient;
use rig::prelude::*;
use rig::providers::anthropic::{self, Anthropic};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Any `HttpClientExt` works; here a configured reqwest client in rig-reqwest's newtype.
    let http_client = ReqwestClient::from(rig::rig_reqwest::reqwest::Client::new());
    let model = Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?)
        .with_http(http_client)
        .completion(anthropic::CLAUDE_SONNET_4_6);
    let agent = AgentBuilder::new(model).build();
    println!("{}", agent.prompt("What is 2 + 2?").await?.output);
    Ok(())
}
```

- **SDK-backed companions:**
  - `rig_bedrock::client::Client` → `BedrockRuntime`
  - `rig_vertexai::client::Client` → `VertexAi` (builder `VertexAiBuilder`)
  - `rig_gemini_grpc::client::Client` → `GeminiGrpc`
  - `rig_fastembed::{Client, EmbeddingModel}` → `Fastembed::load(&m)?.embedding(&m, n)?`
  - Candle models build with `candle.completion()`.
  - `Client::get_inner` → `inner`.

```rust
// 0.42
use rig::bedrock::client::Client;
use rig::bedrock::completion::AMAZON_NOVA_LITE;
use rig::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Client::from_env()?.agent(AMAZON_NOVA_LITE).build();
    println!("{}", agent.prompt("Hello!").await?);
    Ok(())
}
```

```rust
// 0.43
use rig::bedrock::client::BedrockRuntime;
use rig::bedrock::completion::AMAZON_NOVA_LITE;
use rig::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // SDK-backed runtimes build their own models, like the HTTP clients do.
    let model = BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE);
    let agent = AgentBuilder::new(model).build();
    println!("{}", agent.prompt("Hello!").await?.output);
    Ok(())
}
```

- **Model traits are gone.** Code generic over `M: CompletionModel` (or `EmbeddingModel`, …) should take a `DynModel<Completion>` or a `Model<W, T>`. `rig::agent::ModelHandle` is replaced by `DynModel<Completion>` for values and by a `ModelRef` label for routing (see *Choosing models at runtime*).

#### Prompting an agent: results, traits and streaming

- `agent.prompt(p).await?` now returns a `PromptResponse`; read `.output` for the text. `extended_details()` is gone, because you always get the details.
- `agent.chat(p, &mut history).await?` returns a `PromptResponse` too.
- `Prompt`, `Chat`, `TypedPrompt`, `StreamingPrompt` and `StreamingChat` are removed; delete the imports, since the methods are inherent on `Agent`. If you implemented `Prompt` for your own type, give it an inherent `async fn prompt`.
- `PromptRequest`, `StreamingPromptRequest`, `TypedPromptRequest`, `Agent::runner` and the `Standard`/`Extended` markers are removed. `AgentRunner` has every setter.
- `agent.stream_prompt(p).await` → `agent.prompt(p).stream()`, and `agent.stream_chat(p, &history).await` → `agent.prompt(p).history(&history).stream()`.
- Stream items carry `Item<StreamEvent>` instead of `StreamedAssistantContent` (see *Low-level streams*).

```rust
// 0.42
    let mut stream = agent.stream_chat("Another one!", &history).await;
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => print!("{}", text.text),
            MultiTurnStreamItem::FinalResponse(response) => println!("\n\n{}", response.output()),
            _ => {}
        }
    }
```

```rust
// 0.43
    let mut stream = agent.prompt("Another one!").history(&history).stream();
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text { text, .. })) => {
                print!("{text}");
            }
            MultiTurnStreamItem::FinalResponse(response) => println!("\n\n{}", response.output()),
            _ => {}
        }
    }
```

- `MultiTurnStreamItem::ToolExecutionCommitted` lost `internal_call_id`, since tool calls are correlated by `CallId`. `MultiTurnStreamItem::ToolCall { tool_call }` is new.
- `PromptError` variants are unboxed, for example `MaxTurnsError { chat_history: Vec<Message>, prompt }`. `Agent::into_parts()` returns `Result<AgentParts, Agent>`.

#### Choosing models at runtime

0.42's runtime model swapping still works, but hooks now select models by **label** rather than by handle. Register extra models on the agent with `AgentBuilder::model_route(label, model)` (or start from `AgentBuilder::named_model(label, model)`), then return `ModelSelectionAction::select("label")`. `ModelSelection::{default_model, selected_model, previous_model}` are `ModelRef` labels. `agent.set_model(..)`/`with_model(..)` and a run's `using_model_value(..)` take any model value. `using_model(..)` now takes a label, and `set_model_handle`/`with_model_handle` are gone.

```rust
// 0.42
// Hooks selected models by handle.
struct FallbackOnRetry {
    fallback: ModelHandle,
}

impl AgentHook for FallbackOnRetry {
    fn on_model_select(&self, _ctx: &HookContext, event: ModelSelection<'_>) -> ModelSelectionAction {
        if event.previous_model.is_some() {
            ModelSelectionAction::select(self.fallback.clone())
        } else {
            ModelSelectionAction::continue_run()
        }
    }
}
```

```rust
// 0.43
// Hooks select models by the label they were registered under.
#[derive(Clone)]
struct FallbackOnRetry;

impl AgentHook for FallbackOnRetry {
    fn on_model_select(&self, _ctx: &HookContext, event: ModelSelection<'_>) -> ModelSelectionAction {
        // `previous_model` is set once an attempt has been issued in this run.
        if event.previous_model.is_some() {
            ModelSelectionAction::select("fallback")
        } else {
            ModelSelectionAction::continue_run()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_5_2))
        .model_route("fallback", Anthropic::from_env()?.completion(anthropic::CLAUDE_SONNET_4_6))
        .add_hook(FallbackOnRetry)
        .build();

    // Replace the default model for every future run, or override it for one run.
    agent.set_model(Anthropic::from_env()?.completion(anthropic::CLAUDE_SONNET_4_6));
    let answer = agent
        .prompt("Summarize this in one line.")
        .using_model_value(Gemini::from_env()?.completion(gemini::completion::GEMINI_2_5_FLASH))
        .await?;
    println!("{}", answer.output);
    Ok(())
}
```

#### Extractors and typed prompts

```rust
// 0.42
    let client = openai::Client::from_env()?;
    let extractor = client.extractor::<Person>(openai::GPT_5_2).build();

    let person: Person = extractor.extract("Jane Smith is a data scientist.").await?;
    println!("{person:?}");
    let response = extractor.extract_with_usage("John Doe is a chef.").await?;
    println!("{:?} ({} tokens)", response.data, response.usage.total_tokens);
```

```rust
// 0.43
    let client = OpenAI::from_env()?;
    let extractor = ExtractorBuilder::<Person>::new(client.completion(openai::GPT_5_2)).build();

    // `extract` returns a typed run: `.await` it, then read `.output`, `.usage`, `.completion_calls`.
    let response = extractor.extract("Jane Smith is a data scientist.").await?;
    println!("{:?} ({:?} tokens)", response.output, response.usage.total_tokens);
```

The removed methods map as follows:

- `extract_with_usage(t)` → `extract(t).await?`
- `extract_with_chat_history(t, h)` → `extract(t).history(h)`
- `extractor.using_model(m).extract(t)` → `extractor.extract(t).using_model(m)`

`ExtractionResponse { data, usage }` is now `TypedPromptResponse { output, usage, completion_calls }` (also at `rig::TypedPromptResponse`). `ExtractionError` is now `StructuredOutputError`, and `ExtractorRun` is removed. `agent.prompt_typed::<T>(p)` returns a `TypedRun<T>` with the same response.

#### Calling a model directly

`model.completion_request(p)….send()` and `CompletionRequestBuilder` are gone.

```rust
// 0.42
    let reply = model
        .completion_request("And of Italy?")
        .preamble("Answer in one word.".to_string())
        .temperature(0.2)
        .max_tokens(16)
        .send()
        .await?;
    let text: String = reply
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
```

```rust
// 0.43
    let request = CompletionRequest::new("And of Italy?")
        .preamble("Answer in one word.")
        .temperature(0.2)
        .max_tokens(16);
    let reply = model.call(request).await?;
    println!("{} ({:?} output tokens)", reply.text(), reply.usage.output_tokens);
```

A few related changes:

- The `CompletionRequest::preamble` *field* and `without_preamble` are removed. The system prompt is a leading `Message::System` in `chat_history`, and `system_instructions()` reads it. The `.preamble(..)` *setter* still exists.
- Build an assistant history turn with `response.message()`; for a tool result, `Message::tool_result(call_id, tool_name, output)` or `tool_call.result(content)`.
- `raw_completion` and `raw_stream` are removed: read `response.raw`.
- The capability `model.ndims()` is now `model.capabilities().ndims`.
- `NormalizeCompletionResponse` and the provider `TryFrom<…> for CompletionResponse` impls are gone; the wire's decoder is the mapping.

#### Low-level streams

```rust
// 0.42
    let mut stream = model.completion_request("Tell me a three-sentence story.").stream().await?;
    while let Some(chunk) = stream.next().await {
        if let StreamedAssistantContent::Text(text) = chunk? {
            print!("{}", text.text);
        }
    }
```

```rust
// 0.43
    let mut stream = model.stream("Tell me a three-sentence story.")?;
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item? {
            print!("{text}");
        }
    }
    // The same `CompletionResponse` that `model.call(..)` would have returned.
    let response = stream.finish().await?;
```

The mapping:

| 0.42 | 0.43 |
|---|---|
| `StreamingCompletionResponse` | `CompletionStream` = `Streamed<Completion>` |
| `StreamedAssistantContent::Text(text)` | `Item::Event(StreamEvent::Text { text, .. })` |
| `StreamedAssistantContent::ReasoningDelta { reasoning, .. }` | `Item::Event(StreamEvent::Reasoning { text, .. })` |
| `StreamedAssistantContent::Reasoning { reasoning, .. }` | `StreamEvent::End { content: AssistantContent::Reasoning(..), .. }` |
| `StreamedAssistantContent::ToolCallDelta { .. }` | none: a call's arguments arrive once, as `StreamEvent::Arguments` |
| `StreamedAssistantContent::ToolCall { tool_call, .. }` | `StreamEvent::End { content: AssistantContent::ToolCall(..), .. }` (agent streams also emit `MultiTurnStreamItem::ToolCall`) |
| `StreamedAssistantContent::Final(StreamFinal)` | `stream.finish().await` |
| `StreamedAssistantContent::Unknown(payload)` | `Item::Unknown(payload)` |
| `pause()` / `resume()` / `cancel()` / `is_paused()` | stop polling / drop the stream |

A transport that refuses a request now yields the failure as the stream's first item, not as an error from `stream()`.

#### Errors

```rust
// 0.42
    match model.completion_request("Hello").send().await {
        Ok(reply) => println!("{:?}", reply.choice),
        Err(CompletionError::ProviderResponse(response)) => eprintln!("provider said {:?}: {}", response.status, response.body),
        Err(CompletionError::HttpError(error)) => eprintln!("transport failed: {error}"),
        Err(other) => eprintln!("{other}"),
    }
```

```rust
// 0.43
    match model.call("Hello").await {
        Ok(reply) => println!("{}", reply.text()),
        // One error type for every operation; the variant is the classification.
        Err(ProviderError::ProviderResponse(response)) => {
            eprintln!(
                "provider said {:?} (request id {:?}, retryable: {:?}): {}",
                response.status, response.provider_request_id, response.transient, response.body
            );
        }
        Err(ProviderError::Http(error)) => eprintln!("transport failed: {error}"),
        Err(other) => eprintln!("{:?}: {other}", other.kind()),
    }
```

The mapping:

- **Variant renames:** `HttpError`→`Http`, `JsonError`→`Json`, `UrlError`→`Url`, `RequestError`→`Request`, `ResponseError`→`Response`, `ProviderError(msg)`→`Provider(msg)`.
- **Verification:** `VerifyError::InvalidAuthentication` → `ProviderError::InvalidAuthentication(response)`.
- **Model listing:** `ModelListingError::ApiError { status_code, message }` → `ProviderError::ProviderResponse(response)`, with the provider and path in `response.route`.
- **Vector stores:** `VectorStoreError::ReqwestError` → `VectorStoreError::Http(http_client::Error)`.
- **Auth:** `copilot::auth::AuthError::Http` and `chatgpt::auth::AuthError::Http` wrap `http_client::Error`.
- **Struct literals:** `ProviderResponseError` literals need `route: None` and `headers: Option<HeaderMap>` (no longer boxed).
- **Persistence:** `ProviderError` is not `Serialize`, so persist `error.report()`.

#### Token usage

```rust
// 0.42
    let usage = reply.usage;
    // Plain u64 counters; 0 meant both "zero" and "not reported".
    println!("input {} (cached {}), output {}", usage.input_tokens, usage.cached_input_tokens, usage.output_tokens);
```

```rust
// 0.43
    // Counters are `Option<u64>`: `None` = the provider did not report it.
    let usage = reply.usage;
    println!(
        "input {} (cached {}), output {}",
        usage.input_tokens.unwrap_or(0),
        usage.cached_input_tokens.unwrap_or(0),
        usage.output_tokens.unwrap_or(0)
    );
```

- `Usage::new()` → `Usage::default()`, and `has_values()` → `is_reported()`.
- Provider streaming usage fields (OpenAI Chat and Responses) are `Option` too.
- **Values change** for Anthropic, its gateways and Bedrock (`input_tokens` includes cache reads and writes) and for the Gemini family (`output_tokens` includes thoughts). Dashboards that summed `input_tokens + cached_input_tokens` for Anthropic, or `output_tokens + reasoning_tokens` for Gemini, now double count.

#### Vector stores and embeddings

Stores no longer name the embedding model in their type. They take `impl Into<DynModel<Embedding>>`:

```rust
// 0.42
// The index type names the embedding model.
fn dictionary_agent(index: InMemoryVectorIndex<openai::EmbeddingModel, WordDefinition>, client: &openai::Client) -> Agent {
    client.agent(openai::GPT_5_2).preamble("You are a dictionary assistant.").dynamic_context(1, index).build()
}
```

```rust
// 0.43
// The index type no longer names the embedding model.
fn dictionary_agent(index: InMemoryVectorIndex<WordDefinition>, client: &OpenAI) -> Agent {
    AgentBuilder::new(client.completion(openai::GPT_5_2))
        .preamble("You are a dictionary assistant.")
        .dynamic_context(1, index)
        .build()
}
```

The affected types:

- `InMemoryVectorIndex<M, D>` → `<D>`
- `QdrantVectorStore<M>` → `QdrantVectorStore`
- `LanceDbVectorIndex<M>`
- `MongoDbVectorIndex<C, M>` → `<C>`
- `MilvusVectorStore<M>`
- `Neo4jVectorIndex<M>`
- `PostgresVectorStore<Model>`
- `ScyllaDbVectorStore<M>`
- `S3VectorsVectorStore<M>`
- `SqliteVectorStore<E, T>` → `<T>`
- `SurrealVectorStore<C, Model>` → `<C>`
- `HelixDBVectorStore<C, E>`
- `VectorizeVectorStore<M>`

Other changes in this area:

- **Schema constructors take the width:** `SqliteVectorStore::new(conn, ndims)` and `Neo4jClient::create_vector_index(config, label, ndims)`.
- **Filter builders** take `&str` keys and borrowed values (qdrant, milvus, surrealdb, lancedb, neo4j, postgres) or `impl Into<String>` keys (mongodb, scylladb, s3vectors, sqlite).
- **rig-milvus** now exports `Filter` and `MilvusValue`.
- **rig-sqlite** removes `SqliteError`, which was never constructed.
- **Embedding and rerank responses** carry provider, model, identity and `raw`. `RerankResponse::model` is `Option<String>`.
- **rig-memory** keys conversations by `ConversationId`; `conversation("id")` still accepts a string. `HeuristicTokenCounter::{gemini, openai}` are removed.

#### Tools and MCP

```rust
// 0.42
    // The closure took the (mutable) tool context first.
    let add = DynamicTool::new("add", "Add x and y", json!({ "type": "object" }), |_context, args| {
        Box::pin(async move {
            let args: Args = serde_json::from_value(args).map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
            Ok(ToolOutput::json(json!(args.x + args.y)))
        })
    });
```

```rust
// 0.43
    // A context-free dynamic tool: the closure takes only the JSON arguments.
    let add = DynamicTool::new(
        "add",
        "Add x and y",
        json!({
            "type": "object",
            "properties": { "x": { "type": "integer" }, "y": { "type": "integer" } },
            "required": ["x", "y"]
        }),
        |args| {
            Box::pin(async move {
                let args: Args = serde_json::from_value(args)
                    .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
                Ok(ToolOutput::json(json!(args.x + args.y)))
            })
        },
    );
```

- `PortableDynamicTool` merged into `DynamicTool`: `new` for a context-free callback, and `new_with_context` when you need the `ToolContext`. `DynamicTool::from_portable`, `portable_tool_definition` and the `portable_dynamic_tool` builder methods are removed. `PortableTool` and `PortableToolEmbedding` stay.
- **`ToolContext` values are typed and serializable.** Derive `ContextValue` (optionally `#[context(key = "…")]`) on anything you `insert`. `insert`, `get` and `require` return `Result<_, ToolContextError>`, and `MissingToolContext` is gone. Keep live shared state (connections, handles) in the tool struct itself:

```rust
// 0.43
// Values placed in a `ToolContext` declare a key and are serializable, so runs can be persisted.
#[derive(Clone, Serialize, Deserialize, ContextValue)]
#[context(key = "app.user_id")]
struct UserId(String);
```

- **MCP:** rig-agent's `rmcp` feature is gone; use the facade's `rmcp` feature (rig-rmcp). `rig::tool::rmcp::McpClientHandler` with a `ToolServer` handle works exactly as before. `AgentBuilder::rmcp_tool(s)(_with_timeout)` and `ToolServer::rmcp_tools*` are removed; register MCP tools as `DynamicTool`s (`McpTool` converts into one) or through `McpClientHandler`.
- `#[rig::tool_macro]` → `#[rig::rig_tool]`.
- `ToolServerHandle`'s registration methods are synchronous; drop the `.await`.

#### Hooks

- `on_tool_call`, `on_tool_result`, `on_completion_response` and `on_stream_response_finish` are removed. Gate effects (including tool calls) in `on_dispatch` and observe their results in `on_outcome`: `event.completion()` for model replies, or match the outcome for tools.
- The hook event `CompletionCall` is renamed `CompletionCallEvent`; `rig::run::CompletionCall` keeps its name. `ToolCallAction`, `ToolResultAction` and `ToolResultEvent` are gone.
- `InvalidToolCallContext` and `StreamedInvalidToolCall` gained a `reason`, so struct literals must add it. A hook that always returns `Repair` now gets the fail-closed report for malformed arguments.
- `StepEventKind` has new variants; add arms to exhaustive matches.

#### Messages and persisted state

- Tool calls carry a required `CallId` (`Provider` or `Local`) and a `ToolName`. `ToolCallId` is removed, and `Message::tool_result` takes `(CallId, ToolName, output)`. On `wasm32-unknown-unknown`, `LocalCallId` needs `getrandom`'s `js` backend.
- `AssistantContent::Reasoning` holds `Sealed` reasoning with an `Issuer`; read it with `open(&issuer)` or `open_for(&[..])`.
- **Persisted histories and runs from 0.42 may not load.** The old `ToolCallId` shape no longer parses. `AgentRun` and rig-ecs checkpoints carry a format tag and reject unknown fields. `CompletionResponse`, `ModelTurn` and `CompletionCall` constructors take the `raw` document. Re-create persisted state, or migrate it.
- `RunId` moved from `rig::agent` to `rig_core::id` and is a numeric counter id. `conversation(..)` takes `impl Into<ConversationId>`.
- The conversions from provider wire messages into `Message` are removed: `TryFrom<anthropic::completion::Message>`, `TryFrom<cohere::completion::Message>`, `TryFrom<openai::Message>` and `From<ollama::Message>`.

#### Cargo features and dependencies

- **`rig-core`'s default features drop `reqwest`.** A standalone `rig-core` has no HTTP client unless you enable `reqwest` (`cargo add rig-core --features reqwest`). The `rig` facade still enables it by default. rig-core's `socks`, `reqwest-middleware*`, `websocket-rustls`, `websocket-native-tls` and `rayon` features are gone. `websocket` is now the protocol only; `tungstenite` adds the bundled backend.
- **Facade features:**
  - `reqwest` (default) routes `socks` and `reqwest-middleware*` through rig-reqwest.
  - `websocket`, `websocket-rustls` and `websocket-native-tls` pull in rig-tungstenite.
  - `rmcp` points at rig-rmcp and no longer implies `agent`.
  - `typesafeai` is new.
  - `discord-bot` is removed; copy `examples/discord_bot`, which also drops four `rustls-webpki` advisories from your dependency tree.
  - `rayon` is removed; vector math is deterministic and sequential.
- **rig-agent** drops its `rmcp` and `discord-bot` features.
- **rig-vectorize** gains `rustls` (default) and `native-tls`.
- **Toolchain:** the workspace builds on Rust 1.95.0 (was 1.94.0).

#### Transports

- To type-erase a transport, wrap it in `DynHttpClient::new(h)`. `with_middleware` attaches `HttpMiddleware`.
- `rig::http_client::ReqwestClient` used to be a re-export of `reqwest::Client`. It is now rig-reqwest's newtype: wrap a configured client with `ReqwestClient::from(client)` and get it back with `inner()`. `ReqwestClient::default()` is the process-wide client, and if reqwest can't build it (for example, no CA store) every send reports the error instead of panicking.
- `client.responses_websocket(model)` / `responses_websocket_builder(model)` → `openai_client.responses(model).responses_websocket().connect()` with the `websocket` feature, or `.connect_with(&backend)` for another `WebSocketClientExt`. rig-core's own `websocket-rustls`/`websocket-native-tls` features are gone; use the facade's features of the same names, or rig-tungstenite's `rustls`/`native-tls`.
- `Transport` is no longer in the preludes; use `rig::driver::Transport`.

#### Implementing a provider or a custom model

`impl CompletionModel for MyModel` (with hand-written `completion` and `stream`) is replaced by:

- `impl Wire` for your request/response shape (`describe`, `encode`, `decoder`)
- `impl Decoder` over the reply frames

The driver owns the transport, telemetry and observation. For a new operation, implement `Operation` plus a `Fold`. For scripted test models, `test_utils::MockCompletionModel` is now an alias for `Model<MockScript, MockRuntime>`, built with `from_turns(..)`. See the `rig_core::wire` module docs and `examples/frozen_api` (`p5_local_operation.rs`, `p6_http_operation.rs`).

#### Smaller removals and renames

- **Removed model constants:**
  - Deprecated ones: `deepseek::DEEPSEEK_CHAT`/`DEEPSEEK_REASONER` (use `DEEPSEEK_V4_FLASH`/`DEEPSEEK_V4_PRO`), Cohere's `COMMAND*`, and Mistral's `CODESTRAL_MAMBA`, `MISTRAL_NEMO`, `MISTRAL_SABA`, `PIXTRAL_LARGE` and `PIXTRAL_SMALL`.
  - Retired Groq and Moonshot ids. The old strings still work if you pass them yourself, but the providers reject them.
- **llama.cpp:** `providers::llamafile` → `providers::llamacpp`.
- **Modality request builders**:
  - Required values are constructor arguments: `TranscriptionRequestBuilder::new(model, data)` or `from_file(model, path)?`, `ImageGenerationRequestBuilder::new(model, prompt)`, `AudioGenerationRequestBuilder::new(model, text, voice)`.
  - `additional_params(impl Into<Option<Value>>)` merges on every builder, and `additional_params_opt` is removed.
  - Modality responses lost their type parameter and `response` field.
- **Telemetry:** `CompletionOperation` → `GenAiOperation` (it covers all ten operations), and `CompletionSpanBuilder` → `SpanBuilder`. `ProviderResponseExt` is removed.
- **Candle:** the `LlamaModel` and `ModelFamily` aliases are removed; use `CandleModel` and `ConversationProtocol`.
- **ChatGPT and Copilot:** the `AuthContext` credential fields are `Secret`.
- **Responses:** `Output` and `InputContent` gained `Compaction` variants, so exhaustive matches need an arm.
- **rig-core:** the `if_wasm!`/`if_not_wasm!` macros are removed. The `http_client::retry` (`RetryPolicy`, `ExponentialBackoff`, `DEFAULT_RETRY`) and `http_client::sse` modules are removed; retry transient failures in a hook or with `reqwest-middleware`.

### Other API Additions

- `CompletionResponse::{text, reasoning, tool_calls}`; `CompletionRequest` from `&str`/`String`/`Message`/`Vec<Message>`.
- `Model::{name, id, capabilities, erase}` and `DynModel<Op>`.
- `providers::registry::{Format, ProviderId, ProviderConfig, ProviderRef}`.
- `AgentRunner::run_channel`, `RunEvents`; `Agent::resume`; `AgentBuilder::record_to`.
- `ToolServerHandle::{snapshot, static_tool_defs, toolset}` for synchronous registry reads, and `ToolSet: Clone`. `rig::tool` also exports `ToolCatalog`, `ToolLease`, `RegisteredTool`, `ToolDispatch`, `ErasedTool` and `execute_tool`.
- `rig::run` / `rig_agent::run`: `AgentRun`, `RunSpec`, `RunEntry`, `TurnTools`, `prepare_request`.
- `rig_core::transcript::validate_canonical` checks that tool calls and results pair up in externally sourced histories.
- `http_client::HttpMiddleware` and `DynHttpClient::with_middleware`.
- `completion::{CacheCost, CacheRates}`.
- `gemini::cached_content`, `gemini::caching` and `ThoughtReplay`.
- `ErrorReport` and `ErrorKind` at the crate root; `ErrorDetail::MalformedToolInput`.
- `rig_core::observe::{scrub_diagnostic, diagnostic_url_secrets}` for hosts that persist their own diagnostics.
- `openai::wire::Quirks::{without_stream_usage, without_response_format, reliable_reasoning_count}` and `Dialect::with_quirks`, for tuning OpenAI-compatible dialects.
- `rig_derive::ContextValue`.
- `rig_candle::pose`.
- `rig_cassette::http::skip_when_recording`; `CassetteSpec::expects_account_failure`.

### Behavior Changes

These compile as before but act differently:

- **Token usage values** follow one contract (see above): Anthropic/Bedrock input includes cache tokens, and Gemini output includes thoughts. Gemini input includes `toolUsePromptTokenCount`.
- **OpenAI Responses tools** send `strict: false` explicitly. If you relied on OpenAI's implicit strict mode, set strict tools yourself.
- **`claude-sonnet-4-6` defaults to 128K `max_tokens`**.
- **Other providers' reasoning is dropped from requests** instead of being replayed.
- **Truncated streams are errors.** A stream cut before its terminal record, or a reply without a finish reason, is `ProviderError::Truncated` (retryable) instead of a short successful answer.
- **Tool calls stream whole**, and **`choice` is in the order parts started**.
- **Malformed streamed tool arguments** consult `on_invalid_tool_call` before failing.
- **Gemini prompt blocks:** `SAFETY`/`BLOCKLIST`/`PROHIBITED_CONTENT` is a refusal (`refusal: true`, with the reason as the code), and `OTHER` or unknown is transient.
- **Anthropic mid-conversation system messages** are moved, not hoisted.
- **Request-building failures** report `ErrorKind::Request` on every provider.
- **`CompletionResponse::raw`** is the provider's verbatim reply document, not a re-serialization of Rig's parse.
- **Bedrock document names** are `document-<hash>` instead of random UUIDs.
- **The bundled reqwest client** reports a build failure (such as a missing CA store) on the first request instead of panicking.
- **Telemetry:** every modality opens a `gen_ai` span. Completion spans record `rig.provider_request_id`. `gen_ai.system_instructions` is actually recorded now.
- **Vector-store scores** are bit-identical from run to run.

### Bug Fixes

- **Prompt caching could never hit** on Gemini requests with tools, or Cohere requests with documents, because map keys serialized in random order.
- **Gemini Interactions usage** dropped thought, cached and tool-use tokens, so cache hit rates read as zero. Vertex AI hard-coded `reasoning_tokens: 0`.
- **llama.cpp** served a specific-function `tool_choice` as `auto`. Images in tool results were refused for every OpenAI-compatible server.
- **`from_env()` panicked** on hosts with no CA store. The bundled client now reports the failure as an error instead.
- **Tool-call identity:** Vertex, gRPC Gemini and Candle gave every id-less call the same position, so two such calls in one turn could collide.
- **A truncated reasoning-only turn** was committed to history, and **images were dropped** when a streamed turn was regrouped around a tool call.
- **OpenAI Responses** silently enabled strict mode (reported by @KumaCrypto). The Groq and Moonshot constants were all retired upstream (reported by @Muhammad-Owais-Warsi, @loony-lisa and @Parikalp-Bhardwaj). Sonnet 4.6 was capped at 64K (@scadoshi).
- **A Responses reply with odd echoed metadata failed to decode entirely** (@BANG404, @mzdora). **rig-postgres generated invalid SQL** for `member`, single-condition and threshold searches (reported and diagnosed by @jackthepunished).
- **One malformed streamed tool call killed the whole agent run** (@tiswas247).
- **Unary/streaming divergences.** Moving every provider onto one decoder per wire closed the gaps between the two paths (14 in all).
  - Perplexity streams were misread.
  - A unary tool call cut mid-string destroyed the turn.
  - A stalled pagination cursor looped forever.
  - The Responses in-band `error` event was skipped.
  - Mid-stream transport failures lost status, headers and body.
  - OpenRouter cache writes were lost.
  - `embedding(model, None)` dropped `dimensions`.
  - The telemetry span named the wrong model on override.
  - …and more.
- **Ten vector stores could not compile for wasm,** and a standalone `rig-vectorize` build had no TLS.
- **Eleven downstream-consumer defects**:
  - `#[derive(Embed)]` called an unrelated inherent `embed`, and emitted wrong members for tuple fields.
  - `#[rig_tool]` arguments named `args`/`_context` were shadowed.
  - A native-TLS-only `reqwest-middleware` selection hid the client.
  - Generated in-memory store ids overwrote explicit ones.
  - Angular distance was NaN for identical or opposite vectors.
  - rig-memory window truncation kept orphan tool results.
  - The EPUB loader deleted entity references.
  - Vector-store errors lost response facts.
  - Boxed errors lost their `source()` chains.
  - Vertex's raw response type was unnameable.
- **Groq tool loops with reasoning models** failed on the second call, and **Ollama** dropped tool-call ids.
- **Reasoning was replayed to providers that could not read it,** and Bedrock requests with documents were never byte-stable, so they could not hit a prompt cache.
- **Gemini answer-text signatures** moved to a synthesized thought part, Vertex dropped them, and OpenRouter's Responses route dropped Claude signatures.
- **An Interactions stream with `store: false` ended at its first status update,** and an Anthropic stream lost its cache counters when the terminal delta omitted them (with @allenrchan).
- **Rig-caused 400s on the new Anthropic and OpenAI models:** the extractor's forced tool choice, GPT-6 Chat tools, and system-message hoisting.
- **Resolved ChatGPT/Copilot credentials** appeared in `Debug` output.

### Performance, Packaging, Docs and DX

- **Smaller dependency graphs.**
  - rig-core depends on neither reqwest nor tokio.
  - Fourteen unused dependencies are removed across eleven crates, and rig-bedrock no longer builds the `rig-derive` proc-macro for its users.
  - `serenity`, `rayon`, `as-any` and `eventsource-stream` are gone from the workspace.
- **The `rig` package ships its source, not the repository.** The facade tarball is 13 files and 417 KiB.
- **Fewer allocations.** Two ownership audits and three idiomatic-Rust sweeps removed dead clones, whole-batch copies and redundant bounds. The provider layer lost several thousand lines of duplicated code.
- **Docs.** Every production doc comment was rewritten to one style, and doc examples that were `ignore`d now compile as `no_run`. New and updated examples: `agent_no_tokio`, `http_middleware`, `frozen_api`, `candle_pose`, `typesafeai_chat` and `typesafeai_triage`.
- **Contributing is easier.**
  - Rig no longer asks you to open an issue first, keep PRs small, or avoid drafts.
  - Changelog and migration notes go in the PR description instead of conflicting edits to `CHANGELOG.md`.
  - A lean dev profile and a `local` nextest profile cut local build sizes from tens of gigabytes.
  - Local checks are minimal, and CI does the full verification.
- **Tests.**
  - The cassette corpus grew from 1,793 to 2,995 provider recordings, plus 2,058 replayable effect-log goldens.
  - 1,758 provider-matrix tests are generated from descriptors.
  - Inline test modules moved to sibling files; `runner.rs` went from 11.7k lines to 1.1k.

---

## 0.41 → 0.42

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
satisfy it (see [Silent behavior changes](#042)). Its `is_empty()` returned a
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
  `0.41 → 0.42` is superseded.
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
renames. Those rows stay accurate as history for 0.41; on 0.42 the
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

| Constant | 0.41 | 0.42 |
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
releases they document, but the attribute claim no longer holds on 0.42:
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
| `message::Document` | `message::DocumentContent` | 0.47 |
| `message::MediaType::from_mime_type` | `ImageMediaType::from_mime_type` and the other three kinds | 0.47 |
| `RunId::from_raw`, `RunId::to_raw`, `RunId::from_str` | serde (`serde_json::from_value`, `to_value`), `Display` | 0.47 |
| `TurnEnd::Stopped { why, .. }` | `TurnEnd::Stopped { by: Option<StopCause>, .. }` | 0.47 |
| `TurnEnd::Stopped::calls: Vec<CallId>` | `Vec<CallRef>` | 0.47 |
| `Conversation::pending_calls() -> Vec<ToolCall>` | `-> Vec<(CallRef, ToolCall)>` | 0.47 |
| `Conversation::aside_position(aside)` | `aside.after` with `Conversation::index_of` / `start_of` | 0.47 |
| `tool::feedback::tool_result_output(id, name, &result)` | `UserContent::ToolResult(call.answer(&result))` | 0.47 |
| `invalid_call_feedback(&content, &CallId, text)` | `invalid_call_feedback(&content, position, text)` | 0.47 |
| `structured_output::text_satisfies_schema` | `structured_output::validate_text` | 0.47 |
| `redact::json_text`, `redact::tool_result` | `redact::json` on the parsed value | 0.47 |
| `completion::{CacheCost, CacheRates}` | `Usage`'s counters and your own rates | 0.47 |
| `loaders::{file, pdf, epub}::IntoIter<'a, T>` | `Box<dyn Iterator<Item = T> + 'a>` | 0.47 |
| `json_utils::merge_inplace(a, b)` | `*a = json_utils::merge(std::mem::take(a), b)` | 0.47 |
| `cli_chatbot::ChatBotBuilder::new().agent(a).build()` | `cli_chatbot::ChatBot::new(a)` | 0.47 |
| `ChatBot::run() -> Result<(), PromptError>` | `-> Result<(), StreamToStdoutError>` | 0.47 |
| `PreparedRequest::apply(CompletionRequest::new(prompt))` | `prepare_request(.., prompt, ..)?.request` | 0.47 |
| `run::PrepareError::Request(String)` | `ProviderError::Request` | 0.47 |
| `tool::server::ToolServerError` | `ProviderError` | 0.47 |
| `CompletionCall::response_id`, `provider_request_id` (also on `ModelTurn`) | `identity: ResponseIdentity` | 0.47 |
| `ModelTurn::from_response(&resp, &prepared)` | `ModelTurn::from_policy(&resp, prepared.policy.clone())` | 0.47 |
| `StreamedTurn::finish_reason` | `CompletionCall::finish_reason` | 0.47 |
| `Agent::with_model`, `Agent::with_model_label` | `Agent::set_model`, `Agent::set_model_label` | 0.47 |
| `AgentRun::with_validated_history(h)` | `pairing::validate(&h)?` then `AgentRun::with_history(h)` | 0.47 |
| `HookStack::with(hook)` | `HookStack::new()` then `push(hook)` | 0.47 |
| `rig::agent::tool::*` | `rig::tool::*` | 0.47 |
| `http_client::Error::{Instance, Unbuilt}`, `Error::{instance, unbuilt}` | `http_client::Error::{Transient, Permanent}`, `Error::{transient, permanent}` | 0.47 |
| `rig_core::error::transient_transport(&e)` | `http_client::Error::is_transient` | 0.47 |
| `HttpClientExt::send::<T, U>`, `LazyBody<T>` | `HttpClientExt::send(Request<Bytes>)`, `LazyBody` | 0.47 |
| `rig_http::wasm_compat::WasmCompatSendStream` | `Stream<Item = http_client::Result<Bytes>> + WasmCompatSend` | 0.47 |
| `providers::{chatgpt, copilot}::auth::{TokenStore, FileTokenStore, MemoryTokenStore, StoreLock, DeviceCodeHandler, DeviceCodePrompt, AuthContext, AuthError}` | `providers::auth::{..}` | 0.47 |
| `Model::call_observed`, `Model::stream_observed` | `Model::call`, `Model::stream` | 0.47 |
| `serve::Observe`, `Dispatch::with_observer` | `serve::Recorder`, `Dispatch::recorded_by` | 0.47 |
| `rig_core::wire::Event<W>` | `<<W as Wire>::Op as Operation>::Event` | 0.47 |
| `cohere::CohereChat` with `ChatRoute::Native` | `Cohere::native_completion` (`NativeChat`) | 0.47 |
| `cohere::CohereChat` with `ChatRoute::Compatibility` | `Cohere::completion` (`openai::wire::Chat`) | 0.47 |
| `OpenAIConfig::with_alternate_key(dialect, key)` | `OpenAIConfig::with_key(dialect, key).with_auth(Auth::Bearer)` | 0.47 |
| `OpenAIConfig::with_audio_api_version(v)` | the `audio_api_version` field | 0.47 |
| `openai::Usage` (embeddings) | `openai::EmbeddingUsage` | 0.47 |
| `openai::wire::ModelsReply` as `Models`' event | `model::ModelList` | 0.47 |
| `copilot::model_listing::Models::new(config)` | `config.connect(http).list_models()` | 0.47 |
| `azure::{GPT_4_TURBO, GPT_4_32K_0613}` | `azure::{GPT_4, GPT_4_32K}` | 0.47 |
| `anthropic::streaming::MessagesDecoder::default()` | `MessagesDecoder::new(anthropic::wire::Quirks::anthropic())` | 0.47 |
| `anthropic::Messages::with_default_max_tokens(n)` | the `default_max_tokens` field | 0.47 |
| `gemini::InteractionsModels` | `gemini::Models` | 0.47 |
| `gemini::VerifyKeyDecoder` | `rig_core::operation::VerifyDecoder` | 0.47 |
| `gemini::completion::{accepts, encodes, declares_tools, normalize_tool_call_id}` | the `ReplayTarget` methods on a Gemini wire | 0.47 |
| `gemini::extension::{HarmCategory, HarmBlockThreshold}::as_str` | serde (`serde_json::to_value`) | 0.47 |
| `voyageai::Rerank::{with_top_k, with_return_documents}` | the `top_k` and `return_documents` fields | 0.47 |
| `test_utils::RecordingHttpClient::with_error_response` | `with_error_response_headers` | 0.47 |
| `rig_memory::CompactionPolicy` | `CompactionSettings`, `prepare_compaction`, `find_cut_point` | 0.47 |
| `rig_memory::Summarizer::request` | `summary_request`, `turn_prefix_request`, `summarization_request` | 0.47 |
| `rig_memory::Summarizer::summary_text(&reply)` | `summary_text(&reply, label)` | 0.47 |
| `rig_memory::SummaryError` | `SummaryFailure` | 0.47 |
| `rig_memory::{SummaryPlan, SummaryCut}` | `CompactionPreparation` | 0.47 |
| `apply_summary(.., keep_from: Option<MessageId>, ..)` | `apply_summary(.., replaces: &[Arc<Message>], ..)` | 0.47 |
| `SummaryState::tracked` | `SummaryState::{read_files, modified_files}` | 0.47 |
| `rig_memory::Compacted::of`, `COMPACTED` | `summary_state`, `SummaryState::read` | 0.47 |
| `rig_memory::Summarized` (from `summarized_in_log`) | `BTreeMap<MessageId, Vec<Entry>>` | 0.47 |
| `rig_memory::CompactReason` | `rig_basics::compaction::CompactReason` (coding agent only) | 0.47 |
| `rig_memory::TokenCounter::count` | `HeuristicTokenCounter::count` (inherent), or your own `recent::Recent` | 0.47 |
| `HeuristicTokenCounter::{new, anthropic}` | `HeuristicTokenCounter::default()` | 0.47 |
| `TokenBudget { max_tokens, counter }` | `TokenBudget::new(max_tokens)` | 0.47 |
| `rig_memory::{Prepared, next_request_tokens}` | `request_estimate`, `request_tokens` | 0.47 |
| `rig_memory::OUTPUT_MARGIN` | `clamp_max_tokens` | 0.47 |
| `rig_memory::main_order` | `Session::main_agent()` | 0.47 |
| `Session { .. }` literal | `Session::default()`, `Session::read` | 0.47 |
| features `audio`, `image` (`rig`, `rig-core`, `rig-agent`) | always compiled; delete the names | 0.47 |
| feature `websocket` (`rig-core`, `rig-http`) | always compiled; delete the name | 0.47 |
| `rig_cassette::effect_log::EffectFilter::matches` | a closure over `EffectRecord::{scope, kind, turn}` | 0.47 |
| `McpConnection::from(service)` | `McpConnection::new(service)` | 0.47 |
| `world.spawn((Agent, ..))` | `rig_ecs::spawn::spawn_root(world, bundle)`, `Spawner::spawn(Origin, AgentSpec)` | 0.47 |
| `SpawnedBy(parent)`, `spawned_by.0` | `Origin::ChildOf(parent)`, `SpawnedBy::parent()` | 0.47 |
| `ToolAccess::All`, `ToolAccess::Only(vec)` | `ToolAccess::all()`, `ToolAccess::only(names)`, `ToolAccess::names()` | 0.47 |
| `StopTurn { entity, reason }` | `StopTurn::new`, `StopTurn::plugin`, `plugin_alone`, `plugin_done` | 0.47 |
| `TurnOutcome::Stopped(Option<String>)` | `TurnOutcome::Stopped(Stop)`, `Stop::cause()` (`StopCause`) | 0.47 |
| `rig_ecs::restore::RESTARTED` | `StopCause::Restarted` | 0.47 |
| `commands.trigger(ForkAgent { agent, through })` | `commands.spawn(ForkAgent::new(agent, through))`, `ForkResult` | 0.47 |
| `Models::connect` returning `Result<_, String>` | `Result<_, ConnectError>`, `Models::cannot_use` | 0.47 |
| `SetModel { entity, model }`, `SetReasoning { entity, effort }` | the same with `kind: ChoiceKind` | 0.47 |
| `ReasoningSetting::to_string()` | `spec.reasoning.name_of(setting.0)` | 0.47 |
| `rig_ecs::SaveSystems`, `WriteLogs` in `Last` | the `Persist` schedule, `WriteLogs` a set in it | 0.47 |
| `rig_ecs::turn::RETRY_POLICY` | the `rig_ecs::retry::RetryPolicy` resource | 0.47 |
| `rig_ecs::turn::DroppedOptions` | `Adapted::dropped(&ModelRequest)` in `PrepareSystems::Fit` | 0.47 |
| `SavedKeys::iter` | `SavedKeys::path(key)`, `SavedKeys::key(type_path)` | 0.47 |
| `AppSettingsExt::add_project_setting` | `AppSettingsExt::add_setting` (home `settings.toml` only) | 0.47 |
| `ProjectRoot/.rig` for project files | `rig_harness::services::Project` (`root`, `config`) | 0.47 |
| `AppPerCallExt::add_live_tool_per_call` | `add_tool_per_call` with `PerCallTool::LIVE = true` | 0.47 |
| `impl PerCallTool for rig_tools::Shell` | `rig_harness::per_call::PerCallShell::new(shell, world)` | 0.47 |
| `rig_basics::coding_tools::ShellOutput` | `rig_harness::per_call::ShellOutput` | 0.47 |
| `rig_basics::coding_tools::SeenFiles` | `rig_harness::per_call::FilesSeen` (a component of each agent) | 0.47 |
| `PerCall::build(place)` | `PerCallTools::build(name, place)` | 0.47 |
| `Picker::checked` | `PickItem::checked` | 0.47 |
| `rig_basics::print::{Starting, KeepRunning, Work, TOOLS}` | `rig_harness::run::{Starting, KeepRunning, Work, TOOLS}` | 0.47 |
| `ExitReason::Failed(u8)` | `ExitReason::Failed(FailCode)`, `ExitReason::Refused(String)` | 0.47 |
| `rig_harness::host::exit::SIGNAL_EXIT_CODE` | `rig_harness_protocol::SIGNAL_EXIT_CODE` | 0.47 |
| `launcher::build_failure_message(what, summary)` | `launcher::build_failure_message(&home, what, summary)` | 0.47 |
| `rig_basics::sessions::agents::ShownId` | `rig_harness::main_agent::CurrentAgent` | 0.47 |
| `rig_harness::edit_stopped` | `rewrite_stopped` | 0.47 |
| `PluginTimings::totals`, `SystemTimings::systems` | `PluginTimings::last_frame` | 0.47 |
| `PluginSystems::iter` | `PluginSystems::plugin(system, schedule)` | 0.47 |
| `FileChanges::paths` | `FileChanges::event` (`WatchEvent::paths()`) | 0.47 |
| `Home::plugins_toml`, `Home::defaults` | `Home::settings`, `rig_harness_protocol::config::chosen` | 0.47 |
| `rig_harness_protocol::{essential, ESSENTIAL_PLUGINS}` | `rig_harness_protocol::config::{essential, BUILT_IN}` | 0.47 |
| `RunDir::{ready, refusal, screen}`, `SessionDir::switch` | `end::AgentEnd::{read, write}`, `RunDir::{say_ready, heard_ready, say_screen, took_screen}` | 0.47 |
| `rig_tools::{Read, Write, Edit, Search, List, Delete}::{workspace, base, confine}` | `root: rig_tools::workspace::Root` | 0.47 |
| `rig_tools::fs::{resolve, place}` | `<dyn Workspace>::{resolve, place}`, `Root::{resolve, place}` | 0.47 |
| `rig_tools::fs::read_text` | `<dyn Workspace>::read_text` | 0.47 |
| `Workspace::replace` | `Workspace::read`, then `Workspace::write_if` | 0.47 |
| `Workspace::write_if(path, bytes, &hash)` | `Workspace::write_if(path, bytes, Some(&hash))` | 0.47 |
| `Listing::keeps(below)` | `Listing::keeps(below, dir)` | 0.47 |
| `rig_tools::clip(line, n)` | `rig_tools::excerpt(line, None, n).kept()` | 0.47 |
| `ReadArgs::shown` | `ReadArgs::show` | 0.47 |
| `process::{list_group, unlist_group}` | `Process::start`, `Process::start_with_stdio` | 0.47 |
| `process::record_groups(path)` | `process::record_groups(path, adopted)` | 0.47 |
| `attach::Attached::missing` | `attach::Attached::warnings` | 0.47 |
| `Watch` items `Vec<PathBuf>` | `WatchEvent` | 0.47 |
| `rig_basics::sessions::agents::AgentsPlugin` | `rig_basics::sessions::forks::ForksPlugin` (with `agent_picker::AgentPickerPlugin`) | 0.47 |
| `rig_basics::sessions::agents::{AgentColumn, ColumnOf, AgentColumns}` | `rig_basics::agent_picker::{AgentColumn, ColumnOf, AgentColumns}` | 0.47 |
| `rig_basics::compaction::plan` | `Summarizing::plan` / `plan_at`, then `Summarizing::request` | 0.47 |
| `rig_basics::compaction::Summarizing::keep_from` | `Summarizing::replaces` | 0.47 |
| `rig_basics::subagents::Request::asker` | `Request::to` | 0.47 |
| `rig_basics::subagents::OpenRequests` | `subagents::Awaiting`, `rig_ecs::requests::RequestOf` | 0.47 |
| `rig_basics::helpers::AT_ONCE` | the `[helpers] at_once` setting | 0.47 |
| `AgentUsage::record(..)` | trigger `telemetry::usage::Spent` | 0.47 |
| `EffectLog::read_matching(&filter)` | filter `EffectLog::read()` | 0.47 |
| `ReloadStatus::ask(agent)` | `ReloadStatus::ask(agent, Rebuild::Build)` | 0.47 |
| `rig_tui::TuiPlugin` (with status line, hints and title) | `rig_tui::TuiPlugins` | 0.47 |
| `[tui] mark_nonce_env` | `[title] mark_nonce_env` | 0.47 |
| `ToolCallView`, `EntryView`, `ContentView`, `AsideView` and `view()` | `ToolRender`, `EntryRender`, `ContentRender`, `AsideRender` | 0.47 |
| `rig_tui::ContentBlock`, `RenderContent::block()` | `RenderContent`, `RenderContent::kind()` | 0.47 |
| `TranscriptItem::Call(CallId)`, `Decoration(CallId, ..)` | `TranscriptItem::Call(CallItem)`, `Decoration(CallItem, ..)` | 0.47 |
| `DrawnTranscript::item` | `DrawnTranscript::rows_of` with `rows()` | 0.47 |
| `KeyPress::shift(code)` | `KeyPress::new(code, KeyModifiers::SHIFT)` | 0.47 |
| `rig_tui::wrap` (own module), `rig_tui::RowWrap` | re-exports of `rig_markdown::wrap` | 0.47 |
| `McpState::{NotApproved, Denied}` | `McpState::Held` with `McpHold` | 0.47 |
| `rig_steel::AgentSpec` | `rig_ecs::spawn::AgentSpec` | 0.47 |
| `RIG_HOME/plugins.toml` (`[[plugin]]` tables) | `[plugins] enable` / `disable` / `off` and `[[plugins.extra]]` in `RIG_HOME/settings.toml` | 0.47 |
| `RIG_HOME/defaults.json` | `[model] default` / `effort` in `RIG_HOME/settings.toml` | 0.47 |
| `<project>/.rig/settings.toml` | `RIG_HOME/settings.toml` | 0.47 |
| `[compaction] keep_recent` | `[compaction] keep_recent_tokens` | 0.47 |
| `[compaction] limits.reserve` | `[compaction] reserve_tokens` | 0.47 |
| project MCP approval inside `rig_mcp_client::McpPlugin` | `rig_mcp_client::approval::McpApprovalPlugin` (off by default) | 0.47 |
| built-in secret redaction (`[REDACTED]`) | `rig_basics::redact::RedactPlugin` (off by default, `‹SECRET_n›`) | 0.47 |
| `rig-harness = "0.46.0"`, `bevy_ecs = "=0.20.0"` in a plugin's `Cargo.toml` | `rig-harness = "0.47.0"`, `bevy_ecs = "0.20"` | 0.47 |
| `rig acp --tools=read` | `rig --tools=read acp` | 0.47 |
| `--json` lines with `"v":1` | `"v":3` | 0.47 |
| `--json` `tool_call` / `tool_result` `id` (provider call id) | the call's place, such as `m4/1`; pair by `agent` and `id` | 0.47 |
| `/fork`, rewind and `/agents` through `rig_basics::sessions::agents::AgentsPlugin` (on) | `rig_basics::sessions::forks::ForksPlugin` and `rig_basics::agent_picker::AgentPickerPlugin` (off by default) | 0.47 |
| `[subagents] max_depth` | removed; `[subagents] model`, `tools` | 0.47 |
| `OLLAMA_API_BASE_URL=http://host:11434/v1` | `OLLAMA_API_BASE_URL=http://host:11434` | 0.47 |
| `Native::reasoning_blob(&Api)` | `Native::reasoning_blob()`, with `ReplayTarget::blob_slots` | 0.46 |
| `history::{USER_IMAGE_OMITTED, ASSISTANT_IMAGE_OMITTED}` | `history::IMAGE_OMITTED` | 0.46 |
| `ImageDetail::from_str` | serde (`serde_json::from_value`) | 0.46 |
| `rig_tools::attach::attached_file` | `rig_core::conversation::attached_file` | 0.46 |
| `openai::wire::Identity::session_ids` | `Quirks::session_header` | 0.46 |
| `MessagesDecoder::new(bool)` | `MessagesDecoder::new(anthropic::wire::Quirks)` | 0.46 |
| `Compat::chat_tools_need_reasoning_off` | `Compat::chat_tools` (`ChatTools`) | 0.46 |
| `ContextUse::window` | `ContextUse::limit` | 0.46 |
| `xiaomimimo::{MIMO_V2_FLASH, MIMO_V2_OMNI, MIMO_V2_PRO}` | `MIMO_V2_6_FLASH`, `MIMO_V2_6_PRO` | 0.46 |
| `Summarizer::request(.., focus: &str, ..)` | `Summarizer::request(.., &CompactReason, ..)` | 0.46 |
| `rig_memory::output_room` | rig-basics compaction (no public replacement) | 0.46 |
| `DirectRecordingHttpClient` | `ProviderCassette::direct_http()` | 0.46 |
| `rig_ecs::turn::{Adapter, Adapters, ModelStarts}` | `turn::Adapted` in `PrepareSystems::{Adapt, AdaptOutermost}` | 0.46 |
| `rig_ecs::agent::Retry`, `Conversation::resume` | `Deliver::new(agent, text, DeliveryMode::Queue)` | 0.46 |
| `SubagentSystems::StopRestored` | `GateSystems::After`, `restore::CarryOn` | 0.46 |
| `add_tool_per_call(tool, options, closure)` | `add_tool_per_call(tool, options)` with `PerCallTool::at` | 0.46 |
| `rig_tui::AppToolRenderersExt` and its `add_*_renderer` / `add_call_decorator` | systems in `TranscriptRender` writing `RenderBatch` | 0.46 |
| `rig_tui::highlight::highlight`, `ToolCallView::header_spans` | `highlight::Code`, `ToolCallView::header_code` | 0.46 |
| `rig_basics::login_chatgpt::{ChatgptLoginPlugin, PendingLogin}` | `rig_basics::login::{LoginPlugin, PendingLogin}` | 0.46 |
| `rig_basics::coding_tools::ShellSettings`, `Shell::{max_timeout, quiet}`, `Ended::Quiet` | removed (no default shell timeout) | 0.46 |
| `SessionPicker::show`, `SessionItem` | `ResumeLabel`, a system after `SessionPickerSystems` | 0.46 |
| `rig_basics::reload::{SELF_EXTENSION_TAG, self_extension, ReloadAsked, MAX_RELOADS, ORIGIN}` | removed | 0.46 |
| `rig::core`, `rig_agent::core` | `rig`, `rig_core` | 0.45 |
| `rig_core::memory`, `ConversationMemory` and policy wrappers | `rig_core::conversation` plus `rig_memory::{Store, AsyncStore}` | 0.45 |
| `rig_core::transcript::{validate_canonical, TranscriptError}` | `conversation::pairing::{validate, PairingError}` | 0.45 |
| transcript tool-result/feedback helpers | `rig_core::tool::feedback` | 0.45 |
| `PortableTool`, `PortableToolEmbedding`, `tool::portable` | `Tool`, `ToolEmbedding`, `tool` | 0.45 |
| `ToolFn::new` | `DynamicTool::new_with_context` | 0.45 |
| `rig::bus`, `rig_agent::bus`, `Bus::channel`, `AgentBuilder::over_bus` | `handlers::Handlers`, `AgentBuilder::over_handlers` | 0.45 |
| `effect::Outcome` | `effect::Answered` | 0.45 |
| `serve::{Decision, Verdict, Origin}` | `serve::{BeforeServe, AfterServe, DispatchContext}` | 0.45 |
| `streaming::Transcript` | `streaming::StreamRecord` | 0.45 |
| `operation::Turn` | `operation::ReplyFold` | 0.45 |
| `http_client::LazyBytes` | `http_client::LazyBody<Bytes>` | 0.45 |
| HTTP middleware method/URI arguments | `http_client::Exchange` | 0.45 |
| `RunEntry` | `conversation::Aside` | 0.45 |
| `AgentBuilder<WithToolServerHandle>` and other tool-state parameters | `AgentBuilder` | 0.45 |
| `AgentRun::tool_results`, pending-call public fields | `ExecCall::answer` / `MalformedCall::answer`, `AgentRun::{answer, answer_all}` | 0.45 |
| `ModelTurn::from_response_parts`, `PreparedRequest` tool fields | `ModelTurn::from_response`, `prepared.policy` | 0.45 |
| `StreamedResolution::TurnAbandoned { skipped_tool_result }` | `TurnAbandoned`, `run::project` | 0.45 |
| `DocumentSourceKind::String` for binary data | `DocumentSourceKind::Base64` | 0.45 |
| document `data: DocumentSourceKind` | `DocumentData::Text` / `DocumentData::File` | 0.45 |
| `Catalog::{from_json, merge, find}` | `from_overrides` / `from_models_dev`, `with_overrides`, `get` | 0.45 |
| `ProviderRef`, free `registry::connect` | `Catalog::connect` / `connect_with` | 0.45 |
| `ProviderId::api_key_env` | `api_key_envs` | 0.45 |
| `gemini::AutoCache::for_model` | `AutoCache::for_spec` | 0.45 |
| config `.with_http(http)` | `.connect(http)` | 0.45 |
| `CopilotConfig`, `CopilotWire` | shared OpenAI configuration/wire with `copilot::DIALECT` | 0.45 |
| `Authenticator::new` file-path arguments | a `TokenStore` | 0.45 |
| `McpTool::from_mcp_server`, `tools_from_server` | `McpTool::new`, `McpConnection::tools` | 0.45 |
| `McpClientError::{ToolFetch, ToolFetchTimeout}` | `{Request, Timeout}` | 0.45 |
| `McpResponseMeta`, `McpStructuredContent` | `McpCallToolResult` | 0.45 |
| `rig_reqwest::runtime` | `rig_http::runtime::handle` | 0.45 |
| old `rig-ecs` world/driver API | new Bevy kernel, or classic `AgentRun` (no compatibility shim) | 0.45 |
| `rig::harness_protocol` (pre-release) | `rig_harness_protocol`, `rig_harness::harness_protocol` | 0.45 |
| separate default plugin crates (pre-release) | `rig_basics` modules; see [plugin paths](#16-plugins-written-against-pre-release-harness-builds) | 0.45 |
| `rig_basics::headless` (pre-release) | `rig_basics::print` | 0.45 |
| `rig_tools::{numbered, read_lines}` (pre-release) | `rig_tools::show_lines` | 0.45 |
| `PromptResponse::output` (field) and `output() -> &str` | `PromptResponse::output() -> String` | 0.44 |
| `PromptResponse::messages: Option<Vec<Message>>` | `PromptResponse::messages: Vec<Message>` | 0.44 |
| `PromptResponse::with_content` | set the public `content` field | 0.44 |
| `StreamingError` | `PromptError` | 0.44 |
| `PromptError::{CompletionError, MemoryError, MaxTurnsError, PromptCancelled}`, `PromptError::prompt_cancelled` | `PromptError::{Provider, Memory, MaxTurns, Cancelled}`, `PromptError::cancelled` | 0.44 |
| `StructuredOutputError::{PromptError, DeserializationError}` | `StructuredOutputError::{Prompt, Deserialization { output, error }}` | 0.44 |
| `MultiTurnStreamItem::StreamUserItem`, `streaming::StreamedUserContent` | `MultiTurnStreamItem::ToolResult { tool_result }` | 0.44 |
| `ToolCallDelta::call_id` | `ToolCallDelta::part` (the id arrives on the call's `End`) | 0.44 |
| `Message::Assistant { id, content }` | `Message::Assistant(AssistantMessage)` | 0.44 |
| `message::ReasoningContent` | `message::Reasoning { text, redacted, native }` | 0.44 |
| `operation::IfMalformed`, `MalformedToolInput`, `error::ErrorDetail`, `ProviderError::DuplicateCallId`, `require_non_empty*` | none | 0.44 |
| `Operation::validate` | `Operation::prepare` | 0.44 |
| Anthropic `Messages::{with_automatic_caching, with_automatic_caching_1h}` | `.cache(CacheRetention::Short)` / `.cache(CacheRetention::Long)` | 0.44 |
| `openai::wire::Chat::with_prompt_caching`, `OpenAiWire::with_prompt_caching`, Bedrock `Converse::with_prompt_caching` | `.cache(CacheRetention::Short)` | 0.44 |
| Bedrock `Converse::with_guardrail` | `rig_bedrock::extension::BedrockOptions::guardrail` | 0.44 |
| typed provider request/reply types (`openai::completion::Message`, `responses_api::*`, `anthropic::completion::*`, `gemini_api_types::*`, OpenRouter/Venice parameters, …) | `providers::<p>::extension::{…Options, …Extras}` with `provider_option(..)` and `extras::<P>()` | 0.44 |
| `openrouter::ProviderPreferences`, `venice::VeniceParameters` | `openrouter::extension::ProviderPreferences` via `OpenRouterOptions::provider`; `venice::extension::VeniceParameters` via `VeniceOptions::venice_parameters` | 0.44 |
| `ToolDefinition::name: String`, `ToolChoice::Specific::function_names: Vec<String>` | `ToolName`, `Vec<ToolName>` | 0.44 |
| `From<Agent> for DynamicTool`, `From<McpTool> for DynamicTool` | `TryFrom` | 0.44 |
| `McpClientError::{ConnectionError, ToolFetchError}` | `McpClientError::{Connection, ToolFetch}` | 0.44 |
| `UserContent::{audio, video, document}` | `UserContent::{audio_base64, video_base64, document_text}` (and `document_base64`) | 0.44 |
| `VectorStoreIndex::top_n` tuples, `VectorStoreOutput` | `VectorSearchResult<T>`, `VectorSearchIdResult` | 0.44 |
| `VectorSearchRequestBuilder::additional_params`, `VectorStoreError::BuilderError` | none, `VectorStoreError::SamplesOutOfRange` | 0.44 |
| `rig_helixdb::HelixDBClient`, `HelixDBVectorStore<C>` | `HelixDBVectorStore` (always `HelixDB`) | 0.44 |
| `AnthropicConfig::with_dialect(key, &dialect)` | `AnthropicConfig::with_key(&dialect, key)` | 0.44 |
| `BedrockRuntime::with_profile_name`, `DEFAULT_AWS_REGION` | `BedrockRuntime::builder().profile_name(..)` | 0.44 |
| `rig_bedrock::types::{assistant_content, converse_output}` (`RigAssistantContent`, …) | none: read `raw` or `extras::<BedrockExt>()` | 0.44 |
| `GeminiGrpc::from_val` | `GeminiGrpc::new(key).await` | 0.44 |
| `Ollama::completion` (native `/api/chat`) | `Ollama::native_completion` | 0.44 |
| `operation::ContextCache`, `operation::CachedContentFold` | `providers::gemini::cached_content::ContextCache`, `operation::Whole` | 0.44 |
| `effect::RerankRequest` | `operation::RerankRequest` | 0.44 |
| `embeddings::ImageEmbeddingResponse` | `embeddings::EmbeddingResponse` | 0.44 |
| `transcription::NormalizeTranscriptionResponse` | OpenAI's inherent `normalize()`; none for Gemini | 0.44 |
| `CompletionResponse::{provider, model, response_id, message_id}` (fields) | `provider()`, `model()`, `response_id()`, or `origin` | 0.44 |
| `Reasoning::{display_text, first_text, multi, redacted, …}`, `Sealed`, `Issuer` | `Reasoning { text, redacted, native }` | 0.44 |
| `deepseek`/`mistral`/`openrouter`/`llamacpp`/`venice::CompletionResponse` and other typed replies | `extras::<P>()` or `raw` | 0.44 |
| `ollama::wire::{Chat, Embeddings, Models}`, `cohere::wire::{Embeddings, ImageEmbeddings}` | `ollama::{chat::Chat, embedding::Embeddings, model_listing::Models}`, `cohere::embeddings::*` | 0.44 |
| `rig_cassette::http::DirectHeader` | `(impl AsRef<str>, impl AsRef<str>)` pairs | 0.44 |
| `rig_ecs::policy::text::*`, `policy::{turn_is_empty, answer_text, user_text}` | `rig_core::structured_output::*`, `transcript::{is_empty_assistant_turn, assistant_text_from_choice}`, `Message::user` | 0.44 |
| `rig_ecs::systems::{gather_turn, AssemblyInputs}` | `systems::fold_turn` | 0.44 |
| `rig_ecs::agent::checkpoint::CheckpointError` | `TurnHoldError` | 0.44 |
| `GenAiOperation::{ChatStreaming, Interactions, InteractionsStreaming}` (deprecated) | `GenAiOperation::Chat` + `SpanBuilder::streaming(true)` | 0.44 |
| `rig_core::OneOrMany<T>` (and the `one_or_many` module, both prelude re-exports) | `Vec<T>` — no replacement type; see the conversion table in "0.41 → 0.42" | 0.42 |
| `rig_core::EmptyListError` | none — use `message::require_non_empty` where you relied on the rejection | 0.42 |
| `one_or_many::string_or_option_one_or_many` | none — `json_utils::string_or_vec` into a `Vec<T>`, then `message::non_empty` where the `Option` carried "absent" | 0.42 |
| `ToolSet::builder()` / `ToolSetBuilder` (whole type) | `ToolSet::default()` plus `add_tool` / `add_dynamic_tool` / `add_portable_dynamic_tool` / `add_retrieved_tool` | 0.42 |
| `ModelListingError::{RateLimitError, ServiceUnavailable, UnknownError}` and the `auth_error` / `rate_limit_error` / `service_unavailable` / `unknown_error` constructors | `ApiError` (build the surviving variants directly) | 0.42 |
| `telemetry::ProviderResponseExt::{OutputMessage, get_output_messages}` | none — `get_text_response` is kept | 0.42 |
| `json_utils::null_or_vec` | `json_utils::null_or_default` | 0.42 |
| `http_client::with_bearer_auth` | `http_client::bearer_auth_header` on the builder's header map | 0.42 |
| `InMemoryVectorStoreBuilder::documents_with_id_f` | `documents_with_ids`, or the store's own `from_documents_with_id_f` / `add_documents_with_id_f` | 0.42 |
| `InMemoryVectorStore::get_document` | `iter()` | 0.42 |
| `message::Reasoning::optional_id` | the public `id` field, or `with_id` | 0.42 |
| `message::DocumentSourceKind::{raw, unknown}` | `DocumentSourceKind::Raw(..)` / `::Unknown` (the variants stay) | 0.42 |
| `message::Message::tool_result_with_call_id` | `UserContent::tool_result_with_call_id(item_id, call_id, name, content)` wrapped with `Message::from(..)`; `Message::tool_result` itself now takes `(call, name, content)` | 0.42 |
| `message::Image::try_into_url`, `message::Message::assistant_with_id`, `CompletionRequest::{with_provider_tool, with_provider_tools}`, `streaming::RawStreamingToolCall::with_internal_call_id` | none | 0.42 |
| `StreamedTurnEvent::EmitToolCallDelta::id` | none — `internal_call_id` is the correlator, and the provider's id arrives on the completed `ToolCall` | 0.42 |
| `MultiTurnStreamItem::final_response_with_history` | `final_response` | 0.42 |
| `anthropic::completion::apply_cache_control` | none — `apply_prompt_cache_control` is `pub(super)` | 0.42 |
| `TryFrom<message::AssistantContent> for anthropic::completion::Content` | none — convert the whole message with the surviving `TryFrom<message::Message> for anthropic::completion::Message` | 0.42 |
| `anthropic::completion::Citation`'s struct variants (`CharLocation { .. }` … `WebSearchResultLocation { .. }`) | the same five variants as newtypes over `CharLocationCitation`, `PageLocationCitation`, `ContentBlockLocationCitation`, `SearchResultLocationCitation`, `WebSearchResultLocationCitation` — field names and the wire shape unchanged | 0.42 |
| `azure::EmbeddingModel::{new, with_model}(.., Option<usize>)` | the `GenericEmbeddingModel` pair taking `ndims: usize`; `EmbeddingModel::make(.., None)` still infers | 0.42 |
| `azure::{EmbeddingResponse, EmbeddingData, Usage}`, `doubleword::{EmbeddingResponse, EmbeddingData, Usage}` | `openai::embedding::{EmbeddingResponse, EmbeddingData}` plus `openai::completion::Usage` (`openai::embedding` exposes no `Usage` path); for Doubleword, keep a `usage`-optional response type of your own | 0.42 |
| `doubleword::client::doubleword_api_types`, `openai::responses_api::OutputReasoning`, `deepseek::Message::{System, User, ToolResult}`, `ollama::AssistantContent` | none | 0.42 |
| `ollama::{UserContent, ImageUrl, SystemContent, SystemContentType}` | the `openai::completion` types of the same names | 0.42 |
| `copilot::ChatCompletionResponse` / `copilot::ChatChoice` | `openai::completion::CompletionResponse` / `openai::completion::Choice` | 0.42 |
| `mira::MiraError`, mira's inherent `Client::list_models` | `ModelListingClient::list_models` → `Result<ModelList, ModelListingError>` | 0.42 |
| gemini `interactions_api_types::{ImageDelta, AudioDelta, DocumentDelta, VideoDelta, FunctionCallDelta, FunctionResultDelta, CodeExecutionCallDelta, CodeExecutionResultDelta, UrlContextCallDelta, UrlContextResultDelta, GoogleSearchCallDelta, GoogleSearchResultDelta, McpServerToolCallDelta, McpServerToolResultDelta, FileSearchResultDelta}` | the matching `*Content` types, carried directly by `ContentDelta` | 0.42 |
| `rig_bedrock::streaming::BedrockUsage` | `rig_bedrock::types::converse_output::TokenUsage` | 0.42 |
| rig-bedrock's mirror→AWS-SDK `TryFrom` impls (`types::converse_output`) | none — the conversion is one-way now | 0.42 |
| `rig_candle::LlamaModelBuilder<'a>` | `rig_candle::CandleModelBuilder<'a>` | 0.42 |
| `CandleModel::{from_artifacts, from_artifacts_async, from_gguf_async, from_gguf_bytes_async}` | `builder_from_artifacts` / `builder_from_gguf_bytes` plus `build()` / `build_async()` | 0.42 |
| `CandleModel::model_family()` / `CandleModelBuilder::model_family(..)` | `conversation_protocol()` / `conversation_protocol(..)` | 0.42 |
| `rig_s3vectors::document!` | none — build `aws_smithy_types::Document` values directly | 0.42 |
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
| `ToolSetBuilder::dynamic_tool(ToolEmbedding)` | `retrieved_tool` (0.41), then `ToolSet::add_retrieved_tool` (0.42) | 0.41 |
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
