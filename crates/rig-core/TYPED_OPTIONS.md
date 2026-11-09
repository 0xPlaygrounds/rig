# Typed options: guarantees, option mapping per wire, provider options and catalog schema

This reference records what rig-core's typed generation options promise and
how each completion wire answers them. `GenerationOptions` carries the
portable knobs, the model catalog holds the per-model facts the encoders
read, each provider's `Options` and `Extras` types carry its own body keys
and reply fields, every wire's streamed `raw` rebuilds its unary document,
and citations and cost live in core types. The rustdoc of each type is the
API reference; this file holds the cross-wire tables.

"[unverified]" marks a cell no vendor page or recording confirms. Such a
cell is answered `Mapping::Unsupported` with the reason "unverified for
<provider>" until a vendor page or a recording confirms it. The JSON in such
a cell is what the wire sends once it is confirmed.

## 1. The five guarantees

Guarantees 1, 2 and 5 have a type-level core: a compiler error, or a
private constructor that only one function can call. A guard or a test
backs that core where the compiler cannot reach. Guarantees 3 and 4 have no
type-level core: guarantee 3 is the `extras-off-decode-path` source guard,
guarantee 4 a guard test. What in-crate code can still do despite the
mechanism is listed in section 1.1.

| # | guarantee | mechanism (type-level core where there is one) | backed by |
|---|---|---|---|
| 1 | No wire can silently drop an option. | Every completion wire's `ReplayTarget` must implement `fn map_options(&self, request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap`, which has no default, so a missing one is E0046. `Completion::prepare`, which the driver runs before every completion encode and which already refuses a wire that names no replay target, calls it on the routed target and reports each `Unsupported` slot through `on_unsupported` and fails a set option whose slot is `Nothing`, before `encode` runs. This holds for every wire, SDK-backed and local ones included, whatever its `encode` does. `map_options` destructures `OptionFields` with no `..` and returns an `OptionMap` struct literal with one `Mapping` per option. `GenerationOptions::fields` destructures `GenerationOptions` with no `..` inside rig-core. A new option breaks `fields` (E0027), then every wire's pattern (E0027) and every wire's `OptionMap` literal (E0063). The result is typed per field, so a wire cannot answer for one option in another option's slot. | The `options-mapping` guard rejects `..`, `_` and `_`-prefixed bindings in an `OptionFields` pattern, and a `..base` in an `OptionMap` literal. The `option_matrix` golden checks each slot against its section 2 cell and the body the wire sends, so `Omit` passes only where the cell says "omit", and a `Send` the wire never writes into its request fails. |
| 2 | Precedence lives in one place. | `options::request_params` is the one merge: base body, mapped options, provider options and `additional_params`, in that order. It returns a `FinalBody`, whose field is private to `completion::options`, so no wire can construct one. `FinalBody` has no `&mut` access, and `FinalBody::into_body` is how a completion wire gets its request bytes. | The `options-precedence` guard, over the completion-wire files, rejects reading `additional_params` as a field or through a struct pattern, any path to `Body::Bytes` or `Body::Multipart` (an expression or a pattern), an `impl` on `Body` (whose `Self::Bytes` it cannot resolve), a qualified `<Body>::..` path, `Body` itself in a macro's tokens, a `.body_mut()` call, any `.body(..)` call whose argument is not `FinalBody::into_body()` or `Body::empty()`, `FinalBody::deserialize` turbofished to `Value` or `Map`, and `serde_json::to_value`, `to_vec` or `to_string` of a function parameter typed `CompletionRequest`. It sees only the listed files and names, not types, so it does not stop every copy of a `FinalBody` or of the request into JSON, nor a `Body` built by a helper or macro in an unlisted file (section 1.1). |
| 3 | The decoding path cannot depend on typed views. | Each provider's `Options` and `Extras` live in `providers::<p>::extension`, a module no other provider module names. Rust has no visibility that hides a module from its siblings while keeping it public, so the prompt makes a source guard this guarantee's mechanism. | The `extras-off-decode-path` guard parses each file with `syn` and resolves every `use` tree (grouped, aliased, glob, `self`, `super`) before matching. Outside an `extension` module, no non-test file under rig-core's `providers` tree or a companion provider crate's `src` may name a path through an `extension` module, or `ReplyExtras` or `ProviderExtension`. A `pub use` or `pub type` of an extension item is rejected everywhere. The guard's own tests cover each import form. |
| 4 | Every public model constant has a catalog entry. | The catalog is data, and the constants are `pub const`. No type links them, so this guarantee is a guard test, as the prompt specifies. | A guard test walks every public string model constant in rig-core's provider modules and the companion provider crates (a `pub const` or `pub static`, a `pub` associated `const`, any `const` in a trait impl, a `pub` trait's default `const`, and a `pub use` of any of these, named or glob, a glob followed through its target module's own `pub use` items, filed under the re-exporting module's vendor, with `rig_core::` paths resolved in rig-core) and looks it up in `Catalog::builtin()`. An item is a string constant when its type is a reference to `str` by any path or a `type` alias of one, or when its value is a string literal or resolves to a string constant, whatever its type is spelled as. An item-level macro call the guard has not read fails it. A `pub use` whose target the guard cannot resolve to a constant (an external crate's item) is read as no constant. |
| 5 | Streamed and unary `raw` agree for every API. | `Out::raw` moves behind `Emit = Free`, so a completion decoder cannot write `raw` (E0599). Every `Wire` names an associated `type Reassembler: Reassemble<Self::Event> + Serves<Self::Op>`, over its decoder's classified event. `Unreassembled`, which records nothing, serves only `Emit = Free` operations, so a completion wire must name a reassembler that rebuilds its document. The driver, not the decoder, feeds it every known event of a stream and records its `finish()` as `raw`, so no decoder can skip one. | A whole-document parity test runs over every recorded unary and streamed pair, and a guard test fails when the corpus records a turn both ways (the same request body, streaming switches aside, to the same endpoint) that has no parity row. |

Provider options and citations have type checks of their own: misplaced
provider options are E0308, E0609 or E0616, and hand-written spans and
citation lists are E0451 or E0616.

### 1.1 Known limits

These are the bypasses the mechanisms leave open inside a crate. Each is
listed with what covers it.

- **Guarantee 1.** The compiler checks that every wire answers for every
  option, and `prepare` reports every refusal, but not that the answer is
  right or that the wire keeps it. A wire may return `Omit` where its cell
  says `Send`, or return `Send` and leave the value out of what it encodes.
  The second matters for the wires `request_params` does not bind (the
  first guarantee 2 bullet below). The `option_matrix` golden catches both for every wire and
  dialect in its table, because it compares the encoded request with the
  cell.
- **Guarantee 1.** `Descriptor::replay` is an `Option`, so a completion wire
  that names no replay target compiles without `map_options`. It cannot send
  anything: `Completion::prepare` refuses every request it is given.
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
  reaches the request.
- **Guarantee 2.** `Encoded` and `http::Request` keep public constructors,
  because every non-completion wire uses them. `Body::Bytes`,
  `Body::Multipart` and any `.body(..)` on an `http::request::Builder` are
  therefore not a compile error in a completion wire. The
  `options-precedence` guard is the barrier there, and it covers only the
  listed completion-wire files.
- **Guarantee 2.** The guard matches paths, not types or expansions. It
  rejects `Body::Bytes` and `Body::Multipart` by any import, alias or glob,
  an `impl` on `Body`, a qualified `<Body>::..` and `Body` in a macro's
  tokens. A macro defined in an unlisted file and invoked with only a
  variant's name (`b!(Bytes, bytes)`) names no `Body` at the use site, so it
  is the unlisted-helper case below, and nothing but review stops it.
- **Guarantee 2.** The serialize check sees only a function parameter typed
  `CompletionRequest`, by name or as `&name`, because the guard cannot type
  a binding. `let copy = request; serde_json::to_value(copy)`,
  `to_value(&*request)` and `to_value(request.clone())` pass it, and the
  resulting JSON can be read for `additional_params` outside
  `request_params`. Such a read cannot reach the request bytes in a listed
  file (the `Body` rules above), but it can steer what the wire builds, and
  nothing but review stops it.
- **Guarantee 2.** The SDK-backed wires (Vertex, Gemini gRPC) and Candle
  turn the `FinalBody` into their own request type with
  `FinalBody::deserialize::<T>()`, and Bedrock serializes it to bytes for
  its SDK call. The compiler cannot stop a write to that typed value
  afterwards; every such write (Vertex `contents` and `model`) sits in the
  base builder.
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
- **Guarantee 2.** Code reads raw keys through `options::param` and
  `BaseInput::param`. Reading cannot change rank,
  `BaseInput::raw_tools` can take out only `tools`, which every wire
  appends, and `BaseInput::claim` takes a field out of the provider
  layer only, for the base to place itself (Responses `input`).
- **Guarantee 3.** The guard resolves names within a file. A macro that
  expands to an extension path is not seen. No provider module defines such
  a macro, and the guard rejects `macro_rules!` that mention
  `extension`.
- **Guarantee 3.** The guard reads every path only on the decode path
  (rig-core's `providers` tree and the companion provider crates' `src`).
  Elsewhere it checks `pub use` and `pub type` alone, so a crate-private
  helper outside those roots (for example under `completion/`) may call
  `extras::<P>()` or name a marker, and a decoder could call that helper.
  Nothing but review stops it.
- **Guarantee 4.** The guard reads source, not types. A model id held in a
  constant of another type (an `Option<&str>`, an array, a struct) is not
  read, and a `pub use` of an external crate's item is read as no constant.
  The item-level macros it allows (`openai_vendor!`, `anthropic_vendor!`,
  `rest_messages!`, `tonic::include_proto!`) were read by hand; a change to
  one that makes it emit a model constant is not seen.
- **Guarantee 5.** The type ensures every known event reaches a reassembler
  that is not `Unreassembled`. It cannot ensure the reassembler folds each event
  correctly: a wire may name a fold that drops a field, or one that
  records nothing at all, and implement `Serves<Completion>` for it. That
  is the parity test's job, over every recorded pair, and review's for a
  wire with none. Interactions, xAI Chat and the ChatGPT backend have no
  recorded pair; their reassemblers are checked by unit tests and
  hand-built pairs only. The guard finds a pair only when its two requests
  send the same JSON body, `stream` and `stream_options` aside, to the same
  endpoint; pairs whose prompts differ are listed by hand.
- **Guarantee 5.** The driver records `raw` when the reply ends, fails or
  reaches the end of its frames. A caller that stops polling a stream
  before then never reaches that point, so `Streamed::partial()` carries
  no `raw` for it.
- **Guarantee 5.** `CompletionResponse::raw` stays a public field. Code that
  runs after the driver (a session layer, the caller) can still overwrite
  it. The guarantee covers what the driver records.
- **Decision E.** `Text::with_citations` and `Text::span` stay public, so a
  caller can attach a citation built from offsets in the wrong unit. `span`
  takes byte offsets into that text and refuses a range off a character
  boundary, but it cannot know what the offsets meant. Decoders cannot call
  either: the `extras-off-decode-path` guard also rejects them under the
  `providers` tree. `Span` and `Citation` derive `Deserialize`
  for stored history, so `serde_json::from_value` builds either without the
  guard seeing a constructor; `Text::with_citations` still drops a span off
  a character boundary of its text.

## 2. Option mapping per completion wire and dialect

Each cell is the JSON merged into the request body (`Mapping::Send`), "omit"
(`Mapping::Omit`: the provider default already does what was asked, so an
omitted option is told apart from a dropped one), or "unsupported"
(`Mapping::Unsupported`) with its reason. An unsupported cell is
`UnsupportedOption` under `Error` and a warning under `Ignore`. A cell that
puts markers inside arrays (MiniMax, Bedrock) is `Mapping::Place`, and the
base builder writes the markers from `BaseInput::cache`. Every
cell is a row of the `option_matrix` golden for the wires it covers, so a
wire may answer `Omit` only where its cell says "omit". "model" means the catalog row decides (section 4). Doc labels are
defined at the head of each subsection.

### 2.1 Anthropic Messages and its dialects

Docs: [AM] https://platform.claude.com/docs/en/api/messages/create,
[AE] https://platform.claude.com/docs/en/build-with-claude/effort,
[AA] https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking,
[AX] https://platform.claude.com/docs/en/build-with-claude/extended-thinking,
[AG] https://platform.claude.com/docs/en/about-claude/models/migration-guide,
[AP] https://platform.claude.com/docs/en/build-with-claude/prompt-caching,
[AS] https://platform.claude.com/docs/en/api/service-tiers,
[AT] https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview.

Model classes (rig constants at `crates/rig-core/src/providers/anthropic/completion.rs`):

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
| `Effort(Low/Medium/High)` | A2-A8: `"thinking":{"type":"adaptive"},"output_config":{"effort":"<level>"}`. A1: `"output_config":{"effort":..}` only. A0: unsupported | `output_config` deep-merges with `output_config.format`; on binding models the existing `block_binding` joins the same `thinking` object (`crates/rig-core/src/providers/anthropic/completion.rs`) | [AE], [AA] |
| `Effort(XHigh)` | A3-A8 as above with `"xhigh"`; A0-A2 unsupported | | [AE] |
| `Effort(Max)` | A2-A8 as above with `"max"`; A0, A1 unsupported | | [AE] |
| `Budget { tokens }` | A0-A2: `"thinking":{"type":"enabled","budget_tokens":N}`; A3-A8 unsupported | N >= 1024 and N < `max_tokens`, else unsupported. A2 accepts it though deprecated | [AX] |
| `CacheRetention::None` | no `cache_control` anywhere; with `with_prompt_caching()` or `with_static_prefix_cache_ttl(..)` on the wire, unsupported: "the wire places cache markers" | a caller's tool `cache_control` in `additional_params.tools` is kept | [AP] |
| `CacheRetention::Short` | top-level `"cache_control":{"type":"ephemeral"}` (automatic caching) | byte for byte what `with_automatic_caching()` sends. The wire's placement knobs stay as they are and are not options: `Messages::with_prompt_caching()` (tool, system and last message block) and `with_static_prefix_cache_ttl(..)` (tool and system markers). Manual markers take their TTL from `BaseInput::cache`, so `with_prompt_caching()` plus `cache(Long)` sends what `with_prompt_caching().with_automatic_caching_1h()` sends. `with_prompt_caching()` with `cache` unset sends no top-level marker. 4 markers at most. Below the model's minimum prefix the API declines to cache; that is the provider's choice, not a drop | [AP] |
| `CacheRetention::Long` | top-level `"cache_control":{"type":"ephemeral","ttl":"1h"}`; with `with_prompt_caching()`, the base builder's tool and system markers carry `"ttl":"1h"` too | byte for byte `with_automatic_caching_1h()`. 1 h markers precede 5 min markers (check kept, `crates/rig-core/src/providers/anthropic/completion.rs`) | [AP] |
| `ServiceTier::Auto` | `"service_tier":"auto"` | | [AS] |
| `ServiceTier::Default` | `"service_tier":"standard_only"` | | [AS] |
| `ServiceTier::Priority` | unsupported: no "priority only" value; `auto` uses Priority capacity only under a commitment. Also unsupported by model on A4-A7 and Fable 5.1 | the served tier is `usage.service_tier` | [AS] |
| `ServiceTier::Flex` | unsupported: no flex tier | Batch is another endpoint | [AS] |
| `verbosity` | unsupported: no verbosity knob | `effort` changes depth, not verbosity | [AM] |
| `parallel_tool_calls: true` | omit (the default) | | [AT] |
| `parallel_tool_calls: false` | `"tool_choice":{"type":"auto","disable_parallel_tool_use":true}`, or the caller's `any`/`tool` choice with the flag added | merges into the `tool_choice` object (`crates/rig-core/src/providers/anthropic/completion.rs`); with no tools, omit | [AM] |
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
(`crates/rig-core/src/providers/anthropic/wire.rs`). That is a
correctness note outside this work.

### 2.2 OpenAI Chat Completions and its dialects

One wire (`crates/rig-core/src/providers/openai/wire/chat.rs`) serves every
dialect in `crates/rig-core/src/providers/registry.rs`. Mapped options
merge between the typed fields and `additional_params`. The dialect body
rewrites (DeepSeek, Mistral, Moonshot, Perplexity, Mira) run after the merge
as `Rewrite::ChatDialect` and read the merged body, so a mapped option and a
raw key drive them alike.

Cross-cutting rules:
- Reasoning levels depend on model and route: Azure documents `max` only on
  GPT-6 and GPT-5.6 with Responses, `xhigh` only on GPT-6, 5.6, 5.5, 5.4 and
  `gpt-5.1-codex-max`, `minimal` only on GPT-5, `none` from GPT-5.1
  (https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning).
- GPT-5.6 and GPT-6 on Chat reject function tools unless `reasoning_effort` is
  `none`, and the default is `medium`. `validate` must see tools and reasoning
  together. `check_body` refuses it for the rows that cannot turn reasoning
  off, through `on_unsupported`.
- Automatic caching: `Short` = omit where the dialect caches by default;
  `None` = unsupported unless a documented off switch exists; `Long` =
  unsupported unless a longer retention knob exists.
- `stop` limits: OpenAI, OpenRouter, Groq, Venice, Z.AI, MiMo 4; Moonshot 5; DeepSeek 16.
  A longer list is unsupported with the limit as reason.
- Mira rejects every pass-through parameter (`chat.rs`): every option is unsupported.
- A cache key (`prompt_cache_key`, `session_id`) is not in `GenerationOptions`; it is a provider option (section 3).
- Where a cell says "catalog row" for a gateway dialect (OpenRouter `Off`,
  Mistral, Groq and Venice reasoning), one rule reads it
  (`openai::options::listed_reasoning`): a row that lists its reasoning
  options (effort levels, a budget, or a way to turn reasoning off), or that
  marks a model that does not reason, decides with the check
  `ModelSpec::validate` makes; a row that marks a reasoning model but lists
  no option, and an id the catalog does not list, are sent as asked.

**OpenAI** (`"openai"`, Chat route; [OC] https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create, [OK] https://developers.openai.com/api/docs/guides/prompt-caching):

| option | JSON | notes | doc |
|---|---|---|---|
| `Off` | `"reasoning_effort":"none"` | a catalog row whose levels include `none` (GPT-5.1+, not the `-pro` rows, `gpt-5.2-chat-latest`, `gpt-6-astra` or `gpt-6.1-sol`); an unlisted id unless its name is o-series, `-pro` or GPT-5.0; non-reasoning models: omit | [OC] |
| `Effort(e)` | `"reasoning_effort":"minimal"\|"low"\|"medium"\|"high"\|"xhigh"\|"max"` | the levels the catalog row lists; an unlisted id takes every level; non-reasoning model unsupported | [OC] |
| `Budget` | unsupported: effort only | | [OC] |
| `cache None` | GPT-5.6+: `"prompt_cache_options":{"mode":"explicit"}` with no breakpoints; earlier: unsupported (always on) | | [OK] |
| `cache Short` | before 5.6: `"prompt_cache_retention":"in_memory"`; 5.6+: omit (30 min is the only TTL) | GPT-5.5 and 5.5-pro take only `24h`: unsupported there | [OK] |
| `cache Long` | before 5.6, on extended-retention models: `"prompt_cache_retention":"24h"`; 5.6+: unsupported | the 24 h default for non-ZDR orgs makes "omit" differ from `Short` | [OK] |
| `service_tier` | `"service_tier":"auto"\|"default"\|"flex"\|"priority"` | `Flex` is model-limited | [OC] |
| `verbosity` | `"verbosity":"low"\|"medium"\|"high"` | reasoning models (the GPT-5 family); a model that does not reason (catalog `reasoning`, else its id) takes only `medium`: `Medium` omit, `Low` and `High` unsupported | [OC] |
| `parallel_tool_calls` | `"parallel_tool_calls":bool` | | [OC] |
| `top_p` | `"top_p":p` | the catalog row's `sampling`, as on Responses (one rule, `openai::options::openai_top_p`): `never` (o-series, GPT-5.0) unsupported; `reasoning_off` (GPT-5.1 and later) unsupported unless the effective effort is `none`, which with no `reasoning` set is the default of a row that can turn reasoning off and names no default (GPT-5.1 to 5.4), not of GPT-5.6 or GPT-6 (default `medium`); a row with no rule or an unlisted id by its name, as on Responses | [OC] |
| `seed` | `"seed":n` | deprecated, accepted | [OC] |
| `stop` | `"stop":[..]`, at most 4 | unsupported on reasoning models (the o-series and the GPT-5 family, catalog `reasoning`, else the name rule), at every effort including `none` | [OC] |

**Azure OpenAI** (`"azure.openai"`; https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning, https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/prompt-caching):
the OpenAI JSON for every field, with three differences. `Effort(Max)` is
unsupported on Chat. `service_tier` is unsupported [unverified]. rig's default
`api-version` `2024-10-21` (`dialects.rs`) predates `reasoning_effort`,
`verbosity` and `prompt_cache_options` [unverified whether it rejects them]. Azure keeps `OutputCap::Legacy` though its
reasoning models take only `max_completion_tokens`. The `top_p`, `stop` and
`verbosity` model rules apply by the deployment name, which reads as the
catalog id or the name rule; a deployment name that names no model sends
all three.

**OpenRouter** (`"openrouter"`; [OR] https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request, [ORR] https://openrouter.ai/docs/guides/best-practices/reasoning-tokens, [ORC] https://openrouter.ai/docs/guides/best-practices/prompt-caching, [ORT] https://openrouter.ai/docs/guides/features/service-tiers):

| option | JSON | notes | doc |
|---|---|---|---|
| `Off` | `"reasoning":{"effort":"none"}` | unsupported where the upstream's catalog row lists its reasoning options and none turns reasoning off (`openai/gpt-5-nano`, the GPT-5.0 and `-pro` rows, Gemini 2.5 Pro); a row that lists none is sent | [ORR] |
| `Effort(e)` | `"reasoning":{"effort":"<e>"}` | OpenRouter converts effort to an Anthropic budget itself | [ORR] |
| `Budget { tokens }` | `"reasoning":{"max_tokens":N}` | Anthropic and Qwen upstreams; OpenAI and Grok take effort only (model) | [ORR] |
| `cache None` | omit | unsupported on automatic-cache upstreams (OpenAI, DeepSeek, Grok, Groq), by `vendor/` prefix | [ORC] |
| `cache Short` | top-level `"cache_control":{"type":"ephemeral"}`; automatic upstreams: omit | replaces the system-message marker | [ORC] |
| `cache Long` | top-level `"cache_control":{"type":"ephemeral","ttl":"1h"}` | meaningful on Anthropic upstreams | [ORC] |
| `ServiceTier::Auto` | omit (requests use the standard tier unless one is named) | OpenRouter accepts only `default`, `flex`, `priority` (alias `fast`) and `ultrafast`; `auto` is not a value | [ORT] |
| `Default` / `Flex` / `Priority` | `"service_tier":"default"\|"flex"\|"priority"` | `ultrafast` is a provider option (section 3) | [ORT] |
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
`Off` `"reasoning_effort":"none"` and `Effort(e)` `"reasoning_effort":"<e>"`
where the catalog row allows them (`mistral-small-latest` takes `none` and
`high`; a model that does not reason refuses every effort), every value on
an unlisted id; `Budget` unsupported; cache
`Short` omit, `None` and `Long` unsupported; `Auto` `"service_tier":"auto"`,
`Default` `"service_tier":"standard_only"`, `Flex` and `Priority` unsupported;
`verbosity` unsupported; `parallel_tool_calls`, `top_p` as OpenAI; **`seed` is
`"random_seed":n`**; `stop` `"stop":[..]`.

**Groq** (`"groq"`; https://console.groq.com/docs/api-reference, https://console.groq.com/docs/reasoning, https://console.groq.com/docs/prompt-caching):
`Off` `"reasoning_effort":"none"` where the catalog row can turn reasoning
off (Qwen; GPT-OSS cannot, and an unlisted `gpt-oss` id is refused by its
name); `Effort(e)` `"reasoning_effort":"<e>"` for the levels the catalog
row lists, every level on an unlisted id: the API
lists `none`, `default`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max`
and rejects a value outside the model's set with a 400 (GPT-OSS takes
low..high; `qwen/qwen3.8-27b` takes low..high and maps `high` to its native
`xhigh`), so the catalog decides; `Budget` unsupported; cache `Short` omit, `None` and `Long`
unsupported ("cannot be manually disabled"); `Auto` `"auto"`, `Default`
`"on_demand"`, `Flex` `"flex"`, `Priority` `"performance"`; `verbosity`
unsupported; `parallel_tool_calls`, `top_p`, `seed` as OpenAI; `stop` at most 4.

**xAI** (`"xai"`, Chat route; https://docs.x.ai/developers/model-capabilities/text/reasoning, https://docs.x.ai/docs/api-reference):
`Off` `"reasoning_effort":"none"` on a model whose catalog levels include
`none` (grok-4.3), unsupported on other reasoning models ("cannot be
disabled"), omit on models that do not reason (the catalog's `reasoning`;
an unlisted id reasons unless it says `non-reasoning`); `Effort(Low..XHigh)` `"reasoning_effort":..` for
the levels the catalog row lists (grok-4.3 and grok-4.6+ low..xhigh;
grok-4.5 low..high, and xAI silently treats `xhigh` as `high` there, so the
catalog must refuse it; grok-3-mini low and high; none on a row with no
levels such as grok-4-0709), every level on an unlisted id; `Max`, `Minimal`, `Budget` unsupported; cache
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
`Off` `"reasoning_effort":"none"` and `Effort(e)` `"reasoning_effort":"<e>"`
where the catalog row allows them (`qwen3-5-9b` takes `none` and low..high,
so `xhigh` is refused although Venice accepts it), every value on an
unlisted id or a row that lists no reasoning option; `Budget` unsupported;
cache `None` unsupported, `Short`
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
unsupported on a model whose catalog row says it does not think (Command A,
R and R7B, Aya; an unlisted id is sent), other levels unsupported ("only
none and high"); `Budget` unsupported on this route; every `cache` value unsupported (no cache control); `service_tier`,
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
each is answered `Unsupported` until a recording confirms it.
**Hyperbolic** (`"hyperbolic"`), **Doubleword** (`"doubleword"`): no doc page
lists their parameters. `top_p`, `seed`, `stop` as OpenAI [unverified];
every other cell unsupported [unverified].
**Mira** (`"mira"`): every cell unsupported.
**A gateway rig does not ship** (`Dialect::gateway(..)`, and a Messages-format
`compatible(..)` dialect): every cell unsupported, with the reason that no
mapping is known for it; its fields go through `additional_params`.

### 2.3 OpenAI Responses and its dialects

One encoder (`crates/rig-core/src/providers/openai/responses_api/mod.rs`)
serves HTTP and the WebSocket session (`responses_api/websocket.rs`).
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
[OO] https://openrouter.ai/docs/api/api-reference/responses/create-a-response,
[GQR] https://console.groq.com/docs/responses-api,
[MMR] https://platform.minimax.io/docs/api-reference/responses-create.md,
[KMR] https://platform.kimi.ai/docs/api/responses,
[DWA] https://docs.doubleword.ai/inference-api/async-inference,
[DWE] https://docs.doubleword.ai/inference-api/reasoning-controls.

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
| `cache Long` | before 5.6: `"prompt_cache_retention":"24h"`; 5.6+: `"prompt_cache_options":{"ttl":"12h"}` | on 5.6+ `Short` is the default 30 minute TTL and `Long` the 12 hour one, which OpenAI's prompt caching guide documents for Responses only (fetched 2026-10-30) | [RK] |
| `ServiceTier` | `"service_tier":"auto"\|"default"\|"flex"\|"priority"` | `Flex` beta and model-limited; `Priority` not with EU residency on gpt-6 | [RC], [RF], [RP] |
| `verbosity` | `"text":{"verbosity":..}`, deep-merged with `text.format` (`responses_api/mod.rs`) | GPT-5 family and later; a model that does not reason takes only `medium`: `Medium` omit, `Low` and `High` unsupported, as on Chat | [RC] |
| `parallel_tool_calls` | `"parallel_tool_calls":bool` | | [RC] |
| `top_p` | `"top_p":p` | unsupported when the effective effort is not `none`: with no `reasoning` set that is the row's default, `none` on GPT-5.1 to 5.4 (rows that can turn reasoning off and name no default), `medium` on GPT-5.6 and GPT-6; o-series and gpt-5 reject it; gpt-5.5 unsupported by the name rule (OpenAI rejects `top_p` there with no effort set). The same rule as Chat | [RL] |
| `seed`, `stop` | unsupported: no such parameter | | [RC] |

A reasoning request adds `"include":["reasoning.encrypted_content"]` only
where the dialect's `ResponsesQuirks::reasoning_ciphertext` says the
endpoint returns it: not on Groq (which rejects `include`), Doubleword
(which accepts it and returns none), MiniMax or Moonshot (which document
none). Groq and Doubleword return reasoning as the item's own
`reasoning_text` content, so with `store: false` such an item replays as
returned rather than being left out.

From GPT-5.6 on, OpenAI's own Responses API places
`CompletionRequest::cache_points` as `"prompt_cache_breakpoint":{"mode":"explicit"}`
on the last `input_text`, `input_image` or `input_file` part at or before
each point, every point marked: the provider writes the latest three (four
when `prompt_cache_options.mode` is `explicit`) and looks up the rest.
`cache None` places none. Tools and `instructions` take no breakpoint. [RK].

**xAI** (`"xai"`): `Off` unsupported on grok-4.5, 4.6, 4.7 (`{"effort":"none"}`
on grok-4.3 only; omit on models that do not reason); `Effort(Low..XHigh)`
`"reasoning":{"effort":..}` for the levels the catalog row lists, as on Chat
(`XHigh` on grok-4.3 and grok-4.6+); `Minimal`, `Max`,
`Budget` unsupported; cache `Short` omit, `None`, `Long` unsupported; `Default`
`"default"`, `Priority` `"priority"`, `Auto`, `Flex` unsupported; `verbosity`,
`seed`, `stop` unsupported; `parallel_tool_calls`, `top_p` as OpenAI. [XR], [XT].

**OpenRouter Responses** (`"openrouter"`, Responses route): `Off`
`"reasoning":{"effort":"none"}`, unsupported where the upstream's row cannot
turn reasoning off, as on Chat; `Effort(e)` `"reasoning":{"effort":"<e>"}`;
`Budget` `"reasoning":{"max_tokens":N}` (upstream); cache `None` unsupported
on implicit upstreams, `"prompt_cache_options":{"mode":"explicit"}` on gpt-5.6+
upstreams; `Short` `"cache_control":{"type":"ephemeral"}` on Anthropic
upstreams, omit on OpenAI upstreams; `Long` `"cache_control":{"type":"ephemeral","ttl":"1h"}`
on Anthropic upstreams; `Auto` omit, `Default`, `Flex`, `Priority` as
OpenRouter Chat (`auto` is not an OpenRouter value, [ORT]); `verbosity`
`"text":{"verbosity":..}`; `parallel_tool_calls`, `top_p` as OpenAI; `seed`,
`stop` unsupported. [OO].

**Groq Responses** (`"groq"`, Responses route, beta): reasoning as Groq Chat,
spelled `"reasoning":{"effort":..}`; cache, `service_tier`, `verbosity`,
`parallel_tool_calls` and `top_p` as Groq Chat (the route takes Chat's
`on_demand` and `performance` tiers); `seed`, `stop` unsupported (rejected as
unknown fields); no `prompt_cache_key`. [GQR].

**MiniMax Responses** (`"minimax"`, Responses route): `Off`
`"reasoning":{"effort":"none"}` on MiniMax-M3, unsupported elsewhere
(M3.1-Flash rejects it, M2.x ignores it); `Effort(e)` `{"effort":"<e>"}` on
M3.1-Flash except `Minimal`, unsupported elsewhere, as on Chat; `Budget`
unsupported; cache as MiniMax Chat; `Default` `"standard"` (omit on
M3.1-Flash), `Priority` `"priority"` except on M3.1-Flash, `Auto`, `Flex`
unsupported; `verbosity`, `seed`, `stop` unsupported; `parallel_tool_calls`
and `top_p` unsupported (accepted and ignored); `cache_key`
`"prompt_cache_key"`. [MMR].

**Moonshot Responses** (`"moonshot"`, Responses route, Kimi K3 family only):
the input takes no `system` role, so the leading system messages go to
`instructions` and later ones stay in place as `developer` messages;
`Effort(Low/High/Max)` `"reasoning":{"effort":..}` on K3, other levels and
`Off` unsupported; `Budget` unsupported; cache as Moonshot Chat
(`prompt_cache_options`; a `prompt_cache_breakpoint` is rejected);
`service_tier`, `verbosity`, `parallel_tool_calls`, `top_p`, `seed`, `stop`
unsupported; `cache_key` `"prompt_cache_key"`. [KMR].

**Doubleword Responses** (`"doubleword"`, Responses route): `Flex`
`"service_tier":"flex"`, which the call waits for unless
`additional_params` sets `background`; `Priority` `"priority"` (realtime); `Auto`, `Default` unsupported;
`Off` and `Effort(e)` `"reasoning":{"effort":..}` within the catalog row's
levels; `Budget` unsupported; `top_p` `"top_p":p`; `parallel_tool_calls`
unsupported (`false` is rejected); `seed`, `stop`, `verbosity`, `cache`
and `cache_key` unsupported (accepted and ignored, live 2026-10-23). [DWA],
[DWE].

**ChatGPT / Codex backend** (`"chatgpt"`): no public doc; evidence is
`references/codex/codex-rs/codex-api/src/common.rs`,
`references/pi/packages/ai/src/api/openai-codex-responses.ts:553-597` and
the recorded echoes in `crates/rig-cassette/fixtures/cassettes/chatgpt/codex_sessions/`.
`Off` `"reasoning":{"effort":"none"}` (model); `Effort(e)`
`"reasoning":{"effort":..,"summary":"auto"}`, as pi sends, since without a
summary the backend streams no reasoning text (with one, a model may still
return empty summaries); `Budget` unsupported; cache
`Short`, `None` unsupported, `Long` omit (the backend echoes `24h`); `Flex` `"service_tier":"flex"`, `Priority`
`"service_tier":"priority"`, `Auto`, `Default` unsupported; `verbosity`
`"text":{"verbosity":..}`; `parallel_tool_calls` `bool`; `top_p`, `seed`,
`stop` unsupported. The Codex contract deletes
`parallel_tool_calls`, `service_tier` and `text` (`responses_api/mod.rs`),
which Codex itself sends. The Codex base builder still leaves out the typed
`max_tokens`, `temperature` and `output_schema`, which the backend does not
accept. They are request fields, not options, so guarantee 1 does not
cover them.

**Copilot Responses** (`"copilot"`, `*codex*` models): no public doc;
evidence `crates/rig-cassette/fixtures/cassettes/copilot/reasoning_roundtrip/streaming.yaml`.
`Effort(e)` as OpenAI with catalog levels (gpt-5.3-codex has no `none`, so
`Off` unsupported); `Budget` unsupported; cache `Long` omit (echoed default),
`Short`, `None` unsupported; `service_tier`, `verbosity` [unverified];
`parallel_tool_calls`, `top_p` as OpenAI; `seed`, `stop` unsupported.

**WebSocket mode**: the HTTP row of the session's dialect, inside
`{"type":"response.create", ..}`.

### 2.4 Gemini GenerateContent, Vertex, Interactions and gRPC

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
`crates/rig-core/src/providers/gemini/interactions_api/mod.rs`) is a
completion wire with no request body: it reads a stored interaction by id, and
its `encode` takes `_request: CompletionRequest` and uses none of it. Every
`GenerationOptions` field on it is `UnsupportedOption`, reason "a resumed
interaction is read, not created; its options were fixed when it was
created". `additional_params` (its `tools` and provider tools included,
which its base refuses because `request_params` hands them to the base) or
provider options on it are not options, so they are an `EncodeError` under
either policy. Its `map_options`
destructures `OptionFields` with no `..` and answers `Unsupported` for each
set field, so `prepare` refuses a set option before `encode` runs. Its
`encode` still calls `request_params` with an empty base and reads the
result: a non-empty
`FinalBody` (raw or provider keys) is that `EncodeError`. It sends
`Body::empty()`, the one body the `options-precedence` guard accepts besides
`FinalBody::into_body()`, so a `GET` never carries `{}`. So guarantee 1 holds
for it as for any wire: a new field fails to compile there too, and a set
option is an error under `Error` and a warning under `Ignore`, never dropped.

Raw `additionalParams.generationConfig` and `generation_config` keys win
over typed fields, as `request_params` merges them last. gRPC checks
`service_tier` itself before `from_rest`, which would otherwise fail with a
generic request error (`crates/rig-gemini-grpc/src/rest.rs`).
[G38] recommends stripping `temperature`, `top_p` and `top_k` on Gemini 3.x;
that is advice, not a documented 400, so the wire still sends them.

### 2.5 Bedrock Converse, Cohere native, Ollama native and Candle

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
(`crates/rig-bedrock/src/request.rs`).

| option | JSON | notes | doc |
|---|---|---|---|
| `Off` | Claude 5.x, Opus 4.6/4.7, and a Claude id the class table does not name (an application inference profile with `Family::Claude`, which may think by default): `{"additionalModelRequestFields":{"thinking":{"type":"disabled"}}}`; Claude 4.5 (A0, A1): omit; Fable, Mythos: unsupported; Nova 2 Lite: `{"reasoningConfig":{"type":"disabled"}}`; always-on models (DeepSeek R1): unsupported | model | [AD], [ET], [NV] |
| `Effort(Low/Medium/High)` | adaptive Claude: `{"thinking":{"type":"adaptive"},"output_config":{"effort":"<level>"}}`; Nova 2: `{"reasoningConfig":{"type":"enabled","maxReasoningEffort":"<level>"}}`; budget-only Claude: unsupported | Nova with `high` requires `temperature`, `topP` and `maxTokens` unset | [AD], [NV] |
| `Effort(Minimal)` | unsupported: no Bedrock model lists it | | [AD] |
| `Effort(XHigh/Max)` | adaptive Claude where the catalog lists the level [unverified: [AD] and pi disagree on which models take `xhigh`] | | [AD] |
| `Budget { tokens }` | Claude with budgets: `{"thinking":{"type":"enabled","budget_tokens":N}}`, N >= 1024 and < `maxTokens`; Opus 4.7, Claude 5, Fable, Mythos, Nova 2: unsupported | | [ET] |
| `cache None` | omit every `cachePoint` (implicit caching cannot be stopped; "no explicit checkpoints") | | [PC] |
| `cache Short` | `Mapping::Place`: the base builder appends `{"cachePoint":{"type":"default"}}` after the system blocks and at the end of the last message; models without explicit caching: unsupported | after a reasoning turn, only a Claude model the catalog lists gets the message checkpoint; for any other model (Nova 2, a model the catalog does not list) the mapping answers `Place`, and the base builder places the system checkpoint and reports the skipped message checkpoint through `BaseInput::refuse_cache`, never silently (`crates/rig-bedrock/src/request.rs`) | [CP], [PC] |
| `cache Long` | `Mapping::Place`, the same blocks with `"ttl":"1h"`; Claude 3.7 and 3.5 v2: unsupported | 1 h checkpoints precede 5 min ones | [CP], [PC] |
| `Auto` | omit `serviceTier` | | [ST] |
| `Default` / `Flex` / `Priority` | `"serviceTier":{"type":"default"\|"flex"\|"priority"}` | per-model tier support (model) | [ST] |
| `verbosity` | unsupported | | [CV] |
| `parallel_tool_calls` | unsupported: `toolChoice` has no flag | | [CV] |
| `top_p` | `"inferenceConfig":{"topP":p}`; Claude classes A3-A8 (section 2.1, Sonnet 5 included): unsupported | Claude thinking rejects it ([ET]) | [CV] |
| `seed` | unsupported: `inferenceConfig` has no seed | | [CV] |
| `stop` | `"inferenceConfig":{"stopSequences":[..]}` | | [CV] |

**Cohere native** (`"cohere"`, `api "cohere.chat"`):
`Off` `"thinking":{"type":"disabled"}` on models that think (thinking is on
by default; the catalog row's `reasoning`: `command-a-reasoning-08-2025`,
`command-a-plus-05-2026`, `north-mini-code-1-0`; an unlisted id whose name
says `reasoning`), omit otherwise; `Effort(High)` `"thinking":{"type":"enabled"}`,
other levels unsupported; `Budget` `"thinking":{"type":"enabled","token_budget":N}`;
`Effort` and `Budget` unsupported on a model whose catalog row says it does
not think (an unlisted id is sent);
cache `None` omit, `Short`, `Long` unsupported; `service_tier` unsupported
(`priority` is a queue integer, a provider option); `verbosity`,
`parallel_tool_calls` unsupported; `top_p` `"p":p`, 0.01 to 0.99 (1.0 is
unsupported); `seed` `"seed":n`; `stop` `"stop_sequences":[..]`, at most 5.
[CO], [COR].

**Ollama native `/api/chat`** (`"ollama"`, `api "ollama.chat"`):
`Off` `"think":false`; `Effort(Low/Medium/High)` `"think":"<level>"` where
`/api/show` lists the values, boolean-only models unsupported; `Effort(Max)`
model (rig accepts `max`, [OLT] lists none); `Minimal`, `XHigh`,
`Budget` unsupported; cache `None` omit, `Short`, `Long` unsupported
(`keep_alive` keeps the model loaded, a provider option); `service_tier`,
`verbosity`, `parallel_tool_calls` unsupported; `top_p`
`"options":{"top_p":p}`; `seed` `"options":{"seed":n}`; `stop`
`"options":{"stop":[..]}`. [OL], [OLT].

**Candle** (`"candle"`, local; no wire JSON, cells are `GenerationConfig`
fields): `Off` honoured (Qwen3's template already renders no-thinking mode,
`crates/rig-candle/src/protocol.rs`; other models do not reason);
`Effort`, `Budget` unsupported; cache `None` honoured, `Short`, `Long`
unsupported; `service_tier`, `verbosity`, `parallel_tool_calls` unsupported;
`top_p` `GenerationConfig.top_p`, 0 < p <= 1; `seed` `GenerationConfig.seed`;
`stop` unsupported (generation stops only on end tokens).

## 3. Provider extensions

Sources: the 0.43 typed types (`git show v0.43.0:<path>`) and current
provider docs. Every `Options` type is serialize-only; every `Extras` type reads
`raw`. Sections are `"*"` and the route `Api` names.

| provider key | route sections | Options fields | Extras fields | sources |
|---|---|---|---|---|
| `anthropic` | `"*"` | `top_k` (refused where the model fixes its sampling), `metadata_user_id`, `inference_geo`, `speed` (`fast` + beta; refused off Opus 5.5, 5 and 4.8), `task_budget` (`output_config.task_budget` + beta), `fallbacks` (`"default"` or `[{"model"}]`) + beta, `container` (an id or `{id?, skills}`), `context_management` + beta, `mcp_servers` + beta, `diagnostics_previous_message_id` + beta | `stop_reason`, `stop_sequence`, `stop_details`, `cache_creation { ephemeral_5m_input_tokens, ephemeral_1h_input_tokens }`, `service_tier`, `inference_geo`, `speed`, `server_tool_use`, `container`, `fallback_model` | v0.43 `crates/rig-core/src/providers/anthropic/completion.rs` 57-121, 212-230, 350-355; [AM] |
| `zai`, `minimax`, `moonshot`, `xiaomimimo` | `"openai.chat"`, `"anthropic.messages"` | Z.AI Chat: `do_sample`, `request_id`, `user_id`, `thinking.clear_thinking`. MiniMax Chat: `reasoning_split`; Messages: `metadata_user_id` (`metadata.user_id`). Moonshot Chat: `thinking.keep`, `prompt_cache_key`. MiMo: none (web search is a server tool). The Messages route takes no other documented field of these vendors' own | one flat type per vendor. Chat: Z.AI `request_id`; MiniMax `reasoning_details`, `base_resp`; MiMo `message.annotations`; Moonshot choice-level `usage`, `prompt_tokens_details`. Messages: `stop_reason`, `stop_sequence` | dialect docs in 6.1 and 6.2 |
| `openai` | `"*"`, `"openai.chat"`, `"openai.responses"` | `"*"`: `store`, `metadata`, `prompt_cache_key`, `safety_identifier`. Chat: `logit_bias`, `prediction`, `logprobs`, `top_logprobs`, penalties, `modalities`, `audio`, `web_search_options`. Responses: `input` (caller-built items, which the wire claims and places ahead of the request's own), `reasoning.{summary, mode, context}`, `include`, `conversation`, `truncation`, `context_management`, `prompt_cache_options.comparison_response_id`, `background` (refused on the WebSocket route), `max_tool_calls`, `top_logprobs`, `access_programs`. Not added: `n` (the decoder folds candidate 0), `include_obfuscation` (mode-only), `previous_response_id` (the WebSocket chain owns it), `prompt_cache_options.mode`/`.ttl` (`cache` owns them), `.prewarm` (the WebSocket warm-up), `service_tier_extra` (writes `service_tier`) | `OpenAiExtras`, flat: `service_tier` on both routes. Chat: `system_fingerprint`, `prompt_tokens_details`, `completion_tokens_details`, `annotations`. Responses: effective `reasoning` effort, summary, mode and context, `prompt_cache_retention`, `incomplete_details.reason`, `phase` per message item, `billing.payer` (unary only) | v0.43 `crates/rig-core/src/providers/openai/responses_api/mod.rs` 1378-1422, 1563-1610, 1659-1895; [OC], [RC] |
| `azure.openai` | `"openai.chat"` | the `openai` Chat fields, `data_sources`; no Responses section (unverified, and every option is refused there) | `service_tier`, `system_fingerprint`, token details, `prompt_filter_results`, `content_filter_results` | Azure docs in 6.2 |
| `openrouter` | `"*"` | `provider: ProviderPreferences { order, only, ignore, allow_fallbacks, require_parameters, data_collection, zdr, sort, preferred_min_throughput, preferred_max_latency, max_price, quantizations }`, `models` (fallbacks, never empty), `plugins`, `session_id`, `metadata`, `reasoning.exclude`, `reasoning.summary`, `top_k`, `min_p`, `top_a`, `repetition_penalty`, `user`. Not added: `route` (deprecated in the current docs), `trace` (unverified schema), `service_tier_extra` (writes `service_tier`) | `provider`, `native_finish_reason`, `service_tier`, `system_fingerprint`, `cost`, `cost_details`, `is_byok`, `prompt_tokens_details`, `server_tool_use_details`, `openrouter_metadata`, `annotations`, each read per route | v0.43 `crates/rig-core/src/providers/openrouter/completion.rs` 30-621; [OR], https://openrouter.ai/docs/guides/routing/provider-selection |
| `deepseek` | `"*"` | none: `thinking` and `reasoning_effort` are portable | `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, `reasoning_tokens`, `system_fingerprint` | v0.43 `crates/rig-core/src/providers/deepseek.rs` 20-55; [DS] |
| `mistral` | `"*"` | `prompt_mode`, `safe_prompt`, `prompt_cache_key`, penalties, `prediction`. Not added: `n` (candidate 0 only), `guardrails` (unverified schema) | `usage.service_tier`, `prompt_audio_seconds`, `num_cached_tokens`, `prompt_tokens_details` | v0.43 `crates/rig-core/src/providers/mistral/completion.rs` 120-160 |
| `groq` | `"*"` | `reasoning_format`, `include_reasoning` (each builder clears the other), `search_settings`, `citation_options`. Not added: `compound_custom` (unverified) | `x_groq`, `queue_time`, `prompt_time`, `completion_time`, `total_time`, `usage_breakdown`, `service_tier`, `executed_tools`, `system_fingerprint` | https://console.groq.com/docs/api-reference |
| `xai` | `"*"` | `prompt_cache_key`. Not added: `search_parameters` (live search is deprecated; unverified whether still accepted) | `cost_in_usd_ticks`, `num_sources_used`, `num_server_side_tools_used`, `server_side_tool_usage_details`. Not added: `citations` (no recording) | https://docs.x.ai/developers/cost-tracking, [XR] |
| `together` | `"*"` | `chat_template_kwargs`, `top_k`, `min_p`, `repetition_penalty`, `safety_model` | `warnings`, `message.reasoning` | https://docs.together.ai/reference/chat-completions-1 |
| `venice` | `"*"` | `venice_parameters: VeniceParameters { character_slug, strip_thinking_response, enable_web_search, enable_web_scraping, enable_x_search, enable_web_citations, include_search_results_in_stream, return_search_results_as_documents, include_venice_system_prompt }`, `prompt_cache_key` (`disable_thinking` is `Reasoning::Off`) | `venice_parameters` echo with `web_search_citations`, `cost { usd, diem }`. Not added: `cache_creation_input_tokens` (no recording) | v0.43 `crates/rig-core/src/providers/venice/completion.rs` 40-233 |
| `perplexity` | none | none: the Chat API was retired on 2026-09-27 | `citations`, `search_results`, `images`, `related_questions`, `usage.cost`, `search_context_size`, from the recordings | https://docs.perplexity.ai/api-reference/chat-completions-post |
| `llamacpp` | `"*"` | `chat_template_kwargs`, `reasoning_format`, `n_probs`, `samplers`, `top_k`, `min_p`, `typical_p`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `id_slot`, `timings_per_token` | `timings { cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, .. }` | v0.43 `crates/rig-core/src/providers/llamacpp/completion.rs` 20-60 |
| `ollama` | `"*"`, `"ollama.chat"` | `"*"`: `keep_alive` (`/v1` takes it too). `"ollama.chat"`: `options { num_ctx, num_keep, top_k, min_p, repeat_penalty, repeat_last_n, num_gpu, num_thread }` (typed fields, never an open map), `logprobs`, `top_logprobs`. Not added: `truncate`, `shift` (unverified) | native: `model`, `created_at`, `done_reason`, durations, `prompt_eval_count`, `prompt_eval_cached_count`, `eval_count`, `logprobs` | v0.43 `crates/rig-core/src/providers/ollama.rs` 124-294; [OL] |
| `cohere` | `"*"`, `"cohere.chat"` | `"*"`: `frequency_penalty`, `presence_penalty`. `"cohere.chat"`: `citation_mode` (`citation_options.mode`), `safety_mode`, `priority`, `top_k` (`k`), `logprobs`. `strict_tools` stays `with_strict_tools` (an encoder directive) | `id` (both routes); native: `finish_reason`, `billed_units`, `tokens`, `cached_tokens`, `tool_plan`, `logprobs` | v0.43 `crates/rig-core/src/providers/cohere/completion.rs` 26-305; [CO] |
| `copilot` | none | none: `intent` and `copilot_cache_control` are headers | `copilot_usage` (both routes), `prompt_filter_results` (Chat) | recorded parity snapshot `crates/rig-cassette/fixtures/parity/copilot.json` |
| `chatgpt` | `"openai.responses"` | `prompt_cache_key`, `client_metadata`, `access_programs`; no `store` (the wire forces `false`); `session_id` is a header | as `openai` Responses but `billing` | `references/codex/codex-rs/codex-api/src/common.rs` |
| `gcp.gemini` | `"*"`, `"gemini.generate_content"`, `"gemini.interactions"` | `"*"`: `store`, `labels` (both routes spell them the same). GenerateContent: `generationConfig` entries `include_thoughts`, `top_k`, penalties, `response_logprobs`, `logprobs`, `candidate_count` (only 1), `response_modalities`, `image_config`, `speech_config`, `media_resolution`, `enable_enhanced_civic_answers`; `safety_settings`. Interactions: `agent`, `agent_config`, `background`, `previous_interaction_id`, `safety_settings` (`type`/`threshold`), `thinking_summaries`, speech. Not added: `cached_content` (`with_cached_content` stays the one typed way), Interactions `response_modalities` (no longer in [IX-REF]), `response_format`/`response_mime_type` (they duplicate `output_schema`), transcription and video config (no verified shape), `service_tier_deferred` (it writes `service_tier`, which `ServiceTier` owns) | `GeminiExtras`, flat: GenerateContent `model_version`, `response_id`, `service_tier`, the four `usageMetadata` token details, `safety_ratings`, `prompt_feedback`, `finish_message`, `citation_metadata`, `grounding_metadata`, `url_context_metadata`, `avg_logprobs`, `logprobs_result`; Interactions `id`, `status`, `service_tier`, `created`, `updated`, per-modality usage, `grounding_tool_count` | v0.43 `crates/rig-core/src/providers/gemini/completion.rs` 622-680, 1314-1700, 2138-2155; v0.43 `gemini/interactions_api/mod.rs` 378-430, 1727-1935; [GR], [GX] |
| `vertexai` | `"vertexai.generate_content"` | the `generationConfig` entries every GenerateContent route takes, `routing_config`, `audio_timestamp`, `safety_settings` with `method`, `labels`, `model_armor_config`; no `store`, `enableEnhancedCivicAnswers` or `cached_content` | the GenerateContent extras but `service_tier` and `citation_metadata`, plus `traffic_type`, `create_time`, `citationMetadata.citations` | `google-cloud-aiplatform-v1` 1.11.0 `model.rs` |
| `gemini-grpc` | as `gcp.gemini` (`GeminiGrpcOptions` wraps `GeminiOptions`, so the refusal hook lives in the gRPC crate) | the GenerateContent fields the proto declares; `store`, `labels` and a `HARM_CATEGORY_JAILBREAK` safety setting go through `on_unsupported` | GenerateContent extras minus `service_tier`, `grounding_metadata` and `url_context_metadata` (rig's proto omits the first and last, `crates/rig-gemini-grpc/proto/gemini.proto`, and declares of `groundingMetadata` only the chunks and supports the decoder cites, which reach `raw` and `Text::citations()`) | [GPR] |
| `aws_bedrock` | `"*"` | `guardrail { identifier, version, trace }` (`guardrailConfig`, sent on both modes), `performance_latency` (`performanceConfig.latency`), `request_metadata`, `additional_response_field_paths`, `anthropic_beta` and `top_k` (under `additionalModelRequestFields`). Not added: `stream_processing_mode` (mode-only), `service_tier_reserved` (writes `serviceTier`), `thinking_display` | `stop_reason`, `latency_ms`, `cache_details`, `service_tier`, `performance_latency`, `trace`, `invoked_model_id`, `additional_model_response_fields` | v0.43 `crates/rig-bedrock/src/types/converse_output.rs` 26-297, v0.43 `crates/rig-bedrock/src/completion.rs` 128-189; [CV], [CS] |
| `candle` | `"*"` | `top_k`, `repeat_penalty`, `repeat_last_n`. Not added: `disable_top_p` (`top_p` owns the leaf) | `CandleExtras`: the ten `CandleCompletionResponse` fields | `crates/rig-candle/src/generation.rs` |
| `huggingface`, `hyperbolic`, `doubleword`, `mira` | `"*"` | none beyond generic OpenAI fields | none | |

Every field here is a body key. Headers and encoder directives are not
provider options. Server-tool definitions (web
search, web fetch, code execution) stay in `additional_params.tools`, which
every wire appends to rig's tools. Per-block cache
breakpoints are out of scope: a content-level marker belongs to neither
`GenerationOptions` nor a request-level `Options` type. Where a wire places
them, they come from `CompletionRequest::cache_points`.

**Field setters on the top-level `Options` type.** Where a common field
sits two builder levels deep, the top-level type has a setter that writes
the same leaf, through the nested builder, and a test pins it to the
nested form's JSON:
- Gemini `GeminiOptions` and Vertex `VertexOptions`: the
  `generationConfig` entries `include_thoughts`, `top_k`,
  `presence_penalty`, `frequency_penalty`, `response_logprobs`, `logprobs`,
  `candidate_count`, `response_modalities`, `media_resolution`
  (GenerateContent only);
- OpenAI `OpenAiOptions`: `store`, `metadata`, `prompt_cache_key`,
  `safety_identifier` (both routes); `reasoning_summary`, `include`,
  `conversation`, `truncation`, `background`, `max_tool_calls` (Responses,
  OpenAI's default route; each setter's rustdoc names its route). The
  Chat-only fields (`logit_bias`, `prediction`, `frequency_penalty`,
  `presence_penalty`) have no top-level setter: on the default Responses
  route their section is skipped with only a `debug!`, so a shortcut would
  silently do nothing. They stay on `ChatOptions` (`.chat(..)`) and on
  `AzureOptions`, whose one route is Chat;
- Azure `AzureOptions`: `logprobs`, `top_logprobs`, `frequency_penalty`,
  `presence_penalty`, `logit_bias`, `prediction`;
- Venice `VeniceOptions`: the `venice_parameters` entries `character_slug`,
  `strip_thinking_response`, `enable_web_search`, `enable_web_citations`,
  `include_venice_system_prompt`;
- Z.AI, Moonshot and MiniMax: every field of their route sections.

A field two routes spell differently stays on its section, because a
setter on the top-level type would hide which route it writes: Gemini's
safety settings (GenerateContent `safetySettings` vs Interactions
`safety_settings`), OpenAI's `top_logprobs` (on both sections) and Chat's
`logprobs` (Responses asks through `include`). A section setter replaces
the whole section, including fields set before it.

## 4. Catalog schema and the models.dev mapping

The data file is models.dev-shaped JSON: an object of provider keys, each with
a `models` object keyed by model id. `Catalog::from_models_dev` reads the same
shape leniently, and `Catalog::from_overrides` strictly for a user's
overrides; `with_overrides` lets the override's rows win field by field.
`cargo xtask catalog sync` lists under each row's `rig.pinned` the keys
`xtask/src/catalog/review.json` or a generator join set (`limit` and `cost`
by part), and `with_models_dev` lays a fetched models.dev copy on without
replacing them, so a refresh keeps the reviewed facts.

| models.dev field | `ModelSpec` field | rule |
|---|---|---|
| provider key | `provider` | through the key map below |
| `models.<id>.id` | `id` | |
| `name` | `display_name` | |
| `limit.context` | `context_window` | |
| `limit.input` | `max_input_tokens` | compaction and the context meter measure against `input_limit()` |
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
| `description`, `family`, `type`, `canonical_model_id`, `attachment`, `interleaved`, `modalities.output`, `knowledge`, `release_date`, `last_updated`, `open_weights`, `status` | `details` | as listed |
| `cost.{reasoning,input_audio,output_audio,tiers,context_over_200k}` | `details.cost` | as listed; `Pricing::cost` does not apply them |
| `provider`, `experimental.modes` | `details.provider`, `details.modes` | as listed; no wire applies them |

Sync keeps every key of a models.dev row but `id`, which is its key, and
warns about a key rig-core does not read. A reviewed row of a model
models.dev no longer lists must carry its facts and a `name`.

Provider key map: `openai` to `openai`, `azure` to `azure.openai`,
`anthropic` to `anthropic`, `google` to `gcp.gemini`, `google-vertex` to
`vertexai`, `amazon-bedrock` to `aws_bedrock`, `openrouter`, `deepseek`,
`groq`, `mistral`, `xai`, `venice`, `zai`, `minimax`, `cohere`, `perplexity`,
`huggingface` to themselves, `togetherai` to `together`, `moonshotai` to
`moonshot`, `xiaomi` to `xiaomimimo`, `ollama-cloud` to `ollama`,
`github-copilot` to `copilot`. models.dev has no rows for `hyperbolic`,
`doubleword`, `mira`, `llamacpp` or `chatgpt`; those entries are hand-written
or absent.

## 5. Wire-body changes for callers who set no option

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

The P3 rows change only what is sent for a model the catalog lists, by its
exact id or a dated snapshot of it (`-YYYYMMDD`, `-YYYY-MM-DD`; Anthropic's
`-20…` suffixes), and every row covers those snapshots. An id the catalog
does not list (another spelling, case, gateway prefix, Bedrock region or
revision form, `-latest`, a deployment name) is encoded by the naming rule
P2 had for the same decision, restored beside each catalog lookup
(`catalog::reads_images_or` takes the rule as an argument; the Anthropic,
OpenAI, Groq, Gemini and Bedrock option mappings fall back the same way), so
it sends P2's body. The exceptions are listed in their rows: the OpenAI
effort levels and Chat `Off` of an unlisted id, and the Gemini lookup of the
model an id versions. `tests/core/request_bodies.rs` pins what every wire
sends for every catalog id and public model constant, their other
spellings and a set of request shapes, so a later change to any body fails
until it is listed here and the golden is regenerated.

| phase | change | who is affected | Migration line |
|---|---|---|---|
| P2 | Gemini GenerateContent and Interactions: raw `additionalParams.generationConfig` / `generation_config` keys now beat the typed fields (`crates/rig-core/src/providers/gemini/completion.rs:428-437`, `gemini/interactions_api/mod.rs:352-370`) | callers who set a typed field and the same key in `additional_params` | "`additional_params` now overrides `temperature`, `max_tokens` and other typed fields on Gemini; remove the duplicate key to keep the typed value." |
| P2 (no change) | Gemini GenerateContent, Vertex AI and gRPC: a raw top-level `generationConfig: null` is still read as absent (today `None \| Some(Value::Null) => None`, `crates/rig-core/src/providers/gemini/completion.rs:417-418` on the parent), so the typed `temperature`, `maxOutputTokens`, `responseJsonSchema` and the mapped `thinkingConfig` are still sent. It is the one exception to "a `null` is sent as `null`" | none; stated because the merge would otherwise send the `null` and wipe those fields | "On Gemini GenerateContent, Vertex AI and gRPC, a raw `generationConfig: null` is still ignored, as before; a `null` inside it is sent." |
| P2 | OpenAI Responses flips from base-wins to raw-wins (`crates/rig-core/src/providers/openai/responses_api/mod.rs:464-468`): `additional_params.temperature`, `tool_choice`, `text` and any other key now override the typed field. The same loop skips a raw `null` today (`!value.is_null()`, `:465`); it is now sent as `null`, as on every other wire | Responses callers whose `additional_params` repeat a typed field or hold a `null` | "On OpenAI Responses, `additional_params` now overrides typed request fields, and a `null` in it is sent as `null`, as on every other wire. Remove a key to send nothing." |
| P2 | Shallow `extend` becomes a deep merge on Anthropic (`completion.rs:256`), Chat (`chat.rs:384`), Cohere native (`cohere/chat.rs:146-151`) and Bedrock (`crates/rig-bedrock/src/request.rs:133`): an object in `additional_params` such as `output_config`, `thinking`, `tool_choice` or `generationConfig` now merges into the wire's object instead of replacing it | callers who relied on an object in `additional_params` replacing the wire's whole object (for example to drop `output_config.format`) | "`additional_params` objects now merge key by key into the request body. A `null` is still sent as `null` (Gemini GenerateContent still ignores a top-level `generationConfig: null`, as before). To drop a key the wire writes from a typed field, leave the field unset (for `output_config.format`, set no `output_schema`)." |
| P2 | Ollama `/v1`: the `think` to `reasoning_effort` rewrite is deleted (`crates/rig-core/src/providers/openai/wire/chat.rs:688-722`), so an existing `additional_params.think` is sent raw | `/v1` callers using `think` | "On Ollama's OpenAI-compatible route, set `GenerationOptions::reasoning` instead of `additional_params.think`, which is now sent as written." |
| P2 | OpenAI Responses, Codex: the `UNACCEPTED` strip (`responses_api/mod.rs:479-490`) is deleted, but not what it does to typed fields. The Codex base builder keeps omitting the typed fields the backend does not accept, `max_output_tokens` (from `max_tokens`), `temperature` and `text.format` (from `output_schema`), so a request with no `additional_params` sends today's body. Two things change: the mapped `parallel_tool_calls`, `service_tier` and `text.verbosity` are sent when the option is set, and raw `additional_params` keys (`temperature`, `max_output_tokens`, `top_p`, `metadata`, `user`, `background`, `parallel_tool_calls`, `service_tier`, `text`) reach the backend instead of being stripped | ChatGPT callers who put those keys in `additional_params` | "The ChatGPT backend no longer strips `additional_params` keys; remove `temperature`, `top_p`, `metadata`, `user`, `background` and `max_output_tokens` from them or expect the backend's error. Typed `max_tokens`, `temperature` and `output_schema` are still not sent to ChatGPT." |
| P2 | Anthropic: `with_automatic_caching()` and `with_automatic_caching_1h()` are deleted. `with_prompt_caching()` and `with_static_prefix_cache_ttl(..)` are unchanged; their markers take a 1 h TTL from `cache(Long)` | callers of the two deleted methods | "Replace `Messages::with_automatic_caching()` with `GenerationOptions::cache(CacheRetention::Short)` on the request, and `with_automatic_caching_1h()` with `CacheRetention::Long`." |
| P2 | rig-vertexai: the `contents` rewrites and the `model` key, written after the body is built today (`crates/rig-vertexai/src/completion.rs:66-76`), move into the base builder, so a raw `additional_params.model` or `contents` now overrides them | Vertex callers who put `model` or `contents` in `additional_params` | "On Vertex AI, `additional_params.model` and `additional_params.contents` now override the request's own values, as raw params do on every wire." |
| P2 | rig-gemini-grpc: the `model: models/{model}` key, inserted after `request_body` merged `additional_params` today (`crates/rig-gemini-grpc/src/completion.rs:70-71`), moves into the base builder's adjustments, so a raw `additional_params.model` now overrides it | gRPC callers who put `model` in `additional_params` | "On Gemini gRPC, `additional_params.model` now overrides the request's model, as raw params do on every wire." |
| P2 | Bedrock: `cache(Short/Long)` after a reasoning turn, on a model other than a listed Claude model, reports the skipped message checkpoint through `on_unsupported`, an error by default, where `with_prompt_caching` skipped it silently (`crates/rig-bedrock/src/request.rs`) | Bedrock callers caching a conversation with reasoning in its history | "On Bedrock, a cache checkpoint that cannot follow a reasoning turn is an `UnsupportedOption` error; set `OnUnsupported::Ignore` to skip it with a warning." |
| P2 | OpenRouter Chat: `cache(Short)` sends the documented top-level `cache_control` where `with_prompt_caching` marked the system message (`chat.rs:816-843`) | OpenRouter callers of `with_prompt_caching` | "OpenRouter caching now uses the top-level `cache_control` marker." |
| P2 | Ollama native `/api/chat`: the `think` validation is deleted (`crates/rig-core/src/providers/ollama/chat.rs:46-47`, `:117-126`, `:257-269`), so a raw `additional_params.think` is sent as written under `RawAt::Split`: `"HIGH"` is no longer lowercased, and `"bogus"` is the daemon's error, not a local `EncodeError`. The warning on a raw `reasoning_effort` goes; the key still lands in `options` | native callers who set `think` in `additional_params` | "On Ollama's native route, set `GenerationOptions::reasoning` instead of `additional_params.think`, which is now sent as written; use a lower-case level." |
| P2 | Gemini Interactions: a raw `additional_params.tools` that is not an array is an `EncodeError` through `BaseInput::raw_tools`, where it is discarded today (`crates/rig-core/src/providers/gemini/interactions_api/mod.rs:380`) | Interactions callers whose `additional_params.tools` is not an array | "On Gemini Interactions, `additional_params.tools` must be an array, as on every other wire." |
| P2 | Gemini Interactions resume: a set option is refused through `on_unsupported`, and `additional_params` or provider options are an `EncodeError`, where `InteractionResume::encode` ignores the whole request today (`crates/rig-core/src/providers/gemini/interactions_api/mod.rs:247-274`) | callers who reuse a request with options or `additional_params` to resume an interaction | "`InteractionResume` now refuses a request that sets options or `additional_params`, raw or provider tools included; resume with a request that sets neither." |
| P2 | Anthropic: a raw `additional_params.cache_control` is sent as written. Today `top_level_cache_control` (`crates/rig-core/src/providers/anthropic/completion.rs:664-675`) sends a rebuilt `{"type":"ephemeral"}` plus `ttl`, so keys beyond `type` and `ttl` are now sent, and a raw `null` is sent as `null` ("`null` is a value") where today it is dropped. Validation, placement, the budget of four and the order check are unchanged | Anthropic callers whose raw `cache_control` holds extra keys or is `null` | "On Anthropic, `additional_params.cache_control` is sent as written: remove keys other than `type` and `ttl`, and remove the key rather than setting it to `null`." |
| P2 | Anthropic, Chat Completions, Responses and Cohere: a raw `additional_params.tools: null` is read as no tools and sends nothing, where today it is an `EncodeError` (`BaseInput::raw_tools`). Every other value keeps its current handling | callers who sent `tools: null` and relied on the error | a raw `tools: null` no longer fails the request; send an array, or leave the key out |
| P3 | OpenAI Chat: the `max_tokens` to `max_completion_tokens` rename reads the catalog's `reasoning` on the full id (a `vendor/model` id keeps `max_tokens`, as before); the GPT-5-and-later and o-series name rule (`is_openai_reasoning_model`, now `openai::options::named_reasoning`) stays for ids the catalog does not list, so the unlisted ids in OpenAI's own model list (`gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`, `gpt-5.2-codex`, `gpt-5-chat-latest`, `gpt-5.1-chat-latest`, `gpt-5-search-api`, `o3-deep-research`, `o4-mini-deep-research`) keep `max_completion_tokens`. Listed models whose answer differs: `gpt-5.3-chat-latest` (models.dev: no reasoning) now sends `max_tokens`; `gpt-daybreak-blue-latest`, `gpt-daybreak-red-latest` and `gpt-realtime-2.1` now send `max_completion_tokens` (on the OpenAI and ChatGPT dialects, with their snapshots) | OpenAI Chat callers of those four models who set `max_tokens` | "On OpenAI Chat Completions, whether `max_tokens` is sent as `max_completion_tokens` follows the model catalog: `gpt-5.3-chat-latest` now sends `max_tokens`, and `gpt-daybreak-blue-latest`, `gpt-daybreak-red-latest` and `gpt-realtime-2.1` send `max_completion_tokens`." |
| P3 | OpenAI Chat Completions and Responses: the effort levels a listed model takes, and whether it takes `Reasoning::Off`, are its catalog row's `levels` and `can_disable`; on Chat every effort was sent and `Off` was refused only on GPT-5.0 ids, and on Responses version rules refused `minimal` off GPT-5.0, `xhigh` before GPT-5.2 and `max` before GPT-5.6. Now refused on both routes: `minimal` on every row but `gpt-5`, `gpt-5-mini`, `gpt-5-nano` and `gpt-realtime-2.1`; `xhigh` on `gpt-5`, its mini, nano and pro, `gpt-5.1`, `gpt-5.2-chat-latest` and the o-series; `max` before GPT-5.6 (but the daybreak rows); `low` on the `-pro` rows from `gpt-5.2-pro` on and on `gpt-5-pro` (which takes `high` only); every level but `medium` on `gpt-5.2-chat-latest`; every effort on `o1-mini` and `o1-preview` (no levels) and on `gpt-5.3-chat-latest` (models.dev: no reasoning). The rows models.dev marks as not reasoning, the image, embedding, speech and transcription models (`dall-e-2`, `dall-e-3`, `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-1.5`, `gpt-image-2`, `chatgpt-image-latest`, `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002`, `tts-1`, `tts-1-hd`, `whisper-1`), refuse every effort and send nothing for `Off` on both routes, where P2 sent the effort or `none`. `Off` is now refused on Chat on the o-series, the later `-pro` rows, `gpt-5.2-chat-latest`, `gpt-realtime-2.1` (no `none` level), `gpt-6-astra` and `gpt-6.1-sol`, where `reasoning_effort: "none"` was sent, and on Responses on `gpt-5.2-chat-latest` and `gpt-realtime-2.1`. On `gpt-5.3-chat-latest`, `Off` sends nothing and Responses sends `top_p`, where both sent `none` and refused `top_p`. An id the catalog does not list takes every effort; its name decides `Off` on Chat (refused on the o-series, `-pro` and `gpt-5`/`gpt-5-*` ids, such as `o3-mini-high`, `o1-latest` or `gpt-6-sol-pro`, which were sent `none` but for GPT-5.0). Everything else about an unlisted id is P2's name rule: whether it reasons (an earlier numbered GPT does not), its cache retention by GPT generation, Responses `Off` and `top_p` | OpenAI callers who set `reasoning` on those models, or `top_p` on `gpt-5.3-chat-latest` | "OpenAI effort levels now follow the model catalog on Chat Completions and Responses: a listed model refuses a level, or `Reasoning::Off`, that its entry does not list." |
| P3 | Anthropic: the default `max_tokens` of a listed model (its id or a snapshot from `-20`) is the catalog's `max_output_tokens` instead of 128000 for the ten tabled models and 64000 for the `claude-opus-4`, `claude-sonnet-4` and `claude-haiku-4-5` prefixes. `claude-opus-4-0`, `claude-opus-4-1` and their snapshots now send 32000, their documented limit, where 64000 was refused by the API. An unlisted id under those prefixes (`claude-opus-4.6`, `claude-sonnet-4-6@20260101`, `claude-opus-4-1-latest`) keeps 64000 | Anthropic callers who set no `max_tokens` | "Claude Opus 4 and 4.1 now default to 32000 output tokens, their documented limit." |
| P3 | Image input on every Chat dialect (OpenAI, Azure, ChatGPT, Copilot and the gateways: OpenRouter, Together, Venice, HuggingFace, Hyperbolic, Ollama `/v1`, Doubleword, Perplexity, Mistral and the rest), Responses, the Anthropic dialects, Cohere native and Bedrock Converse: whether a model the catalog lists reads user images comes from its entry's `input.image` (`catalog::reads_images_or`, `crates/rig-core/src/catalog/mod.rs`, called from `openai/wire/chat.rs`, `responses_api/wire.rs`, `anthropic/wire.rs`, `cohere/chat.rs` and `crates/rig-bedrock/src/completion.rs`; OpenRouter reads its own row). A model the catalog does not list keeps P2's naming rule, which `reads_images_or` takes as an argument: the dialect's rule (DeepSeek, Groq, Mistral, OpenAI and Azure, xAI, Cohere, Z.AI, Moonshot, MiniMax, MiMo), the vendor prefix's rule on OpenRouter, P2's `TEXT_ONLY` list on Bedrock, and images on the other gateways. A listed model whose entry has no image input now gets the history adapter's placeholder text `(image omitted: model does not support images)` instead of the image, on gateways that filtered nothing before too: OpenRouter 126, HuggingFace 60, Venice 54, Together 39, Ollama 30, Hyperbolic 19, Mistral 15 (`magistral-medium-latest` among them), Doubleword 11, Copilot 6 (`gpt-4`, `o3-mini`, `mai-code-1-flash-picker` and the three `text-embedding-*` rows), Perplexity 2 (`sonar`, `sonar-deep-research`); OpenAI's `dall-e-3`, `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002`, `tts-1`, `tts-1-hd` and `whisper-1` on Chat (OpenAI and Azure) and on Responses (OpenAI, OpenRouter, ChatGPT and Copilot, read past an `openai/` prefix); ChatGPT's `gpt-5.3-codex-spark`; xAI's `tts-1`; and Bedrock's `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0`, `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `stability.stable-image-core-v1:1` and `stability.stable-image-ultra-v1:1`. For example `meta-llama/llama-3.3-70b-instruct` and `mistralai/mistral-large` on OpenRouter, `meta-llama/Llama-3.3-70B-Instruct-Turbo` on Together, `llama-3.3-70b` on Venice and `llama3.2` on Ollama. Fifteen listed rows now get the image where P2 sent the placeholder: Cohere `command-a-plus-05-2026` and its embedding rows `embed-v4.0`, `embed-english-v3.0`, `embed-english-light-v3.0`, `embed-multilingual-v3.0` and `embed-multilingual-light-v3.0` (Compatibility and native API), Groq `qwen/qwen3.6-27b` and `qwen/qwen3.8-27b`, Z.AI `glm-5.3-flash` and `glm-5.3-flashx` (both formats, the coding plan too), and OpenRouter `cohere/command-a-plus`, `deepseek/deepseek-v4-flash-vision-exp`, `deepseek/deepseek-v4.1-flash`, `z-ai/glm-5.3-flash` and `z-ai/glm-5.3-flashx`. The llama.cpp server row `LLaMA_CPP` reads images, since the id names no model. Rig never refused an image locally on any of these routes. Kept, not restored: the gateways' own recorded model listings declare these models text-only (`crates/rig-cassette/fixtures/cassettes/openrouter/models/list_models_smoke.yaml`: `meta-llama/llama-3.3-70b-instruct` and `mistralai/mistral-large` have `input_modalities` without `image`; `crates/rig-cassette/fixtures/cassettes/venice/model_listing/list_models_smoke.yaml`: `llama-3.3-70b` has `supportsVision: false`), so the P2 body sent them an image they cannot read | callers sending images to those models | "Image input now follows the model catalog on every Chat Completions provider and gateway, Responses, Anthropic, Cohere and Bedrock: a listed model the catalog says cannot read images gets the text `(image omitted: model does not support images)` in place of the image, also on gateways (OpenRouter, Together, Venice, HuggingFace, Hyperbolic, Ollama, Doubleword, Perplexity, Mistral, Copilot) that sent every image before. A model the catalog does not list keeps the naming rule it had." |
| P3 | xAI Chat and Responses: whether a Grok model reasons, and which effort levels it takes, come from the catalog instead of the `non-reasoning` name rule, which stays only for ids the catalog does not list. `Reasoning::Off` on grok-4.3 (levels include `none`) now sends `"reasoning_effort":"none"` on Chat, as the Responses route already did, where it was refused; on listed models that do not reason (the grok-2 and grok-3 ids, `grok-3-fast`, the `grok-imagine-*` image and video rows and `tts-1`) it now sends nothing where it was refused, and their `stop` is now sent on Chat where it was refused. Listed models refuse levels they do not list, where every Grok id but grok-4.5 took `low` to `xhigh`: `low`, `medium`, `high` and `xhigh` on `grok-4-0709`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning` and `grok-build-0.1` (no levels) and on the rows that do not reason, and `medium` and `xhigh` on `grok-3-mini` and `grok-3-mini-fast` (`low` and `high` only). Snapshots of these rows follow them. grok-4.3 still takes `xhigh` (docs.x.ai/docs/models/grok-4.3) | xAI callers asking those models for `Reasoning::Off`, `stop` or those efforts | "On xAI, effort levels follow the model catalog on Chat and Responses: a listed model refuses a level it does not list. `Reasoning::Off` turns reasoning off on Grok 4.3 on Chat too, and is accepted on models that do not reason." |
| P3 | Bedrock Converse (superseded by the catalog row below, which replaces `claude_spec`): a Claude id is read as the Anthropic wire reads it (`anthropic::completion::claude_spec`: past a region prefix and `-v1:N`, a `-20…` snapshot, `.` as `-`), and any other id by its exact Bedrock row, else P2's rules. `Reasoning::Off` on Claude Opus 4, Opus 4.1 and Sonnet 4 (`anthropic.claude-opus-4-1-20250805-v1:0` and the `us.`, `eu.`, `apac.` and `global.` `claude-sonnet-4-20250514` profiles) sends nothing (their catalog rows take no adaptive thinking, and thinking is off unless asked for), where it sent `{"thinking":{"type":"disabled"}}` because the class table did not name them | Bedrock callers asking those models for `Reasoning::Off` | "On Bedrock, `Reasoning::Off` on Claude Opus 4, Opus 4.1 and Sonnet 4 sends nothing, as on the other models whose thinking is off by default." |
| P3 | Gemini GenerateContent (REST, Vertex, gRPC): thinking follows the catalog instead of the prefix table; an id it does not list is looked up as the model it versions (`gemini-2.0-flash-001` as `gemini-2.0-flash`, `gemini-2.5-flash-preview-09-2025` as `gemini-2.5-flash`), and an unlisted id before 2.5 still does not think. Failing that, the prefix table decides, as in P2 (`gemini-2.5-flash-latest`, a `-tts` id the catalog does not list). Rows that differ from the prefix table: `gemini-2.5-flash-image` does not think (its model page), so `Off` sends nothing where it sent `thinkingBudget: 0` and a budget is refused, and so do the ids that version it (`gemini-2.5-flash-image-001`, `-exp`, `-preview-09-2025`); Gemma 4 (`gemma-4-26b-a4b-it`, `gemma-4-31b-it`) takes `Off` as `thinkingLevel: "minimal"` and `Effort(High)` as `"high"`, where `Off` was refused and every level and budget was sent; the listed rows that do not think (`gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`, `gemini-3.5-live-translate-preview`, the embedding rows `gemini-embedding-001`, `gemini-embedding-2` and `text-embedding-004` and the ids that version them, `lyria-3-clip-preview`, `lyria-3-pro-preview` and the `veo-3.1-*` previews) refuse every effort and budget and send nothing for `Off`, where the table sent them, or refused `Off` on an id it did not name; the listed rows with thinking levels the table did not name (`deep-research-preview-04-2026`, `deep-research-max-preview-04-2026`, `gemini-2.5-computer-use-preview-10-2025`, `gemini-3.1-flash-tts-preview`, `gemini-3.1-flash-live-preview`, `gemini-omni-flash-preview`, `gemini-flash-lite-latest`, and `gemini-3.1-flash-image`, `gemini-3.1-flash-image-preview` and `gemini-flash-latest` below) refuse a budget, and those without `high` refuse it, on Interactions too; `gemini-3.1-flash-image` and `gemini-3.1-flash-lite-image` (levels `minimal` and `high`) now refuse `low` and `medium`, and `gemini-flash-latest` (`low` to `high`) refuses `minimal`, all of which were sent | Gemini callers of those models with `reasoning` set | "Gemini thinking now follows the model catalog: Gemma 4 turns thinking off with `Reasoning::Off`, and Gemini 2.5 Flash Image, which does not think, no longer takes a budget." |
| P3 | OpenAI Chat Completions and Responses: the rows from `gpt-5.1` to `gpt-5.6-*` (`gpt-5.1`, `gpt-5.2`, `gpt-5.2-pro`, `gpt-5.2-chat-latest`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.4` and its mini, nano and pro, `gpt-5.6` and its sol, luna and terra) carry `sampling: reasoning_off`, the rule #2616 applied to GPT-6 only. `check_body` now refuses `temperature`, `top_p` or `top_logprobs` (and on Chat `logprobs`), typed or raw, when the body's effort is set to anything but `none` and the request sets a `GenerationOptions`, where all were sent. The refusal goes through `on_unsupported`: under `Ignore` it is a warning, and a typed value is left out. A request with no `GenerationOptions` (a raw effort and a `temperature` alone, say) is sent as before. With no effort set GPT-5.1 to 5.4 are not checked (their default effort is `none`); the `gpt-5.6` rows and the GPT-6 rows, whose default is `medium` (`reasoning_default: medium`; gpt-5.6-luna echoes `medium` with no effort set, live smoke 2026-10-06), are, as the next row describes. `check_body` keys on the bare OpenAI id and runs on every Chat dialect, ChatGPT and Copilot included, so an Azure deployment named `gpt-5.4` is checked too. Kept, not restored: OpenAI's model guidance says these parameters are "only supported" at effort `none` and that other settings "will raise an error" (https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2, `?model=gpt-5.4`, fetched 2026-10-06), so the old body failed at the API | OpenAI callers who set an effort and a sampling parameter on GPT-5.1 or later | "On OpenAI GPT-5.1 and later, `temperature`, `top_p` and `logprobs` with an effort other than `Reasoning::Off` (or a raw effort other than `none`) are refused before sending, as the API refuses them, when the request sets a `GenerationOptions`; on GPT-5.6, which reasons by default, this holds with no effort set too. Under `OnUnsupported::Ignore` they are a warning instead." |
| P3 | OpenAI Chat Completions and Responses, GPT-6 (`gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra`, `gpt-6.1-sol`; P2 had no GPT-6 check but the Responses `Off` refusal): `openai::options::check_body` (`crates/rig-core/src/providers/openai/options.rs`) reads each row's `sampling: reasoning_off`, `chat_tools_need_reasoning_off` and `reasoning_default: medium`. With no effort set the model reasons by default, so the body is checked even when the options set no effort, on every Chat Completions dialect (the gateways included, by the bare id) and on Responses (OpenAI, OpenRouter, xAI, ChatGPT and Copilot), for these ids and their dated snapshots: `temperature`, `top_p` and `top_logprobs` (and on Chat `logprobs`), typed or raw, are refused on all four ids on both routes unless the effort is `none`; and on Chat Completions a request with tools is refused on `gpt-6-astra` and `gpt-6.1-sol`, whose reasoning cannot be turned off. All were sent. These refusals apply only to a request that sets a `GenerationOptions`, and go through `on_unsupported`: a request that sets no option is sent as before, so these ids' bodies for callers who set nothing are unchanged. Sol and Luna with Chat tools at the default effort are still sent (OpenAI's 400 is recorded, `crates/rig-cassette/fixtures/cassettes/openai/models/gpt_6_luna/session.yaml:2647`). Kept, not restored: these are the rules of #2616, which this phase supersedes, and OpenAI's model guidance says "When reasoning effort is not `none`, remove `temperature`, `top_p`, and `top_logprobs`" and that on GPT-6 Astra and GPT-6.1 Sol "tool calling requires Responses" (https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-sol, fetched 2026-10-06), so the old body failed at the API | OpenAI callers of GPT-6 who set a sampling parameter, or Chat tools on Astra or 6.1 Sol | "On OpenAI GPT-6 (`gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra`, `gpt-6.1-sol`), `temperature`, `top_p`, `top_logprobs` and `logprobs`, typed or in `additional_params`, are refused before sending on Chat Completions and Responses when the request sets a `GenerationOptions` and reasoning is not off (`Reasoning::Off`, or a raw `reasoning_effort: none`), even when no effort is set, because these models reason by default. On Chat Completions, tools are refused on `gpt-6-astra` and `gpt-6.1-sol` in the same case; use the Responses API. Under `OnUnsupported::Ignore` each refusal is a warning instead. A request with no `GenerationOptions` is sent unchanged." |
| P3 | Anthropic Messages: `Reasoning::Effort(Low..Max)` on `claude-opus-4-0`, `claude-opus-4-1`, `claude-sonnet-4-0` and their dated snapshots, in every spelling the wire reads a Claude id in (`anthropic/claude-opus-4.1`, `us.anthropic.claude-sonnet-4-20250514-v1:0`), is refused ("takes a thinking budget, not an effort level"), where P2 sent `{"thinking":{"type":"adaptive"},"output_config":{"effort":..}}` because its class table did not name them. Kept, not restored: Anthropic's effort page lists the models that take `effort` (Opus 4.5 and later, Sonnet 4.6 and later) and none of these (https://platform.claude.com/docs/en/build-with-claude/effort, fetched 2026-10-06), and they take no adaptive thinking, so the old body failed at the API. A budget is still sent | Anthropic callers asking those models for an effort | "On Anthropic, Claude Opus 4, Opus 4.1 and Sonnet 4 refuse an effort level before sending, as the API does; use `Reasoning::Budget` on them." |
| P6 | OpenAI Chat Completions (OpenAI and Azure) and Responses: `top_p` follows one rule on both routes, the catalog row's `sampling` (`openai::options::openai_top_p`). On Chat a typed `top_p` is now refused on rows whose `sampling` is `never` (GPT-5.0, its mini, nano and pro, the o-series), on `reasoning_off` rows unless the effective effort is `none`, and by the name rule on ids the catalog does not list or that carry no rule (`gpt-5.5`), where Chat sent it for every model; on GPT-6 the refusal now names `top_p` before `check_body` names `temperature`. On Responses, `top_p` with no `reasoning` set is now sent on the `reasoning_off` rows that can turn reasoning off and name no default (GPT-5.1 to 5.4, whose default effort is `none`), where it was refused; GPT-5.6 and GPT-6 (default `medium`) still refuse it. Chat `stop` is refused on every reasoning model (catalog `reasoning`, else the name rule) at every effort, where it was sent. `verbosity` other than `medium` is refused on both routes on a model that does not reason, and `Medium` sends nothing there. Kept, not restored: OpenAI answered the old bodies with 400 `unsupported_parameter` for `top_p` and `stop` on gpt-5-mini and gpt-5.4-nano and `unsupported_value` for `verbosity: low` on gpt-4.1-mini, and accepted `top_p` with no effort on gpt-5.4-nano Responses, but rejected it on gpt-5.5 and gpt-5.6-luna, whose default effort is not `none` (live smoke, 2026-10-06). The request-body golden moves only in its `sampling` shape: `chat\|openai` and `chat\|azure.openai` refuse `top_p` on the ids above, and `resp\|openai` and `route\|openai` send it on GPT-5.1 to 5.4 | OpenAI callers who set `top_p`, `stop` or `verbosity` on those models | "On OpenAI Chat Completions, `top_p` follows the model's sampling rule as on Responses, and `stop` is refused on reasoning models (o-series, GPT-5 family); on both routes a model that does not reason refuses `verbosity` other than `medium`. On Responses, `top_p` with no reasoning option set is now sent to GPT-5.1 to 5.4, whose default effort is `none`." |
| P6 | OpenRouter Chat and Responses, Mistral, Groq and Venice: reasoning follows the catalog row through one rule (`openai::options::listed_reasoning`). OpenRouter refuses `Reasoning::Off` where the upstream's row lists reasoning options and none turns reasoning off (71 corpus ids: the GPT-5.0, `-pro`, o-series `-high`, GPT-OSS and GPT-6 Astra rows, Claude Fable 5 and Opus/Sonnet 5.5, Gemini 2.5 Pro and 3.x, Grok 4.5 to 4.7 and others), where `{"effort":"none"}` was sent. Mistral, Groq and Venice refuse what `ModelSpec::validate` refuses on a row that lists its reasoning options or marks a model that does not reason: an effort on the models that do not reason (Mistral 44 corpus ids, Groq 19, Venice 77), a level the row does not list (`qwen/qwen3.6-27b` on Groq, `qwen-3-8-27b` on Venice), and `Off` where the row cannot turn reasoning off (Venice 56, Mistral `zai-glm-5-3`; Groq's GPT-OSS rows keep their refusal with the catalog's reason). A row that marks a reasoning model and lists no option, and an id the catalog does not list, are sent as before; Groq's `gpt-oss` name rule stays for unlisted ids. Kept, not restored: OpenRouter answered `Off` on `openai/gpt-5-nano` with 400 "Reasoning is mandatory", Mistral `low` on `mistral-small-latest` and Groq `xhigh` on `openai/gpt-oss-20b` with 400 (live smoke, 2026-10-06). The golden moves only in its `off` and `high` shapes on `chat\|openrouter`, `resp\|openrouter`, `chat\|mistral`, `chat\|groq` and `chat\|venice` | callers asking those models for `Reasoning::Off` or an effort the catalog row does not list | "On OpenRouter, `Reasoning::Off` is refused where the upstream model cannot turn reasoning off. On Mistral, Groq and Venice, reasoning options follow the model catalog: an effort the model does not list, an effort on a model that does not reason, or `Reasoning::Off` on a model that cannot turn reasoning off is refused before sending." |
| P6 | Cohere native and Compatibility: whether a model thinks is its catalog row's `reasoning` (`cohere::thinks`), the `reasoning` name rule staying for ids the catalog does not list. Native `Reasoning::Off` on `command-a-plus-05-2026` and `north-mini-code-1-0` (and their snapshots) now sends `{"thinking":{"type":"disabled"}}`, where it sent nothing and the model thought anyway. A listed model that does not think (Command A, Command A Translate and Vision, Command R, R+ and R7B, Aya, the embed rows: 41 corpus ids) refuses `Effort` and `Budget` natively and `Effort(High)` on the Compatibility route, where `thinking` or `reasoning_effort` was sent. Kept, not restored: Cohere answered `thinking` and `reasoning_effort: high` on `command-r7b-12-2024` with 400 "`thinking` parameter is not supported", and `command-a-plus-05-2026` spent 110 reasoning tokens with `Off` dropped (live smoke, 2026-10-06). The golden moves in `cohere\|native` `off`, `high` and `budget`, and in `chat\|cohere` and `cohere\|chat` `high` | Cohere callers asking those models for reasoning | "On Cohere, `Reasoning::Off` now turns thinking off on Command A Plus and North Mini Code, and a model that does not think refuses a reasoning effort or budget before sending." |
| Catalog | References: `minimax/<model>` (and any reference that names no format) connects in the family the built-in catalog row names, else the vendor's primary (`providers::registry::PRIMARY`: Anthropic's Messages for MiniMax), where it took the vendor's first registered family (OpenAI Chat for MiniMax) | callers of `ProviderRef::parse`, `Catalog::connect` or the reference deserializer with a bare `minimax/...` | "`minimax/<model>` now connects over MiniMax's Anthropic-format endpoint. Write `minimax/openai:<model>`, or pass the format in `ConnectOptions`, to keep the OpenAI-format endpoint." |
| Catalog | References: `vendor:model` (`deepseek:deepseek-chat`) no longer parses; only `vendor/model` and the older `vendor/format:model` are read | callers and config files that wrote the colon form | "Write `deepseek/deepseek-chat` instead of `deepseek:deepseek-chat`." |
| Catalog | `ProviderRef` `Display` writes `vendor/model`, followed by ` (format)` when the format is not the one the catalog row names (`zai/glm-5 (anthropic)`); that text does not parse back. Serialization writes the string `vendor/model` in the row's format and the object `{"model": "vendor/model", "format": ...}` otherwise, which 0.44 cannot read; the `vendor/format:model` strings 0.44 wrote are still read | code that round-trips a `ProviderRef` through `to_string` or shares serialized references with 0.44 | "Serialize a `ProviderRef` with serde rather than `to_string` to read it back; references that select a non-default format are written as an object that rig 0.44 does not read." |
| Catalog | Reasoning: a reasoning row that lists no `reasoning_options` is `ReasoningSupport::Unknown` and refuses no reasoning option (the wire's own family rule answers), where every option was refused; a row marked `"reasoning_control": "none"` still refuses them. 34 golden cells move (xAI `grok-4.20-0309-reasoning`, `grok-build-0.1`; Gemini deep-research, computer-use, TTS and omni previews) | callers who relied on the refusal | "A model whose catalog row lists no reasoning controls no longer refuses `reasoning`; the request is sent as the wire maps it." |
| Catalog | Validation: `ModelSpec::refusals` is the one rule `validate`, `DynModel::check` and every wire apply, so the catalog's sampling rule (`sampling`, `reasons_by_default`) now refuses `temperature`, `top_p` and `top_logprobs` on every wire that sends them, including Anthropic, Gemini, Bedrock, OpenRouter, Together, Ollama, Vertex and Copilot, and its reasoning rule runs on the gateways too (884 golden cells now refused, 516 refusals reworded) | callers who sent sampling options or reasoning to a model whose row refuses them through a wire that did not check | "A request the catalog says the model refuses is now refused on every wire, not only the vendor's own; drop the option, or override the catalog row." |
| Catalog | Gateway rows: OpenRouter and Vertex AI Claude rows carry Anthropic's `rig` facts (Vertex also its reasoning options), and Azure OpenAI and Copilot rows of an OpenAI model carry OpenAI's `sampling` and `reasoning_default`, joined by models.dev's `canonical_model_id` | callers of those gateways who sent options the origin model refuses | "Claude through OpenRouter or Vertex AI, and OpenAI models through Azure or Copilot, now follow the origin vendor's catalog facts." |
| Catalog | Bedrock Converse: a model id is read through Bedrock's own catalog rows (each Bedrock Claude row carries Anthropic's reasoning options and facts), falling back past a cross-region prefix (`jp.`) and a `-vN[:M]` revision, replacing `claude_spec` from the P3 row above. A Claude id no Bedrock row lists, under either fallback, is unlisted: `high` is refused, `Off` sends `thinking: disabled` (35 golden cells); listed profiles and dated ids now find their row (88 cells in all) | Bedrock callers using an id the catalog does not list | "Use a Bedrock model or profile id the catalog lists, or add a row with `Catalog::insert` or an override file." |
| Catalog | Cost: a catalog-priced cost whose cache read or write tokens have no listed rate leaves that part `None`, and `Cost::is_complete()` is false, where the part was priced at the input rate; the `total` sums the known parts | callers reading `cost.cache_read`, `cost.cache_write` or `cost.total` from catalog pricing | "Check `cost.is_complete()` before treating `total` as the whole charge; a cache part is `None` when the catalog has no rate for it." |
| Catalog | Refresh: `Catalog::with_models_dev` lays a fetched models.dev copy over the built-in catalog without replacing the keys the built-in rows pin under `rig.pinned` (the hand-reviewed facts and the generator's joins); `with_overrides` replaces every key it is given | applications that refresh the catalog from models.dev | "Lay a fetched models.dev copy on with `with_models_dev`, not `with_overrides`, so rig's reviewed facts survive the refresh." |
| P4 | Bedrock: the guardrail is sent on streams as well as unary requests (`crates/rig-bedrock/src/request.rs:135` filters it to unary today) | streaming callers of `with_guardrail` | "A Bedrock guardrail now also applies to streamed requests." |
| P3 | ChatGPT (`chatgpt` wire): the catalog now lists the current ChatGPT-plan models (`gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-*`, `gpt-5.5`) with the reasoning controls of their OpenAI rows, so `Reasoning::Off` is refused for the ones that cannot turn reasoning off (`gpt-6.1-sol`, `gpt-6-astra` and their dated ids) instead of being sent as `"reasoning":{"effort":"none"}`, and every `chatgpt` row lists only the backend's 24-hour cache | ChatGPT-plan callers who send `Reasoning::Off` to those models | "`gpt-6.1-sol` and `gpt-6-astra` cannot turn reasoning off; choose an effort level instead of `Reasoning::Off`." |
| Fix | ChatGPT (`chatgpt` wire, Chat and Responses): the session header is `session-id`, the spelling the Codex CLI sends, where it was `session_id`; its value is unchanged (the request's cache key, else a fresh id). Every `chat\|chatgpt` and `resp\|chatgpt` golden cell moves by that header name only | none: no caller sets the header | "The ChatGPT dialect now sends `session-id` instead of `session_id`." |
| 0.45 review (#2848) | rig-gemini-grpc: `Reasoning::Effort` maps to `thinkingConfig.thinkingLevel` as on the REST route, instead of the refusal "Google's proto has no thinking level" (`crates/rig-core/src/providers/gemini/options.rs`); and `gemini-grpc` now lists every `gcp.gemini` catalog row (`xtask/src/catalog.rs` mirror), so a Gemini 3 id resolves on gRPC with the same facts as on REST | gRPC callers who set an effort, and callers of a current Gemini model over gRPC | "On rig-gemini-grpc, `GenerationOptions::reasoning` effort is sent as Gemini's thinking level instead of being refused, and every model the Gemini REST catalog lists resolves on gRPC." |
| Catalog refresh (#2878) | The catalog follows models.dev as of 2026-10-22. Ids models.dev retired are no longer listed and take each wire's rule for an unlisted id: OpenRouter `kwaipilot/kat-coder-pro-v2.5`, `microsoft/wizardlm-2-8x22b`, `qwen/qwen3-coder-plus`, `qwen/qwen3-max`, `qwen/qwen3-vl-32b-instruct`, `sao10k/l3.3-euryale-70b`, `unbiased/pareto` and `unbiased/pareto-26.10-preview` send an effort or budget where it was refused, and they and `apodex/apodex-1.1-mini:free`, `inclusionai/ling-3.0-flash-fin` and `inclusionai/ling-3.0-flash-sante:free` send images; Mistral `open-mistral-7b`, `open-mixtral-8x7b` and `open-mixtral-8x22b` send `reasoning_effort`. Listed rows whose models.dev facts changed: Mistral `magistral-medium-latest` and `ministral-3b-latest`/`ministral-8b-latest` (and snapshots) send images, `codestral-2508` refuses an effort; Together `Qwen/Qwen3.7-Max` sends an effort, `MiniMaxAI/MiniMax-M3` refuses one, `deepseek-ai/DeepSeek-V4.1-Flash` and `moonshotai/Kimi-K3` send `Reasoning::Off`; Z.ai, Ollama and Ollama native `glm-5.2` send `Off` where it was refused; Ollama `minimax-m2.7` sends an effort; OpenRouter `openai/gpt-5.1-codex-mini` refuses `Off`; Vertex AI `mistral-small-2503` and `qwen/qwen3-235b-a22b-instruct-2507-maas` refuse reasoning, `deepseek-ai/deepseek-v3.1-maas`/`v3.2-maas` refuse an effort and a budget, `xai/grok-4.6` refuses a budget. OpenAI `gpt-5.2-chat-latest` and `gpt-5.3-chat-latest` and Venice `qwen3-5-9b`, which models.dev no longer lists, keep their last rows in `xtask/src/catalog/review.json`. The golden blanks the user agent's platform (`<os> <arch>`) | callers of those models who set reasoning or send images | "The model catalog follows models.dev as of 2026-10-22: reasoning and image support changed for the models listed in TYPED_OPTIONS.md section 5 (Catalog refresh), and models models.dev retired follow each provider's rule for an unlisted model." |
| Fix | Responses on Groq, Doubleword, MiniMax and Moonshot (`ResponsesQuirks::reasoning_ciphertext` false): with `store: false`, an earlier turn's reasoning item goes back as returned, where it was left out and the item after it lost its id. These endpoints return no ciphertext and read the item's own text (Groq and Doubleword live, 2026-10-23; Moonshot and MiniMax by their docs) | callers replaying reasoning turns with `store: false` on those vendors | "On Groq, Doubleword, MiniMax and Moonshot Responses, a `store: false` conversation now sends earlier reasoning back to the model." |
| Fix | MiniMax China: `minimax::CHINA_API_BASE_URL` and `CHINA_ANTHROPIC_API_BASE_URL` are on `api.minimax.cn`, the host MiniMax's China docs name, where they were on `api.minimaxi.com` (which answers the same). The golden's MiniMax China rows move only in their URL | MiniMax China callers | "MiniMax's China base URLs now use `api.minimax.cn`, the host its docs name." |
| Fix | ChatGPT (`chatgpt` wire, Responses): `Reasoning::Effort` sends `"reasoning":{"effort":..,"summary":"auto"}` where it sent the effort alone, so the backend summarises the reasoning of the models that summarise (`gpt-6-astra` does; `gpt-6.1-sol` returned empty summaries live, 2026-10-26); `Reasoning::Off` and unset reasoning are unchanged, and a raw `reasoning.summary` still wins. The golden's `resp\|chatgpt` cells move at the `high` shape only, and `chatgpt/codex_sessions/reasoning_session_two_tool_calls_streaming.yaml` is re-recorded | ChatGPT callers who set an effort | "On the ChatGPT backend, an effort level now also asks for an `auto` reasoning summary, which the models that summarise stream as reasoning text; set `additional_params` `{"reasoning":{"summary":..}}` to choose another." |
| Catalog refresh (#2956) | The catalog follows models.dev as of 2026-10-30. Ids models.dev retired are no longer listed and take each wire's rule for an unlisted id: OpenAI `gpt-image-1` (its snapshots and `openai/gpt-image-1`) sends an effort, and `Off` sends `none`, on the OpenAI Chat, Responses and route wires; OpenRouter `ibm-granite/granite-4.0-h-micro`, `openai/gpt-3.5-turbo`, `openai/gpt-3.5-turbo-instruct`, `openai/gpt-4-turbo` (and `-2024-04-09`) and `openai/gpt-5.2-chat` send an effort, `granite-4.0-h-micro` also a budget, `minimax/minimax-m3.1-flash:free`, `openai/o3-mini-high` and `openai/o4-mini-high` send `Off`, and `arcee-ai/trinity-large-thinking`, `granite-4.0-h-micro`, `nvidia/nemotron-3-super-120b-a12b:free`, `qwen/qwen3-next-80b-a3b-thinking` and `upstage/solar-pro-3` send images; Mistral `devstral-medium-2507`, `devstral-small-2505`, `devstral-small-2507`, `mistral-large-2411`, `mistral-nemo` and `pixtral-large-latest` send an effort, `mistral-large-2411` and `mistral-nemo` images; Groq `groq/compound` and `groq/compound-mini` send an effort; Ollama `glm-5.1`, `kimi-k2.5` and `qwen3.5:397b` send an effort (Chat, Messages and native), and `deepseek-v4-flash` (and its snapshots and `:0731`), `glm-5.1` and `minimax-m2.5` send images; HuggingFace `Qwen/Qwen3-Next-80B-A3B-Thinking`, `XiaomiMiMo/MiMo-V2-Flash` and `zai-org/GLM-4.5` and Copilot `mai-code-1-flash-picker` send images; Gemini `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview` and `veo-3.1-lite-generate-preview` send a thinking level or budget and refuse `Off` on REST, gRPC and Vertex, and send `generation_config.thinking_level` on Interactions. Listed rows whose models.dev facts changed: Bedrock `global.zai.glm-5.3` and `us.zai.glm-5.3` and Mistral `mistral-large-4` refuse `Off` (their effort levels no longer list `none`); Mistral `deepseek-v4-1-flash-1017`, now listed, sends no images. Xiaomi retired `mimo-v2-flash`, `mimo-v2-omni` and `mimo-v2-pro` on 2026-06-30, and their constants are gone. `gcp.gemini/gemini-3-pro-image-preview` leaves `xtask/src/catalog/review.json` with its models.dev row. Venice `minimax-m3-1-flash`, which models.dev lists with no effort levels, keeps Venice's `low` to `max` in `review.json` | callers of those models who set reasoning or send images | "The model catalog follows models.dev as of 2026-10-30: reasoning and image support changed for the models listed in TYPED_OPTIONS.md section 5 (Catalog refresh, #2956), and models models.dev retired follow each provider's rule for an unlisted model. `xiaomimimo::MIMO_V2_FLASH`, `MIMO_V2_OMNI` and `MIMO_V2_PRO` are removed (Xiaomi retired them); use `MIMO_V2_6_FLASH` or `MIMO_V2_6_PRO`." |
| Fix (#2956) | OpenAI Responses, GPT-5.6 and later (`prompt_cache_options` rows): `cache(Long)` sends `"prompt_cache_options":{"ttl":"12h"}`, OpenAI's 12 hour retention, where it sent `"30m"`, the default that `Short` already gets by sending nothing (https://developers.openai.com/api/docs/guides/prompt-caching, fetched 2026-10-30). Chat Completions still refuses `Long` on these models. Every `resp\|openai` and `route\|openai` golden cell at the `cache_long` shape moves (60 ids each) | Responses callers who set `CacheRetention::Long` on GPT-5.6 or GPT-6 | "On OpenAI Responses, `CacheRetention::Long` on GPT-5.6 and later now asks for the 12 hour cache (`ttl: 12h`), which OpenAI bills at 2.5x the input rate for cache writes; use `Short` for the 30 minute default." |
| Catalog cleanup | ChatGPT: the catalog no longer lists `gpt-5.3-chat-latest`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.3-instant`, `gpt-5.4`, `gpt-5.4-pro` or `gpt-5.5`, which no ChatGPT account offers. Direct wire callers using these ids follow the rules for unlisted models: Responses sends `high` for `gpt-5.3-chat-latest` and `gpt-5.3-instant`, and `Off` for `gpt-5.4-pro`; Chat sends images for `gpt-5.3-codex-spark`. These changes also apply to their dated ids. OpenAI and other providers keep their rows. | callers resolving those ChatGPT catalog entries or using the removed `chatgpt::GPT_5_*` constants | "Use `chatgpt::DEFAULT_MODEL` or a retained ChatGPT catalog entry, such as `chatgpt/gpt-6.1-sol`." |
| Fix (gateway ids) | Anthropic Messages and the Anthropic-compatible wires that read Claude's catalog row (MiniMax, Moonshot, Ollama, OpenCode, OpenCode Go, Xiaomi MiMo, Z.ai): a gateway's `anthropic/` spelling of a listed id (`anthropic/claude-opus-5`) takes that id's catalog facts (`anthropic::completion::spec`), where it was unlisted. The golden's `anthropic/claude-*` rows for the eleven listed models move: with no `max_tokens` the Anthropic wire sends the model's default (128000, or 64000 for `claude-haiku-4-5`) where it refused with "`max_tokens` must be set"; the reasoning options the model's row refuses are refused (`budget` on effort-only models, `Off` on `claude-fable-5`, `claude-fable-5-1` and `claude-opus-5-5`, `high` on `claude-haiku-4-5`) and `Off` on `claude-sonnet-5-5` sends `between_tools`; a `top_p` refusal gives the row's reason; the context-binding models (`claude-fable-5-1`, `claude-opus-5`, `claude-opus-5-5`, `claude-sonnet-5-5`) send adaptive thinking with `block_binding` and its beta header; and models that read a later system message in place keep it there in `reasoning_history`. Another vendor's prefix (`openai/claude-opus-5`) and an unlisted id are unchanged | callers of a gateway that spells Claude ids with `anthropic/` | "A gateway's `anthropic/<model>` id now gets that model's defaults and checks, so `max_tokens` may be omitted; options the model refuses are refused before sending." |
| Fix (#2986) | DeepSeek Chat: `deepseek-flash`, `deepseek-v4-flash` (and their snapshots) and `deepseek-v4-flash-vision-exp` send a user image as an `image_url` part, where they sent the image-omitted placeholder. `deepseek-v4-pro` and unlisted ids keep the placeholder | callers of those models who send images | "DeepSeek's catalog vision models now receive user images instead of a placeholder." |
| Fix (API key header) | Gemini GenerateContent (`generateContent`, `streamGenerateContent`): the API key goes in the `x-goog-api-key` header, as on Interactions, where it went in the URL's `key` query parameter, which proxies, traces and errors keep. The body is unchanged. Embeddings, image generation, transcription, cached contents, model listing and the credential check move the same way | callers behind a gateway that accepts only `?key=` | "Gemini requests send the API key in the `x-goog-api-key` header and never in the URL. A gateway that reads only `?key=` must also accept that header." |
