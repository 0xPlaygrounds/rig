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
