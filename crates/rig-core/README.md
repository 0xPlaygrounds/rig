# Rig
Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

More information about this crate can be found in the [crate documentation](https://docs.rs/rig-core/latest/rig_core/).
## Table of contents

- [Rig](#rig)
  - [Table of contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [WASM target support](#wasm-target-support)
  - [Simple example:](#simple-example)
  - [Integrations](#integrations)
  - [Who is using Rig?](#who-is-using-rig)

## Features
- Portable contracts for agent runtimes, including completions, messages, tools, and memory
- Full [GenAI Semantic Convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) compatibility
- 20+ model providers, all under one singular unified interface
- Built-in providers selectable as data: `providers::registry` names a vendor and a protocol family (`deepseek/openai:deepseek-chat`) or carries a whole typed configuration, and both round-trip through serde without a credential. Model references discard embedded credentials and reject empty identifiers; a configuration's `id()` returns a catalog selection only when its dialect name is registered. Providers outside this catalog can use `CompletionModel` and `CompletionAdapter` directly.
- 10+ vector store integrations, all under one singular unified interface
- Full support for LLM completion and embedding workflows
- Support for transcription, audio generation and image generation model capabilities
- Integrate LLMs in your app with minimal boilerplate
- Full WASM compatibility (core library only)

## Installation
```bash
cargo add rig-core
```

## WASM target support

`rig-core` supports the browser-oriented `wasm32-unknown-unknown` target. When
the `pdf` feature is enabled, the host must provide the Web Crypto API's
`Crypto.getRandomValues` implementation, as modern browsers, Web Workers, and
Node.js 19 or later do. WASI targets are not supported.

## Simple example
```rust
use rig_core::{
    completion::{AssistantContent, CompletionModel},
    providers::openai::{self, OpenAI},
};
// rig-core ships no transport; `.bound()` builds the bundled `reqwest` one.
use rig_reqwest::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read `OPENAI_API_KEY` into the provider's configuration, bind it to a
    // transport, and pick a model: a model is a wire plus its socket.
    // OpenAI's default completion route is the Responses API;
    // `.with_route(Route::Chat)` on the configuration selects Chat Completions.
    let model = OpenAI::from_env()?.bound()?.completion(openai::GPT_5_2);

    let request = model.completion_request("Who are you?").build();
    let response = model.completion(request).await?;
    for item in response.choice {
        if let AssistantContent::Text(text) = item {
            println!("{}", text.text);
        }
    }

    Ok(())
}
```
Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

You can find more examples in the repository-level `examples/` directory. Many provider-specific examples now also live as ignored live integration tests under the repository-level `tests/providers` directory, organized by provider. When running those provider-backed tests, prefer provider-specific targets such as `cargo test -p rig --test openai -- --ignored --test-threads=1` to avoid rate-limiting. More detailed walkthroughs are regularly published on our Dev.to blog and added to Rig's official documentation at `docs.rig.rs`.

## Integrations
Rig supports the following LLM providers out of the box:

- Anthropic
- Azure OpenAI
- ChatGPT and GitHub Copilot auth-backed clients
- Cohere
- DeepSeek
- Gemini
- Groq
- Hugging Face
- Hyperbolic
- llama.cpp (`llama-server`, and llamafile)
- MiniMax
- Mira
- Mistral
- Moonshot
- Ollama
- OpenAI
- OpenRouter
- Perplexity
- Together
- Venice
- Voyage AI
- xAI
- Xiaomi MiMo
- Z.ai

Vector stores are available as separate companion-crates and as feature-gated modules on the root `rig` facade:

```toml
rig = { version = "0.36.0", features = ["lancedb", "fastembed"] }
```

- MongoDB: [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-mongodb)
- LanceDB: [`rig-lancedb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-lancedb)
- Neo4j: [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-neo4j)
- Qdrant: [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-qdrant)
- SQLite: [`rig-sqlite`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-sqlite)
- SurrealDB: [`rig-surrealdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-surrealdb)
- Milvus: [`rig-milvus`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-milvus)
- ScyllaDB: [`rig-scylladb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-scylladb)
- AWS S3Vectors: [`rig-s3vectors`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-s3vectors)
- HelixDB: [`rig-helixdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-helixdb)
- Cloudflare Vectorize: [`rig-vectorize`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-vectorize)

The following providers are available as separate companion-crates:

- AWS Bedrock: [`rig-bedrock`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-bedrock)
- Fastembed: [`rig-fastembed`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-fastembed)
- Google Gemini gRPC: [`rig-gemini-grpc`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-gemini-grpc)
- Google Vertex: [`rig-vertexai`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-vertexai)

## Who is using Rig?
Below is a non-exhaustive list of companies and people who are using Rig:
- [St Jude](https://www.stjude.org/) - Using Rig for a chatbot utility as part of [`proteinpaint`](https://github.com/stjude/proteinpaint), a genomics visualisation tool.
- [Coral Protocol](https://www.coralprotocol.org/) - Using Rig extensively, both internally as well as part of the [Coral Rust SDK.](https://github.com/Coral-Protocol/coral-rs)
- [VT Code](https://github.com/vinhnx/vtcode) - VT Code is a Rust-based terminal coding agent with semantic code intelligence via Tree-sitter and ast-grep. VT Code uses `rig` for simplifying LLM calls and implement model picker.
- [Dria](https://dria.co/) - a decentralised AI network. Currently using Rig as part of their [compute node.](https://github.com/firstbatchxyz/dkn-compute-node)
- [Nethermind](https://www.nethermind.io/) - Using Rig as part of their [Neural Interconnected Nodes Engine](https://github.com/NethermindEth/nine) framework.
- [Neon](https://neon.com) - Using Rig for their [app.build](https://github.com/neondatabase/appdotbuild-agent) V2 reboot in Rust.
- [Listen](https://github.com/piotrostr/listen) - A framework aiming to become the go-to framework for AI portfolio management agents. Powers [the Listen app.](https://app.listen-rs.com/)
- [Cairnify](https://cairnify.com/) - helps users find documents, links, and information instantly through an intelligent search bar. Rig provides the agentic foundation behind Cairnify’s AI search experience, enabling tool-calling, reasoning, and retrieval workflows.
- [Ironclaw](https://github.com/nearai/ironclaw) - A secure personal AI assistant

Are you also using Rig in production? [Open an issue](https://www.github.com/0xPlaygrounds/rig/issues) to have your name added!

## Provider selection persistence

Registry references distinguish registered presets from explicit configurations.
Qualified `vendor/format:model` strings remain unambiguous when a vendor gains
another protocol family, but do not freeze preset defaults, endpoint availability,
or model names. Explicit configurations retain host, route, and typed options and
serialize as objects rather than lossy labels. References remove credentials;
standalone configurations may hold them at runtime. Deserialization requires a
self-describing format such as JSON.

Copilot presets accept already-exchanged session tokens and derive their endpoint
from those tokens. Explicit configurations keep their host and instruction
placement. Token exchange and asynchronous SDK setup belong to the host. The
catalog is not exhaustive: other models can use `CompletionAdapter` directly.
Registry-qualified names do not replace the provider names used in telemetry.

## Provider implementation

Provider configuration and endpoint wires are separate from transports. A `Has*`
implementation exposes each supported capability, while `Bound` supplies shared
execution. Chat-compatible dialects use the shared `Chat` wire and decoder with
`Dialect` data and `BodyRewrite` hooks rather than duplicating request conversion.
This keeps normalization, retry classification, and telemetry consistent.

Each wire encodes requests without transport access and creates a fresh decoder
for each reply. Buffered and streaming replies use the same classifier and event
mapping. A buffered response with a distinct shape is another classified event,
not a separate normalization path. The driver owns framing and asynchronous I/O;
decoders can be exercised directly with frames without opening a connection.
Credentials stored in configuration use `Secret`, whose serialized redaction
must be replaced with a credential when the host reloads the configuration.

Construct normalized completion responses through their builders so finish
reasons reconcile with tool output. Preserve unknown terminal reasons in `Other`,
use the selected descriptor's provider name, and retain the decoded provider
payload in `raw` for typed inspection without a second request. Preserve error
bodies through `ProviderError::from_http_response` for failed HTTP responses and
`ProviderError::from_provider_body` for error envelopes on successful HTTP
responses.
Credentials require redacted debug output. Provider changes need coverage for
supported streaming, usage, tool and multimodal content, with examples and facade
exposure matching the configured capabilities.

Request serialization must use stable map ordering. Randomized `HashMap`
iteration can change request bytes and serialized schemas embedded in prompts,
reducing prefix-cache reuse. Compare raw serialized bytes when testing this;
canonicalized cassette JSON can hide ordering differences. The JSON helpers sort
map keys and recursively sort rendered values regardless of serde_json's
`preserve_order` feature.

## Anthropic prompt caching

Manual caching marks the system prompt, final tool definition, and final message
block. Automatic caching delegates the moving conversation breakpoint to the
provider. Combined mode retains static-prefix markers within the four-marker
budget, including any markers already supplied on provider-specific tools.

A longer TTL can keep shared tools and instructions cached across conversations
without paying the longer storage lifetime for every conversation tail. Set
`with_static_prefix_cache_ttl(CacheTtl::OneHour)` with automatic five-minute
caching for this arrangement. One-hour markers must precede five-minute markers;
request encoding rejects the inverse ordering. Cache-write prices differ by TTL,
so choose a lifetime based on actual reuse.

Caching is skipped when the prefix through a marker is below the model's minimum:

| Model | Minimum tokens |
| --- | ---: |
| `claude-opus-4-7`, `claude-opus-4-6`, `claude-opus-4-5` | 4096 |
| `claude-sonnet-4-6` | 2048 |
| `claude-sonnet-4-5`, `claude-opus-4-1`, `claude-opus-4`, `claude-sonnet-4` | 1024 |
| `claude-haiku-4-5` | 4096 |

## Gemini explicit caching

Explicit caching uploads reusable content and returns a `cachedContents` handle.
Requests can reuse it immediately, including across conversations. Storage is
billed per token-hour until deletion or expiry, in addition to cached-input
charges. Implicit prefix caching is automatic and best-effort; it does not
provide a handle or guarantee a warm first request.

A request using an explicit cache cannot also send `systemInstruction`, `tools`,
or `toolConfig`. Rig rejects these conflicts before sending. Agents therefore
need no preamble, advertised tools, or configured tool choice on those turns.
An empty `RequestPatch::active_tools` allow-list suppresses tool advertisements,
but does not make cached function declarations executable by the agent. Agent
advertisement and dispatch share a registry snapshot. To execute functions
specified in a cache, drive `GenerateContent` directly and append matching
function responses yourself. Provider-hosted tools such as `codeExecution`
need no caller-side dispatch.

Native structured output and context documents can accompany a cache because
neither adds those conflicting fields. `OutputMode::Tool` adds a synthetic tool
and extends the preamble; `OutputMode::Prompted` adds schema instructions to the
preamble. Neither can accompany a cache. Extractors select tool output mode.

Cache resource paths accept a bare id or a `cachedContents/` handle. Validation
rejects path separators, query delimiters, fragments, and traversal segments
rather than escaping them, so lookup, expiry updates, and deletion cannot be
retargeted by an invalid handle. A 403 or 404 on an existing handle maps to
`ProviderError::CacheExpired`, preserving the provider's reply because 403 can
also indicate credential or quota problems. Cache creation does not apply this
mapping, so authorization failures do not become recreation loops.

## Provider observations

`observe::AdapterContext` carries a caller-owned operation identity and a
`Witness` sink. Pass it separately from request data through
`CompletionModel::completion_with_context(request, Some(context))` or
`stream_with_context(request, Some(context))`. Ordinary `completion(request)`
and `stream(request)` are the required provider methods. The observed methods
have defaults that delegate to them without emitting facts, so ordinary providers
need no context boilerplate. Request builders and
request literals contain only provider request data.

Bus-backed `ModelHandle` calls use `complete_with_context` and
`stream_with_context`. `CompletionAdapter` forwards `Dispatch::adapter_context`;
explicit caller context takes precedence over Recorder/Observe context for
that invocation, including through handler layers. It is never serialized into
provider data or effect records and is not inherited by child calls.

Clone the context for attempts of the same operation;
use a distinct non-sensitive identity for another logical call.
`for_host_attempt(subject, ordinal)` rebinds a host retry to its current
dispatch subject while sharing the logical operation and HTTP send counter.
Facts carry the optional host ordinal separately: multiple HTTP sends within
one dispatch share its host ordinal. Already-running attempts keep their
original subjects and ordinals. A changed logical request needs a new context.

The Gemini unary and SSE paths emit `Action::Adapter` facts for send, HTTP
response status, provider metadata, error envelopes and closure through their
shared drivers. Unary closure
reports decoding, error or drop. SSE closure distinguishes terminal, EOF,
partial frame, error and drop; recoverable corrupt frames are separate facts.
Usage facts retain provider-reported optional counts even when decoding or
normalization rejects a response. Each fact is a cumulative snapshot for its
attempt: replace earlier snapshots instead of summing them, keep absent counts
unknown, and do not add overlapping token categories to invent a total.
Provider verdicts retain reported finish/block reasons and model versions;
error-envelope fields remain separate from HTTP status and Rig retryability.
Sparse verdict fields update only when present. Response IDs and allowlisted
headers are bounded, scrubbed analysis data, excluded from semantic comparison.
An ID-only payload attaches its latest ID to the next verdict or attempt
closure without creating an empty semantic event. This also preserves the ID
on error and consumer drop. Known request credentials are redacted before
persistence; arbitrary response bodies and headers are not copied wholesale.
Credential extraction includes URL userinfo and known query keys, including
origin-form request URIs. Diagnostic comparisons account for percent escapes
and control removal without rewriting safe diagnostic text. Hosts can reuse
`observe::diagnostic_url_secrets` for endpoint configuration; its returned
secrets must stay in memory and must never be persisted or logged.
The normalized Gemini unary API closes after normalization, so an empty
decoded response closes as an error; the raw API retains its decode boundary.
Error closures also preserve the original typed error boundary before report
conversion erases HTTP subtypes: request construction, provider response,
response decoding, typed transport termination, or unknown. An opaque client
error stays unknown; its message is never used to infer a boundary. This field
does not change the error returned to the caller or its retryability.
`ObservationLog::with_clock` optionally stamps individual facts with host elapsed
time. Rig does not calculate run, handler or transport intervals. These stamps
describe the executing environment, including replay, and do not establish live
provider performance.
Partial-frame facts count raw bytes after the last blank SSE delimiter without
retaining their contents. Complete-frame counts include only frames passed to
the adapter driver, excluding provider-recognized analysis-only frames (Gemini
frames containing only a valid string `responseId`). Unknown, malformed and
otherwise-empty frames still count. Corrupt-frame ordinals start at one using
the same counting rule. `TransportEof`
records body completeness independently: a provider terminal followed by
an incomplete trailing frame retains both its terminal closure and partial
EOF evidence. The context does not promise cross-execution identity. The ECS completion adapter
forwards its driver's context when the invocation supplies no explicit context. Response bodies and
credentials are not copied into these boundary facts.
