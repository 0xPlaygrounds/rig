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
    client::CompletionClient,
    completion::{AssistantContent, CompletionModel},
    providers::openai,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an OpenAI client and completion model.
    // This requires the `OPENAI_API_KEY` environment variable to be set.
    let openai_client = openai::Client::from_env()?;

    let model = openai_client.completion_model(openai::GPT_5_2);
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

## Provider observations

`observe::AdapterContext` carries a caller-owned operation identity and a
`Witness` sink. Attach it through `CompletionRequestBuilder::observation` or
`CompletionRequest::observation`; it is skipped by serde and never becomes
provider request data. Clone the context for attempts of the same operation;
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
When the witness supplies a clock through `Witness::elapsed`, the closure's
analysis includes `AdapterTiming`: send-to-closure duration and, for an observed
response body, send-to-first-byte duration. `ObservationLog::with_clock` supplies
this seam. Empty chunks and response headers are not first-byte boundaries.
For unary requests, the adapter installs a `ResponseBodyObserver` request
extension; `rig-reqwest` notifies it before buffering successful and failed
response bodies. Custom transports can notify the same extension at their
body boundary; transports without that hook leave first-byte timing absent. Missing
clocks produce no timing payload; measurements never affect semantic comparison.
These intervals describe the executing transport environment, including replay,
and do not establish live provider performance.
Partial-frame facts count raw bytes after the last blank SSE delimiter without
retaining their contents. Complete-frame counts include only frames passed to
the adapter driver, excluding provider-recognized analysis-only frames (Gemini
frames containing only a valid string `responseId`). Unknown, malformed and
otherwise-empty frames still count. Corrupt-frame ordinals start at one using
the same counting rule. `TransportEof`
records body completeness independently: a provider terminal followed by
an incomplete trailing frame retains both its terminal closure and partial
EOF evidence. The context does not promise cross-execution identity. The ECS completion adapter
forwards its driver's context when the request has none. Response bodies and
credentials are not copied into these boundary facts.
