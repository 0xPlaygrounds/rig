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
Rig powers coding agents and developer tools, desktop and terminal AI assistants, retrieval-augmented generation and semantic search, agent memory systems, multi-agent and workflow orchestration frameworks, and production services ranging from genomics research and incident management to decentralised compute networks.

The full list of companies, applications, libraries, and articles lives in [awesome-rig](https://github.com/0xPlaygrounds/awesome-rig), a community-maintained list. Using Rig? Open a pull request there to add your project.
