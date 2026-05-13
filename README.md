<p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/rig-rebranded-logo-white.svg">
    <source media="(prefers-color-scheme: light)" srcset="img/rig-rebranded-logo-black.svg">
    <img src="img/rig-rebranded-logo-white.svg" style="width: 40%; height: 40%;" alt="Rig logo">
</picture>
<br>
<br>
<a href="https://docs.rig.rs"><img src="https://img.shields.io/badge/📖 docs-rig.rs-dca282.svg" /></a> &nbsp;
<a href="https://docs.rs/rig/latest/rig/"><img src="https://img.shields.io/badge/docs-API Reference-dca282.svg" /></a> &nbsp;
<a href="https://crates.io/crates/rig"><img src="https://img.shields.io/crates/v/rig.svg?color=dca282" /></a>
&nbsp;
<a href="https://crates.io/crates/rig"><img src="https://img.shields.io/crates/d/rig-core.svg?color=dca282" /></a>
&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/crates/l/rig.svg?color=dca282" /></a>
</br>
<a href="https://discord.gg/playgrounds"><img src="https://img.shields.io/discord/511303648119226382?color=%236d82cc&label=Discord&logo=discord&logoColor=white" /></a>
&nbsp;
<a href=""><img src="https://img.shields.io/badge/built_with-Rust-dca282.svg?logo=rust" /></a>
&nbsp;
<a href="https://github.com/0xPlaygrounds/rig"><img src="https://img.shields.io/github/stars/0xPlaygrounds/rig?style=social" alt="stars - rig" /></a>
<br>

<br>
</p>
&nbsp;


<div align="center">

[📑 Docs](https://docs.rig.rs)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[🌐 Website](https://rig.rs)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[🤝 Contribute](https://github.com/0xPlaygrounds/rig/issues/new)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[✍🏽 Blogs](https://docs.rig.rs/guides)

</div>

✨ If you would like to help spread the word about Rig, please consider starring the repo!

> [!WARNING]
> Here be dragons! As we plan to ship a torrent of features in the following months, future updates **will** contain **breaking changes**. With Rig evolving, we'll annotate changes and highlight migration paths as we encounter them.

## Table of contents

- [Table of contents](#table-of-contents)
- [What is Rig?](#what-is-rig)
- [Features](#features)
- [Who's using Rig?](#who-is-using-rig)
- [Get Started](#get-started)
  - [Simple example](#simple-example)
- [Integrations](#supported-integrations)

## What is Rig?
Rig is a Rust library for building scalable, modular, and ergonomic **LLM-powered** applications.

More information about this crate can be found in the [official](https://docs.rig.rs) and [crate](https://docs.rs/rig/latest/rig/) API reference documentation.

## Features
- Agentic workflows that can handle multi-turn streaming and prompting
- Full [GenAI Semantic Convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) compatibility
- 20+ model providers, all under one singular unified interface
- 10+ vector store integrations, all under one singular unified interface
- Full support for LLM completion and embedding workflows
- Support for transcription, audio generation and image generation model capabilities
- Integrate LLMs in your app with minimal boilerplate
- Full WASM compatibility (core library only)

## Who is using Rig?
Below is a non-exhaustive list of companies and people who are using Rig:
- [St Jude](https://www.stjude.org/) - Using Rig for a chatbot utility as part of [`proteinpaint`](https://github.com/stjude/proteinpaint), a genomics visualisation tool.
- [Coral Protocol](https://www.coralprotocol.org/) - Using Rig extensively, both internally as well as part of the [Coral Rust SDK.](https://github.com/Coral-Protocol/coral-rs)
- [VT Code](https://github.com/vinhnx/vtcode) - VT Code is a Rust-based terminal coding agent with semantic code intelligence via Tree-sitter and ast-grep. VT Code uses `rig` for simplifying LLM calls and implement model picker.
- [Con](https://github.com/nowledge-co/con) - Con is a GPU-accelerated terminal emulator with a built-in AI agent harness. It uses Rig as the provider abstraction layer for its integrated coding agents.
- [Dria](https://dria.co/) - a decentralised AI network. Currently using Rig as part of their [compute node.](https://github.com/firstbatchxyz/dkn-compute-node)
- [Nethermind](https://www.nethermind.io/) - Using Rig as part of their [Neural Interconnected Nodes Engine](https://github.com/NethermindEth/nine) framework.
- [Neon](https://neon.com) - Using Rig for their [app.build](https://github.com/neondatabase/appdotbuild-agent) V2 reboot in Rust.
- [Listen](https://github.com/piotrostr/listen) - A framework aiming to become the go-to framework for AI portfolio management agents. Powers [the Listen app.](https://app.listen-rs.com/)
- [Cairnify](https://cairnify.com/) - helps users find documents, links, and information instantly through an intelligent search bar. Rig provides the agentic foundation behind Cairnify’s AI search experience, enabling tool-calling, reasoning, and retrieval workflows.
- [Ryzome](https://ryzome.ai) - Ryzome is a visual AI workspace that lets you build interconnected canvases of thoughts, research, and AI agents to orchestrate complex knowledge work.
- [deepwiki-rs](https://github.com/sopaco/deepwiki-rs) - Turn code into clarity. Generate accurate technical docs and AI-ready context in minutes—perfectly structured for human teams and intelligent agents.
- [Cortex Memory](https://github.com/sopaco/cortex-mem) - The production-ready memory system for intelligent agents. A complete solution for memory management, from extraction and vector search to automated optimization, with a REST API, MCP, CLI, and insights dashboard out-of-the-box.
- [Ironclaw](https://github.com/nearai/ironclaw) - A secure personal AI assistant
- [ilert](https://www.ilert.com/) - Incident management & alerting platform. Uses Rig as the multi-provider abstraction in its agentic LLM proxy powering ilert AI.

For a full list, check out our [ECOSYSTEM.md file.](https://www.github.com/0xPlaygrounds/rig/tree/main/ECOSYSTEM.md)

Are you also using Rig? [Open an issue](https://www.github.com/0xPlaygrounds/rig/issues) to have your name added!

## Get Started
Use the root `rig` facade when you want feature-gated access to companion crates,
or use `rig-core` directly when you only need the core provider abstractions.

```bash
cargo add rig
# or: cargo add rig-core
```

### Simple example
```rust
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = openai::Client::from_env()?;

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(openai::GPT_5_2)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;

    println!("{response}");

    Ok(())
}
```
Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

You can find more examples in each crate's `examples` directory (for example, [`examples`](./examples)). Many provider-specific examples now also live as ignored live integration tests under [`tests/providers`](./tests/providers), organized by provider. When running those provider-backed tests, prefer provider-specific targets such as `cargo test -p rig --test openai -- --ignored --test-threads=1` to avoid rate-limiting. More detailed use case walkthroughs are regularly published on our [Dev.to Blog](https://dev.to/0thtachi) and added to Rig's official documentation at [docs.rig.rs](https://docs.rig.rs).

## Supported Integrations

Vector stores are available as separate companion-crates:
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

The root `rig` facade also exposes these companion crates behind one feature per integration:

```toml
rig = { version = "0.36.0", features = ["lancedb", "fastembed"] }
```

Available facade features include `bedrock`, `fastembed`, `gemini-grpc`,
`helixdb`, `lancedb`, `memory`, `milvus`, `mongodb`, `neo4j`, `postgres`,
`qdrant`, `s3vectors`, `scylladb`, `sqlite`, `surrealdb`, `vectorize`, and
`vertexai`.
With those features enabled, use the ergonomic root modules such as
`rig::lancedb`, `rig::mongodb`, `rig::bedrock`, and `rig::fastembed`.

We also have some other associated crates that have additional functionality you may find helpful when using Rig:
- `rig-onchain-kit` - the [Rig Onchain Kit.](https://github.com/0xPlaygrounds/rig-onchain-kit) Intended to make interactions between Solana/EVM and Rig much easier to implement.


<p align="center">
<br>
<br>
<img src="img/built-by-playgrounds.svg" alt="Build by Playgrounds" width="30%">
</p>
