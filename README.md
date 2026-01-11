<p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/rig-rebranded-logo-white.svg">
    <source media="(prefers-color-scheme: light)" srcset="img/rig-rebranded-logo-black.svg">
    <img src="img/rig-rebranded-logo-white.svg" style="width: 40%; height: 40%;" alt="Rig logo">
</picture>
<br>
<br>
<a href="https://docs.rig.rs"><img src="https://img.shields.io/badge/📖 docs-rig.rs-dca282.svg" /></a> &nbsp;
<a href="https://docs.rs/rig-core/latest/rig/"><img src="https://img.shields.io/badge/docs-API Reference-dca282.svg" /></a> &nbsp;
<a href="https://crates.io/crates/rig-core"><img src="https://img.shields.io/crates/v/rig-core.svg?color=dca282" /></a>
&nbsp;
<a href="https://crates.io/crates/rig-core"><img src="https://img.shields.io/crates/d/rig-core.svg?color=dca282" /></a>
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
- [High-level features](#high-level-features)
- [Who's using Rig?](#who-is-using-rig)
- [Get Started](#get-started)
  - [Simple example](#simple-example)
- [Integrations](#supported-integrations)

## What is Rig?
Rig is a Rust library for building scalable, modular, and ergonomic **LLM-powered** applications.

More information about this crate can be found in the [official](https://docs.rig.rs) & [crate](https://docs.rs/rig-core/latest/rig/) (API Reference) documentations.

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
- [Dria](https://dria.co/) - a decentralised AI network. Currently using Rig as part of their [compute node.](https://github.com/firstbatchxyz/dkn-compute-node)
- [Nethermind](https://www.nethermind.io/) - Using Rig as part of their [Neural Interconnected Nodes Engine](https://github.com/NethermindEth/nine) framework.
- [Neon](https://neon.com) - Using Rig for their [app.build](https://github.com/neondatabase/appdotbuild-agent) V2 reboot in Rust.
- [Listen](https://github.com/piotrostr/listen) - A framework aiming to become the go-to framework for AI portfolio management agents. Powers [the Listen app.](https://app.listen-rs.com/)
- [Cairnify](https://cairnify.com/) - helps users find documents, links, and information instantly through an intelligent search bar. Rig provides the agentic foundation behind Cairnify’s AI search experience, enabling tool-calling, reasoning, and retrieval workflows.
- [Ryzome](https://ryzome.ai) - Ryzome is a visual AI workspace that lets you build interconnected canvases of thoughts, research, and AI agents to orchestrate complex knowledge work.

For a full list, check out our [ECOSYSTEM.md file.](https://www.github.com/0xPlaygrounds/rig/tree/main/ECOSYSTEM.md)

Are you also using Rig? [Open an issue](https://www.github.com/0xPlaygrounds/rig/issues) to have your name added!

## Get Started
```bash
cargo add rig-core
```

### Simple example
```rust
use rig::{client::CompletionClient, completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    // This requires the `OPENAI_API_KEY` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```
Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

You can find more examples each crate's `examples` (ie. [`rig-core/examples`](./rig-core/examples)) directory. More detailed use cases walkthroughs are regularly published on our [Dev.to Blog](https://dev.to/0thtachi) and added to Rig's official documentation [(docs.rig.rs)](http://docs.rig.rs).

## Supported Integrations

Vector stores are available as separate companion-crates:
- MongoDB: [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-mongodb)
- LanceDB: [`rig-lancedb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-lancedb)
- Neo4j: [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j)
- Qdrant: [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/rig-qdrant)
- SQLite: [`rig-sqlite`](https://github.com/0xPlaygrounds/rig/tree/main/rig-sqlite)
- SurrealDB: [`rig-surrealdb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-surrealdb)
- Milvus: [`rig-milvus`](https://github.com/0xPlaygrounds/rig/tree/main/rig-milvus)
- ScyllaDB: [`rig-scylladb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-scylladb)
- AWS S3Vectors: [`rig-s3vectors`](https://github.com/0xPlaygrounds/rig/tree/main/rig-s3vectors)
- HelixDB: [`rig-helixdb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-helixdb)

The following providers are available as separate companion-crates:
- AWS Bedrock: [`rig-bedrock`](https://github.com/0xPlaygrounds/rig/tree/main/rig-bedrock)
- Fastembed: [`rig-fastembed`](https://github.com/0xPlaygrounds/rig/tree/main/rig-fastembed)
- Eternal AI: [`rig-eternalai`](https://github.com/0xPlaygrounds/rig/tree/main/rig-eternalai)
- Google Vertex: [`rig-vertexai`](https://github.com/0xPlaygrounds/rig/tree/main/rig-vertexai)

We also have some other associated crates that have additional functionality you may find helpful when using Rig:
- `rig-onchain-kit` - the [Rig Onchain Kit.](https://github.com/0xPlaygrounds/rig-onchain-kit) Intended to make interactions between Solana/EVM and Rig much easier to implement.


<p align="center">
<br>
<br>
<img src="img/built-by-playgrounds.svg" alt="Build by Playgrounds" width="30%">
</p>
