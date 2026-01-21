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
<a href="https://github.com/BlockRunAI/rig"><img src="https://img.shields.io/github/stars/BlockRunAI/rig?style=social" alt="stars - rig" /></a>
<br>

<br>
</p>
&nbsp;


<div align="center">

[📑 Docs](https://docs.rig.rs)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[🌐 BlockRun](https://blockrun.ai)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[⚡ x402 Protocol](https://x402.org)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[✍🏽 Blogs](https://docs.rig.rs/guides)

</div>

---

## BlockRun: Pay-Per-Request AI — No API Keys Needed

This fork adds **[BlockRun](https://blockrun.ai)** support to Rig, enabling pay-per-request access to **30+ AI models** via [x402 micropayments](https://x402.org). Fund a wallet with USDC on Base and start building—no API keys, no subscriptions, no rate limits.

### Why BlockRun?

| Traditional API Keys | BlockRun x402 |
|---------------------|---------------|
| Apply for API access | Fund a wallet with USDC |
| Wait for approval | Start immediately |
| Monthly minimums | Pay only for what you use |
| Rate limits | No artificial limits |
| Manage multiple keys | One wallet, all providers |
| Key rotation & security | Cryptographic signatures |

### Quick Start

```toml
[dependencies]
rig-core = { git = "https://github.com/BlockRunAI/rig", features = ["blockrun"] }
```

```rust
use rig::prelude::*;
use rig::providers::blockrun::{self, CLAUDE_SONNET_4, GPT_4O, DEEPSEEK_CHAT};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create client from wallet private key (no API keys!)
    let client = blockrun::Client::from_env();

    // Use Claude, GPT-4, DeepSeek — same interface, one wallet
    let agent = client.agent(CLAUDE_SONNET_4)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = agent.prompt("What is x402?").await?;
    println!("{answer}");

    Ok(())
}
```

**Setup:** Set `BLOCKRUN_PRIVATE_KEY=0x...` with a wallet funded with USDC on Base.

### On-Chain Transaction Proof (Base Mainnet)

Every API request is settled on-chain with cryptographic proof:

| Model | Cost | Latency | Transaction |
|-------|------|---------|-------------|
| Claude Sonnet 4 | 0.016239 USDC | ~5s | [0x6b2e42f5...](https://basescan.org/tx/0x6b2e42f5341bbf51df123756789553d05621db877cebecdcc5bddf00fdd1fd34) |
| GPT-4o | 0.010821 USDC | ~4s | [0x2ebec29e...](https://basescan.org/tx/0x2ebec29ef5b2ed0706dc64c5320d79613ec8d469b008da477f3bf4908128233a) |
| DeepSeek | 0.001 USDC | ~3s | [0xb960c54e...](https://basescan.org/tx/0xb960c54e34a65b8811100672a3ebb3d29cda764423c68969f043a028ddf5e193) |
| Claude + Tools | 0.01621 USDC | ~4s | [0xf1a5c831...](https://basescan.org/tx/0xf1a5c8318c1e75e84ffaa10a12013cd2ddecfbabb59ea126688c3594300b52b6) |

**Test Wallet:** [0x4069560641ec74acfc74ddec64181f588c64e3a7](https://basescan.org/address/0x4069560641ec74acfc74ddec64181f588c64e3a7#tokentxns)

### Available Models

**Anthropic:** Claude Opus 4, Claude Sonnet 4, Claude Sonnet 3.5, Claude Haiku 3.5
**OpenAI:** GPT-4o, GPT-4o Mini, GPT-o1, GPT-o1 Mini, GPT-o3 Mini
**DeepSeek:** DeepSeek Chat, DeepSeek Reasoner
**Meta:** Llama 3.3 70B, Llama 3.1 405B/70B/8B
**Google:** Gemini 2.5 Pro, Gemini 2.0 Flash, Gemini 1.5 Pro
**Mistral:** Mistral Large, Codestral, Ministral 8B
**Image:** Flux 1.1 Pro, Flux 1.1 Pro Ultra

### Upstream PR

This BlockRun provider has been submitted upstream: [0xPlaygrounds/rig#1294](https://github.com/0xPlaygrounds/rig/pull/1294)

---

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
- [St Jude](https://www.stjude.org/) - Using Rig for a chatbot utility as part of [`proteinpaint`](https://github.com/stjude/proteinpaint)
- [Coral Protocol](https://www.coralprotocol.org/) - Using Rig extensively, part of the [Coral Rust SDK](https://github.com/Coral-Protocol/coral-rs)
- [VT Code](https://github.com/vinhnx/vtcode) - Rust-based terminal coding agent
- [Dria](https://dria.co/) - Decentralised AI network, part of their [compute node](https://github.com/firstbatchxyz/dkn-compute-node)
- [Nethermind](https://www.nethermind.io/) - Part of their [Neural Interconnected Nodes Engine](https://github.com/NethermindEth/nine)
- [Neon](https://neon.com) - Using Rig for their [app.build](https://github.com/neondatabase/appdotbuild-agent) V2
- [Listen](https://github.com/piotrostr/listen) - AI portfolio management agents
- [Cairnify](https://cairnify.com/) - Intelligent search bar with agentic AI
- [Ryzome](https://ryzome.ai) - Visual AI workspace

For the full list, see [ECOSYSTEM.md](https://www.github.com/0xPlaygrounds/rig/tree/main/ECOSYSTEM.md)

## Get Started (Standard Providers)

```bash
cargo add rig-core
```

```rust
use rig::providers::openai;
use rig::completion::Prompt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env();
    let agent = client.agent(openai::GPT_4O)
        .preamble("You are a comedian.")
        .build();

    let response = agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}
```

## Supported Integrations

**Vector Stores:**
[MongoDB](https://github.com/0xPlaygrounds/rig/tree/main/rig-mongodb) •
[LanceDB](https://github.com/0xPlaygrounds/rig/tree/main/rig-lancedb) •
[Neo4j](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j) •
[Qdrant](https://github.com/0xPlaygrounds/rig/tree/main/rig-qdrant) •
[SQLite](https://github.com/0xPlaygrounds/rig/tree/main/rig-sqlite) •
[SurrealDB](https://github.com/0xPlaygrounds/rig/tree/main/rig-surrealdb) •
[Milvus](https://github.com/0xPlaygrounds/rig/tree/main/rig-milvus) •
[ScyllaDB](https://github.com/0xPlaygrounds/rig/tree/main/rig-scylladb) •
[S3Vectors](https://github.com/0xPlaygrounds/rig/tree/main/rig-s3vectors) •
[HelixDB](https://github.com/0xPlaygrounds/rig/tree/main/rig-helixdb)

**Additional Providers:**
[AWS Bedrock](https://github.com/0xPlaygrounds/rig/tree/main/rig-bedrock) •
[Fastembed](https://github.com/0xPlaygrounds/rig/tree/main/rig-fastembed) •
[Eternal AI](https://github.com/0xPlaygrounds/rig/tree/main/rig-eternalai) •
[Google Vertex](https://github.com/0xPlaygrounds/rig/tree/main/rig-vertexai)

**Built-in Providers:**
BlockRun • Anthropic • Azure • Cohere • DeepSeek • Galadriel • Gemini • Groq • Huggingface • Hyperbolic • Mira • Mistral • Moonshot • Ollama • OpenAI • OpenRouter • Perplexity • Together • Voyage AI • xAI

---

<p align="center">
<br>
<a href="https://blockrun.ai"><img src="https://img.shields.io/badge/Powered%20by-BlockRun-blue" alt="Powered by BlockRun" /></a>
&nbsp;
<a href="https://x402.org"><img src="https://img.shields.io/badge/x402-Micropayments-green" alt="x402 Micropayments" /></a>
<br>
<br>
<img src="img/built-by-playgrounds.svg" alt="Build by Playgrounds" width="30%">
</p>
