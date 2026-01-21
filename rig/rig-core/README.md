# Rig + BlockRun: Pay-Per-Request AI with No API Keys

This is [BlockRun's fork](https://github.com/BlockRunAI/rig) of [Rig](https://github.com/0xPlaygrounds/rig) with native **x402 micropayment** support. Access 30+ AI models (Claude, GPT-4, DeepSeek, Llama, and more) with pay-per-request pricing—no API keys, no subscriptions, no rate limits.

## Why BlockRun?

| Traditional API Keys | BlockRun x402 |
|---------------------|---------------|
| Apply for API access | Fund a wallet with USDC |
| Wait for approval | Start immediately |
| Monthly minimums | Pay only for what you use |
| Rate limits | No artificial limits |
| Manage multiple keys | One wallet, all providers |
| Key rotation & security | Cryptographic signatures |

## Installation

```toml
[dependencies]
rig-core = { git = "https://github.com/BlockRunAI/rig", features = ["blockrun"] }
```

## Quick Start

```rust
use rig::prelude::*;
use rig::providers::blockrun::{self, CLAUDE_SONNET_4, GPT_4O, DEEPSEEK_CHAT};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create client from wallet private key (no API keys needed!)
    let client = blockrun::Client::from_env();

    println!("Wallet: {}", client.address());

    // Use Claude
    let claude = client.agent(CLAUDE_SONNET_4)
        .preamble("You are a helpful assistant.")
        .build();
    let answer = claude.prompt("What is x402?").await?;
    println!("Claude: {answer}");

    // Use GPT-4o (same wallet, same interface)
    let gpt = client.agent(GPT_4O)
        .preamble("You are a helpful assistant.")
        .build();
    let answer = gpt.prompt("What is x402?").await?;
    println!("GPT-4o: {answer}");

    // Use DeepSeek (cost-effective)
    let deepseek = client.agent(DEEPSEEK_CHAT)
        .preamble("You are a helpful assistant.")
        .build();
    let answer = deepseek.prompt("What is x402?").await?;
    println!("DeepSeek: {answer}");

    Ok(())
}
```

**Setup:**
1. Generate or use an existing Ethereum wallet
2. Fund it with USDC on Base (even $1 works)
3. Set `BLOCKRUN_PRIVATE_KEY=0x...` environment variable
4. Run your code

## On-Chain Transaction Proof

Every request is settled on Base mainnet with cryptographic proof:

| Model | Cost | Latency | Transaction |
|-------|------|---------|-------------|
| Claude Sonnet 4 | 0.016239 USDC | ~5s | [0x6b2e42f5...](https://basescan.org/tx/0x6b2e42f5341bbf51df123756789553d05621db877cebecdcc5bddf00fdd1fd34) |
| GPT-4o | 0.010821 USDC | ~4s | [0x2ebec29e...](https://basescan.org/tx/0x2ebec29ef5b2ed0706dc64c5320d79613ec8d469b008da477f3bf4908128233a) |
| DeepSeek | 0.001 USDC | ~3s | [0xb960c54e...](https://basescan.org/tx/0xb960c54e34a65b8811100672a3ebb3d29cda764423c68969f043a028ddf5e193) |
| Claude + Tools | 0.01621 USDC | ~4s | [0xf1a5c831...](https://basescan.org/tx/0xf1a5c8318c1e75e84ffaa10a12013cd2ddecfbabb59ea126688c3594300b52b6) |

Test wallet: [0x4069560641ec74acfc74ddec64181f588c64e3a7](https://basescan.org/address/0x4069560641ec74acfc74ddec64181f588c64e3a7#tokentxns)

## Available Models

### Anthropic
- `CLAUDE_OPUS_4` - Most capable
- `CLAUDE_SONNET_4` - Balanced performance
- `CLAUDE_SONNET_3_5` - Fast and efficient
- `CLAUDE_HAIKU_3_5` - Fastest, most economical

### OpenAI
- `GPT_4O` - Latest GPT-4
- `GPT_4O_MINI` - Cost-effective
- `GPT_O1` - Reasoning model
- `GPT_O1_MINI` - Compact reasoning
- `GPT_O3_MINI` - Latest mini

### DeepSeek
- `DEEPSEEK_CHAT` - General purpose
- `DEEPSEEK_REASONER` - Enhanced reasoning

### Meta Llama
- `LLAMA_3_3_70B`
- `LLAMA_3_1_405B`
- `LLAMA_3_1_70B`
- `LLAMA_3_1_8B`

### Google
- `GEMINI_2_5_PRO`
- `GEMINI_2_0_FLASH`
- `GEMINI_1_5_PRO`

### Mistral
- `MISTRAL_LARGE`
- `CODESTRAL`
- `MINISTRAL_8B`

### Image Generation
- `FLUX_1_1_PRO`
- `FLUX_1_1_PRO_ULTRA`

## How x402 Works

1. **Request** - Your app sends an API request
2. **402 Response** - Server returns payment requirements (amount, recipient)
3. **Sign** - Client signs a USDC authorization (EIP-712)
4. **Retry** - Request sent with payment header
5. **Settle** - Payment settled on Base, response delivered

All signing happens locally. Your private key never leaves your machine.

## Agent with Tools

```rust
use rig::prelude::*;
use rig::providers::blockrun::{self, CLAUDE_SONNET_4};
use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, Serialize)]
struct Calculator;

#[derive(Deserialize)]
struct CalcArgs { x: i32, y: i32 }

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

impl Tool for Calculator {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = CalcArgs;
    type Output = i32;

    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add two numbers".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = blockrun::Client::from_env();

    let agent = client.agent(CLAUDE_SONNET_4)
        .preamble("You are a calculator.")
        .tool(Calculator)
        .build();

    let answer = agent.prompt("What is 15 + 27?").await?;
    println!("{answer}");

    Ok(())
}
```

## About This Fork

This fork adds the `blockrun` provider to Rig. We've submitted a [pull request](https://github.com/0xPlaygrounds/rig/pull/1294) to upstream. Until merged, use this fork directly.

**Upstream PR:** [#1294](https://github.com/0xPlaygrounds/rig/pull/1294)

---

# Rig Core Features

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## Features
- Agentic workflows with multi-turn streaming and prompting
- Full [GenAI Semantic Convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) compatibility
- 20+ model providers under one unified interface
- 10+ vector store integrations
- Full support for completion and embedding workflows
- Support for transcription, audio generation and image generation
- Full WASM compatibility (core library only)

## All Providers

Rig supports these providers out of the box:
- **BlockRun** (pay-per-request via x402)
- Anthropic
- Azure
- Cohere
- Deepseek
- Galadriel
- Gemini
- Groq
- Huggingface
- Hyperbolic
- Mira
- Mistral
- Moonshot
- Ollama
- OpenAI
- OpenRouter
- Perplexity
- Together
- Voyage AI
- xAI

## Vector Stores

Available as companion crates:
- MongoDB: [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-mongodb)
- LanceDB: [`rig-lancedb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-lancedb)
- Neo4j: [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j)
- Qdrant: [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/rig-qdrant)
- SQLite: [`rig-sqlite`](https://github.com/0xPlaygrounds/rig/tree/main/rig-sqlite)
- SurrealDB: [`rig-surrealdb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-surrealdb)
- Milvus: [`rig-milvus`](https://github.com/0xPlaygrounds/rig/tree/main/rig-milvus)
- ScyllaDB: [`rig-scylladb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-scylladb)
- AWS S3Vectors: [`rig-s3vectors`](https://github.com/0xPlaygrounds/rig/tree/main/rig-s3vectors)

## Links

- [BlockRun](https://blockrun.ai) - Pay-per-request AI
- [x402 Protocol](https://www.x402.org) - HTTP 402 payments
- [Rig Documentation](https://docs.rig.rs)
- [Rig Examples](https://github.com/0xPlaygrounds/rig/tree/main/rig-core/examples)
