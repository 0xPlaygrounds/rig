## Rig-BlockRun

This companion crate integrates [BlockRun](https://blockrun.ai) with Rig, providing pay-per-request access to 30+ AI models via x402 micropayments.

### Features

- **No API keys**: Uses wallet signatures for payment instead of traditional API keys
- **x402 Protocol**: Implements HTTP 402 Payment Required flow with EIP-712 signed USDC authorizations
- **Multi-model access**: Supports Claude, GPT-4o, Gemini, DeepSeek, Grok, and more through a single provider
- **Tool calling support**: Full compatibility with Rig's tool/agent system

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-blockrun = "0.1.0"
rig-core = "0.29.0"
```

You can also run `cargo add rig-blockrun rig-core` to add the most recent versions of the dependencies to your project.

### Setup

1. Generate a wallet private key or use an existing one
2. Fund it with USDC on Base (even $1 is enough to get started)
3. Set the `BLOCKRUN_WALLET_KEY` environment variable

### Example

```rust
use rig::prelude::*;
use rig::completion::Prompt;
use rig_blockrun::{Client, CLAUDE_SONNET_4, GPT_4O, DEEPSEEK_CHAT};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create client from wallet private key (no API keys needed!)
    let client = Client::from_env();

    println!("Wallet: {}", client.address());

    // Use Claude, GPT-4, DeepSeek - same interface, one wallet
    let agent = client.agent(CLAUDE_SONNET_4)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = agent.prompt("What is x402?").await?;
    println!("{answer}");

    Ok(())
}
```

See the [`/examples`](./examples) folder for more usage examples.
