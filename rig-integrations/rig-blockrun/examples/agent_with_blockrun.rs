//! Example of using BlockRun provider with Rig.
//!
//! BlockRun provides pay-per-request access to 30+ AI models via x402 micropayments.
//! No API keys needed - just fund a wallet with USDC on Base.
//!
//! # Setup
//!
//! 1. Generate a wallet private key or use an existing one
//! 2. Fund it with USDC on Base (even $1 is enough to get started)
//! 3. Set `BLOCKRUN_WALLET_KEY` environment variable
//!
//! # Running
//!
//! ```bash
//! BLOCKRUN_WALLET_KEY=0x... cargo run --example agent_with_blockrun
//! ```

use rig::{
    completion::{Prompt, ToolDefinition},
    tool::Tool,
};
use rig_blockrun::{Client, CLAUDE_SONNET_4, DEEPSEEK_CHAT, GPT_4O};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Create BlockRun client from environment variable
    // This reads BLOCKRUN_WALLET_KEY - your wallet key for signing payments
    let client = Client::from_env();

    // Print the wallet address (useful for funding)
    println!("Wallet address: {}", client.address());
    println!("Fund this address with USDC on Base to use BlockRun\n");

    // Simple prompt with Claude Sonnet
    println!("=== Using Claude Sonnet 4 ===");
    let claude_agent = client
        .agent(CLAUDE_SONNET_4)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = claude_agent.prompt("What is x402 in one sentence?").await?;
    println!("Claude: {answer}\n");

    // Try GPT-4o for comparison
    println!("=== Using GPT-4o ===");
    let gpt_agent = client
        .agent(GPT_4O)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = gpt_agent.prompt("What is x402 in one sentence?").await?;
    println!("GPT-4o: {answer}\n");

    // Use DeepSeek for cost-effective inference
    println!("=== Using DeepSeek (cost-effective) ===");
    let deepseek_agent = client
        .agent(DEEPSEEK_CHAT)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = deepseek_agent
        .prompt("What is x402 in one sentence?")
        .await?;
    println!("DeepSeek: {answer}\n");

    // Agent with tools
    println!("=== Calculator Agent with Tools ===");
    let calculator_agent = client
        .agent(CLAUDE_SONNET_4)
        .preamble("You are a calculator. Use the provided tools to perform calculations.")
        .max_tokens(1024)
        .tool(Adder)
        .tool(Multiplier)
        .build();

    let answer = calculator_agent
        .prompt("What is (15 + 7) * 3?")
        .await?;
    println!("Calculator: {answer}");

    Ok(())
}

// Tool definitions

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add two numbers together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "First number"
                    },
                    "y": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool] add({}, {}) = {}", args.x, args.y, args.x + args.y);
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Multiplier;

impl Tool for Multiplier {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "multiply".to_string(),
            description: "Multiply two numbers together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "First number"
                    },
                    "y": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!(
            "[tool] multiply({}, {}) = {}",
            args.x,
            args.y,
            args.x * args.y
        );
        Ok(args.x * args.y)
    }
}
