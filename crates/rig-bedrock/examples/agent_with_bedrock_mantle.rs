//! Minimal agent against Bedrock Mantle (OpenAI-compatible Responses API).
//!
//! Requires AWS credentials (or `AWS_BEARER_TOKEN_BEDROCK`) and Mantle model access
//! in the target region.
//!
//! ```shell
//! export AWS_REGION=us-east-1
//! # optional: export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...
//! cargo run -p rig-bedrock --example agent_with_bedrock_mantle
//! ```

use rig_bedrock::mantle::{ClientBuilder, OPENAI_GPT_OSS_20B};
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = ClientBuilder::from_env().await?;
    let agent = client
        .agent(OPENAI_GPT_OSS_20B)
        .preamble("You are a concise assistant.")
        .build();

    let response = agent.prompt("Say hello in one short sentence.").await?;
    info!("{response}");

    Ok(())
}
