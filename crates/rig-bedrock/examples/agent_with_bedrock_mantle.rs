//! Minimal agent against Bedrock Mantle (OpenAI-compatible API).
//!
//! Defaults to the **Responses** path for GPT-OSS on
//! `https://bedrock-mantle.{region}.api.aws/v1` with `store: false`.
//!
//! Completions alternative (also on `/v1`):
//!
//! ```ignore
//! let client = mantle::from_env_completions().await?;
//! let agent = client.agent(OPENAI_GPT_OSS_20B).build();
//! ```
//!
//! GPT-5.x Responses use the alternate base:
//! `ClientBuilder::from_env().base_url(openai_gpt5_base_url(&region)).build().await?`
//!
//! Requires AWS credentials (or `AWS_BEARER_TOKEN_BEDROCK`) and Mantle model access
//! in the target region.
//!
//! ```shell
//! export AWS_REGION=us-east-1
//! # optional: export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...
//! cargo run -p rig-bedrock --example agent_with_bedrock_mantle
//! ```

use rig_bedrock::mantle::{self, OPENAI_GPT_OSS_20B};
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Responses API on default /v1 (GPT-OSS). Token is snapshotted at build (12h TTL).
    let client = mantle::from_env().await?;
    let agent = client
        .agent(OPENAI_GPT_OSS_20B)
        .preamble("You are a concise assistant.")
        .additional_params(serde_json::json!({"store": false}))
        .build();

    let response = agent.prompt("Say hello in one short sentence.").await?;
    info!("{response}");

    Ok(())
}
