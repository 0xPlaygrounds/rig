//! `ExtractorBuilder<T>` is gone; extraction runs over plain data. The fluent
//! [`ExtractionRunner`] carries the same configuration, and `.classic()`
//! reproduces the old builder's `submit`-tool protocol byte for byte. The
//! extracted type is chosen at `.run::<T>()`, not on the runner.
use std::sync::Arc;

use rig_agent::agent::AgentConfig;
use rig_agent::extract::ExtractionRunner;
use rig_agent::provider::{ProviderConfig, Runtime};
use rig_bedrock::completion::AMAZON_NOVA_LITE;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub job: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Bedrock authenticates through the AWS SDK's default credential chain,
    // so the provider is expressed as plain configuration (the config-level
    // equivalent of `Client::from_env`).
    let person = ExtractionRunner::new(
        AgentConfig::new(),
        ProviderConfig::Bedrock(rig_bedrock::functions::Config::new(AMAZON_NOVA_LITE)),
        Arc::new(Runtime::new()),
        "Hello my name is John Doe! I am a software engineer.",
    )
    .classic()
    .run::<Person>()
    .await?;

    info!("AWS Bedrock: {}", serde_json::to_string_pretty(&person)?);
    Ok(())
}
