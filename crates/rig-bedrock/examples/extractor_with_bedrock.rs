use rig_agent::prelude::*;
use rig_bedrock::Client;
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

    let client = Client::from_env();
    let agent = client.agent(AMAZON_NOVA_LITE).build();
    let person: Person = agent
        .extractor("Hello my name is John Doe! I am a software engineer.")
        .classic()
        .run()
        .await?;

    info!("AWS Bedrock: {}", serde_json::to_string_pretty(&person)?);
    Ok(())
}
