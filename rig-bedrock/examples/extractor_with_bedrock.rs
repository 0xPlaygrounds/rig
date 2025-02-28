use rig::providers::anthropic::{
    completion::CompletionModel, ClientBuilder as AnthropicClientBuilder,
};
use rig_bedrock::client::ClientBuilder;
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

    let client = ClientBuilder::new().build().await;
    let anthropic_client = AnthropicClientBuilder::new("").build();
    let completion_model = anthropic_client.completion_model("claude-3-5-sonnet-20240620-v1:0");
    let data_extractor = client
        .extractor::<Person, CompletionModel>(completion_model, "claude-3-5-sonnet-20240620-v1:0")
        .build();
    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    info!(
        "AWS Bedrock: {}",
        serde_json::to_string_pretty(&person).unwrap()
    );
    Ok(())
}
