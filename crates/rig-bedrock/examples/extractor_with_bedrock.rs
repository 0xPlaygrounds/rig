use rig_agent::extractor::ExtractorBuilder;
use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::completion::{AMAZON_NOVA_LITE, Converse};
use rig_core::Model;
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

    let model = Model::new(Converse::new(AMAZON_NOVA_LITE), BedrockRuntime::from_env());
    let data_extractor = ExtractorBuilder::<Person>::new(model).build();
    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?
        .output;

    info!("AWS Bedrock: {}", serde_json::to_string_pretty(&person)?);
    Ok(())
}
