use rig_bedrock::{client::ClientBuilder, completion::AMAZON_NOVA_LITE};
use rig_shared::fixtures::person::Person;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = ClientBuilder::new().build().await;

    let data_extractor = client.extractor::<Person>(AMAZON_NOVA_LITE).build();
    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    info!(
        "AWS Bedrock: {}",
        serde_json::to_string_pretty(&person).unwrap()
    );
    Ok(())
}
