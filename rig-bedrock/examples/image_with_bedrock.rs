use reqwest::Client;

use base64::{prelude::BASE64_STANDARD, Engine};
use rig::{
    completion::message::Image,
    message::{ContentFormat, ImageMediaType},
};
use rig::{
    completion::{Preamble, Prompt},
    providers::anthropic::ClientBuilder as AnthropicClientBuilder,
};
use rig_bedrock::client::ClientBuilder;
use tracing::info;

const IMAGE_URL: &str = "https://playgrounds.network/assets/PG-Logo.png";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = ClientBuilder::new().build().await;
    let anthropic_client = AnthropicClientBuilder::new("").build();
    let completion_model = anthropic_client.completion_model("claude-3-5-sonnet-20240620-v1:0");
    let agent = client
        .agent(completion_model, "claude-3-5-sonnet-20240620-v1:0")
        .preamble(vec![Preamble::new(
            "You are an image describer.".to_string(),
        )])
        .temperature(0.5)
        .build();

    // Grab image and convert to base64
    let reqwest_client = Client::new();
    let image_bytes = reqwest_client.get(IMAGE_URL).send().await?.bytes().await?;
    let image_base64 = BASE64_STANDARD.encode(image_bytes);

    // Compose `Image` for prompt
    let image = Image {
        data: image_base64,
        media_type: Some(ImageMediaType::PNG),
        format: Some(ContentFormat::Base64),
        ..Default::default()
    };

    // Prompt the agent and print the response
    let response = agent.prompt(image).await?;
    info!("{}", response);

    Ok(())
}
