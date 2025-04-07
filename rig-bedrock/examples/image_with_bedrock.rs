use reqwest::Client;

use rig::{
    completion::{message::Image, Prompt},
    message::{ContentFormat, ImageMediaType},
};

use base64::{prelude::BASE64_STANDARD, Engine};
use rig_bedrock::{client::ClientBuilder, completion::AMAZON_NOVA_LITE};
use tracing::info;

const IMAGE_URL: &str = "https://playgrounds.network/assets/PG-Logo.png";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = ClientBuilder::new().build().await;
    let agent = client
        .agent(AMAZON_NOVA_LITE)
        .preamble("You are an image describer.")
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
