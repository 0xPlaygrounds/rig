use reqwest::Client;

use rig::{
    completion::{message::Image, Prompt},
    message::{ContentFormat, ImageMediaType},
    providers::anthropic::{self, CLAUDE_3_5_SONNET},
};

use base64::{prelude::BASE64_STANDARD, Engine};

const IMAGE_URL: &str =
    "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create Anthropic client
    let client = anthropic::Client::from_env();

    // Create agent with a single context prompt
    let agent = client
        .agent(CLAUDE_3_5_SONNET)
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
        media_type: Some(ImageMediaType::JPEG),
        format: Some(ContentFormat::Base64),
        ..Default::default()
    };

    // Prompt the agent and print the response
    let response = agent.prompt(image).await?;
    println!("{}", response);

    Ok(())
}
