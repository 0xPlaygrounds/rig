use base64::{Engine, prelude::BASE64_STANDARD};
use rig::message::DocumentSourceKind;
use rig::prelude::*;
use rig::providers::ollama;
use rig::{
    completion::{Prompt, message::Image},
    message::ImageMediaType,
};
use tokio::fs;

const IMAGE_FILE_PATH: &str = "rig-core/examples/images/camponotus_flavomarginatus_ant.jpg";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create ollama client
    let client = ollama::Client::from_env();

    // Create agent with a single context prompt
    let agent = client
        .agent("llava")
        .preamble("describe this image and make sure to include anything notable about it (include text you see in the image)")
        .temperature(0.5)
        .build();

    // Read image and convert to base64
    let image_bytes = fs::read(IMAGE_FILE_PATH).await?;
    let image_base64 = BASE64_STANDARD.encode(image_bytes);

    // Compose `Image` for prompt
    let image = Image {
        data: DocumentSourceKind::base64(&image_base64),
        media_type: Some(ImageMediaType::JPEG),
        ..Default::default()
    };

    // Prompt the agent and print the response
    let response = agent.prompt(image).await?;

    println!("{response}");

    Ok(())
}
