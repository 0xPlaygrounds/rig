//! Migrated from `examples/image_ollama.rs`.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::completion::message::Image;
use rig::message::DocumentSourceKind;
use rig::message::ImageMediaType;
use rig::providers::ollama;

use crate::support::{
    IMAGE_FIXTURE_PATH, assert_contains_any_case_insensitive, assert_nonempty_response,
};

#[tokio::test]
#[ignore = "requires a local Ollama server with a multimodal model"]
async fn multimodal_image_prompt() {
    let client = ollama::Client::from_env();
    let agent = client
        .agent("llava")
        .preamble("Describe this image and include anything notable about it.")
        .temperature(0.5)
        .build();

    let image_bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
    let image = Image {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(image_bytes)),
        media_type: Some(ImageMediaType::JPEG),
        ..Default::default()
    };
    let response = agent.prompt(image).await.expect("prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["ant", "insect"]);
}
