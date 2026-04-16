//! Migrated from `examples/image.rs`.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::completion::message::Image;
use rig::message::DocumentSourceKind;
use rig::message::ImageMediaType;
use rig::providers::anthropic;
use tokio::fs;

use crate::support::{
    IMAGE_FIXTURE_PATH, assert_contains_any_case_insensitive, assert_nonempty_response,
};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn image_prompt_from_fixture() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You are an image describer.")
        .temperature(0.5)
        .build();

    let image_bytes = fs::read(IMAGE_FIXTURE_PATH)
        .await
        .expect("fixture image should be readable");
    let image = Image {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(image_bytes)),
        media_type: Some(ImageMediaType::JPEG),
        ..Default::default()
    };

    let response = agent
        .prompt(image)
        .await
        .expect("image prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["ant", "insect"]);
}
