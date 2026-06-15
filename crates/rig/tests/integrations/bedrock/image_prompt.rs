//! AWS Bedrock image prompt smoke test inspired by Anthropic image tests.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::message::{ImageMediaType, Message, UserContent};
use tokio::fs;

use super::{
    BEDROCK_COMPLETION_MODEL, client,
    support::{IMAGE_FIXTURE_PATH, assert_contains_any_case_insensitive, assert_nonempty_response},
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock vision model access"]
async fn image_prompt_from_fixture() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble("You are an image describer.")
        .temperature(0.5)
        .build();

    let image_bytes = fs::read(IMAGE_FIXTURE_PATH)
        .await
        .expect("fixture image should be readable");
    let response = agent
        .prompt(Message::User {
            content: OneOrMany::many(vec![
                UserContent::image_base64(
                    BASE64_STANDARD.encode(image_bytes),
                    Some(ImageMediaType::JPEG),
                    None,
                ),
                UserContent::text("Describe the image in one sentence."),
            ])
            .expect("content should be non-empty"),
        })
        .await
        .expect("image prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["ant", "insect"]);
}
