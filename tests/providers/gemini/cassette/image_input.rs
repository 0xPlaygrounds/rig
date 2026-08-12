//! Gemini inline image *input* (the suite already covers image generation;
//! this records the ingestion direction: a base64 `inline_data` part in the
//! request and a grounded description in the response).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
use base64::{Engine, prelude::BASE64_STANDARD};
use rig::completion::Prompt;
use rig::completion::message::Image;
use rig::message::DocumentSourceKind;
use rig::message::ImageMediaType;
use rig::prelude::*;
use rig::providers::gemini;
use tokio::fs;

use super::super::support::with_gemini_cassette;
use crate::support::{
    IMAGE_FIXTURE_PATH, assert_contains_any_case_insensitive, assert_nonempty_response,
};

#[tokio::test]
async fn image_prompt_from_fixture() {
    with_gemini_cassette(
        "image_input/image_prompt_from_fixture",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble("You are an image describer.")
                .temperature(0.0)
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
        },
    )
    .await;
}
