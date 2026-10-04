//! Cassette-backed OpenRouter multimodal prompts.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::message::{
    AudioMediaType, DocumentSourceKind, Image, ImageMediaType, Message, UserContent, VideoMediaType,
};

use crate::support::{
    AUDIO_FIXTURE_PATH, IMAGE_FIXTURE_PATH, VIDEO_FIXTURE_PATH, assert_nonempty_response,
};

use super::super::support::with_openrouter_cassette;

const VISION_MODEL: &str = "google/gemini-2.5-flash";

fn image_message() -> Image {
    let bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
    Image {
        data: DocumentSourceKind::base64(BASE64_STANDARD.encode(bytes)),
        media_type: Some(ImageMediaType::JPEG),
        detail: None,
        native: None,
    }
}

/// Builds base64 video content via the `UserContent::video_base64` helper.
fn video_content() -> UserContent {
    let bytes = std::fs::read(VIDEO_FIXTURE_PATH).expect("fixture video should be readable");
    UserContent::video_base64(BASE64_STANDARD.encode(bytes), Some(VideoMediaType::MP4))
}

/// Builds base64 audio content via the `UserContent::audio_base64` helper.
fn audio_content() -> UserContent {
    let bytes = std::fs::read(AUDIO_FIXTURE_PATH).expect("fixture audio should be readable");
    UserContent::audio_base64(BASE64_STANDARD.encode(bytes), Some(AudioMediaType::MP3))
}

#[tokio::test]
async fn mixed_multimodal_prompt() {
    with_openrouter_cassette("multimodal/mixed_multimodal_prompt", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
            .preamble("You are a helpful assistant.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: vec![
                    UserContent::text("I have two questions:"),
                    UserContent::text("1. What colors do you see in this image?"),
                    UserContent::Image(image_message()),
                    UserContent::text("2. What is the main subject?"),
                ],
            })
            .await
            .expect("mixed content prompt should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}

#[tokio::test]
async fn video_analysis_prompt() {
    with_openrouter_cassette("multimodal/video_analysis_prompt", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
            .preamble("You are a helpful assistant that describes videos.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: vec![
                    UserContent::text("What do you see in this short video? Describe it briefly."),
                    video_content(),
                ],
            })
            .await
            .expect("video prompt should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}

#[tokio::test]
async fn audio_analysis_prompt() {
    with_openrouter_cassette("multimodal/audio_analysis_prompt", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
            .preamble("You are a helpful assistant that transcribes and describes audio.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: vec![
                    UserContent::text("What is said in this audio clip? Transcribe it briefly."),
                    audio_content(),
                ],
            })
            .await
            .expect("audio prompt should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}
