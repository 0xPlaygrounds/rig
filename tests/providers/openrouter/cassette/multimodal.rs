//! Cassette-backed OpenRouter multimodal prompts.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::message::{
    AudioMediaType, Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType,
    Message, UserContent, VideoMediaType,
};

use crate::support::{
    AUDIO_FIXTURE_PATH, IMAGE_FIXTURE_PATH, PDF_FIXTURE_PATH, VIDEO_FIXTURE_PATH,
    assert_nonempty_response,
};

use super::super::support::with_openrouter_cassette;

const VISION_MODEL: &str = "google/gemini-2.5-flash";

fn image_message() -> Image {
    let bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
    Image {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(bytes)),
        media_type: Some(ImageMediaType::JPEG),
        detail: None,
        additional_params: None,
    }
}

fn pdf_document() -> Document {
    let bytes = std::fs::read(PDF_FIXTURE_PATH).expect("fixture pdf should be readable");
    Document {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(bytes)),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    }
}

/// Builds base64 video content via the `UserContent::video` helper.
fn video_content() -> UserContent {
    let bytes = std::fs::read(VIDEO_FIXTURE_PATH).expect("fixture video should be readable");
    UserContent::video(BASE64_STANDARD.encode(bytes), Some(VideoMediaType::MP4))
}

/// Builds base64 audio content via the `UserContent::audio` helper.
fn audio_content() -> UserContent {
    let bytes = std::fs::read(AUDIO_FIXTURE_PATH).expect("fixture audio should be readable");
    UserContent::audio(BASE64_STANDARD.encode(bytes), Some(AudioMediaType::MP3))
}

#[tokio::test]
async fn image_analysis_prompt() {
    with_openrouter_cassette("multimodal/image_analysis_prompt", |client| async move {
        let agent = client
            .agent(VISION_MODEL)
            .preamble("You are a helpful assistant that describes images in detail.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: OneOrMany::many(vec![
                    UserContent::text("What do you see in this image? Describe it in detail."),
                    UserContent::Image(image_message()),
                ])
                .expect("content should be non-empty"),
            })
            .await
            .expect("image prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn pdf_analysis_prompt() {
    with_openrouter_cassette("multimodal/pdf_analysis_prompt", |client| async move {
        let agent = client
            .agent(VISION_MODEL)
            .preamble("You are a helpful assistant that summarizes documents.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: OneOrMany::many(vec![
                    UserContent::text("Please summarize the key points of this document."),
                    UserContent::Document(pdf_document()),
                ])
                .expect("content should be non-empty"),
            })
            .await
            .expect("pdf prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn mixed_multimodal_prompt() {
    with_openrouter_cassette("multimodal/mixed_multimodal_prompt", |client| async move {
        let agent = client
            .agent(VISION_MODEL)
            .preamble("You are a helpful assistant.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: OneOrMany::many(vec![
                    UserContent::text("I have two questions:"),
                    UserContent::text("1. What colors do you see in this image?"),
                    UserContent::Image(image_message()),
                    UserContent::text("2. What is the main subject?"),
                ])
                .expect("content should be non-empty"),
            })
            .await
            .expect("mixed content prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn video_analysis_prompt() {
    with_openrouter_cassette("multimodal/video_analysis_prompt", |client| async move {
        let agent = client
            .agent(VISION_MODEL)
            .preamble("You are a helpful assistant that describes videos.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: OneOrMany::many(vec![
                    UserContent::text("What do you see in this short video? Describe it briefly."),
                    video_content(),
                ])
                .expect("content should be non-empty"),
            })
            .await
            .expect("video prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn audio_analysis_prompt() {
    with_openrouter_cassette("multimodal/audio_analysis_prompt", |client| async move {
        let agent = client
            .agent(VISION_MODEL)
            .preamble("You are a helpful assistant that transcribes and describes audio.")
            .build();

        let response = agent
            .prompt(Message::User {
                content: OneOrMany::many(vec![
                    UserContent::text("What is said in this audio clip? Transcribe it briefly."),
                    audio_content(),
                ])
                .expect("content should be non-empty"),
            })
            .await
            .expect("audio prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
