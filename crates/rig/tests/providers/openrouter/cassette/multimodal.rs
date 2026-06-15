//! Cassette-backed OpenRouter multimodal prompts.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType, Message, UserContent,
};

use crate::support::{IMAGE_FIXTURE_PATH, PDF_FIXTURE_PATH, assert_nonempty_response};

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
