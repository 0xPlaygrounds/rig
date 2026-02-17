//! Example demonstrating OpenRouter's multimodal support for images and PDFs.
//!
//! This example shows how to send images and PDF documents to models via OpenRouter's API.
//!
//! OpenRouter supports:
//! - Images via URL or base64 data URI
//! - PDF files via URL or base64 data URI
//!
//! To run this example, set your OpenRouter API key:
//! ```bash
//! export OPENROUTER_API_KEY=your_api_key
//! cargo run --example openrouter_multimodal
//! ```

use rig::OneOrMany;
use rig::completion::Prompt;
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType, Message, UserContent,
};
use rig::prelude::*;
use rig::providers::openrouter;

/// Model that supports vision (images)
const VISION_MODEL: &str = "google/gemini-2.5-flash";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Create OpenRouter client
    let client = openrouter::Client::from_env();

    // Example 1: Analyze an image from URL
    println!("=== Example 1: Image Analysis (URL) ===\n");
    analyze_image_url(&client).await?;

    // Example 2: Send a PDF document from URL
    println!("\n=== Example 2: PDF Analysis (URL) ===\n");
    analyze_pdf_url(&client).await?;

    // Example 3: Mixed content - text with image
    println!("\n=== Example 3: Mixed Content ===\n");
    mixed_content(&client).await?;

    Ok(())
}

/// Example: Analyze an image from a URL
async fn analyze_image_url(client: &openrouter::Client) -> Result<(), anyhow::Error> {
    let agent = client
        .agent(VISION_MODEL)
        .preamble("You are a helpful assistant that describes images in detail.")
        .build();

    // Create an image from URL using Rig's standard message types
    let image_message = Message::User {
        content: OneOrMany::many(vec![
            UserContent::text("What do you see in this image? Describe it in detail."),
            UserContent::Image(Image {
                data: DocumentSourceKind::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string()
                ),
                media_type: Some(ImageMediaType::JPEG),
                detail: None,
                additional_params: None,
            }),
        ])?,
    };

    let response = agent.prompt(image_message).await?;
    println!("Response:\n{response}");

    Ok(())
}

/// Example: Analyze a PDF document from a URL
async fn analyze_pdf_url(client: &openrouter::Client) -> Result<(), anyhow::Error> {
    let agent = client
        .agent(VISION_MODEL)
        .preamble("You are a helpful assistant that summarizes documents.")
        .build();

    // Create a document from URL using Rig's standard message types
    // OpenRouter will automatically convert this to its file format
    let pdf_message = Message::User {
        content: OneOrMany::many(vec![
            UserContent::text("Please summarize the key points of this document."),
            UserContent::Document(Document {
                data: DocumentSourceKind::Url("https://bitcoin.org/bitcoin.pdf".to_string()),
                media_type: Some(DocumentMediaType::PDF),
                additional_params: None,
            }),
        ])?,
    };

    let response = agent.prompt(pdf_message).await?;
    println!("Response:\n{response}");

    Ok(())
}

/// Example: Mixed content with text and image
async fn mixed_content(client: &openrouter::Client) -> Result<(), anyhow::Error> {
    let agent = client
        .agent(VISION_MODEL)
        .preamble("You are a helpful assistant.")
        .build();

    // Multiple content items in a single message
    let message = Message::User {
        content: OneOrMany::many(vec![
            UserContent::text("I have two questions:"),
            UserContent::text("1. What colors do you see in this image?"),
            UserContent::Image(Image {
                data: DocumentSourceKind::Url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png".to_string()
                ),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                additional_params: None,
            }),
            UserContent::text("2. What is the main subject?"),
        ])?,
    };

    let response = agent.prompt(message).await?;
    println!("Response:\n{response}");

    Ok(())
}
