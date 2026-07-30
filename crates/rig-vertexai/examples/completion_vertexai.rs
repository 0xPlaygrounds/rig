//! A single Vertex AI completion through the crate's free-function face.
//!
//! `functions::Config` is plain data (project / location / credential
//! source); `functions::client_from_config` resolves ADC credentials into the
//! live `Client` handle, and `functions::complete` takes that handle plus a
//! model identifier.

use anyhow::Context;
use rig_vertexai::{completion::GEMINI_2_5_FLASH_LITE, functions};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Uses ADC credentials and expects GOOGLE_CLOUD_PROJECT to be set. See
    // `Config::with_project` / `with_location` / `with_impersonated_service_account`
    // for more granular control.
    let config = functions::Config::new(GEMINI_2_5_FLASH_LITE);
    let client = functions::client_from_config(&config)?;

    let request = rig_core::completion::CompletionRequest {
        max_tokens: Some(1024),
        ..rig_core::completion::CompletionRequest::from_prompt("What is the capital of France?")
    };

    let response = functions::complete(&client, &config.model, request)
        .await
        .context("Failed to get completion")?;

    let mut response_text = String::new();
    for content in response.choice.iter() {
        if let rig_core::message::AssistantContent::Text(rig_core::message::Text { text, .. }) =
            content
        {
            response_text.push_str(text);
        }
    }

    println!("Response: {}", response_text);

    Ok(())
}
