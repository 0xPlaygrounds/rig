use anyhow::Context;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig_vertexai::{Client, GEMINI_2_5_FLASH_LITE};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Uses ADC credentials and expects GOOGLE_CLOUD_PROJECT to be set. See ClientBuilder for more granular control.
    let client = Client::from_env();
    let model = client.completion_model(GEMINI_2_5_FLASH_LITE);

    let request = model
        .completion_request("What is the capital of France?")
        .preamble("You always end a response with exactly three smiley faces".to_string())
        .build();

    let response = model
        .completion(request)
        .await
        .context("Failed to get completion")?;

    let mut response_text = String::new();
    for content in response.choice.iter() {
        if let rig::message::AssistantContent::Text(rig::message::Text { text }) = content {
            response_text.push_str(text);
        }
    }

    println!("Response: {}", response_text);

    Ok(())
}
