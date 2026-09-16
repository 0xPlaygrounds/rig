use anyhow::Context;
use rig_core::completion::CompletionModel;
use rig_core::driver::CompletionProvider;
use rig_vertexai::{Client, completion::GEMINI_2_5_FLASH_LITE};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Uses ADC credentials and expects GOOGLE_CLOUD_PROJECT to be set. See
    // `rig_vertexai::ClientBuilder` for more granular control.
    let client = Client::from_env()?;
    let model = client.completion(GEMINI_2_5_FLASH_LITE);

    let request = model
        .completion_request("What is the capital of France?")
        .max_tokens(1024)
        .build();

    let response = model
        .completion(request)
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

    println!("Response: {response_text}");

    Ok(())
}
