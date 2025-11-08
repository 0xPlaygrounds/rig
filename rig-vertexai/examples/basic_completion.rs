use anyhow::Context;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig_vertexai::Client;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Vertex AI client using implicit credentials
    let client = Client::from_env();

    // Create a completion model (using gemini-2.5-flash-lite as an example)
    let model = client.completion_model("gemini-2.5-flash-lite");

    // Build a completion request with a preamble (system instruction)
    let request = model
        .completion_request("What is the capital of France?")
        .preamble("You always end a response with exactly three smiley faces".to_string())
        .build();

    // Get the completion response
    let response = model
        .completion(request)
        .await
        .context("Failed to get completion")?;

    // Extract text from the response
    let mut response_text = String::new();
    for content in response.choice.iter() {
        if let rig::message::AssistantContent::Text(rig::message::Text { text }) = content {
            response_text.push_str(text);
        }
    }

    // Print the response
    println!("Response: {}", response_text);

    Ok(())
}
