use anyhow::Context;
use rig_core::completion::CompletionRequest;
use rig_core::message::AssistantContent;
use rig_vertexai::VertexAi;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Uses ADC credentials and expects GOOGLE_CLOUD_PROJECT to be set, and
    // GOOGLE_CLOUD_LOCATION when not "global".
    let model = VertexAi::from_env()?.anthropic("claude-sonnet-4-6");

    let request = CompletionRequest::new("What is the capital of France?").max_tokens(1024);
    let response = model
        .call(request)
        .await
        .context("Failed to get completion")?;

    for content in response.choice.iter() {
        if let AssistantContent::Text(text) = content {
            println!("{}", text.text);
        }
    }
    Ok(())
}
