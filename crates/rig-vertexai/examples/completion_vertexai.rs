use anyhow::Context;
use rig_core::Model;
use rig_vertexai::{
    VertexAi,
    completion::{GEMINI_2_5_FLASH_LITE, GenerateContent},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Uses ADC credentials and expects GOOGLE_CLOUD_PROJECT to be set. See
    // `rig_vertexai::VertexAiBuilder` for more granular control.
    let model = Model::new(
        GenerateContent::new(GEMINI_2_5_FLASH_LITE),
        VertexAi::from_env()?,
    );

    let request = model
        .completion_request("What is the capital of France?")
        .max_tokens(1024)
        .build();

    let response = model
        .call(request, None)
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
