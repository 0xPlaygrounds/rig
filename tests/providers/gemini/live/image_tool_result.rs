use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::gemini,
};
use rig_agent::test_utils::MockImageGeneratorTool;

/// Verifies that Gemini can process an image returned by a classic tool call.
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY environment variable"]
async fn test_gemini_agent_with_image_tool_result_e2e() -> anyhow::Result<()> {
    let client = gemini::Client::from_env()?;

    let agent = client
        .agent("gemini-3-flash-preview")
        .preamble(
            "You are a helpful assistant. When asked about images, use the \
             generate_test_image tool to create one, then describe what you see in the image.",
        )
        .tool(MockImageGeneratorTool)
        .build();

    let response_text = agent
        .prompt("Please generate a test image and tell me what color the pixel is.")
        .await?;
    println!("Response: {response_text}");
    anyhow::ensure!(!response_text.is_empty(), "response should not be empty");
    Ok(())
}
