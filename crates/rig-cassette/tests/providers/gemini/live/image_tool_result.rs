use rig_agent::test_utils::MockImageGeneratorTool;
use rig_test_support::cassette_models::GeminiModels;

/// Verifies that Gemini can process an image returned by a classic tool call.
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY environment variable"]
async fn test_gemini_agent_with_image_tool_result_e2e() -> anyhow::Result<()> {
    let client = GeminiModels::from_env()?;

    let agent = rig::AgentBuilder::new(client.completion("gemini-3-flash-preview"))
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
    anyhow::ensure!(
        !response_text.output().is_empty(),
        "response should not be empty"
    );
    Ok(())
}
