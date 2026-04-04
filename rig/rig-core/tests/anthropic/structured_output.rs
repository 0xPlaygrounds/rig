//! Anthropic structured output smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::anthropic;

use crate::support::{STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn structured_output_smoke() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent("claude-sonnet-4-5")
        .output_schema::<SmokeStructuredOutput>()
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("structured output prompt should succeed");
    let structured: SmokeStructuredOutput =
        serde_json::from_str(&response).expect("structured output should deserialize");

    assert_nonempty_response(&structured.title);
    assert_nonempty_response(&structured.category);
    assert_nonempty_response(&structured.summary);
}
