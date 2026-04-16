//! Anthropic structured output smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};

use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn structured_output_smoke() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(CLAUDE_SONNET_4_6)
        .output_schema::<SmokeStructuredOutput>()
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("structured output prompt should succeed");
    let structured: SmokeStructuredOutput =
        serde_json::from_str(&response).expect("structured output should deserialize");

    assert_smoke_structured_output(&structured);
}
