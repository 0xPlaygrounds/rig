//! Anthropic structured output smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn structured_output_smoke() {
    let (cassette, client) =
        super::super::support::anthropic_cassette("structured_output/structured_output_smoke")
            .await;
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

    cassette.finish().await;
}
