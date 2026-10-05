//! Dedicated Claude Opus 4.7 live smoke tests.

use rig::providers::anthropic::completion::CLAUDE_OPUS_4_7;

use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn messages_structured_output_smoke() {
    super::super::support::with_anthropic_cassette(
        "opus_4_7/messages_structured_output_smoke",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_OPUS_4_7))
                .output_schema::<SmokeStructuredOutput>()
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed")
                .output();
            let structured: SmokeStructuredOutput =
                serde_json::from_str(&response).expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}
