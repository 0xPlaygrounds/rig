//! Gemini structured output smoke test.

use rig::client::AgentClientExt;
use rig::completion::Prompt;
use rig::test_utils::decode_structured_output;

use super::super::support::with_gemini_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn structured_output_smoke() {
    with_gemini_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client
                .agent("gemini-3-flash-preview")
                .output_schema::<SmokeStructuredOutput>()
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput =
                decode_structured_output("gemini_structured_output_smoke", &response)
                    .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}
