//! Cassette-backed Doubleword structured output coverage.

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};
use rig::wire::Wire as _;

#[tokio::test]
async fn structured_output_smoke() {
    with_doubleword_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let response: SmokeStructuredOutput =
                rig::AgentBuilder::new(client.completion(DEFAULT_MODEL).on(rig::transport()))
                    .build()
                    .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
                    .await
                    .expect("structured output prompt should succeed")
                    .output;
            assert_smoke_structured_output(&response);
        },
    )
    .await;
}
