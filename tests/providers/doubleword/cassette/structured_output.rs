//! Cassette-backed Doubleword structured output coverage.

use rig::completion::TypedPrompt;
use rig::prelude::AgentClientExt;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn structured_output_smoke() {
    with_doubleword_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let response: SmokeStructuredOutput = client
                .agent(DEFAULT_MODEL)
                .build()
                .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            assert_smoke_structured_output(&response);
        },
    )
    .await;
}
