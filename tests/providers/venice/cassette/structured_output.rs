//! Cassette-backed Venice structured output coverage.

use rig::completion::TypedPrompt;
use rig::prelude::*;

// Recorded against `TOOL_MODEL`: Venice's `qwen3-5-9b` capacity for
// `response_format: json_schema` requests answered 429 ("model is currently
// overloaded") while its plain completions path stayed healthy.
use super::super::{TOOL_MODEL, support::with_venice_cassette};
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn structured_output_smoke() {
    with_venice_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let response: SmokeStructuredOutput = client
                .agent(TOOL_MODEL)
                .build()
                .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            assert_smoke_structured_output(&response);
        },
    )
    .await;
}
