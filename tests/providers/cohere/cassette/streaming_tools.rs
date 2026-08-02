//! Cassette-backed Cohere streaming tool-call coverage.

use rig::prelude::*;
use rig::streaming::StreamingPrompt;

use super::super::{
    CASSETTE_MODEL,
    support::{IntegerAdder, IntegerSubtract, with_cohere_cassette},
};
use crate::support::{
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, assert_mentions_expected_number,
    collect_stream_observation,
};

#[tokio::test]
async fn streaming_tool_call_roundtrip() {
    with_cohere_cassette(
        "streaming_tools/streaming_tool_call_roundtrip",
        |client| async move {
            let agent = client
                .agent(CASSETTE_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(IntegerAdder)
                .tool(IntegerSubtract)
                .default_max_turns(2)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
            let observation = collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "streaming tool prompt should not error: {:?}",
                observation.errors
            );
            assert_eq!(observation.tool_calls, vec!["subtract".to_string()]);
            assert_eq!(observation.tool_results, 1);

            let response = observation
                .final_response_text
                .expect("stream should yield a final response");
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}
