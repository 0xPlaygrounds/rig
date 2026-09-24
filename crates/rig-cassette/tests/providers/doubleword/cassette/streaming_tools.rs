//! Cassette-backed Doubleword streaming tool coverage.

use rig::prelude::*;

use super::super::{TOOL_MODEL, support::with_doubleword_cassette};
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_tools_smoke() {
    with_doubleword_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .endpoint(|provider| provider.completion(TOOL_MODEL))
                .into_agent_builder()
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();
            let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}
