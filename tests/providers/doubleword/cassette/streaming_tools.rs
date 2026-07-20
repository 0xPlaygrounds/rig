//! Cassette-backed Doubleword streaming tool coverage.

use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

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
                .agent(TOOL_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();
            let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}
