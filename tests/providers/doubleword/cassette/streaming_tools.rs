//! Cassette-backed Doubleword streaming tool coverage.

use super::super::{TOOL_MODEL, support::with_doubleword_cassette};
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};
use rig::prelude::*;

#[tokio::test]
async fn streaming_tools_smoke() {
    with_doubleword_cassette("streaming_tools/streaming_tools_smoke", |env| async move {
        let agent = AgentBuilder::new(env.provider(TOOL_MODEL))
            .preamble(STREAMING_TOOLS_PREAMBLE)
            .tool(Adder)
            .tool(Subtract)
            .default_max_turns(2)
            .build();
        let mut stream = Box::pin(agent.runner(STREAMING_TOOLS_PROMPT).stream_run());
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming tool prompt should succeed");
        assert_mentions_expected_number(&response, -3);
    })
    .await;
}
