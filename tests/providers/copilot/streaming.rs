//! Copilot streaming coverage, including the migrated example path.

use rig::prelude::*;

use crate::copilot::{LIVE_MODEL, with_copilot_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_smoke() {
    with_copilot_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(LIVE_MODEL)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn example_streaming_prompt() {
    with_copilot_cassette("streaming/example_streaming_prompt", |client| async move {
        let agent = client
            .agent(LIVE_MODEL)
            .preamble("Be precise and concise.")
            .temperature(0.5)
            .build();

        let mut stream = Box::pin(
            agent
                .runner("When and where and what type is the next solar eclipse?")
                .stream_run(),
        );
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
