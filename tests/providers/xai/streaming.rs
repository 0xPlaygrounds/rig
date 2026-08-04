//! xAI streaming smoke test.

use super::support::with_xai_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::xai;

#[tokio::test]
async fn streaming_smoke() {
    with_xai_cassette("streaming/streaming_smoke", |env| async move {
        let agent = env
            .agent(xai::completion::GROK_3_MINI)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
