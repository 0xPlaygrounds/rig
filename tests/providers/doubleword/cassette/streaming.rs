//! Cassette-backed Doubleword streaming coverage.

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;

#[tokio::test]
async fn streaming_smoke() {
    with_doubleword_cassette("streaming/streaming_smoke", |env| async move {
        let agent = AgentBuilder::new(env.provider(DEFAULT_MODEL))
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
