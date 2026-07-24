//! Cassette-backed Doubleword streaming coverage.

use rig::prelude::*;
use rig::streaming::StreamingPrompt;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_smoke() {
    with_doubleword_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(DEFAULT_MODEL)
            .preamble(STREAMING_PREAMBLE)
            .build();
        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");
        assert_nonempty_response(&response);
    })
    .await;
}
