//! Perplexity streaming completion cassette coverage.

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::perplexity;

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn streaming_smoke() {
    with_perplexity_cassette("streaming/streaming_smoke", |env| async move {
        let agent = env
            .agent(perplexity::SONAR)
            .preamble(STREAMING_PREAMBLE)
            .max_tokens(16)
            .build();

        let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
