//! Perplexity streaming completion cassette coverage.

use rig::providers::perplexity;
use rig::wire::Wire as _;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn streaming_smoke() {
    with_perplexity_cassette("streaming/streaming_smoke", |client| async move {
        let agent =
            rig::AgentBuilder::new(client.completion(perplexity::SONAR).on(rig::transport()))
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(16)
                .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
