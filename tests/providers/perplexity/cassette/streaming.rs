//! Perplexity streaming completion cassette coverage.

use rig::client::CompletionClient;
use rig::providers::perplexity;
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn streaming_smoke() {
    with_perplexity_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(perplexity::SONAR)
            .preamble(STREAMING_PREAMBLE)
            .max_tokens(16)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
