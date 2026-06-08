//! xAI streaming smoke test.

use rig::client::CompletionClient;
use rig::providers::xai;
use rig::streaming::StreamingPrompt;

use super::support::with_xai_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_smoke() {
    with_xai_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(xai::completion::GROK_3_MINI)
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
