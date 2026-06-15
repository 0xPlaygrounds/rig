//! DeepSeek streaming smoke test.

use rig::client::CompletionClient;
use rig::providers::deepseek::DEEPSEEK_V4_FLASH;
use rig::streaming::StreamingPrompt;

use super::support::with_deepseek_cassette;
use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
async fn streaming_prompt_smoke() {
    with_deepseek_cassette("streaming/streaming_prompt_smoke", |client| async move {
        let agent = client
            .agent(DEEPSEEK_V4_FLASH)
            .preamble("You are a helpful assistant.")
            .build();

        let mut stream = agent.stream_prompt("Tell me a joke").await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
