//! DeepSeek streaming smoke test.

use rig::providers::deepseek::DEEPSEEK_V4_FLASH;
use rig::wire::Wire as _;

use super::support::with_deepseek_cassette;
use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
async fn streaming_prompt_smoke() {
    with_deepseek_cassette("streaming/streaming_prompt_smoke", |client| async move {
        let agent =
            rig::AgentBuilder::new(client.completion(DEEPSEEK_V4_FLASH).on(rig::transport()))
                .preamble("You are a helpful assistant.")
                .build();

        let mut stream = agent.prompt("Tell me a joke").stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
