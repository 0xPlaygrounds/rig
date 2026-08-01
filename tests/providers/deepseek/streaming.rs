//! DeepSeek streaming smoke test.

use rig::prelude::*;
use rig::providers::deepseek::DEEPSEEK_V4_FLASH;

use super::support::with_deepseek_cassette;
use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
async fn streaming_prompt_smoke() {
    with_deepseek_cassette("streaming/streaming_prompt_smoke", |env| async move {
        let agent = env
            .agent(DEEPSEEK_V4_FLASH)
            .preamble("You are a helpful assistant.")
            .build();

        let mut stream = agent.runner("Tell me a joke").stream_run();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
