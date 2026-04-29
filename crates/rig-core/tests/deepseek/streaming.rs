//! DeepSeek streaming smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::deepseek::{self, DEEPSEEK_V4_FLASH};
use rig_core::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn streaming_prompt_smoke() {
    let client = deepseek::Client::from_env().expect("client should build");
    let agent = client
        .agent(DEEPSEEK_V4_FLASH)
        .preamble("You are a helpful assistant.")
        .build();

    let mut stream = agent.stream_prompt("Tell me a joke").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
