//! Together streaming smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::together;
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn streaming_smoke() {
    let client = together::Client::from_env();
    let agent = client
        .agent(together::LLAMA_3_8B_CHAT_HF)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
