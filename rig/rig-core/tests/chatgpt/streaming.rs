//! ChatGPT streaming smoke tests.

use rig::client::CompletionClient;
use rig::providers::chatgpt;
use rig::streaming::StreamingPrompt;

use crate::chatgpt::live_client;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming_smoke() {
    let agent = live_client()
        .agent(chatgpt::GPT_5_3_CHAT_LATEST)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("ChatGPT stream should succeed");

    assert_nonempty_response(&response);
}
