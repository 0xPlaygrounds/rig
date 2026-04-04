//! Migrated from `examples/deepseek_streaming.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Message;
use rig::providers::deepseek::{self, DEEPSEEK_CHAT};
use rig::streaming::{StreamingChat, StreamingPrompt};

use crate::support::{
    Adder, Subtract, assert_mentions_expected_number, assert_nonempty_response,
    collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn streaming_prompt_smoke() {
    let client = deepseek::Client::from_env();
    let agent = client
        .agent(DEEPSEEK_CHAT)
        .preamble("You are a helpful assistant.")
        .build();

    let mut stream = agent.stream_prompt("Tell me a joke").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn streaming_chat_with_tools() {
    let client = deepseek::Client::from_env();
    let agent = client
        .agent(DEEPSEEK_CHAT)
        .preamble("You are a calculator here to help the user perform arithmetic operations.")
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let history: &[Message] = &[];
    let mut stream = agent.stream_chat("Calculate 2 - 5", history).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming chat should succeed");

    assert_mentions_expected_number(&response, -3);
}
