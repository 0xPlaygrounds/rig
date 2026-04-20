//! DeepSeek streaming tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Message};
use rig::message::ToolChoice;
use rig::providers::deepseek::{self, DEEPSEEK_CHAT};
use rig::streaming::StreamingChat;

use crate::support::{
    Adder, REQUIRED_ZERO_ARG_TOOL_PROMPT, Subtract, assert_mentions_expected_number,
    assert_stream_contains_zero_arg_tool_call_named, collect_stream_final_response,
    zero_arg_tool_definition,
};

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

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = deepseek::Client::from_env();
    let model = client.completion_model(DEEPSEEK_CHAT);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}
