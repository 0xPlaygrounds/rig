//! Groq streaming tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::providers::groq;

use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    zero_arg_tool_definition,
};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = groq::Client::from_env();
    let model = client.completion_model(groq::LLAMA_3_1_8B_INSTANT);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}
