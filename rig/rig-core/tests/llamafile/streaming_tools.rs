//! Llamafile streaming tools smoke test.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::providers::llamafile;

use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    zero_arg_tool_definition,
};

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = llamafile::Client::from_url("http://localhost:8080");
    let model = client.completion_model(llamafile::LLAMA_CPP);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}
