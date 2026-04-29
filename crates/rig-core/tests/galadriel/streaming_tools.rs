//! Galadriel streaming tools smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::CompletionModel;
use rig_core::message::ToolChoice;
use rig_core::providers::galadriel;

use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    zero_arg_tool_definition,
};

#[tokio::test]
#[ignore = "requires GALADRIEL_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = galadriel::Client::from_env().expect("galadriel client should build");
    let model = client.completion_model(galadriel::GPT_4O);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}
