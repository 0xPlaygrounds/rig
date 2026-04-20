//! xAI streaming tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::providers::xai;
use rig::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    assert_tool_call_precedes_later_text, collect_stream_observation, zero_arg_tool_definition,
};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = xai::Client::from_env();
    let model = client.completion_model(xai::completion::GROK_4);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn responses_stream_preserves_tool_result_flow() {
    let client = xai::Client::from_env();
    let agent = client
        .agent(xai::completion::GROK_4)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
        .multi_turn(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(
        &observation,
        "lookup_harbor_label",
        &[ALPHA_SIGNAL_OUTPUT],
    );
}
