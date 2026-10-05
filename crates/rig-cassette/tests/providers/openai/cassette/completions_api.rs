//! Migrated from `examples/openai_agent_completions_api.rs`.
use rig::message::ToolChoice;
use rig::providers::openai;

use super::super::support::with_openai_completions_cassette;
use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    zero_arg_tool_definition,
};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn completions_api_raw_stream_emits_required_zero_arg_tool_call() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_stream_emits_required_zero_arg_tool_call",
        |client| async move {
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required);
            let stream = model.stream(request).expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
        },
    )
    .await;
}
