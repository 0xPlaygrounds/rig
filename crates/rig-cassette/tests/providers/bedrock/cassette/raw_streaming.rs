//! AWS Bedrock raw streaming cassette coverage ported from OpenAI completions tests.

use rig::bedrock;
use rig::message::ToolChoice;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    RAW_TEXT_RESPONSE_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_raw_stream_text_contains,
    assert_stream_contains_zero_arg_tool_call_named, collect_raw_stream_observation,
    zero_arg_tool_definition,
};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    with_bedrock_cassette(
        "raw_streaming/raw_stream_emits_required_zero_arg_tool_call",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required);
            let stream = model.stream(request).expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", false).await;
        },
    )
    .await;
}

#[tokio::test]
async fn raw_stream_text_response_smoke() {
    with_bedrock_cassette(
        "raw_streaming/raw_stream_text_response_smoke",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request = CompletionRequest::new(RAW_TEXT_RESPONSE_PROMPT)
                .preamble("Reply with exactly the requested text.")
                .temperature(0.0);

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .expect("raw Bedrock stream should start"),
            )
            .await;

            assert!(
                observation.tool_calls.is_empty(),
                "plain raw stream should not emit tool calls: {:?}",
                observation.tool_calls
            );
            assert_raw_stream_text_contains(&observation, &["cedar", "maple"]);
        },
    )
    .await;
}
