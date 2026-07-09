//! AWS Bedrock raw streaming cassette coverage ported from OpenAI completions tests.

use rig::bedrock;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::ToolChoice;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
    RAW_TEXT_RESPONSE_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, assert_stream_contains_zero_arg_tool_call_named,
    collect_raw_stream_observation, zero_arg_tool_definition,
};

#[tokio::test]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    with_bedrock_cassette(
        "raw_streaming/raw_stream_emits_required_zero_arg_tool_call",
        |client| async move {
            let model = client.completion_model(bedrock::completion::AMAZON_NOVA_LITE);
            let request = model
                .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .build();
            let stream = model.stream(request).await.expect("stream should start");

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
            let model = client.completion_model(bedrock::completion::AMAZON_NOVA_LITE);
            let request = model
                .completion_request(RAW_TEXT_RESPONSE_PROMPT)
                .preamble("Reply with exactly the requested text.".to_string())
                .temperature(0.0)
                .build();

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
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

#[tokio::test]
async fn raw_stream_emits_tool_call_before_text() {
    with_bedrock_cassette(
        "raw_streaming/raw_stream_emits_tool_call_before_text",
        |client| async move {
            let model = client.completion_model(bedrock::completion::AMAZON_NOVA_LITE);
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["lookup_harbor_label".to_string()],
                })
                .build();

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("raw Bedrock stream should start"),
            )
            .await;

            assert_raw_stream_tool_call_precedes_text(&observation, "lookup_harbor_label");
        },
    )
    .await;
}
