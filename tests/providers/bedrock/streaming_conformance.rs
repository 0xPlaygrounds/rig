//! Wire-conformance suite for the Bedrock Converse typed-event wire.
//!
//! Events-first (`WireInput::Event`): fixture frames are already-typed SDK
//! events driven through `rig::bedrock::streaming::stream_from_events` — the
//! shared driver, canonical grammar, and terminal normalization — with no AWS
//! transport. Frame-level malformed/unknown scenarios self-report as skipped:
//! the SDK surfaces decode failures as transport errors, and its
//! non-exhaustive `Unknown` union variant is not constructible from outside
//! the SDK.

use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::completion::{CompletionError, FinishReason};
use rig_core::test_utils::streaming_conformance::{
    self as conformance, ProviderWireFixture, WireDriver, WireInput, event_frame, fixtures::drain,
};

fn driver() -> WireDriver {
    WireDriver::new("aws_bedrock", |chunks| {
        Box::pin(async move {
            let events: Vec<Result<aws_bedrock::ConverseStreamOutput, CompletionError>> = chunks
                .into_iter()
                .map(|chunk| match chunk {
                    Ok(frame) => frame
                        .downcast_event::<aws_bedrock::ConverseStreamOutput>()
                        .cloned()
                        .ok_or_else(|| {
                            CompletionError::ProviderError(
                                "bedrock conformance frames must be Converse events".to_string(),
                            )
                        }),
                    Err(error) => Err(CompletionError::HttpError(error)),
                })
                .collect();
            let stream = rig::bedrock::streaming::stream_from_events(futures::stream::iter(events));
            Ok(drain(stream).await)
        })
    })
}

fn text_delta(index: i32, text: &str) -> WireInput {
    event_frame(aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::Text(text.to_string()))
            .build()
            .expect("content block delta should build"),
    ))
}

fn tool_start(index: i32, id: &str, name: &str) -> WireInput {
    event_frame(aws_bedrock::ConverseStreamOutput::ContentBlockStart(
        aws_bedrock::ContentBlockStartEvent::builder()
            .content_block_index(index)
            .start(aws_bedrock::ContentBlockStart::ToolUse(
                aws_bedrock::ToolUseBlockStart::builder()
                    .tool_use_id(id)
                    .name(name)
                    .build()
                    .expect("tool use start should build"),
            ))
            .build()
            .expect("content block start should build"),
    ))
}

fn tool_delta(index: i32, input: &str) -> WireInput {
    event_frame(aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::ToolUse(
                aws_bedrock::ToolUseBlockDelta::builder()
                    .input(input)
                    .build()
                    .expect("tool use delta should build"),
            ))
            .build()
            .expect("content block delta should build"),
    ))
}

fn block_stop(index: i32) -> WireInput {
    event_frame(aws_bedrock::ConverseStreamOutput::ContentBlockStop(
        aws_bedrock::ContentBlockStopEvent::builder()
            .content_block_index(index)
            .build()
            .expect("content block stop should build"),
    ))
}

fn message_stop(reason: aws_bedrock::StopReason) -> WireInput {
    event_frame(aws_bedrock::ConverseStreamOutput::MessageStop(
        aws_bedrock::MessageStopEvent::builder()
            .stop_reason(reason)
            .build()
            .expect("message stop should build"),
    ))
}

fn metadata(usage: Option<aws_bedrock::TokenUsage>) -> WireInput {
    let mut builder = aws_bedrock::ConverseStreamMetadataEvent::builder();
    if let Some(usage) = usage {
        builder = builder.usage(usage);
    }
    event_frame(aws_bedrock::ConverseStreamOutput::Metadata(builder.build()))
}

fn usage(input: i32, output: i32, total: i32) -> aws_bedrock::TokenUsage {
    aws_bedrock::TokenUsage::builder()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(total)
        .build()
        .expect("token usage should build")
}

fn fixture() -> ProviderWireFixture {
    ProviderWireFixture {
        driver: driver(),
        text_frames: vec![text_delta(0, "hi")],
        expected_texts: vec!["hi"],
        // The block stop completes the call; the stream terminal (the
        // Metadata event) is deliberately absent.
        tool_call_frames: vec![
            tool_start(0, "call_1", "get_weather"),
            tool_delta(0, "{\"city\":\"Tokyo\"}"),
            block_stop(0),
        ],
        expected_tool_name: "get_weather",
        partial_tool_call_frames: Some(vec![
            tool_start(0, "call_1", "get_weather"),
            tool_delta(0, "{\"cit"),
        ]),
        terminal_frames: vec![
            message_stop(aws_bedrock::StopReason::EndTurn),
            metadata(Some(usage(10, 5, 15))),
        ],
        expected_usage_total: 15,
        expected_finish_reason: Some(FinishReason::Stop),
        zero_usage_terminal_frames: Some(vec![
            message_stop(aws_bedrock::StopReason::EndTurn),
            metadata(None),
        ]),
        bare_terminal_frames: None,
        // The AWS SDK owns event-stream decoding: a corrupt frame surfaces as
        // a receive (transport) error, so no frame-level malformed input can
        // be spelled — the scenario reports a visible skip.
        malformed_frame: None,
        // The SDK's `Unknown` union variant is non-exhaustive and cannot be
        // constructed outside the SDK; the classify triage for it is pinned
        // by `classify_typed_event`'s unit tests in `wire.rs`.
        unknown_event_frame: None,
        defective_known_frame: None,
        delta_less_prelude_frame: None,
        refusal: None,
    }
}

macro_rules! conformance_test {
    ($name:ident, $scenario:path) => {
        #[tokio::test]
        async fn $name() {
            let fixture = fixture();
            $scenario(&fixture).await.expect("scenario should hold");
        }
    };
}

conformance_test!(
    truncation_preserves_content_without_terminal,
    conformance::truncation_preserves_content_without_terminal
);
conformance_test!(
    transport_error_after_tool_call_yields_err_then_end,
    conformance::transport_error_after_tool_call_yields_err_then_end
);
conformance_test!(
    malformed_frame_surfaces_err_and_terminal_still_completes,
    conformance::malformed_frame_surfaces_err_and_terminal_still_completes
);
conformance_test!(
    unknown_event_is_skipped,
    conformance::unknown_event_is_skipped
);
conformance_test!(
    defective_known_event_surfaces_err,
    conformance::defective_known_event_surfaces_err
);
conformance_test!(
    delta_less_choice_prelude_is_a_noop,
    conformance::delta_less_choice_prelude_is_a_noop
);
conformance_test!(
    refusal_frames_deliver_text_without_error,
    conformance::refusal_frames_deliver_text_without_error
);
conformance_test!(
    bare_terminal_after_only_unparseable_frames_fabricates_nothing,
    conformance::bare_terminal_after_only_unparseable_frames_fabricates_nothing
);
conformance_test!(
    usage_variants_are_reported_or_zero_sentinel,
    conformance::usage_variants_are_reported_or_zero_sentinel
);
