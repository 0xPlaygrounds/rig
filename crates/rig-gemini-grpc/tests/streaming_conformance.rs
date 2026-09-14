//! Wire-conformance suite for the Gemini gRPC typed-event wire.
//!
//! Events-first (`WireInput::Event`): fixture frames are already-typed
//! protobuf responses driven through
//! [`rig_gemini_grpc::streaming::stream_from_events`] — the shared driver,
//! canonical grammar, and terminal normalization — with no gRPC transport.
//! Frame-level malformed/unknown scenarios self-report as skipped: prost
//! surfaces decode failures as transport `Status` errors, and its
//! unknown-variant signal is the sub-frame `part.data` oneof decoding to
//! `None`.

use rig_core::completion::{CompletionError, FinishReason};
use rig_core::test_utils::streaming_conformance::{
    InterleavedReasoningFixture, ProviderWireFixture, WireDriver, WireInput, event_frame,
    fixtures::drain,
};
use rig_gemini_grpc::proto;

fn driver() -> WireDriver {
    WireDriver::new("gemini-grpc", |chunks| {
        Box::pin(async move {
            let events: Vec<Result<proto::GenerateContentResponse, CompletionError>> = chunks
                .into_iter()
                .map(|chunk| match chunk {
                    Ok(frame) => frame
                        .downcast_event::<proto::GenerateContentResponse>()
                        .cloned()
                        .ok_or_else(|| {
                            CompletionError::ProviderError(
                                "gemini-grpc conformance frames must be protobuf responses"
                                    .to_string(),
                            )
                        }),
                    Err(error) => Err(CompletionError::HttpError(error)),
                })
                .collect();
            let stream =
                rig_gemini_grpc::streaming::stream_from_events(futures::stream::iter(events));
            Ok(drain(stream).await)
        })
    })
}

fn response(parts: Vec<proto::Part>, finish_reason: i32) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts,
                role: "model".to_string(),
            }),
            finish_reason,
            index: None,
            finish_message: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: "gemini-2.5-pro".to_string(),
        response_id: "resp-1".to_string(),
    }
}

fn text_part(text: &str) -> proto::Part {
    proto::Part {
        thought: false,
        thought_signature: Vec::new(),
        part_metadata: None,
        data: Some(proto::part::Data::Text(text.to_string())),
    }
}

fn function_call_part(name: &str) -> proto::Part {
    let args = proto::Struct {
        fields: [(
            "city".to_string(),
            proto::Value {
                kind: Some(proto::value::Kind::StringValue("Tokyo".to_string())),
            },
        )]
        .into_iter()
        .collect(),
    };
    proto::Part {
        thought: false,
        thought_signature: Vec::new(),
        part_metadata: None,
        data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
            name: name.to_string(),
            args: Some(args),
            id: "call-1".to_string(),
        })),
    }
}

fn thought_part(text: &str) -> proto::Part {
    proto::Part {
        thought: true,
        thought_signature: Vec::new(),
        part_metadata: None,
        data: Some(proto::part::Data::Text(text.to_string())),
    }
}

/// FinishReason::Stop in the proto enumeration.
const FINISH_STOP: i32 = 1;

fn terminal(usage: Option<proto::UsageMetadata>) -> WireInput {
    let mut response = response(Vec::new(), FINISH_STOP);
    response.usage_metadata = usage;
    event_frame(response)
}

fn usage(prompt: i32, candidates: i32, total: i32) -> proto::UsageMetadata {
    proto::UsageMetadata {
        prompt_token_count: prompt,
        candidates_token_count: candidates,
        total_token_count: total,
        cached_content_token_count: 0,
    }
}

fn fixture() -> ProviderWireFixture {
    ProviderWireFixture {
        driver: driver(),
        text_frames: vec![event_frame(response(vec![text_part("hi")], 0))],
        expected_texts: vec!["hi"],
        tool_call_frames: vec![event_frame(response(
            vec![function_call_part("get_weather")],
            0,
        ))],
        expected_tool_name: "get_weather",
        // The gRPC wire delivers function calls whole; arguments never stream.
        partial_tool_call_frames: None,
        terminal_frames: vec![terminal(Some(usage(5, 2, 7)))],
        expected_usage_total: 7,
        expected_finish_reason: Some(FinishReason::Stop),
        zero_usage_terminal_frames: Some(vec![terminal(None)]),
        bare_terminal_frames: None,
        // prost owns wire decoding: a corrupt frame surfaces as a transport
        // `Status` error, so no frame-level malformed input can be spelled.
        malformed_frame: None,
        // The unknown-variant signal is a sub-frame oneof (`part.data` =
        // `None`), warn-skipped by the adapter; the classify triage is pinned
        // by `classify_typed_event`'s unit tests in `wire.rs`.
        unknown_event_frame: None,
        defective_known_frame: None,
        delta_less_prelude_frame: None,
        refusal: None,
        // Thought → interleaved function call → thought: the constant-id
        // boundary the adapter owns by synthesizing reasoning ends.
        interleaved_reasoning: Some(InterleavedReasoningFixture {
            frames: vec![
                event_frame(response(vec![thought_part("before tool")], 0)),
                event_frame(response(vec![function_call_part("get_weather")], 0)),
                event_frame(response(vec![thought_part("after tool")], 0)),
                terminal(Some(usage(5, 2, 7))),
            ],
            first_reasoning: "before tool",
            tool_name: "get_weather",
            second_reasoning: "after tool",
        }),
    }
}

rig_core::streaming_conformance_suite! {
    provider: "gemini_grpc",
    fixture: fixture(),
    manifest: [zero_usage_terminal, interleaved_reasoning],
}

/// Compile-linked manifest of the wire families this binary covers.
///
/// This suite lives outside the `rig` facade's `core` test binary, so the
/// workspace registry cannot link it: it lists `gemini_grpc` in
/// `OUT_OF_BINARY_FAMILIES` and relies on the "Test out-of-facade streaming
/// conformance and structural guards" CI step to execute this binary. The test
/// below keeps the family name honest at the definition site, which is the
/// direction the registry loses for out-of-binary suites (#2258 F3).
const SUITE_FAMILIES: &[&str] = &[WIRE_FAMILY];

#[test]
fn suite_families_are_registered_wire_families() {
    for family in SUITE_FAMILIES {
        assert!(
            rig_core::test_utils::streaming_conformance::WIRE_FAMILIES.contains(family),
            "suite names wire family {family:?}, absent from WIRE_FAMILIES"
        );
    }
}
