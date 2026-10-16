//! Wire-conformance suite for the xAI gRPC typed-event wire.
//!
//! Events-first (`WireInput::Event`): fixture frames are already-typed
//! protobuf chunks replayed by a scripted transport through the
//! `GetCompletion` wire, with no gRPC channel. Frame-level malformed and
//! unknown scenarios self-report as skipped: prost surfaces decode failures
//! as transport `Status` errors, and every chunk is a known message.

use rig_core::completion::{CompletionRequest, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::test_utils::streaming_conformance::{
    ProviderWireFixture, WireDriver, WireInput, event_frame, fixtures::drain,
};
use rig_xai_grpc::completion::{GetCompletion, XaiFrame};
use rig_xai_grpc::proto;

/// Replays scripted protobuf chunks as a streamed reply.
#[derive(Clone)]
struct Scripted(std::sync::Arc<std::sync::Mutex<Vec<Result<XaiFrame, ProviderError>>>>);

impl Transport<GetCompletion> for Scripted {
    fn send(
        &self,
        _request: proto::GetCompletionsRequest,
        _exchange: Exchange,
    ) -> Opening<XaiFrame> {
        let chunks = match self.0.lock() {
            Ok(mut chunks) => std::mem::take(&mut *chunks),
            Err(_) => {
                return Opening::failed(ProviderError::Provider("script lock poisoned".to_owned()));
            }
        };
        Opening::ready(Opened::new(futures::stream::iter(chunks)))
    }
}

fn driver() -> WireDriver {
    WireDriver::new("xai-grpc", |chunks| {
        Box::pin(async move {
            let events: Vec<Result<XaiFrame, ProviderError>> = chunks
                .into_iter()
                .map(|chunk| match chunk {
                    Ok(frame) => frame.downcast_event::<XaiFrame>().cloned().ok_or_else(|| {
                        ProviderError::Provider(
                            "xai-grpc conformance frames must be protobuf chunks".to_owned(),
                        )
                    }),
                    Err(error) => Err(ProviderError::Http(error.into())),
                })
                .collect();
            let stream = Model::new(
                GetCompletion::new("grok-4.3"),
                Scripted(std::sync::Arc::new(std::sync::Mutex::new(events))),
            )
            .stream(CompletionRequest::new("hello"))?;
            Ok(drain(stream).await)
        })
    })
}

fn chunk(delta: proto::Delta, finish: proto::FinishReason) -> proto::GetChatCompletionChunk {
    proto::GetChatCompletionChunk {
        id: "resp-1".to_owned(),
        model: "grok-4.3".to_owned(),
        outputs: vec![proto::CompletionOutputChunk {
            delta: Some(proto::Delta {
                role: proto::MessageRole::RoleAssistant as i32,
                ..delta
            }),
            finish_reason: finish as i32,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn frame(chunk: proto::GetChatCompletionChunk) -> WireInput {
    event_frame(XaiFrame::Chunk(chunk))
}

fn text(text: &str) -> WireInput {
    frame(chunk(
        proto::Delta {
            content: text.to_owned(),
            ..Default::default()
        },
        proto::FinishReason::ReasonInvalid,
    ))
}

/// A client-side call to `get_weather`: whole, or a fragment of one
/// streamed in pieces.
fn call(arguments: &str, fragment: bool) -> WireInput {
    let status = if fragment {
        proto::ToolCallStatus::InProgress
    } else {
        proto::ToolCallStatus::Completed
    };
    frame(chunk(
        proto::Delta {
            tool_calls: vec![proto::ToolCall {
                id: "call-1".to_owned(),
                r#type: proto::ToolCallType::ClientSideTool as i32,
                status: status as i32,
                index: fragment.then_some(0),
                tool: Some(proto::tool_call::Tool::Function(proto::FunctionCall {
                    name: "get_weather".to_owned(),
                    arguments: arguments.to_owned(),
                })),
                ..Default::default()
            }],
            ..Default::default()
        },
        proto::FinishReason::ReasonInvalid,
    ))
}

fn terminal(usage: Option<proto::SamplingUsage>) -> WireInput {
    let mut last = chunk(proto::Delta::default(), proto::FinishReason::ReasonStop);
    last.usage = usage;
    frame(last)
}

fn fixture() -> ProviderWireFixture {
    ProviderWireFixture {
        driver: driver(),
        text_frames: vec![text("hi")],
        expected_texts: vec!["hi"],
        tool_call_frames: vec![call(r#"{"city":"Tokyo"}"#, false)],
        expected_tool_name: "get_weather",
        partial_tool_call_frames: Some(vec![call(r#"{"city":"#, true)]),
        terminal_frames: vec![terminal(Some(proto::SamplingUsage {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 7,
            ..Default::default()
        }))],
        expected_usage_total: 7,
        expected_finish_reason: Some(FinishReason::Stop),
        zero_usage_terminal_frames: Some(vec![terminal(None)]),
        bare_terminal_frames: None,
        // prost owns wire decoding: a corrupt frame surfaces as a transport
        // `Status` error, so no frame-level malformed input can be spelled.
        malformed_frame: None,
        unknown_event_frame: None,
        defective_known_frame: None,
        delta_less_prelude_frame: None,
        refusal: None,
        interleaved_reasoning: None,
    }
}

rig_core::streaming_conformance_suite! {
    provider: "xai_grpc",
    fixture: fixture(),
    manifest: [partial_tool_args, zero_usage_terminal],
}

/// The wire families this binary covers, which the workspace registry
/// lists as out of its binary.
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
