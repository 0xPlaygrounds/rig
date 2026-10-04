//! Wire-conformance suite for the Bedrock Converse wire.
//!
//! Events-first (`WireInput::Event`): fixture frames are Converse stream
//! events as JSON, the frames the transport hands the decoder, replayed by
//! a scripted transport through the Converse wire and the shared driver,
//! with no AWS client. Frame-level malformed and unknown scenarios
//! self-report as skipped: the SDK reads the event stream, and surfaces a
//! corrupt frame as a transport error.

use rig::bedrock::completion::{Converse, ConverseFrame, ConverseRequest};
use rig_core::completion::{CompletionRequest, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::test_utils::streaming_conformance::{
    ProviderWireFixture, WireDriver, WireInput, event_frame, fixtures::drain,
};
use serde_json::{Value, json};

type Events = Vec<Result<Value, ProviderError>>;

/// Replays scripted Converse events.
#[derive(Clone)]
struct Scripted(std::sync::Arc<std::sync::Mutex<Events>>);

impl Transport<Converse> for Scripted {
    fn send(&self, _payload: ConverseRequest, _exchange: Exchange) -> Opening<ConverseFrame> {
        let events = match self.0.lock() {
            Ok(mut events) => std::mem::take(&mut *events),
            Err(_) => {
                return Opening::failed(ProviderError::Provider("script lock poisoned".to_owned()));
            }
        };
        Opening::ready(Opened::new(futures::stream::iter(
            events
                .into_iter()
                .map(|event| event.map(ConverseFrame::Event)),
        )))
    }
}

fn driver() -> WireDriver {
    WireDriver::new("aws_bedrock", |chunks| {
        Box::pin(async move {
            let events: Events = chunks
                .into_iter()
                .map(|chunk| match chunk {
                    Ok(frame) => frame.downcast_event::<Value>().cloned().ok_or_else(|| {
                        ProviderError::Provider(
                            "bedrock conformance frames must be Converse events".to_string(),
                        )
                    }),
                    Err(error) => Err(ProviderError::Http(error.into())),
                })
                .collect();
            let model = Model::new(
                Converse::new("amazon.nova-lite-v1:0"),
                Scripted(std::sync::Arc::new(std::sync::Mutex::new(events))),
            );
            let stream = model.stream(CompletionRequest::new("hi"))?;
            Ok(drain(stream).await)
        })
    })
}

fn text_delta(index: i32, text: &str) -> WireInput {
    event_frame(json!({ "contentBlockDelta": {
        "contentBlockIndex": index, "delta": { "text": text },
    } }))
}

fn tool_start(index: i32, id: &str, name: &str) -> WireInput {
    event_frame(json!({ "contentBlockStart": {
        "contentBlockIndex": index, "start": { "toolUse": { "toolUseId": id, "name": name } },
    } }))
}

fn tool_delta(index: i32, input: &str) -> WireInput {
    event_frame(json!({ "contentBlockDelta": {
        "contentBlockIndex": index, "delta": { "toolUse": { "input": input } },
    } }))
}

fn block_stop(index: i32) -> WireInput {
    event_frame(json!({ "contentBlockStop": { "contentBlockIndex": index } }))
}

fn message_stop(reason: &str) -> WireInput {
    event_frame(json!({ "messageStop": { "stopReason": reason } }))
}

fn metadata(usage: Option<Value>) -> WireInput {
    let metadata = match usage {
        Some(usage) => json!({ "usage": usage }),
        None => json!({}),
    };
    event_frame(json!({ "metadata": metadata }))
}

fn usage(input: i32, output: i32, total: i32) -> Value {
    json!({ "inputTokens": input, "outputTokens": output, "totalTokens": total })
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
        terminal_frames: vec![message_stop("end_turn"), metadata(Some(usage(10, 5, 15)))],
        expected_usage_total: 15,
        expected_finish_reason: Some(FinishReason::Stop),
        zero_usage_terminal_frames: Some(vec![message_stop("end_turn"), metadata(None)]),
        bare_terminal_frames: None,
        // The AWS SDK owns event-stream decoding: a corrupt frame surfaces as
        // a receive (transport) error, so no frame-level malformed input can
        // be spelled; the scenario reports a visible skip.
        malformed_frame: None,
        // An event type the decoder does not know is skipped with a warning,
        // which the decoder's own tests hold.
        unknown_event_frame: None,
        defective_known_frame: None,
        delta_less_prelude_frame: None,
        refusal: None,
        interleaved_reasoning: None,
    }
}

rig_core::streaming_conformance_suite! {
    provider: "bedrock",
    fixture: fixture(),
    manifest: [partial_tool_args, zero_usage_terminal],
}

/// Compile-linked manifest of the wire families this binary covers.
///
/// This suite is feature-gated into the `bedrock` test binary, not the `core`
/// one that hosts the workspace registry, so the registry cannot link it: it
/// lists `bedrock` in `OUT_OF_BINARY_FAMILIES` (executed by the workspace-wide
/// `--all-features` nextest run). The test below keeps the family name honest
/// at the definition site, which is the direction the registry loses for
/// out-of-binary suites (#2258 F3).
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
