//! Wire-conformance suite for candle's in-process typed-event wire.
//!
//! Events-first (`WireInput::Event`): fixture frames are already-typed
//! canonical-grammar events driven through [`rig_candle::stream_from_events`]
//! — the shared driver, grammar, and terminal normalization — with no model
//! load. This family never produces `Unknown` and has no frame-level decode,
//! so the malformed/unknown scenarios self-report as skipped.

use rig_candle::{CandleCompletionResponse, FinishReason as CandleFinishReason};
use rig_core::completion::{CompletionError, FinishReason};
use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall};
use rig_core::test_utils::streaming_conformance::{
    ProviderWireFixture, WireDriver, event_frame, fixtures::drain,
};

type CandleEvent = RawStreamingChoice<CandleCompletionResponse>;

fn driver() -> WireDriver {
    WireDriver::new("candle", |chunks| {
        Box::pin(async move {
            let events: Vec<Result<CandleEvent, CompletionError>> = chunks
                .into_iter()
                .map(|chunk| match chunk {
                    Ok(frame) => frame
                        .downcast_event::<CandleEvent>()
                        .cloned()
                        .ok_or_else(|| {
                            CompletionError::ProviderError(
                                "candle conformance frames must be generation events".to_string(),
                            )
                        }),
                    Err(error) => Err(CompletionError::HttpError(error)),
                })
                .collect();
            let stream = rig_candle::stream_from_events(futures::stream::iter(events));
            Ok(drain(stream).await)
        })
    })
}

fn terminal_response() -> CandleCompletionResponse {
    CandleCompletionResponse {
        text: "hi".to_string(),
        prompt_tokens: 10,
        generated_tokens: 5,
        requested_max_tokens: 256,
        effective_max_tokens: 256,
        finish_reason: CandleFinishReason::Eos,
        prefill_duration_ms: 1,
        time_to_first_token_ms: Some(1),
        generation_duration_ms: 2,
        tokens_per_second: None,
    }
}

fn fixture() -> ProviderWireFixture {
    ProviderWireFixture {
        driver: driver(),
        text_frames: vec![event_frame::<CandleEvent>(RawStreamingChoice::Message(
            "hi".to_string(),
        ))],
        expected_texts: vec!["hi"],
        tool_call_frames: vec![event_frame::<CandleEvent>(RawStreamingChoice::ToolCall(
            RawStreamingToolCall::new(
                "call_1".to_string(),
                "get_weather".to_string(),
                serde_json::json!({"city": "Tokyo"}),
            ),
        ))],
        expected_tool_name: "get_weather",
        // Local generation delivers tool calls whole after the buffered turn
        // is parsed; arguments never stream.
        partial_tool_call_frames: None,
        terminal_frames: vec![event_frame::<CandleEvent>(
            RawStreamingChoice::FinalResponse(terminal_response()),
        )],
        // prompt 10 + generated 5.
        expected_usage_total: 15,
        expected_finish_reason: Some(FinishReason::Stop),
        // Local usage is always computed from token counts; there is no
        // usage-less genuine terminal to spell.
        zero_usage_terminal_frames: None,
        bare_terminal_frames: None,
        // No decode exists between the generator and the driver, so no
        // frame-level malformed input can be spelled; generation failures are
        // transport errors on the channel.
        malformed_frame: None,
        // This family never produces `Unknown`: the producer sends
        // already-typed grammar events, so every frame is `Modeled`.
        unknown_event_frame: None,
        defective_known_frame: None,
        delta_less_prelude_frame: None,
        refusal: None,
        interleaved_reasoning: None,
    }
}

rig_core::streaming_conformance_suite! {
    provider: "candle",
    fixture: fixture(),
    // Typed local wire: no malformed/unknown/defective shapes exist,
    // so the derived set is genuinely empty.
    manifest: [],
}

/// Compile-linked manifest of the wire families this binary covers.
///
/// This suite lives outside the `rig` facade's `core` test binary, so the
/// workspace registry cannot link it: it lists `candle` in
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
