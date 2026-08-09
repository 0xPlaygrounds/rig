//! Deterministic runs of the shared empty-turn conformance scenarios.
//!
//! The scenarios in `test_utils::model_conformance` are provider-neutral;
//! these tests drive them with scripted mock turns so the three empty-turn
//! shapes — a textless tool-call turn, a turn ending with no content after a
//! tool-result round trip, and an empty assistant turn already in
//! caller-supplied history — are pinned on both the buffered and streaming
//! surfaces without any provider in the loop. Provider cassette suites drive
//! the same scenarios against recorded live traffic.
#![cfg(feature = "test-utils")]
#![allow(clippy::expect_used)]

use rig_agent::test_utils::{
    MockCompletionModel, MockStreamEvent, MockTurn, empty_history_turn_is_dropped_before_send,
    empty_turn_after_tool_round_trip, textless_tool_call_turn,
};
use serde_json::json;

/// Shape 1: the model answers with a lone function call and no text part, on
/// both surfaces.
#[tokio::test]
async fn textless_tool_call_turn_holds_on_both_surfaces() {
    let model = MockCompletionModel::from_turns_and_stream_turns(
        [MockTurn::tool_call(
            "call-1",
            "record_value",
            json!({"value": 7}),
        )],
        [[
            MockStreamEvent::tool_call("call-2", "record_value", json!({"value": 7})),
            MockStreamEvent::final_response_with_default_usage(),
        ]],
    );
    textless_tool_call_turn(model)
        .await
        .expect("the textless tool-call scenario should hold on the mock");
}

/// Shape 2: after the tool result round trip the model ends its turn with no
/// content at all. The streaming empty turn is a *cleanly finished* stream
/// with zero content parts — representable now, no fabricated empty-text
/// part.
#[tokio::test]
async fn empty_turn_after_tool_round_trip_holds_on_both_surfaces() {
    let model = MockCompletionModel::from_turns_and_stream_turns(
        [
            MockTurn::tool_call("call-1", "ping", json!({})),
            MockTurn::from_contents([]),
        ],
        [
            vec![
                MockStreamEvent::tool_call("call-2", "ping", json!({})),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            vec![MockStreamEvent::final_response_with_default_usage()],
        ],
    );
    empty_turn_after_tool_round_trip(model, |builder| builder)
        .await
        .expect("the empty-terminal-turn scenario should hold on the mock");
}

/// Shape 3: an empty assistant turn in caller-supplied history is dropped
/// before the next request is built, so the request constructs, nothing
/// content-less reaches the model, and the run completes on both surfaces.
#[tokio::test]
async fn empty_history_turn_is_dropped_on_both_surfaces() {
    let model = MockCompletionModel::from_turns_and_stream_turns(
        [MockTurn::text("lantern")],
        [[
            MockStreamEvent::text("lantern"),
            MockStreamEvent::final_response_with_default_usage(),
        ]],
    );
    empty_history_turn_is_dropped_before_send(model)
        .await
        .expect("the poisoned-history scenario should hold on the mock");
}
