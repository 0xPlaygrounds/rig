//! Empty-content conformance, driven by the shared scenarios against
//! recorded Anthropic traffic.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
//!
//! The `empty_end_turn` suite in this directory carries the older,
//! anthropic-specific coverage of the empty terminal turn on the buffered
//! surface; the scenarios here add the shared cross-provider shapes and the
//! streaming leg.

use rig::client::CompletionClient;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_agent::test_utils::{
    empty_history_turn_is_dropped_before_send, empty_turn_after_tool_round_trip,
    textless_tool_call_turn,
};

use super::super::support::with_anthropic_cassette;

/// A forced tool choice plus a no-prose instruction yields a lone tool_use
/// block with no text part on both surfaces.
#[tokio::test]
async fn textless_tool_call_turn_matches_across_surfaces() {
    with_anthropic_cassette(
        "empty_content/textless_tool_call_turn",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            textless_tool_call_turn(model)
                .await
                .expect("the textless tool-call scenario should hold");
        },
    )
    .await;
}

/// Anthropic is the provider that produces the genuinely empty terminal turn
/// on request (an `end_turn` with zero content blocks after the tool result —
/// the shape its deleted fabrication site papered over), so both surfaces of
/// the shared scenario are recordable live here, including the cleanly
/// finished empty stream.
#[tokio::test]
async fn empty_turn_after_tool_round_trip_matches_across_surfaces() {
    with_anthropic_cassette(
        "empty_content/empty_turn_after_tool_round_trip",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            empty_turn_after_tool_round_trip(model, |builder| builder)
                .await
                .expect("the empty-terminal-turn scenario should hold");
        },
    )
    .await;
}

/// An empty assistant turn in caller-supplied history is dropped before the
/// request is built; the recorded exchange proves Anthropic accepted the
/// payload rig actually sent after the drop.
#[tokio::test]
async fn empty_history_turn_is_dropped_before_send_on_the_wire() {
    with_anthropic_cassette(
        "empty_content/empty_history_turn_is_dropped",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            empty_history_turn_is_dropped_before_send(model)
                .await
                .expect("the poisoned-history scenario should hold");
        },
    )
    .await;
}
