//! Empty-content conformance, driven by the shared scenarios against
//! recorded Gemini traffic.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
//!
//! No length-starved empty-turn recording here on purpose: rig deliberately
//! treats a part-less Gemini candidate as a malformed response (the restored
//! inbound guard, pinned by `a_candidate_with_no_parts_is_rejected`), so the
//! genuinely-empty completed turn is recorded on Anthropic instead, where an
//! empty `end_turn` is a legal wire shape.

use rig::client::CompletionClient;
use rig::providers::gemini;
use rig_agent::test_utils::{empty_history_turn_is_dropped_before_send, textless_tool_call_turn};

use super::super::support::with_gemini_cassette;

/// A forced tool choice plus a no-prose instruction yields a lone
/// functionCall part with no text on both surfaces.
#[tokio::test]
async fn textless_tool_call_turn_matches_across_surfaces() {
    with_gemini_cassette(
        "empty_content/textless_tool_call_turn",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            textless_tool_call_turn(model)
                .await
                .expect("the textless tool-call scenario should hold");
        },
    )
    .await;
}

/// An empty assistant turn in caller-supplied history is dropped before the
/// request is built; the recorded exchange proves Gemini accepted the payload
/// rig actually sent after the drop.
#[tokio::test]
async fn empty_history_turn_is_dropped_before_send_on_the_wire() {
    with_gemini_cassette(
        "empty_content/empty_history_turn_is_dropped",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            empty_history_turn_is_dropped_before_send(model)
                .await
                .expect("the poisoned-history scenario should hold");
        },
    )
    .await;
}
