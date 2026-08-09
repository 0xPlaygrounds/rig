//! Empty-content conformance, driven by the shared scenarios against
//! recorded OpenAI traffic.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::providers::openai;
use rig_agent::test_utils::{empty_history_turn_is_dropped_before_send, textless_tool_call_turn};

use super::super::support::with_openai_cassette;

/// A forced tool choice plus a no-prose instruction yields a lone function
/// call with no text part on both surfaces — the turn the old non-empty
/// container could only represent by fabricating an empty text block.
#[tokio::test]
async fn textless_tool_call_turn_matches_across_surfaces() {
    with_openai_cassette(
        "empty_content/textless_tool_call_turn",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            textless_tool_call_turn(model)
                .await
                .expect("the textless tool-call scenario should hold");
        },
    )
    .await;
}

/// An empty assistant turn in caller-supplied history is dropped before the
/// request is built; the recorded exchange proves OpenAI accepted the payload
/// rig actually sent after the drop.
#[tokio::test]
async fn empty_history_turn_is_dropped_before_send_on_the_wire() {
    with_openai_cassette(
        "empty_content/empty_history_turn_is_dropped",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            empty_history_turn_is_dropped_before_send(model)
                .await
                .expect("the poisoned-history scenario should hold");
        },
    )
    .await;
}
