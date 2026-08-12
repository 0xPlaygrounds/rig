//! Anthropic `stop_sequences` boundary behavior, recorded from the real API.
//!
//! Locks down that a `stop_sequences` hit (passed through `additional_params`,
//! which is how callers reach the parameter today) surfaces as
//! `FinishReason::Stop`, that the emitted text stops before the sequence, and
//! that the raw wire response preserves `stop_reason: stop_sequence` together
//! with the matched `stop_sequence` value.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
use rig::completion::{CompletionModel, FinishReason, NormalizeCompletionResponse};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;

#[tokio::test]
async fn stop_sequence_hit_surfaces_stop_finish_reason_and_matched_sequence() {
    with_anthropic_cassette(
        "stop_sequences/stop_sequence_hit_surfaces_stop_finish_reason_and_matched_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = model
                .completion_request(
                    "Count upward from one, writing each number as a lowercase word \
                     separated by spaces: one two three ...",
                )
                .max_tokens(64)
                .temperature(0.0)
                .additional_params(serde_json::json!({
                    "stop_sequences": ["five"]
                }))
                .build();

            let raw = model
                .raw_completion(request)
                .await
                .expect("completion should succeed");

            assert_eq!(raw.stop_reason.as_deref(), Some("stop_sequence"));
            assert_eq!(raw.stop_sequence.as_deref(), Some("five"));

            let response = raw
                .normalize("anthropic")
                .expect("response should normalize");
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));

            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            assert!(
                text.contains("four"),
                "the model should have counted up to the stop sequence: {text:?}"
            );
            assert!(
                !text.contains("five"),
                "the emitted text must stop before the stop sequence: {text:?}"
            );
        },
    )
    .await;
}
