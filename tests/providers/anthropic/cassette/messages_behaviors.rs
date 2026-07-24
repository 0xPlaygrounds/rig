//! Anthropic Messages API behavior regression tests.
//!
//! Locks down provider-response preservation for unusual but valid response
//! shapes: `max_tokens` truncation surfacing partial output with its stop
//! reason intact.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;

#[tokio::test]
async fn max_tokens_truncation_preserves_stop_reason_and_partial_text() {
    with_anthropic_cassette(
        "messages_behaviors/max_tokens_truncation_preserves_stop_reason_and_partial_text",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(
                    "Write a story of at least 150 words about a lighthouse keeper.",
                )
                .preamble("You are a storyteller.".to_string())
                .max_tokens(64)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a truncated response should still convert, not error");

            assert_eq!(
                response.raw_response.stop_reason.as_deref(),
                Some("max_tokens"),
                "hitting max_tokens should preserve the max_tokens stop reason"
            );
            let text: String = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !text.trim().is_empty(),
                "partial output text should still be surfaced"
            );
            assert!(
                response.usage.output_tokens > 0 && response.usage.output_tokens <= 64,
                "output tokens should reflect the truncation cap, got {:?}",
                response.usage
            );
        },
    )
    .await;
}
