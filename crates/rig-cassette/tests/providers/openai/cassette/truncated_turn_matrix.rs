//! Edge matrix for a **truncated** OpenAI turn that carried no content.
//!
//! **Bug.** A reasoning model whose output-token cap is spent entirely on
//! hidden reasoning answers with an empty message and the provider's own
//! diagnostic attached:
//!
//! ```json
//! {"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}
//! ```
//!
//! The Chat Completions normalizer rejected that wholesale —
//! `Response contained no message or tool call (empty)` — throwing away both
//! the finish reason and the usage, so a caller could not tell "you hit the
//! cap" from "the provider misbehaved". The other three paths for the same
//! situation already got it right:
//!
//! * the Responses API deliberately lets a contentless `status: incomplete`
//!   turn through so its `Length` reaches the caller;
//! * the chat-completions *streaming* path yields a terminal record with the
//!   reason regardless of what the stream produced;
//! * and the same reasoning model driven through the Responses surface
//!   therefore behaved differently from the Chat Completions surface for one
//!   and the same rig-level request.
//!
//! The fix gives the unary path the same rule, named once on
//! [`FinishReason::truncated_output`]: a turn the provider *cut short* may be
//! empty, a turn that *ran to completion* may not.
//!
//! **How these cells fail on `origin/main`.** The `chat_*_budget_exhausted`
//! cells replay their recorded fixture into `main`'s normalizer and get
//! `CompletionError(ResponseError("Response contained no message or tool call
//! (empty)"))` where the cell expects `FinishReason::Length`.
//!
//! | # | cell | surface | transport | model | shape | status |
//! |---|------|---------|-----------|-------|-------|--------|
//! | 9 | `chat_streaming_reasoning_budget_exhausted` | chat | streaming | gpt-5-nano | empty + length | recorded |
//! | 13 | `responses_blocking_reasoning_budget_exhausted` | responses | blocking | gpt-5-nano | control | recorded |
//! | 15 | `responses_blocking_partial_text_truncation` | responses | blocking | gpt-4o-mini | control | recorded |
//!
//! Unit cells — the accept/reject rule across the whole finish-reason
//! vocabulary, which live traffic cannot enumerate (no prompt reliably
//! produces an empty `content_filter` or an empty `tool_calls` turn) — live
//! beside the predicate itself
//! (`crates/rig-core/src/providers/internal/openai_chat_completions_compatible/tests.rs`,
//! `truncated_output_covers_only_the_cut_short_reasons`) and beside the
//! wire that consumes it in
//! `crates/rig-core/src/providers/openai/wire/chat/tests.rs`
//! (`an_empty_turn_the_provider_cut_short_keeps_its_reason_and_usage`,
//! `an_empty_turn_that_ran_to_completion_is_a_provider_defect`). They moved
//! there with the wire cutover: the guard that used to live in the client
//! layer's normalizer is now the chat decoder's whole-body entry point, and
//! it reads the same predicate rather than restating the set.
//!
//! Every cell re-reads its own fixture and fails if the recorded turn stopped
//! having the shape the cell is about.

use rig::completion::FinishReason;
use rig::providers::openai;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_openai_truncation_cassette;
use crate::cassettes;
use crate::support::{assistant_text_response, collect_text_and_terminal};
use rig::completion::CompletionRequest;

/// OpenAI's documented floor for the field, and far below what any reasoning
/// model needs to say anything — so the whole budget goes to hidden reasoning
/// and the turn comes back empty.
const TINY_CAP: u64 = 16;
const LONG_PROMPT: &str = "Write a 500 word essay about maple trees.";

// ---------------------------------------------------------------------------
// Chat Completions — the surface that threw the diagnostic away.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Streaming: the transport that was already right, pinned as the parity anchor.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_streaming_reasoning_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_streaming_reasoning_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_streaming_reasoning_budget_exhausted",
        |client| async move {
            let model = client.openai.chat("gpt-5-nano");
            let request = CompletionRequest::new(LONG_PROMPT).max_tokens(TINY_CAP);

            let stream = model.stream(request).expect("stream should connect");
            let (text, terminal) = collect_text_and_terminal(stream).await;

            assert!(text.is_empty());
            assert_eq!(
                terminal.expect("terminal record").finish_reason(),
                Some(FinishReason::Length)
            );
        },
    )
    .await;

    assert_recorded_truncated_chat_stream(SCENARIO);
}

// ---------------------------------------------------------------------------
// Responses API — the rule this fix copied, pinned so it stays the reference.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_blocking_reasoning_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/responses_blocking_reasoning_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/responses_blocking_reasoning_budget_exhausted",
        |client| async move {
            let model = client.openai.completion("gpt-5-nano");
            let request = CompletionRequest::new(LONG_PROMPT).max_tokens(TINY_CAP);

            let response = model.call(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
        },
    )
    .await;

    assert_recorded_incomplete_responses_turn(SCENARIO);
}

#[tokio::test]
async fn responses_blocking_partial_text_truncation() {
    const SCENARIO: &str = "truncated_turn_matrix/responses_blocking_partial_text_truncation";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/responses_blocking_partial_text_truncation",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O_MINI);
            let request = CompletionRequest::new(LONG_PROMPT).max_tokens(TINY_CAP);

            let response = model.call(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(
                assistant_text_response(&response.choice)
                    .is_some_and(|text| !text.trim().is_empty())
            );
        },
    )
    .await;

    assert_recorded_incomplete_responses_turn(SCENARIO);
}

// ---------------------------------------------------------------------------
// Fixture-premise checks.
// ---------------------------------------------------------------------------

fn recorded_response_bodies(scenario: &str) -> Vec<String> {
    let path = cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
        .filter_map(|interaction| {
            interaction
                .get("then")
                .and_then(|then| then.get("body"))
                .and_then(serde_yaml::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect()
}

/// The streaming premise: a terminal chunk reporting `length`.
fn assert_recorded_truncated_chat_stream(scenario: &str) {
    let found = recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| body.lines())
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .any(|chunk| {
            chunk
                .get("choices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|choice| choice.get("finish_reason").and_then(Value::as_str) == Some("length"))
        });

    assert!(
        found,
        "cassette {scenario} no longer records a `length` finish reason on the stream"
    );
}

/// The Responses premise: `status: incomplete` with
/// `incomplete_details.reason == "max_output_tokens"`.
fn assert_recorded_incomplete_responses_turn(scenario: &str) {
    let found = recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| {
            // Both the unary body and each SSE frame's `response` object.
            let frames = body
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .filter_map(|frame| frame.get("response").cloned())
                .collect::<Vec<_>>();
            serde_json::from_str::<Value>(body)
                .into_iter()
                .chain(frames)
                .collect::<Vec<_>>()
        })
        .any(|response| {
            response.get("status").and_then(Value::as_str) == Some("incomplete")
                && response
                    .get("incomplete_details")
                    .and_then(|details| details.get("reason"))
                    .and_then(Value::as_str)
                    == Some("max_output_tokens")
        });

    assert!(
        found,
        "cassette {scenario} no longer records an `incomplete` Responses turn"
    );
}
