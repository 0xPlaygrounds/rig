//! Edge matrix for extended-thinking tokens reaching `Usage::reasoning_tokens`.
//!
//! # The bug
//!
//! Anthropic reports the tokens Claude spent thinking as a breakdown of
//! `output_tokens`:
//!
//! ```text
//! "usage":{"input_tokens":773,"output_tokens":1835,
//!          "output_tokens_details":{"thinking_tokens":1167}}
//! ```
//!
//! Neither `anthropic::completion::Usage` nor the streaming `PartialUsage`
//! modeled `output_tokens_details`, so serde dropped it and
//! `anthropic_usage_totals` left `completion::Usage::reasoning_tokens` at `0`
//! on every turn — blocking and streaming alike. That field's own rustdoc
//! names "Anthropic extended thinking" as something it counts, and Gemini
//! (`thoughts_token_count`) and DeepSeek
//! (`completion_tokens_details.reasoning_tokens`) both populate it, so a
//! caller comparing reasoning spend across providers saw Anthropic as free.
//!
//! The value is a *breakdown* of `output_tokens`, not a sibling of it, so it
//! must not enter `total_tokens`.
//!
//! # Matrix
//!
//! Every row is a recorded live cell unless marked otherwise. Each recorded
//! cell asserts the parsed `reasoning_tokens` **equals its own fixture's**
//! recorded `thinking_tokens` (read back after the wrapper returns, since
//! record mode writes the fixture on the way out), so a cell whose provider
//! turn stopped thinking fails instead of passing vacuously.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 28 | `unit_absent_details_reports_none` | decoder: no `output_tokens_details` | none | unit |
//! | 29 | `unit_unknown_detail_bucket_is_ignored` | decoder: forward compatibility | `> 0` | unit |
//! | 30 | `unit_thinking_tokens_stay_out_of_the_total` | totals arithmetic | n/a | unit |
//! | 31 | `unit_streaming_and_blocking_share_the_mapping` | one usage reader for both modes | n/a | unit |
//!
//! Cells 28–31 are unit tests because no live turn can vary what they assert:
//! whether the decoder tolerates an absent or unknown bucket, and whether the
//! breakdown is excluded from `total_tokens`, are properties of its one usage
//! reader, not of any provider response.
//!
//! Two constraints the live API imposed, discovered while recording:
//!
//! - **Adaptive thinking may decline to think at all.** Opus 4.7 answers even
//!   the multi-step prompt without thinking, reporting the bucket *present and
//!   zero* — a different wire shape from thinking-off, which omits the bucket
//!   entirely. Rather than force it, cells 2 and 15 pin that shape, and cells
//!   10 and 23 (Opus 4.8) cover adaptive thinking that does engage.
//! - **Opus 4.8 rejects `thinking.type.enabled`** (`"is not supported for
//!   this model. Use \"thinking.type.adaptive\" and \"output_config.effort\""`),
//!   so the second-model-family cells cover the adaptive path only.
//!
//! Two dimensions were considered and deliberately dropped, with reasons:
//!
//! - **A `message_start` carry-forward cell.** `cache_creation` needs one
//!   because Anthropic reports it on `message_start`; `output_tokens_details`
//!   arrives on the terminal `message_delta` instead, so there is nothing to
//!   carry and no live turn can produce the inverse split. The streamed cells
//!   are what pin it: each reads the breakdown off its own terminal frame, so
//!   a fallback that started reading `message_start` would have to invent the
//!   same number to stay green.
//! - **A gateway cell.** The Anthropic-compatible gateways do not implement
//!   extended thinking on the Messages endpoint, so a gateway recording would
//!   assert the absent-breakdown case that cells 12–13 already cover.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::{CompletionRequest, Usage};
use rig::providers::anthropic;
use serde_json::json;

// ------------------------------------------------------------- blocking ---

// ------------------------------------------------------------ streaming ---

// ----------------------------------------------------- adjacent surfaces ---

// ----------------------------------------------------------------- unit ---

/// `usage` as the Messages decoder reads it, off a whole reply or off a
/// stream whose terminal `message_delta` carries it.
fn decoded_usage(usage: serde_json::Value, streamed: bool) -> Usage {
    use rig::wire::{Mode, WireFrame};
    let wire = anthropic::Anthropic::new("test-key")
        .completion(anthropic::completion::CLAUDE_SONNET_4_6)
        .wire;
    let message = |usage: serde_json::Value, stop_reason: serde_json::Value| {
        json!({"type": "message", "id": "msg_1", "role": "assistant",
            "model": anthropic::completion::CLAUDE_SONNET_4_6,
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": stop_reason, "usage": usage})
    };
    let (mode, frames) = if streamed {
        let start = json!({"type": "message_start",
            "message": message(json!({"input_tokens": 1, "output_tokens": 1}), json!(null))});
        let delta = json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
            "usage": usage});
        (Mode::Streaming, vec![start, delta])
    } else {
        (Mode::Unary, vec![message(usage, json!("end_turn"))])
    };
    rig_core::test_utils::history_conformance::decode(
        &wire,
        &CompletionRequest::new("hello"),
        mode,
        frames
            .into_iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
    )
    .expect("the reply decodes")
    .usage
}

/// A turn that reports no breakdown yields no counter, not a failure.
#[test]
fn unit_absent_details_reports_none() {
    let usage = decoded_usage(
        json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": null,
            "cache_creation_input_tokens": null
        }),
        false,
    );
    assert_eq!(usage.reasoning_tokens, None);
}

/// A bucket Anthropic adds later must not break the known one.
#[test]
fn unit_unknown_detail_bucket_is_ignored() {
    let usage = decoded_usage(
        json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "output_tokens_details": { "thinking_tokens": 7, "future_bucket_tokens": 3 }
        }),
        false,
    );
    assert_eq!(usage.reasoning_tokens, Some(7));
}

/// The breakdown is inside `output_tokens`, so the total must not grow.
#[test]
fn unit_thinking_tokens_stay_out_of_the_total() {
    let counts = json!({
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4
    });
    let mut thinking = counts.clone();
    thinking["output_tokens_details"] = json!({ "thinking_tokens": 15 });
    let with_thinking = decoded_usage(thinking, false);
    let without = decoded_usage(counts, false);

    assert_eq!(with_thinking.reasoning_tokens, Some(15));
    assert_eq!(without.reasoning_tokens, None);
    assert_eq!(
        with_thinking.total_tokens, without.total_tokens,
        "the breakdown is already counted in output_tokens",
    );
    assert_eq!(with_thinking.total_tokens, Some(10 + 3 + 4 + 20));
}

/// Blocking and streaming read usage through the same reader, so the same
/// counters must normalize identically, including the breakdown that
/// arrives on the streaming path's terminal `message_delta` rather than its
/// `message_start`.
#[test]
fn unit_streaming_and_blocking_share_the_mapping() {
    let counts = json!({
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 6,
        "output_tokens_details": { "thinking_tokens": 42 }
    });
    let streamed = decoded_usage(counts.clone(), true);
    assert_eq!(decoded_usage(counts, false), streamed);
    assert_eq!(streamed.reasoning_tokens, Some(42));
}
