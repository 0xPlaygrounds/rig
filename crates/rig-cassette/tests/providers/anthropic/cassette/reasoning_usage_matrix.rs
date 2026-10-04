//! Extended-thinking tokens reaching `Usage::reasoning_tokens`.
//!
//! Anthropic reports the tokens Claude spent thinking as a breakdown of
//! `output_tokens`, so the decoder maps `output_tokens_details.thinking_tokens`
//! to `completion::Usage::reasoning_tokens` and keeps it out of `total_tokens`:
//!
//! ```text
//! "usage":{"input_tokens":773,"output_tokens":1835,
//!          "output_tokens_details":{"thinking_tokens":1167}}
//! ```
//!
//! These cells are unit tests on decoded usage because no live turn can vary
//! what they assert: an absent or unknown bucket, the total's arithmetic, and
//! one usage reader shared by blocking and streaming replies.
//!
//! | # | Cell | Dimension | expected |
//! |---|------|-----------|----------|
//! | 28 | `unit_absent_details_reports_none` | decoder: no `output_tokens_details` | none |
//! | 29 | `unit_unknown_detail_bucket_is_ignored` | decoder: forward compatibility | `> 0` |
//! | 30 | `unit_thinking_tokens_stay_out_of_the_total` | totals arithmetic | n/a |
//! | 31 | `unit_streaming_and_blocking_share_the_mapping` | one usage reader for both modes | n/a |

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
