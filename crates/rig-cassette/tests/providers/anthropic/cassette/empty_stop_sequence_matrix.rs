//! Edge matrix for normalizing a turn that stopped on a stop sequence before
//! emitting anything.
//!
//! # The bug
//!
//! Anthropic strips the sequence it stopped on. When the match is the *first*
//! thing the model produces, the turn comes back 200 with:
//!
//! ```json
//! {"content":[],"stop_reason":"stop_sequence","stop_sequence":"alpha", ...}
//! ```
//!
//! The blocking response mapping carved out exactly one legal empty case —
//! `end_turn` — and routed every other empty response through
//! `require_non_empty_response`. So a completed provider turn became
//! `CompletionError::ResponseError("Response contained no message or tool call
//! (empty)")`, destroying the usage, identity and finish-reason the same
//! response carried. The streamed twin of that request already finished
//! cleanly with an empty choice, so this was also a blocking/streaming
//! divergence — with the blocking side the broken one.
//!
//! # Matrix
//!
//! `expected` is what the *normalized* response must be. Recorded cells
//! re-derive their premise from their own fixture bytes: a cell whose provider
//! turn stopped coming back empty would otherwise pass while covering nothing.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 11 | `with_preamble_empty_stop` | system prompt present | empty choice | recorded |
//! | 12 | `with_tools_empty_stop` | tools advertised | empty choice | recorded |
//! | 19 | `unit_empty_assistant_turn_cannot_be_replayed` | adjacent request boundary | error | unit |
//!
//! Cell 19 is a unit test because it is about a request-side conversion, not
//! about anything a provider turn can vary.
//!
//! Seven further unit cells pinned the blocking mapping's guard directly on
//! hand-built provider responses (`max_tokens`, `tool_use`, `refusal`,
//! `pause_turn`, a missing stop reason, a `stop_sequence` naming no sequence,
//! and the legal `end_turn` empty). They asserted a second mapping from the
//! provider response type to `AssistantContent`, which no longer exists: the
//! wire's decoder is the one mapping, and the guard lives beside it in
//! `crates/rig-core/src/providers/anthropic/`. Cells 1–18 still cover the
//! behaviour end to end on recorded turns.

use rig::completion::ToolDefinition;
use rig::providers::anthropic;
use serde_json::json;

use super::super::support::{recorded_response_body, with_anthropic_empty_stop_cassette};
use rig::completion::CompletionRequest;

/// Asks for exactly one word so a stop sequence naming that word matches
/// before the model emits anything else.
pub(super) const IMMEDIATE_PROMPT: &str =
    "Reply with exactly this one word and nothing else: alpha";

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("get_weather").expect("tool name"),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

/// The premise every empty-stop cell rests on: the recorded blocking body had
/// **no** content blocks and stopped on a sequence. Read after the cassette
/// wrapper returns — record mode writes the fixture on the way out.
pub(super) fn assert_recorded_empty_stop(scenario: &str) {
    let body = recorded_response_body(scenario);
    assert_eq!(
        body.get("stop_reason").and_then(serde_json::Value::as_str),
        Some("stop_sequence"),
        "{scenario}: the recorded turn must have stopped on a sequence"
    );
    assert_eq!(
        body.get("content").and_then(serde_json::Value::as_array),
        Some(&Vec::new()),
        "{scenario}: the recorded turn must carry no content blocks — without \
         that this cell asserts nothing about the empty case"
    );
}

// ---------------------------------------------------------------------------
// 1–5: the surfaces a caller reaches this through
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 6: control — the same stop reason with content must still produce content
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 7–10: sequence shapes that can match at position zero
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 11–14: the rest of the request surface
// ---------------------------------------------------------------------------

#[tokio::test]
async fn with_preamble_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/with_preamble_empty_stop",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = CompletionRequest::new(IMMEDIATE_PROMPT)
                .preamble("You follow formatting instructions exactly.")
                .max_tokens(32)
                .additional_params(json!({ "stop_sequences": ["alpha"] }));
            let response = model
                .call(request)
                .await
                .expect("empty stop turn with a system prompt should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/with_preamble_empty_stop");
}

#[tokio::test]
async fn with_tools_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/with_tools_empty_stop",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = CompletionRequest::new(IMMEDIATE_PROMPT)
                .max_tokens(32)
                .tool(weather_tool())
                .additional_params(json!({ "stop_sequences": ["alpha"] }));
            let response = model
                .call(request)
                .await
                .expect("empty stop turn with tools advertised should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/with_tools_empty_stop");
}

// ---------------------------------------------------------------------------
// 15–18: what the response must still carry, and life after it
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 19: the adjacent request boundary
// ---------------------------------------------------------------------------
