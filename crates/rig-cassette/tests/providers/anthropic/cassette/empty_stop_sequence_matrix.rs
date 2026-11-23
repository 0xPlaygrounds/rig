//! Edge matrix for normalizing a turn that stopped on a stop sequence before
//! emitting anything.
//!
//! Anthropic strips the sequence it stopped on. When the match is the first
//! thing the model produces, the turn comes back 200 with `"content": []` and
//! `"stop_reason": "stop_sequence"`. The blocking mapping must normalize that
//! to an empty choice that keeps the usage, identity and finish reason, as the
//! streamed twin does, rather than an empty-response error. The mapping's
//! guard lives beside the wire's decoder in
//! `crates/rig-core/src/providers/anthropic/`.
//!
//! Recorded cells re-derive their premise from their own fixture bytes, so a
//! provider turn that stopped coming back empty fails the cell. The replay
//! boundary cell is a unit test because it is about a request-side
//! conversion, not anything a provider turn can vary.

use rig::completion::ToolDefinition;
use rig::providers::anthropic;
use serde_json::json;

use super::super::support::with_anthropic_cassette;
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
        annotations: Default::default(),
    }
}

/// The premise every empty-stop cell rests on: the recorded blocking body had
/// **no** content blocks and stopped on a sequence. Read after the cassette
/// wrapper returns — record mode writes the fixture on the way out.
pub(super) fn assert_recorded_empty_stop(scenario: &str) {
    let body = crate::cassettes::recorded_json_response("anthropic", scenario);
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
    with_anthropic_cassette(
        "empty_stop_sequence_matrix/with_preamble_empty_stop",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = CompletionRequest::new(IMMEDIATE_PROMPT)
                .preamble("You follow formatting instructions exactly.")
                .max_tokens(32)
                .stop(["alpha"]);
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
    with_anthropic_cassette(
        "empty_stop_sequence_matrix/with_tools_empty_stop",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = CompletionRequest::new(IMMEDIATE_PROMPT)
                .max_tokens(32)
                .tool(weather_tool())
                .stop(["alpha"]);
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
