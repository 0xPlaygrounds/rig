//! Edge matrix for the streamed terminal's `stop_sequence`.
//!
//! # The bug
//!
//! Anthropic's terminal `message_delta` reports *which* of the caller's
//! `stop_sequences` matched:
//!
//! ```text
//! {"delta":{"stop_details":null,"stop_reason":"stop_sequence","stop_sequence":"charlie"},...}
//! ```
//!
//! The decoder read that field and then dropped it: the terminal record
//! had no slot for it, so the streamed terminal
//! answered a request with strictly less than the blocking response answered
//! the same request with (`CompletionResponse::stop_sequence` has carried it
//! all along).
//! Anthropic strips the matched sequence from the text, so the wire frame is
//! the only place the value exists — a streamed caller could learn that *a*
//! sequence fired and never which one.
//!
//! # Matrix
//!
//! Every row is a recorded live cell unless marked otherwise. `expected` is the
//! terminal record's `stop_sequence`. Every recorded cell also re-derives its
//! premise from its own fixture bytes (see
//! [`assert_recorded_terminal_stop_sequence`]) so a cell whose provider turn
//! stopped producing the shape it is about fails instead of passing vacuously.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 16 | `raw_with_tools_sequence_fires` | tools advertised | `charlie` | recorded |
//! | 18 | `raw_with_preamble_sequence_fires` | system prompt present | `charlie` | recorded |

use futures::StreamExt;
use rig::completion::{CompletionRequest, ToolDefinition};
use rig::driver::Model;
use rig::providers::anthropic;
use rig::providers::anthropic::wire::Messages;
use serde_json::json;

use super::super::support::with_anthropic_stop_sequence_cassette;

type AnthropicModel = Model<Messages>;

/// Emits `alpha`, `bravo`, `charlie`, `delta` on separate lines, so a stop
/// sequence naming any of them cuts the turn at a known point.
pub(super) const LIST_PROMPT: &str =
    "Repeat exactly these four words, one per line, and nothing else: alpha bravo charlie delta";

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

/// Drain a provider-native stream and return its terminal record.
async fn raw_terminal(model: &AnthropicModel, request: CompletionRequest) -> serde_json::Value {
    let mut stream = model
        .stream(request)
        .expect("stop-sequence stream should open");
    while let Some(item) = stream.next().await {
        item.expect("stream item should not error");
    }
    let record = stream
        .finish()
        .await
        .expect("stream should yield a terminal record");
    record.raw
}

/// Assert the terminal record a streamed cell produced.
///
/// The cell's *premise* — that the recorded `message_delta` really carried this
/// value — is asserted separately, after the cassette wrapper returns: in
/// record mode the fixture is written by `finish_after_test`, so an in-body
/// read would assert against the previous recording.
fn assert_terminal(
    terminal: &serde_json::Value,
    expected_sequence: Option<&str>,
    expected_reason: &str,
) {
    assert_eq!(
        terminal["stop_sequence"].as_str(),
        expected_sequence,
        "the terminal record must carry the sequence the wire reported"
    );
    assert_eq!(
        terminal["stop_reason"].as_str(),
        Some(expected_reason),
        "unexpected stop reason"
    );
}

/// The `stop_sequence` a cassette's terminal `message_delta` frame records.
///
/// Read back from the fixture rather than trusted from the parsed struct: a
/// cell whose provider turn quietly stopped matching would otherwise keep
/// passing while covering nothing.
fn recorded_terminal_stop_sequence(scenario: &str) -> Option<String> {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let frame = contents
        .lines()
        .find(|line| line.contains(r#""type":"message_delta""#))
        .unwrap_or_else(|| {
            panic!(
                "cassette {} should record a message_delta frame",
                path.display()
            )
        });
    let (_, after) = frame.split_once(r#""stop_sequence":"#).unwrap_or_else(|| {
        panic!(
            "message_delta in {} should report stop_sequence",
            path.display()
        )
    });

    json_prefix(after, &path)
}

/// Decode the first JSON value at the start of `text` (either `null` or a
/// string), so escapes and non-ASCII survive the read-back intact.
fn json_prefix(text: &str, path: &std::path::Path) -> Option<String> {
    let value = serde_json::Deserializer::from_str(text)
        .into_iter::<serde_json::Value>()
        .next()
        .unwrap_or_else(|| panic!("{} should record a stop_sequence value", path.display()))
        .unwrap_or_else(|err| {
            panic!(
                "stop_sequence in {} should be valid JSON: {err}",
                path.display()
            )
        });

    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(value) => Some(value),
        other => panic!(
            "stop_sequence in {} should be a string or null, got {other}",
            path.display()
        ),
    }
}

pub(super) fn assert_recorded_terminal_stop_sequence(scenario: &str, expected: Option<&str>) {
    assert_eq!(
        recorded_terminal_stop_sequence(scenario).as_deref(),
        expected,
        "{scenario}: the recorded message_delta must itself carry this value — \
         without it the cell asserts nothing about the wire"
    );
}

// ---------------------------------------------------------------------------
// 1–5: which sequence fired
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 6–10: sequence content shapes that must survive verbatim
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 11–14: controls — every other terminal must report no sequence
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 15–20: interaction with the rest of the request surface
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_with_tools_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_with_tools_sequence_fires",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);
            // Tools advertised but unused: the streaming body's `tool_choice`
            // reconciliation runs, and the terminal must be unaffected.
            let request = CompletionRequest::new(LIST_PROMPT)
                .max_tokens(64)
                .tool(weather_tool())
                .additional_params(json!({ "stop_sequences": ["charlie"] }));
            let terminal = raw_terminal(&model, request).await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_with_tools_sequence_fires",
        Some("charlie"),
    );
}

#[tokio::test]
async fn raw_with_preamble_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_with_preamble_sequence_fires",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = CompletionRequest::new(LIST_PROMPT)
                .preamble("You follow formatting instructions exactly.")
                .max_tokens(64)
                .additional_params(json!({ "stop_sequences": ["charlie"] }));
            let terminal = raw_terminal(&model, request).await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_with_preamble_sequence_fires",
        Some("charlie"),
    );
}

// ---------------------------------------------------------------------------
// 21–24: blocking twins — the surface that was already correct
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 25–26: adjacent paths sharing the terminal construction
// ---------------------------------------------------------------------------
