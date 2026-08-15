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
//! `MessageDelta` parsed that field and the adapter then dropped it:
//! `StreamingCompletionResponse` had no slot for it, so `raw_stream` answered a
//! request with strictly less than `raw_completion` answered the same request
//! with (`CompletionResponse::stop_sequence` has carried it all along).
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
//! | 1 | `raw_single_sequence_fires` | one sequence, matches | `charlie` | recorded |
//! | 2 | `raw_first_listed_sequence_fires` | first list entry matches | `charlie` | recorded |
//! | 3 | `raw_second_listed_sequence_fires` | second list entry matches | `charlie` | recorded |
//! | 4 | `raw_earliest_in_output_wins` | list order vs output order | `bravo` | recorded |
//! | 5 | `raw_four_sequences` | maximum breadth of the list | `charlie` | recorded |
//! | 6 | `raw_newline_sequence` | sequence containing `\n` | `\nbravo` | recorded |
//! | 7 | `raw_whitespace_sequence` | sequence containing a space | `bravo charlie` | recorded |
//! | 8 | `raw_unicode_sequence` | non-ASCII sequence | `🌊` | recorded |
//! | 9 | `raw_punctuation_sequence` | punctuation-only sequence | `###` | recorded |
//! | 10 | `raw_case_sensitive_sequence` | uppercase sequence matches | `BRAVO` | recorded |
//! | 11 | `raw_case_mismatch_does_not_fire` | control: case mismatch | `None` | recorded |
//! | 12 | `raw_end_turn_control` | control: natural stop | `None` | recorded |
//! | 13 | `raw_max_tokens_control` | control: truncation | `None` | recorded |
//! | 14 | `raw_tool_use_control` | control: tool-call terminal | `None` | recorded |
//! | 15 | `raw_empty_content_sequence` | match before any output | `alpha` | recorded |
//! | 16 | `raw_with_tools_sequence_fires` | tools advertised | `charlie` | recorded |
//! | 17 | `raw_with_prompt_caching_sequence_fires` | manual prompt caching | `charlie` | recorded |
//! | 18 | `raw_with_preamble_sequence_fires` | system prompt present | `charlie` | recorded |
//! | 19 | `raw_sonnet_sequence_fires` | second model | `charlie` | recorded |
//! | 20 | `raw_long_multi_token_sequence` | multi-token phrase | phrase | recorded |
//! | 21 | `blocking_single_sequence_parity` | blocking twin of #1 | `charlie` | recorded |
//! | 22 | `blocking_second_listed_sequence_parity` | blocking twin of #3 | `charlie` | recorded |
//! | 23 | `blocking_end_turn_control` | blocking twin of #12 | `None` | recorded |
//! | 24 | `blocking_max_tokens_control` | blocking twin of #13 | `None` | recorded |
//! | 25 | `normalized_stream_single_sequence` | adjacent path: normalized stream | n/a | recorded |
//! | 26 | `agent_stream_single_sequence` | adjacent path: agent stream | n/a | recorded |
//! | 27 | `terminal_record_round_trips_stop_sequence` | serde of the record itself | `charlie` | unit |
//! | 28 | `terminal_record_omits_absent_stop_sequence` | serde of the record itself | `None` | unit |
//!
//! Cells 27–28 are unit tests because they are about the *type's* serde
//! contract, not about anything a provider turn can vary: no recording can
//! witness whether `stop_sequence` is skipped when `None`.
//!
//! Cells 25–26 assert the adjacent surfaces that share the same terminal
//! construction still behave: rig's normalized [`StreamFinal`] deliberately has
//! no `stop_sequence` (it is provider-specific and lives on the raw record), so
//! those cells pin the normalized shape rather than the new field.

use futures::StreamExt;
use rig::completion::CompletionModel as _;
use rig::completion::{CompletionRequest, FinishReason, ToolDefinition};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::streaming::StreamingCompletionResponse;
use rig::streaming::{RawStreamingChoice, StreamingPrompt};
use serde_json::json;

use super::super::support::with_anthropic_stop_sequence_cassette;

type AnthropicModel = anthropic::completion::CompletionModel;

/// Emits `alpha`, `bravo`, `charlie`, `delta` on separate lines, so a stop
/// sequence naming any of them cuts the turn at a known point.
const LIST_PROMPT: &str =
    "Repeat exactly these four words, one per line, and nothing else: alpha bravo charlie delta";
/// The same four words on one line, so a sequence may span a space.
const LINE_PROMPT: &str =
    "Reply with exactly this single line and nothing else: alpha bravo charlie delta";
const UNICODE_PROMPT: &str = "Reply with exactly this and nothing else: alpha 🌊 bravo";
const PUNCTUATION_PROMPT: &str = "Reply with exactly this and nothing else: alpha ### bravo";
const MIXED_CASE_PROMPT: &str = "Reply with exactly this and nothing else: alpha BRAVO charlie";
const IMMEDIATE_PROMPT: &str = "Reply with exactly this one word and nothing else: alpha";
const TOOL_PROMPT: &str = "What is the weather in Paris? Use the get_weather tool.";

fn request(
    model: &AnthropicModel,
    prompt: &str,
    stop_sequences: &[&str],
    max_tokens: u64,
) -> CompletionRequest {
    model
        .completion_request(prompt)
        .max_tokens(max_tokens)
        .additional_params(json!({ "stop_sequences": stop_sequences }))
        .build()
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

/// Drain a provider-native stream and return its terminal record.
async fn raw_terminal(
    model: &AnthropicModel,
    request: CompletionRequest,
) -> StreamingCompletionResponse {
    let mut stream = model
        .raw_stream(request)
        .await
        .expect("stop-sequence stream should open");
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let RawStreamingChoice::FinalResponse(record) =
            item.expect("stream item should not error")
        {
            terminal = Some(record);
        }
    }
    terminal.expect("stream should yield a terminal record")
}

/// Assert the terminal record a streamed cell produced.
///
/// The cell's *premise* — that the recorded `message_delta` really carried this
/// value — is asserted separately, after the cassette wrapper returns: in
/// record mode the fixture is written by `finish_after_test`, so an in-body
/// read would assert against the previous recording.
fn assert_terminal(
    terminal: &StreamingCompletionResponse,
    expected_sequence: Option<&str>,
    expected_reason: &str,
) {
    assert_eq!(
        terminal.stop_sequence.as_deref(),
        expected_sequence,
        "the terminal record must carry the sequence the wire reported"
    );
    assert_eq!(
        terminal.stop_reason.as_deref(),
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

/// The `stop_sequence` a blocking cassette's response body records.
fn recorded_response_stop_sequence(scenario: &str) -> Option<String> {
    let body = super::super::support::recorded_response_body(scenario);
    match body.get("stop_sequence") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(other) => panic!("{scenario}: stop_sequence should be a string or null, got {other}"),
    }
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

/// Whether a cassette's recorded *request* body contains `needle`.
///
/// Used to assert that a cell's request-side premise (a cache_control marker,
/// say) is really in the recording rather than only in the test's intent.
fn recorded_request_contains(scenario: &str, needle: &str) -> bool {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));
    // Fail closed: without a `then:` split this would search the *response*
    // half and could assert the opposite of its name.
    let (request, _) = contents.split_once("\nthen:\n").unwrap_or_else(|| {
        panic!(
            "cassette {} should have a `then:` response section",
            path.display()
        )
    });
    request.contains(needle)
}

fn assert_recorded_terminal_stop_sequence(scenario: &str, expected: Option<&str>) {
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

#[tokio::test]
async fn raw_single_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_single_sequence_fires",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, LIST_PROMPT, &["charlie"], 64)).await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_single_sequence_fires",
        Some("charlie"),
    );
}

#[tokio::test]
async fn raw_first_listed_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_first_listed_sequence_fires",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal = raw_terminal(
                &model,
                request(&model, LIST_PROMPT, &["charlie", "delta"], 64),
            )
            .await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_first_listed_sequence_fires",
        Some("charlie"),
    );
}

#[tokio::test]
async fn raw_second_listed_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_second_listed_sequence_fires",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal = raw_terminal(
                &model,
                request(&model, LIST_PROMPT, &["zulu", "charlie"], 64),
            )
            .await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_second_listed_sequence_fires",
        Some("charlie"),
    );
}

#[tokio::test]
async fn raw_earliest_in_output_wins() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_earliest_in_output_wins",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            // `charlie` is listed first but `bravo` appears first in the output:
            // the reported sequence must be the one that actually matched.
            let terminal = raw_terminal(
                &model,
                request(&model, LIST_PROMPT, &["charlie", "bravo"], 64),
            )
            .await;
            assert_terminal(&terminal, Some("bravo"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_earliest_in_output_wins",
        Some("bravo"),
    );
}

#[tokio::test]
async fn raw_four_sequences() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_four_sequences",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal = raw_terminal(
                &model,
                request(
                    &model,
                    LIST_PROMPT,
                    &["zulu", "yankee", "xray", "charlie"],
                    64,
                ),
            )
            .await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_four_sequences",
        Some("charlie"),
    );
}

// ---------------------------------------------------------------------------
// 6–10: sequence content shapes that must survive verbatim
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_newline_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_newline_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, LIST_PROMPT, &["\nbravo"], 64)).await;
            assert_terminal(&terminal, Some("\nbravo"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_newline_sequence",
        Some("\nbravo"),
    );
}

#[tokio::test]
async fn raw_whitespace_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_whitespace_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, LINE_PROMPT, &["bravo charlie"], 64)).await;
            assert_terminal(&terminal, Some("bravo charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_whitespace_sequence",
        Some("bravo charlie"),
    );
}

#[tokio::test]
async fn raw_unicode_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_unicode_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal = raw_terminal(&model, request(&model, UNICODE_PROMPT, &["🌊"], 64)).await;
            assert_terminal(&terminal, Some("🌊"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_unicode_sequence",
        Some("🌊"),
    );
}

#[tokio::test]
async fn raw_punctuation_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_punctuation_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, PUNCTUATION_PROMPT, &["###"], 64)).await;
            assert_terminal(&terminal, Some("###"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_punctuation_sequence",
        Some("###"),
    );
}

#[tokio::test]
async fn raw_case_sensitive_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_case_sensitive_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, MIXED_CASE_PROMPT, &["BRAVO"], 64)).await;
            assert_terminal(&terminal, Some("BRAVO"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_case_sensitive_sequence",
        Some("BRAVO"),
    );
}

// ---------------------------------------------------------------------------
// 11–14: controls — every other terminal must report no sequence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_case_mismatch_does_not_fire() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_case_mismatch_does_not_fire",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            // The output says `BRAVO`; a lowercase sequence must not match.
            let terminal =
                raw_terminal(&model, request(&model, MIXED_CASE_PROMPT, &["bravo"], 64)).await;
            assert_terminal(&terminal, None, "end_turn");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_case_mismatch_does_not_fire",
        None,
    );
}

#[tokio::test]
async fn raw_end_turn_control() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_end_turn_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal = raw_terminal(&model, request(&model, LIST_PROMPT, &["zulu"], 64)).await;
            assert_terminal(&terminal, None, "end_turn");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_end_turn_control",
        None,
    );
}

#[tokio::test]
async fn raw_max_tokens_control() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_max_tokens_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal = raw_terminal(&model, request(&model, LIST_PROMPT, &["zulu"], 3)).await;
            assert_terminal(&terminal, None, "max_tokens");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_max_tokens_control",
        None,
    );
}

#[tokio::test]
async fn raw_tool_use_control() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_tool_use_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = model
                .completion_request(TOOL_PROMPT)
                .max_tokens(256)
                .tool(weather_tool())
                .additional_params(json!({ "stop_sequences": ["zulu"] }))
                .build();
            let terminal = raw_terminal(&model, request).await;
            assert_terminal(&terminal, None, "tool_use");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_tool_use_control",
        None,
    );
}

// ---------------------------------------------------------------------------
// 15–20: interaction with the rest of the request surface
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_empty_content_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_empty_content_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, IMMEDIATE_PROMPT, &["alpha"], 32)).await;
            assert_terminal(&terminal, Some("alpha"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_empty_content_sequence",
        Some("alpha"),
    );
}

#[tokio::test]
async fn raw_with_tools_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_with_tools_sequence_fires",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            // Tools advertised but unused: the streaming body's `tool_choice`
            // reconciliation runs, and the terminal must be unaffected.
            let request = model
                .completion_request(LIST_PROMPT)
                .max_tokens(64)
                .tool(weather_tool())
                .additional_params(json!({ "stop_sequences": ["charlie"] }))
                .build();
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
async fn raw_with_prompt_caching_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_with_prompt_caching_sequence_fires",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_HAIKU_4_5)
                .with_prompt_caching();
            let terminal =
                raw_terminal(&model, request(&model, LIST_PROMPT, &["charlie"], 64)).await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    let scenario = "stop_sequence_terminal_matrix/raw_with_prompt_caching_sequence_fires";
    assert_recorded_terminal_stop_sequence(scenario, Some("charlie"));
    // The cell's other premise: prompt caching was actually engaged. Asserted
    // on the recorded *request* — `usage.cache_creation` is reported on every
    // `message_start` whether or not caching is on (the non-caching cells
    // record the same all-zero buckets), so reading it back would pass with
    // `with_prompt_caching()` removed and prove nothing.
    assert!(
        recorded_request_contains(scenario, "cache_control"),
        "this cell must record a request carrying a cache_control marker"
    );
}

#[tokio::test]
async fn raw_with_preamble_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_with_preamble_sequence_fires",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = model
                .completion_request(LIST_PROMPT)
                .preamble("You follow formatting instructions exactly.".to_string())
                .max_tokens(64)
                .additional_params(json!({ "stop_sequences": ["charlie"] }))
                .build();
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

#[tokio::test]
async fn raw_sonnet_sequence_fires() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_sonnet_sequence_fires",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let terminal =
                raw_terminal(&model, request(&model, LIST_PROMPT, &["charlie"], 64)).await;
            assert_terminal(&terminal, Some("charlie"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_sonnet_sequence_fires",
        Some("charlie"),
    );
}

#[tokio::test]
async fn raw_long_multi_token_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/raw_long_multi_token_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let terminal =
                raw_terminal(&model, request(&model, LINE_PROMPT, &["charlie delta"], 64)).await;
            assert_terminal(&terminal, Some("charlie delta"), "stop_sequence");
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/raw_long_multi_token_sequence",
        Some("charlie delta"),
    );
}

// ---------------------------------------------------------------------------
// 21–24: blocking twins — the surface that was already correct
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_single_sequence_parity() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/blocking_single_sequence_parity",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let raw = model
                .raw_completion(request(&model, LIST_PROMPT, &["charlie"], 64))
                .await
                .expect("blocking stop-sequence request should succeed");
            assert_eq!(raw.stop_reason.as_deref(), Some("stop_sequence"));
            assert_eq!(raw.stop_sequence.as_deref(), Some("charlie"));
        },
    )
    .await;

    assert_eq!(
        recorded_response_stop_sequence(
            "stop_sequence_terminal_matrix/blocking_single_sequence_parity"
        )
        .as_deref(),
        Some("charlie"),
        "the recorded blocking body must itself report the matched sequence"
    );
}

#[tokio::test]
async fn blocking_second_listed_sequence_parity() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/blocking_second_listed_sequence_parity",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let raw = model
                .raw_completion(request(&model, LIST_PROMPT, &["zulu", "charlie"], 64))
                .await
                .expect("blocking stop-sequence request should succeed");
            assert_eq!(raw.stop_sequence.as_deref(), Some("charlie"));
        },
    )
    .await;

    assert_eq!(
        recorded_response_stop_sequence(
            "stop_sequence_terminal_matrix/blocking_second_listed_sequence_parity"
        )
        .as_deref(),
        Some("charlie")
    );
}

#[tokio::test]
async fn blocking_end_turn_control() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/blocking_end_turn_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let raw = model
                .raw_completion(request(&model, LIST_PROMPT, &["zulu"], 64))
                .await
                .expect("blocking control request should succeed");
            assert_eq!(raw.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(raw.stop_sequence, None);
        },
    )
    .await;

    assert_eq!(
        recorded_response_stop_sequence("stop_sequence_terminal_matrix/blocking_end_turn_control"),
        None
    );
}

#[tokio::test]
async fn blocking_max_tokens_control() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/blocking_max_tokens_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let raw = model
                .raw_completion(request(&model, LIST_PROMPT, &["zulu"], 3))
                .await
                .expect("blocking control request should succeed");
            assert_eq!(raw.stop_reason.as_deref(), Some("max_tokens"));
            assert_eq!(raw.stop_sequence, None);
        },
    )
    .await;

    assert_eq!(
        recorded_response_stop_sequence(
            "stop_sequence_terminal_matrix/blocking_max_tokens_control"
        ),
        None
    );
}

// ---------------------------------------------------------------------------
// 25–26: adjacent paths sharing the terminal construction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn normalized_stream_single_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/normalized_stream_single_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let mut stream = rig::completion::CompletionModel::stream(
                &model,
                request(&model, LIST_PROMPT, &["charlie"], 64),
            )
            .await
            .expect("normalized stream should open");
            while stream.next().await.is_some() {}

            let final_record = stream
                .response
                .as_ref()
                .expect("normalized stream should carry a terminal");
            // rig's normalized terminal deliberately has no `stop_sequence`:
            // the field is provider-specific and lives on the raw record. What
            // must hold here is that a stop-sequence stop still normalizes as a
            // natural stop.
            assert_eq!(final_record.finish_reason, Some(FinishReason::Stop));
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/normalized_stream_single_sequence",
        Some("charlie"),
    );
}

#[tokio::test]
async fn agent_stream_single_sequence() {
    with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/agent_stream_single_sequence",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                .max_tokens(64)
                .additional_params(json!({ "stop_sequences": ["charlie"] }))
                .build();

            let mut stream = agent.stream_prompt(LIST_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                crate::support::collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("agent stream should succeed");

            assert_eq!(provider_final.finish_reason, Some(FinishReason::Stop));
        },
    )
    .await;

    assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/agent_stream_single_sequence",
        Some("charlie"),
    );
}

// ---------------------------------------------------------------------------
// 27–28: serde of the terminal record itself (no provider turn can vary this)
// ---------------------------------------------------------------------------

#[test]
fn terminal_record_round_trips_stop_sequence() {
    let record = StreamingCompletionResponse {
        stop_reason: Some("stop_sequence".to_string()),
        stop_sequence: Some("charlie".to_string()),
        ..Default::default()
    };

    let encoded = serde_json::to_value(&record).expect("terminal record should serialize");
    assert_eq!(encoded["stop_sequence"], json!("charlie"));

    let decoded: StreamingCompletionResponse =
        serde_json::from_value(encoded).expect("terminal record should round-trip");
    assert_eq!(decoded.stop_sequence.as_deref(), Some("charlie"));
}

#[test]
fn terminal_record_omits_absent_stop_sequence() {
    let record = StreamingCompletionResponse {
        stop_reason: Some("end_turn".to_string()),
        ..Default::default()
    };

    let encoded = serde_json::to_value(&record).expect("terminal record should serialize");
    assert!(
        encoded.get("stop_sequence").is_none(),
        "an absent sequence must not be written as an explicit null: {encoded}"
    );

    let decoded: StreamingCompletionResponse =
        serde_json::from_value(encoded).expect("terminal record should round-trip");
    assert_eq!(decoded.stop_sequence, None);
}
