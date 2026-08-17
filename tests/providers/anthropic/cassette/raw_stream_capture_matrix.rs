//! Matrix for raw provider response capture on the streaming path:
//! `StreamFinal::raw` beside the normalized terminal fields.
//!
//! # The feature
//!
//! Capture is always on. `stream()` opens `raw_stream` and hands it to
//! `normalize_stream`, which serializes the provider-native terminal —
//! `anthropic::streaming::StreamingCompletionResponse`, the `R` of
//! `raw_stream` — onto the terminal `StreamFinal::raw` before mapping it. So
//! `raw` is the **terminal record only**: what `raw_stream` would have yielded
//! as its `FinalResponse`, not the stream's frames. Anthropic's terminal is
//! assembled from `message_start` (id, model) and the closing `message_delta`
//! (`stop_reason`, `stop_sequence`, usage), plus the transport `request-id`
//! header the driver stamps. `raw` is `Value::Null` only on a `StreamFinal`
//! built by hand, with no provider terminal behind it; `Value::Null` never
//! means "not requested". Cells 1–3 pin the terminal round trip, a
//! terminal-only field, and normalized/raw agreement on text streams; cells 4
//! and 5 repeat the round trip on an extended-thinking stream and a forced
//! tool-call stream.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `terminal_raw_round_trips_into_provider_type` | plain text request | terminal `raw` populated; deserializes into the Anthropic terminal type and re-serializes equal; terminal-only shape (no frames) | recorded |
//! | 2 | `raw_exposes_stop_sequence` | streamed twin of the `stop_sequences: ["alpha"]` request | `raw["stop_sequence"] == "alpha"`, `raw["stop_reason"] == "stop_sequence"` (verbatim spelling) | recorded |
//! | 3 | `normalized_terminal_matches_raw_renormalized` | plain text request | `StreamFinal::from(("anthropic", StreamingCompletionResponse::deserialize(raw)))` reproduces `identity()`, `finish_reason`, `model`, `usage` | recorded |
//! | 4 | `terminal_raw_round_trips_for_thinking_stream` | streamed twin of the `thinking.enabled` request | terminal `raw` round-trips; `raw["usage"]["output_tokens_details"]["thinking_tokens"]` is the terminal `message_delta`'s, verbatim; normalized terminal folds it into `reasoning_tokens` and never spells `thinking`; the stream yielded the `thinking` block as `Reasoning` | recorded |
//! | 5 | `terminal_raw_round_trips_for_tool_use_stream` | streamed twin of the forced tool call | terminal `raw` round-trips; `raw["stop_reason"] == "tool_use"` verbatim; normalized terminal `finish_reason == ToolCalls`; the stream yielded the `tool_use` block as a `ToolCall` whose provider id is the frame's | recorded |
//!
//! Every recorded cell re-derives its premise from its own SSE frames: the
//! stream opens with a `message_start` naming a `msg_…` id, closes with a
//! `message_delta` carrying `usage`, and the response carries a `request-id`
//! header. Cell 3 is not cell 1 restated: cell 1 proves `raw` is lossless
//! against the *provider* terminal type; cell 3 proves rig's own mapping of
//! that value agrees with the normalized terminal delivered beside it — the
//! single-stream form of the parity contract `raw_completion_parity_matrix.rs`
//! records across two exchanges. Cells 4 and 5 take the terminal round trip
//! off the text-only path, with the request shapes their blocking twins in
//! `raw_capture_matrix.rs` use (the `thinking.enabled` request from
//! `reasoning_usage_matrix.rs`; the `weather_tool` + `tool_choice` pattern
//! from `empty_stop_sequence_matrix.rs`). Because `raw` is the terminal
//! record only, the reasoning block and the tool-call block themselves are
//! not in it — the frames are; so each cell premise-asserts its frames carry
//! the block (`content_block_start` of `type: "thinking"` with a
//! `signature_delta`; `content_block_start` of `type: "tool_use"`), and pins
//! what the terminal *does* carry: the usage bucket and the verbatim
//! `stop_reason`.

use futures::StreamExt;
use rig::completion::{CompletionModel as _, FinishReason, ToolDefinition};
use rig::message::{ReasoningContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::streaming::StreamingCompletionResponse;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{
    assert_ids_match_recording, recorded_request_id_headers, with_anthropic_cassette,
};

const ANTHROPIC_PROVIDER: &str = "anthropic";
const PROMPT: &str = "Reply with exactly: raw stream capture probe";
/// From `empty_stop_sequence_matrix.rs`: one word, so the `alpha` sequence
/// matches and Anthropic names it on the terminal `message_delta`.
const IMMEDIATE_PROMPT: &str = "Reply with exactly this one word and nothing else: alpha";
/// From `reasoning_usage_matrix.rs`: costs a few thinking tokens without a
/// long answer.
const THINKING_PROMPT: &str = "What is 17 * 23? Reply with just the number.";
/// A question the forced tool answers, so the `tool_use` block's `input`
/// carries a real `city`.
const TOOL_PROMPT: &str = "What is the weather in Paris right now?";

const ROUND_TRIP_SCENARIO: &str =
    "raw_stream_capture_matrix/terminal_raw_round_trips_into_provider_type";
const STOP_SEQUENCE_SCENARIO: &str = "raw_stream_capture_matrix/raw_exposes_stop_sequence";
const RENORMALIZED_SCENARIO: &str =
    "raw_stream_capture_matrix/normalized_terminal_matches_raw_renormalized";
const THINKING_SCENARIO: &str =
    "raw_stream_capture_matrix/terminal_raw_round_trips_for_thinking_stream";
const TOOL_USE_SCENARIO: &str =
    "raw_stream_capture_matrix/terminal_raw_round_trips_for_tool_use_stream";

type AnthropicModel = anthropic::completion::CompletionModel;

fn probe_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(32).build()
}

/// From `reasoning_usage_matrix.rs`: extended thinking on, minimum budget,
/// `max_tokens` above it as Anthropic requires.
fn thinking_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(THINKING_PROMPT)
        .max_tokens(2048)
        .additional_params(json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
        .build()
}

/// From `empty_stop_sequence_matrix.rs`.
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

/// `tool_choice: required` (Anthropic `any`) so the turn is a `tool_use`
/// terminal by construction, not by the model's mood.
fn tool_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .max_tokens(256)
        .tool(weather_tool())
        .tool_choice(ToolChoice::Required)
        .build()
}

/// What a cell observed on the stream: every non-terminal item, and the
/// terminal record.
struct Streamed {
    items: Vec<StreamedAssistantContent>,
    terminal: StreamFinal,
}

type TerminalSink = std::sync::Arc<std::sync::Mutex<Option<StreamFinal>>>;
type StreamedSink = std::sync::Arc<std::sync::Mutex<Option<Streamed>>>;

async fn drain_stream(mut stream: rig::streaming::StreamingCompletionResponse) -> Streamed {
    let mut items = Vec::new();
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should succeed") {
            StreamedAssistantContent::Final(final_record) => terminal = Some(final_record),
            item => items.push(item),
        }
    }
    Streamed {
        items,
        terminal: terminal.expect("the stream should yield a terminal record"),
    }
}

async fn drain_terminal(stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    drain_stream(stream).await.terminal
}

/// The body of cells 1–3: open the stream the cell's request describes and
/// keep its terminal record for the assertions that run after the wrapper has
/// written the fixture.
async fn probe_body(
    client: anthropic::Client,
    build: impl FnOnce(&AnthropicModel) -> rig::completion::CompletionRequest,
    sink: TerminalSink,
) {
    let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
    let stream = model
        .stream(build(&model))
        .await
        .expect("stream should open");
    *sink.lock().expect("sink") = Some(drain_terminal(stream).await);
}

/// The body of cells 4 and 5: the same, on the model the cell names, keeping
/// the streamed items as well as the terminal.
async fn streamed_body(
    client: anthropic::Client,
    model_name: &str,
    build: impl FnOnce(&AnthropicModel) -> rig::completion::CompletionRequest,
    sink: StreamedSink,
) {
    let model = client.completion_model(model_name);
    let stream = model
        .stream(build(&model))
        .await
        .expect("stream should open");
    *sink.lock().expect("sink") = Some(drain_stream(stream).await);
}

fn take_terminal(sink: &TerminalSink) -> StreamFinal {
    let terminal = sink.lock().expect("sink").take();
    terminal.expect("the cell body ran")
}

fn take_streamed(sink: &StreamedSink) -> Streamed {
    let streamed = sink.lock().expect("sink").take();
    streamed.expect("the cell body ran")
}

/// Cells 4 and 5 share this: terminal `raw` is populated, and reads back into
/// the provider terminal type without loss.
fn assert_raw_round_trips(raw: &Value) -> StreamingCompletionResponse {
    assert!(
        !raw.is_null(),
        "every terminal `stream()` yields carries `raw`"
    );
    let typed = StreamingCompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::streaming::StreamingCompletionResponse");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "raw ↔ typed must round-trip without loss"
    );
    typed
}

/// Every string a JSON value contains — object keys and string leaves — so a
/// cell can prove a wire spelling (`"thinking"`, `"tool_use"`) is absent from
/// the normalized terminal *anywhere*, not just at one path.
fn collect_strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
    match value {
        Value::String(text) => out.push(text.as_str()),
        Value::Array(items) => items.iter().for_each(|item| collect_strings(item, out)),
        Value::Object(fields) => {
            for (key, field) in fields {
                out.push(key.as_str());
                collect_strings(field, out);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn contains_string(value: &Value, needle: &str) -> bool {
    let mut strings = Vec::new();
    collect_strings(value, &mut strings);
    strings.contains(&needle)
}

/// The normalized terminal, serialized, minus `raw` — what a cell inspects to
/// prove a wire spelling is not part of the normalized vocabulary.
fn normalized_without_raw(terminal: &StreamFinal) -> Value {
    let mut normalized = serde_json::to_value(terminal).expect("terminal serializes");
    normalized
        .as_object_mut()
        .expect("the terminal serializes as an object")
        .remove("raw")
        .expect("the terminal carries `raw`");
    normalized
}

/// The premise every cell rests on, read from its own frames: `message_start`
/// names the id and model, the terminal `message_delta` names the stop reason
/// and carries the final usage, and the response carried a `request-id`.
struct RecordedStream {
    message_id: Option<String>,
    model: Option<String>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    output_tokens: u64,
    input_tokens: u64,
    request_id: Option<String>,
    frame_types: Vec<String>,
}

fn recorded_frames(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames(ANTHROPIC_PROVIDER, scenario)
}

fn recorded_stream(scenario: &str) -> RecordedStream {
    let frames = recorded_frames(scenario);
    let start = frames
        .iter()
        .find(|frame| frame["type"] == "message_start")
        .unwrap_or_else(|| panic!("{scenario}: premise — the stream opens with message_start"));
    let delta = frames
        .iter()
        .find(|frame| frame["type"] == "message_delta")
        .unwrap_or_else(|| panic!("{scenario}: premise — the stream closes with message_delta"));
    let request_ids = recorded_request_id_headers(scenario);
    assert_eq!(request_ids.len(), 1, "{scenario}: one recorded interaction");
    RecordedStream {
        message_id: start["message"]["id"].as_str().map(str::to_string),
        model: start["message"]["model"].as_str().map(str::to_string),
        stop_reason: delta["delta"]["stop_reason"].as_str().map(str::to_string),
        stop_sequence: delta["delta"]["stop_sequence"].as_str().map(str::to_string),
        output_tokens: delta["usage"]["output_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{scenario}: premise — terminal frame carries usage")),
        input_tokens: delta["usage"]["input_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{scenario}: premise — terminal frame carries usage")),
        request_id: request_ids[0].clone(),
        frame_types: frames
            .iter()
            .filter_map(|frame| frame["type"].as_str().map(str::to_string))
            .collect(),
    }
}

/// The normalized terminal reports what its own recording says.
fn assert_terminal_matches_fixture(
    scenario: &str,
    terminal: &StreamFinal,
    recorded: &RecordedStream,
) {
    assert!(
        recorded
            .message_id
            .as_deref()
            .is_some_and(|id| id.starts_with("msg_")),
        "{scenario}: premise — message_start names a msg_ id"
    );
    assert!(
        recorded.request_id.is_some(),
        "{scenario}: premise — the response carries a request-id header"
    );
    assert_ids_match_recording(
        std::slice::from_ref(&terminal.message_id),
        std::slice::from_ref(&recorded.message_id),
        scenario,
    );
    assert_ids_match_recording(
        std::slice::from_ref(&terminal.provider_request_id),
        std::slice::from_ref(&recorded.request_id),
        scenario,
    );
    assert_eq!(terminal.model, recorded.model);
    assert_eq!(terminal.usage.input_tokens, recorded.input_tokens);
    assert_eq!(terminal.usage.output_tokens, recorded.output_tokens);
    assert_eq!(terminal.provider, ANTHROPIC_PROVIDER);
}

// ---------------------------------------------------------------------------
// 1: typed round trip, terminal-only shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminal_raw_round_trips_into_provider_type() {
    let sink = TerminalSink::default();
    with_anthropic_cassette(
        "raw_stream_capture_matrix/terminal_raw_round_trips_into_provider_type",
        {
            let sink = sink.clone();
            move |client| probe_body(client, probe_request, sink)
        },
    )
    .await;
    let terminal = take_terminal(&sink);
    let raw: &Value = &terminal.raw;
    assert!(
        !raw.is_null(),
        "every terminal `stream()` yields carries `raw`"
    );

    // Typed access is recoverable and lossless.
    let typed = StreamingCompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::streaming::StreamingCompletionResponse");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "raw ↔ typed must round-trip without loss"
    );

    // Terminal record only. Pin the exact key set: the fields the Anthropic
    // terminal type carries for an `end_turn` text stream (no
    // `stop_sequence`, which is skipped when absent) — and nothing frame-shaped.
    let mut keys: Vec<&str> = raw
        .as_object()
        .expect("terminal raw is an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "message_id",
            "model",
            "provider_request_id",
            "stop_reason",
            "usage"
        ],
        "the terminal record's own fields, and only those"
    );
    for frame_key in [
        "content_block_delta",
        "content_block_start",
        "content",
        "delta",
        "type",
    ] {
        assert!(
            raw.get(frame_key).is_none(),
            "`raw` is the terminal record, not the frames: found `{frame_key}`"
        );
    }

    // Wire-derived fields equal what the recorded frames say; the transport
    // id is the header the driver stamped.
    let recorded = recorded_stream(ROUND_TRIP_SCENARIO);
    assert!(
        recorded
            .frame_types
            .iter()
            .any(|kind| kind == "content_block_delta"),
        "premise: the recorded stream did carry frames `raw` must not contain"
    );
    assert_ids_match_recording(
        &[raw["message_id"].as_str().map(str::to_string)],
        std::slice::from_ref(&recorded.message_id),
        ROUND_TRIP_SCENARIO,
    );
    assert_ids_match_recording(
        &[raw["provider_request_id"].as_str().map(str::to_string)],
        std::slice::from_ref(&recorded.request_id),
        ROUND_TRIP_SCENARIO,
    );
    assert_eq!(raw["model"].as_str(), recorded.model.as_deref());
    assert_eq!(raw["stop_reason"].as_str(), recorded.stop_reason.as_deref());
    assert_eq!(recorded.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(raw["usage"]["output_tokens"], json!(recorded.output_tokens));
    assert_eq!(raw["usage"]["input_tokens"], json!(recorded.input_tokens));
    assert_eq!(typed.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(typed.usage.output_tokens as u64, recorded.output_tokens);

    // The normalized view beside it reports what the fixture recorded.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_terminal_matches_fixture(ROUND_TRIP_SCENARIO, &terminal, &recorded);
}

// ---------------------------------------------------------------------------
// 2: a terminal-only field, verbatim
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_stop_sequence() {
    let sink = TerminalSink::default();
    with_anthropic_cassette("raw_stream_capture_matrix/raw_exposes_stop_sequence", {
        let sink = sink.clone();
        move |client| {
            probe_body(
                client,
                |model| {
                    model
                        .completion_request(IMMEDIATE_PROMPT)
                        .max_tokens(32)
                        .additional_params(json!({ "stop_sequences": ["alpha"] }))
                        .build()
                },
                sink,
            )
        }
    })
    .await;
    let terminal = take_terminal(&sink);
    let raw: &Value = &terminal.raw;
    assert!(
        !raw.is_null(),
        "every terminal `stream()` yields carries `raw`"
    );

    // Premise from the frames: the terminal `message_delta` stopped on the
    // sequence and named it.
    let recorded = recorded_stream(STOP_SEQUENCE_SCENARIO);
    assert_eq!(
        recorded.stop_reason.as_deref(),
        Some("stop_sequence"),
        "premise: the recorded stream stopped on a sequence"
    );
    assert_eq!(
        recorded.stop_sequence.as_deref(),
        Some("alpha"),
        "premise: the recorded terminal names the sequence"
    );

    // Normalized: folded into `Stop`; the provider's spelling and the
    // sequence itself are only on `raw`.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    let normalized = serde_json::to_value(&terminal).expect("terminal serializes");
    let normalized_keys: Vec<&str> = normalized
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .filter(|key| *key != "raw")
        .collect();
    assert!(
        !normalized_keys.contains(&"stop_sequence") && !normalized_keys.contains(&"stop_reason"),
        "the normalized terminal has neither `stop_sequence` nor a verbatim `stop_reason` \
         ({normalized_keys:?}) — `raw` is the only way to read them"
    );
    assert_eq!(raw["stop_reason"], "stop_sequence");
    assert_eq!(raw["stop_sequence"], "alpha");
    let typed = StreamingCompletionResponse::deserialize(raw).expect("typed access");
    assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
    assert_eq!(typed.stop_reason.as_deref(), Some("stop_sequence"));
    assert_terminal_matches_fixture(STOP_SEQUENCE_SCENARIO, &terminal, &recorded);
}

// ---------------------------------------------------------------------------
// 3: raw and the normalized terminal tell one story
// ---------------------------------------------------------------------------

/// The normalized terminal and `raw` describe the same stream: reading `raw`
/// back into the provider terminal type and mapping it through the public
/// `StreamFinal::from((&str, StreamingCompletionResponse))` — the same
/// mapping `stream()` applies — reproduces every normalized field delivered
/// beside it: identity, finish reason, model, usage. And each of those is
/// what the fixture recorded.
#[tokio::test]
async fn normalized_terminal_matches_raw_renormalized() {
    let sink = TerminalSink::default();
    with_anthropic_cassette(
        "raw_stream_capture_matrix/normalized_terminal_matches_raw_renormalized",
        {
            let sink = sink.clone();
            move |client| probe_body(client, probe_request, sink)
        },
    )
    .await;
    let terminal = take_terminal(&sink);
    let raw: &Value = &terminal.raw;
    assert!(
        !raw.is_null(),
        "every terminal `stream()` yields carries `raw`"
    );

    let typed = StreamingCompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::streaming::StreamingCompletionResponse");
    let renormalized = StreamFinal::from((ANTHROPIC_PROVIDER, typed));
    assert_eq!(
        renormalized.identity(),
        terminal.identity(),
        "identity (message id, transport id) survives raw → typed → StreamFinal"
    );
    assert_eq!(renormalized.finish_reason, terminal.finish_reason);
    assert_eq!(renormalized.model, terminal.model);
    assert_eq!(renormalized.usage, terminal.usage);
    assert_eq!(renormalized.provider, terminal.provider);

    // …and none of that is vacuous: the normalized terminal is the fixture's.
    let recorded = recorded_stream(RENORMALIZED_SCENARIO);
    assert_eq!(recorded.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_terminal_matches_fixture(RENORMALIZED_SCENARIO, &terminal, &recorded);
}

// ---------------------------------------------------------------------------
// 4: an extended-thinking stream — terminal round trip beside a thinking block
// ---------------------------------------------------------------------------

/// The streamed twin of `raw_capture_matrix::raw_exposes_thinking_block_and_signature`.
/// `raw` is the terminal record, so the thinking block itself is not in it —
/// it was streamed as frames and delivered as `Reasoning` items. What the
/// terminal does carry from the thinking turn is the usage bucket, and `raw`
/// carries it in the provider's spelling: `usage.output_tokens_details.thinking_tokens`,
/// verbatim from the terminal `message_delta`. The normalized terminal folds
/// it into `reasoning_tokens` and never spells `thinking`.
#[tokio::test]
async fn terminal_raw_round_trips_for_thinking_stream() {
    let sink = StreamedSink::default();
    with_anthropic_cassette(
        "raw_stream_capture_matrix/terminal_raw_round_trips_for_thinking_stream",
        {
            let sink = sink.clone();
            move |client| {
                streamed_body(
                    client,
                    anthropic::completion::CLAUDE_SONNET_4_6,
                    thinking_request,
                    sink,
                )
            }
        },
    )
    .await;
    let Streamed { items, terminal } = take_streamed(&sink);
    let raw: &Value = &terminal.raw;
    let typed = assert_raw_round_trips(raw);

    // Premise, from the frames: the stream opened a `thinking` content block,
    // signed it, and its terminal `message_delta` reports the thinking bucket.
    let frames = recorded_frames(THINKING_SCENARIO);
    assert!(
        frames.iter().any(|frame| {
            frame["type"] == "content_block_start" && frame["content_block"]["type"] == "thinking"
        }),
        "premise: the recorded stream opened a `thinking` content block"
    );
    let recorded_thinking_text = frames
        .iter()
        .filter(|frame| {
            frame["type"] == "content_block_delta" && frame["delta"]["type"] == "thinking_delta"
        })
        .filter_map(|frame| frame["delta"]["thinking"].as_str())
        .collect::<String>();
    assert!(
        !recorded_thinking_text.is_empty(),
        "premise: the recorded stream carried `thinking_delta` text"
    );
    let recorded_signature = frames
        .iter()
        .find(|frame| {
            frame["type"] == "content_block_delta" && frame["delta"]["type"] == "signature_delta"
        })
        .and_then(|frame| frame["delta"]["signature"].as_str())
        .map(str::to_string);
    assert!(
        recorded_signature
            .as_deref()
            .is_some_and(|sig| !sig.is_empty()),
        "premise: the recorded stream signed the thinking block (`signature_delta`)"
    );
    let recorded = recorded_stream(THINKING_SCENARIO);
    let recorded_thinking_tokens = frames
        .iter()
        .find(|frame| frame["type"] == "message_delta")
        .and_then(|frame| frame["usage"]["output_tokens_details"]["thinking_tokens"].as_u64())
        .expect(
            "premise: the terminal `message_delta` carries `output_tokens_details.thinking_tokens`",
        );
    assert!(
        recorded_thinking_tokens > 0,
        "premise: the recorded turn actually spent thinking tokens"
    );
    assert_eq!(recorded.stop_reason.as_deref(), Some("end_turn"), "premise");

    // `raw` carries the provider's usage breakdown, verbatim.
    assert_eq!(
        raw["usage"]["output_tokens_details"]["thinking_tokens"],
        json!(recorded_thinking_tokens),
        "the terminal `raw` carries `thinking_tokens` as the wire spelled it"
    );
    assert_eq!(
        typed
            .usage
            .output_tokens_details
            .map(|details| details.thinking_tokens),
        Some(recorded_thinking_tokens)
    );
    assert_eq!(raw["stop_reason"], "end_turn");

    // The normalized terminal folds it into `reasoning_tokens` and never
    // spells `thinking` — only `raw` does.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.usage.reasoning_tokens, recorded_thinking_tokens);
    let normalized = normalized_without_raw(&terminal);
    assert!(
        !contains_string(&normalized, "thinking")
            && !contains_string(&normalized, "thinking_tokens")
            && !contains_string(&normalized, "output_tokens_details"),
        "the normalized terminal never spells `thinking` / `output_tokens_details`: {normalized}"
    );
    assert_terminal_matches_fixture(THINKING_SCENARIO, &terminal, &recorded);

    // And the block `raw` does not carry was delivered as normalized items:
    // the stream's `Reasoning` text is the frames' `thinking_delta` text.
    let streamed_reasoning = items
        .iter()
        .filter_map(|item| match item {
            StreamedAssistantContent::Reasoning { reasoning, .. } => Some(reasoning),
            _ => None,
        })
        .flat_map(|reasoning| reasoning.content.iter())
        .filter_map(|content| match content {
            ReasoningContent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(
        streamed_reasoning, recorded_thinking_text,
        "the stream delivered the thinking block as `Reasoning`"
    );
}

// ---------------------------------------------------------------------------
// 5: a forced tool-call stream — terminal round trip beside a tool_use block
// ---------------------------------------------------------------------------

/// The streamed twin of `raw_capture_matrix::raw_exposes_tool_use_block`. The
/// terminal `raw` round-trips and names the stop reason as the wire spelled
/// it — `tool_use` — while the normalized terminal reports
/// `FinishReason::ToolCalls`; the `tool_use` block itself was streamed as
/// frames and delivered as a `ToolCall` item whose provider id is the frame's.
#[tokio::test]
async fn terminal_raw_round_trips_for_tool_use_stream() {
    let sink = StreamedSink::default();
    with_anthropic_cassette(
        "raw_stream_capture_matrix/terminal_raw_round_trips_for_tool_use_stream",
        {
            let sink = sink.clone();
            move |client| {
                streamed_body(
                    client,
                    anthropic::completion::CLAUDE_HAIKU_4_5,
                    tool_request,
                    sink,
                )
            }
        },
    )
    .await;
    let Streamed { items, terminal } = take_streamed(&sink);
    let raw: &Value = &terminal.raw;
    let typed = assert_raw_round_trips(raw);

    // Premise, from the frames: the stream opened a `tool_use` block for the
    // forced tool and its terminal `message_delta` stopped on `tool_use`.
    let frames = recorded_frames(TOOL_USE_SCENARIO);
    let recorded_tool_use = frames
        .iter()
        .find(|frame| {
            frame["type"] == "content_block_start" && frame["content_block"]["type"] == "tool_use"
        })
        .map(|frame| &frame["content_block"])
        .expect("premise: the recorded stream opened a `tool_use` content block");
    assert_eq!(recorded_tool_use["name"], "get_weather", "premise");
    let recorded_tool_id = recorded_tool_use["id"].as_str().map(str::to_string);
    assert!(
        recorded_tool_id
            .as_deref()
            .is_some_and(|id| id.starts_with("toolu_")),
        "premise: the recorded `tool_use` block has a `toolu_…` id"
    );
    let recorded_input: Value = serde_json::from_str(
        &frames
            .iter()
            .filter(|frame| {
                frame["type"] == "content_block_delta"
                    && frame["delta"]["type"] == "input_json_delta"
            })
            .filter_map(|frame| frame["delta"]["partial_json"].as_str())
            .collect::<String>(),
    )
    .expect("premise: the recorded `input_json_delta` frames assemble into JSON");
    assert!(
        recorded_input["city"].is_string(),
        "premise: the recorded input names a `city`: {recorded_input}"
    );
    let recorded = recorded_stream(TOOL_USE_SCENARIO);
    assert_eq!(
        recorded.stop_reason.as_deref(),
        Some("tool_use"),
        "premise: the recorded stream stopped to call a tool"
    );

    // `raw` names the stop reason as the wire spelled it; the normalized
    // terminal maps it onto `ToolCalls` and never spells `tool_use`.
    assert_eq!(raw["stop_reason"], "tool_use");
    assert_eq!(typed.stop_reason.as_deref(), Some("tool_use"));
    assert_eq!(terminal.finish_reason, Some(FinishReason::ToolCalls));
    let normalized = normalized_without_raw(&terminal);
    assert!(
        !contains_string(&normalized, "tool_use") && !contains_string(&normalized, "stop_reason"),
        "the normalized terminal has neither the spelling `tool_use` nor a verbatim \
         `stop_reason` — `raw` is the only way to read them: {normalized}"
    );
    assert_terminal_matches_fixture(TOOL_USE_SCENARIO, &terminal, &recorded);

    // And the block `raw` does not carry was delivered as a normalized item:
    // one `ToolCall` for the forced tool, its arguments the assembled input,
    // its provider id the frame's `toolu_…`.
    let streamed_calls = items
        .iter()
        .filter_map(|item| match item {
            StreamedAssistantContent::ToolCall { tool_call, .. } => Some(tool_call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        streamed_calls.len(),
        1,
        "the stream delivered exactly one tool call: {streamed_calls:?}"
    );
    let call = streamed_calls[0];
    assert_eq!(call.function.name, "get_weather");
    assert_eq!(call.function.arguments, recorded_input);
    assert_ids_match_recording(
        &[call
            .provider
            .as_ref()
            .map(|provider| provider.call_id.clone())],
        std::slice::from_ref(&recorded_tool_id),
        TOOL_USE_SCENARIO,
    );
}
