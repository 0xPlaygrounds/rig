//! Feature matrix for raw provider response capture on the Gemini REST
//! (`streamGenerateContent?alt=sse`) streaming seam.
//!
//! # The feature
//!
//! Raw capture is always on: the GenerateContent decoder builds its terminal
//! record at EOF — a JSON object assembled from the stream's last
//! `finishReason`, usage and metadata — and puts it onto the terminal
//! [`rig::completion::CompletionResponse::raw`]. There is no opt-in and nothing
//! about it reaches the wire; `raw` is `Value::Null` only on a terminal
//! constructed without a provider stream behind it, never because capture
//! "was not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the terminal record. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded stream must end with a frame carrying
//! `usageMetadata` and a natural `finishReason`, or the terminal it asserts on
//! is not the one this matrix is about.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 3 | `raw_terminal_keeps_stop_on_forced_function_call` | forced tool call (`ToolChoice::Specific`), streamed | terminal `raw` is the record; raw `finish_reason` spelled `"STOP"` and `finish_message` == wire while the normalized terminal reports `ToolCalls`; the recorded frames carry `functionCall` | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `streamGenerateContent` route.
//!
//! Gemini's terminal record is assembled by the decoder from the stream's
//! frames (usage is cumulative per chunk; `finishReason` arrives on the last
//! content frame), so the "wire" side of each premise is the last frame that
//! carries `usageMetadata`, read with the same `data:` framing the streaming
//! tests use.
//!
//! Cell 3 is the tool-turn twin: Gemini spells a call-only turn's finish
//! `"STOP"` on the wire and the decoder reconciles the terminal to
//! `ToolCalls` from the tool call it saw, so this is the one place the
//! terminal `raw` and the normalized terminal legitimately disagree — `raw`
//! must keep the wire spelling and the terminal must report the upgrade.

use futures::StreamExt;
use rig::completion::FinishReason;
use rig::message::{AssistantContent, ToolCall, ToolChoice};
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use rig::tool::Tool;
use serde_json::Value;

use super::super::support::with_gemini_cassette;
use crate::support::{Adder, Observed};
use rig::completion::CompletionRequest;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded stream stays short.
const MODEL: &str = "gemini-2.5-flash-lite";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded stream carries a `functionCall`.
fn forced_tool_request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .temperature(0.0)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new(Adder::NAME).expect("tool name")],
        })
}

/// What a drained model stream carried: its tool calls and its single
/// terminal record.
struct Drained {
    tool_calls: Vec<ToolCall>,
    terminal: rig::completion::CompletionResponse,
}

/// Drain a model stream, keeping every tool call it yielded.
///
/// Local rather than one of the shared `raw_capture::capture_*` stream
/// helpers: cell 3 is about a streamed `functionCall`, so it needs the tool
/// calls the stream yielded *beside* the terminal record, which no shared
/// helper parks. The sole-terminal claim it makes is the same one
/// [`capture_text_and_terminal`](crate::raw_capture::capture_text_and_terminal)
/// makes for this file's text-only cells.
async fn drain_stream<
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
>(
    model: &rig_core::driver::Model<W, T>,
    request: rig::completion::CompletionRequest,
) -> Drained {
    let mut stream = model.stream(request).expect("stream should open");
    let mut tool_calls = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            content: AssistantContent::ToolCall(tool_call),
            ..
        }) = item.expect("stream item should succeed")
        {
            tool_calls.push(tool_call)
        }
    }
    let terminal = stream
        .finish()
        .await
        .expect("stream should yield a terminal record");
    Drained {
        tool_calls,
        terminal,
    }
}

/// The premise of the forced-tool cell: the recorded stream carries a
/// `functionCall` part naming `add`, and its finish is still spelled `"STOP"`
/// — the wire shape the adapter's terminal mapping reconciles to `ToolCalls`. Returns the
/// last frame carrying `usageMetadata`, as [`last_usage_frame`] does.
fn last_usage_frame_of_function_call_stream(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let called_add = frames.iter().any(|frame| {
        frame
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts.iter().any(|part| {
                    part.pointer("/functionCall/name") == Some(&Value::String("add".into()))
                })
            })
    });
    assert!(
        called_add,
        "{scenario}: no recorded frame carries a functionCall part naming `add`, so this cell \
         does not exercise the tool-turn shape it claims to cover"
    );
    let stopped = frames.iter().any(|frame| {
        frame.pointer("/candidates/0/finishReason") == Some(&Value::String("STOP".to_string()))
    });
    assert!(
        stopped,
        "{scenario}: Gemini spells a call-only turn's finishReason STOP; this cell exists to \
         show the terminal raw keeps that spelling while the normalized terminal reports \
         ToolCalls"
    );
    frames
        .iter()
        .rev()
        .find(|frame| frame.get("usageMetadata").is_some())
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: no recorded frame carries usageMetadata"))
}

// ---------------------------------------------------------------------------
// 1: the terminal record is recoverable
// ---------------------------------------------------------------------------

/// The fields of the terminal record the decoder builds at EOF.
const TERMINAL_RECORD_FIELDS: &[&str] = &[
    "usage_metadata",
    "finish_reason",
    "finish_message",
    "model_version",
    "response_id",
];

/// Assert `raw` is the decoder's terminal record: an object holding its
/// usage and only the record's fields. Return its total token count.
fn terminal_record_total_tokens(raw: &Value) -> Option<u64> {
    let record = raw
        .as_object()
        .expect("raw must be Gemini's streaming terminal record");
    for key in record.keys() {
        assert!(
            TERMINAL_RECORD_FIELDS.contains(&key.as_str()),
            "the terminal record carries only its own fields, found `{key}` in {raw}"
        );
    }
    for key in [
        "finish_message",
        "model_version",
        "response_id",
        "finish_reason",
    ] {
        assert!(
            record.get(key).is_none_or(Value::is_string),
            "the terminal record's `{key}` is a string: {raw}"
        );
    }
    let usage = record
        .get("usage_metadata")
        .and_then(Value::as_object)
        .expect("the terminal record carries its usage_metadata");
    Some(
        usage
            .get("totalTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
    )
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields are readable and match the wire
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 3: a forced tool call keeps the wire's STOP while the terminal says ToolCalls
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_terminal_keeps_stop_on_forced_function_call() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/raw_terminal_keeps_stop_on_forced_function_call";
    let observed: Observed<Drained> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_stream_capture_matrix/raw_terminal_keeps_stop_on_forced_function_call",
        |client| async move {
            let model = client.completion(MODEL);
            sink.put(drain_stream(&model, forced_tool_request()).await);
        },
    )
    .await;

    let drained = observed.take();

    // The stream carried the forced call as a typed ToolCall.
    let call = drained
        .tool_calls
        .iter()
        .find(|call| call.function.name == Adder::NAME)
        .expect("the stream should carry the forced add call");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({ "x": 2, "y": 3 })
    );

    let terminal = &drained.terminal;
    let raw = &terminal.raw;

    // The record shape holds for a tool turn's terminal too.
    let total_tokens = terminal_record_total_tokens(raw);
    assert_eq!(
        raw.get("response_id").and_then(Value::as_str),
        terminal.response_id()
    );
    assert_eq!(total_tokens, terminal.usage.total_tokens);

    // The normalized terminal reports the reconciled ToolCalls …
    assert_eq!(terminal.finish_reason(), Some(FinishReason::ToolCalls));
    // … while raw keeps Gemini's own STOP.
    assert_eq!(
        raw.get("finish_reason"),
        Some(&Value::String("STOP".to_string())),
        "raw keeps Gemini's finishReason spelling on a call-only turn"
    );

    let last = last_usage_frame_of_function_call_stream(SCENARIO);
    assert_eq!(
        raw.pointer("/usage_metadata/totalTokenCount"),
        last.pointer("/usageMetadata/totalTokenCount"),
        "{SCENARIO}: the captured terminal usage must be the last frame's total"
    );
    assert_eq!(
        raw.pointer("/usage_metadata/promptTokensDetails"),
        last.pointer("/usageMetadata/promptTokensDetails"),
        "raw must carry the terminal frame's promptTokensDetails on the tool turn too"
    );
    // Gemini annotates a call-only STOP with a `finishMessage`; the terminal
    // record keeps it as `finish_message`, and the normalized terminal has no
    // home for it.
    assert!(
        last.pointer("/candidates/0/finishMessage")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.is_empty()),
        "{SCENARIO}: the recorded call-only turn should carry Gemini's finishMessage"
    );
    assert_eq!(
        raw.get("finish_message"),
        last.pointer("/candidates/0/finishMessage"),
        "raw must carry the wire's finishMessage untouched"
    );
}
