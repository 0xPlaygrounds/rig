//! Feature matrix for raw provider response capture on the Gemini REST
//! (`streamGenerateContent?alt=sse`) streaming seam.
//!
//! # The feature
//!
//! Raw capture is always on: the GenerateContent decoder builds its terminal
//! record at EOF — Gemini's own
//! [`StreamingCompletionResponse`], assembled from the stream's last
//! `finishReason`, usage and metadata — and serializes it onto the terminal
//! [`rig::streaming::StreamFinal::raw`]. There is no opt-in and nothing
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
//! | 1 | `raw_roundtrips_streaming_completion_response` | typed access | `StreamingCompletionResponse::deserialize(&raw)` re-serializes equal and agrees with the normalized terminal | recorded |
//! | 2 | `raw_exposes_terminal_only_fields` | un-normalized terminal fields | `finish_reason` spelled `"STOP"`, `usage_metadata.promptTokensDetails` == last frame, absent from the normalized terminal | recorded |
//! | 3 | `raw_terminal_keeps_stop_on_forced_function_call` | forced tool call (`ToolChoice::Specific`), streamed | terminal `raw` round-trips; raw `finish_reason` spelled `"STOP"` and `finish_message` == wire while the normalized terminal reports `ToolCalls`; the recorded frames carry `functionCall` | recorded |
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
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, ToolCall, ToolChoice};
use rig::providers::gemini::streaming::StreamingCompletionResponse;
use rig::streaming::{StreamEvent, StreamFinal};
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_gemini_cassette;
use crate::raw_capture::{capture_text_and_sole_terminal, stream_normalized_without_raw};
use crate::support::{Adder, Observed, json_contains_key};

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded stream stays short.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: streamed";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

fn request(model: &(impl CompletionModel + Clone)) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded stream carries a `functionCall`.
fn forced_tool_request(
    model: &(impl CompletionModel + Clone),
) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .temperature(0.0)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Specific {
            function_names: vec![Adder::NAME.to_string()],
        })
        .build()
}

/// What a drained model stream carried: its tool calls and its single
/// terminal record.
struct Drained {
    tool_calls: Vec<ToolCall>,
    terminal: StreamFinal,
}

/// Drain a model stream, keeping every tool call it yielded.
///
/// Local rather than one of the shared `raw_capture::capture_*` stream
/// helpers: cell 3 is about a streamed `functionCall`, so it needs the tool
/// calls the stream yielded *beside* the terminal record, which no shared
/// helper parks. The sole-terminal claim it makes is the same one
/// [`capture_text_and_sole_terminal`](crate::raw_capture::capture_text_and_sole_terminal)
/// makes for this file's text-only cells.
async fn drain_stream(
    model: &(impl CompletionModel + Clone),
    request: rig::completion::CompletionRequest,
) -> Drained {
    let mut stream = model.stream(request).await.expect("stream should open");
    let mut terminal = None;
    let mut tool_calls = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should succeed") {
            StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            } => tool_calls.push(tool_call),
            StreamEvent::Final(final_record) => {
                assert!(
                    terminal.replace(final_record).is_none(),
                    "a stream yields exactly one terminal record"
                );
            }
            _ => {}
        }
    }
    Drained {
        tool_calls,
        terminal: terminal.expect("stream should yield a terminal record"),
    }
}

/// The last recorded frame carrying `usageMetadata` — Gemini's usage is
/// cumulative, so this is the frame whose numbers the terminal must report.
fn last_usage_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let last = frames
        .iter()
        .rev()
        .find(|frame| frame.get("usageMetadata").is_some())
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: no recorded frame carries usageMetadata"));
    // The premise proper: the recorded stream finished naturally.
    let finished = frames.iter().any(|frame| {
        frame.pointer("/candidates/0/finishReason") == Some(&Value::String("STOP".to_string()))
    });
    assert!(
        finished,
        "{scenario}: the recorded stream should carry a finishReason of STOP; without a \
         natural finish the terminal this cell asserts on is not the shape under test"
    );
    assert!(
        last.pointer("/usageMetadata/promptTokensDetails")
            .and_then(Value::as_array)
            .is_some_and(|details| !details.is_empty()),
        "{scenario}: the terminal usage frame should carry promptTokensDetails, the \
         un-normalized field cell 2 reads through `raw`"
    );
    last
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
// 1: typed access is recoverable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_streaming_completion_response() {
    const SCENARIO: &str = "raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response";
    let observed: Observed<(String, StreamFinal)> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response",
        |client| async move {
            capture_text_and_sole_terminal(client.completion(MODEL), request, sink)
                .await
                .expect("stream should open");
        },
    )
    .await;

    let (text, terminal) = observed.take();
    assert!(!text.is_empty(), "the stream should have carried text");
    let raw = &terminal.raw;

    // `raw` is the terminal record the decoder built, serialized: Gemini's
    // own terminal type reads it back and re-serializes to the same value.
    let typed = StreamingCompletionResponse::deserialize(raw)
        .expect("raw must deserialize into Gemini's streaming terminal type");
    assert_eq!(
        serde_json::to_value(&typed).expect("typed raw re-serializes"),
        *raw,
        "StreamingCompletionResponse must round-trip through its own Serialize/Deserialize"
    );

    // And the typed value agrees with the normalized terminal next to it.
    assert_eq!(typed.model_version, terminal.model);
    assert_eq!(typed.response_id, terminal.response_id);
    assert_eq!(
        Some(typed.usage_metadata.total_token_count as u64),
        terminal.usage.total_tokens
    );

    let last = last_usage_frame(SCENARIO);
    assert_eq!(
        raw.pointer("/usage_metadata/totalTokenCount"),
        last.pointer("/usageMetadata/totalTokenCount"),
        "{SCENARIO}: the captured terminal usage must be the last frame's total"
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_terminal_only_fields() {
    const SCENARIO: &str = "raw_stream_capture_matrix/raw_exposes_terminal_only_fields";
    let observed: Observed<(String, StreamFinal)> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_stream_capture_matrix/raw_exposes_terminal_only_fields",
        |client| async move {
            capture_text_and_sole_terminal(client.completion(MODEL), request, sink)
                .await
                .expect("stream should open");
        },
    )
    .await;

    let (text, terminal) = observed.take();
    assert!(!text.is_empty(), "the stream should have carried text");

    // The normalized terminal provably lacks these: `finish_reason` is rig's
    // vocabulary (`stop`), and the per-modality breakdown has no normalized
    // home.
    let normalized = stream_normalized_without_raw(&terminal);
    assert!(!json_contains_key(&normalized, "promptTokensDetails"));
    assert_ne!(
        normalized.get("finish_reason"),
        Some(&Value::String("STOP".to_string())),
        "the normalized finish reason is rig's spelling, not Gemini's"
    );
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));

    let raw = &terminal.raw;
    let last = last_usage_frame(SCENARIO);
    assert_eq!(
        raw.get("finish_reason"),
        Some(&Value::String("STOP".to_string())),
        "raw keeps Gemini's own finishReason spelling"
    );
    assert_eq!(
        raw.pointer("/usage_metadata/promptTokensDetails"),
        last.pointer("/usageMetadata/promptTokensDetails"),
        "raw must carry the terminal frame's promptTokensDetails exactly as the wire sent it"
    );
    assert_eq!(
        raw.pointer("/usage_metadata/candidatesTokenCount"),
        last.pointer("/usageMetadata/candidatesTokenCount"),
        "raw must carry the terminal frame's candidatesTokenCount untouched"
    );
}

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
            sink.put(drain_stream(&model, forced_tool_request(&model)).await);
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
        call.function.arguments,
        serde_json::json!({ "x": 2, "y": 3 })
    );

    let terminal = &drained.terminal;
    let raw = &terminal.raw;

    // The typed round trip holds for a tool turn's terminal too.
    let typed = StreamingCompletionResponse::deserialize(raw)
        .expect("raw must deserialize into Gemini's streaming terminal type");
    assert_eq!(
        serde_json::to_value(&typed).expect("typed raw re-serializes"),
        *raw,
        "StreamingCompletionResponse must round-trip a tool turn's terminal"
    );
    assert_eq!(typed.response_id, terminal.response_id);
    assert_eq!(
        Some(typed.usage_metadata.total_token_count as u64),
        terminal.usage.total_tokens
    );

    // The normalized terminal reports the reconciled ToolCalls …
    assert_eq!(terminal.finish_reason, Some(FinishReason::ToolCalls));
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
    // type keeps it as `finish_message`, and the normalized terminal has no
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
