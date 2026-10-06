//! Feature matrix for raw provider response capture on the Gemini REST
//! (`streamGenerateContent?alt=sse`) streaming seam.
//!
//! # The feature
//!
//! Raw capture is always on: the GenerateContent reassembler rebuilds the
//! `generateContent` body the same turn has unary from the stream's chunks
//! (parts joined, the last finish, usage and ids) and the driver puts it
//! onto the terminal [`rig::completion::CompletionResponse::raw`]. There is
//! no opt-in and nothing about it reaches the wire.
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
//! | 3 | `raw_terminal_keeps_stop_on_forced_function_call` | forced tool call (`ToolChoice::Specific`), streamed | terminal `raw` is the unary document; raw `finishReason` spelled `"STOP"` and `finishMessage` == wire while the normalized terminal reports `ToolCalls`; raw carries the streamed `functionCall`; the typed extras equal the unary recording's | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `streamGenerateContent` route.
//!
//! Usage is cumulative per chunk and `finishReason` arrives on the last
//! content frame, so the "wire" side of each premise is the last frame that
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
use rig::providers::gemini::extension::Gemini;
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
    let candidate = raw
        .pointer("/candidates/0")
        .expect("raw is the unary document, with its candidate");
    let called: Vec<&Value> = candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("functionCall"))
        .collect();
    assert_eq!(
        called,
        [&serde_json::json!({ "name": "add", "args": { "x": 2, "y": 3 } })],
        "raw carries the streamed call part"
    );
    assert_eq!(
        raw.get("responseId").and_then(Value::as_str),
        terminal.response_id()
    );

    // The normalized terminal reports the reconciled ToolCalls …
    assert_eq!(terminal.finish_reason(), Some(FinishReason::ToolCalls));
    // … while raw keeps Gemini's own STOP.
    assert_eq!(
        candidate.get("finishReason"),
        Some(&Value::String("STOP".to_string())),
        "raw keeps Gemini's finishReason spelling on a call-only turn"
    );

    let last = last_usage_frame_of_function_call_stream(SCENARIO);
    assert_eq!(
        raw.get("usageMetadata"),
        last.get("usageMetadata"),
        "{SCENARIO}: the captured usage must be the last frame's"
    );
    assert_eq!(
        raw.pointer("/usageMetadata/totalTokenCount")
            .and_then(Value::as_u64),
        terminal.usage.total_tokens
    );
    // Gemini annotates a call-only STOP with a `finishMessage`, which the
    // normalized terminal has no home for.
    assert!(
        last.pointer("/candidates/0/finishMessage")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.is_empty()),
        "{SCENARIO}: the recorded call-only turn should carry Gemini's finishMessage"
    );
    assert_eq!(
        candidate.get("finishMessage"),
        last.pointer("/candidates/0/finishMessage"),
        "raw must carry the wire's finishMessage untouched"
    );

    // The typed extras read the streamed document as they read the unary
    // recording of the same prompt; only the minted response id differs.
    let extras = |raw: Value| {
        let mut response = terminal.clone();
        response.raw = raw;
        response
            .extras::<Gemini>()
            .expect("a Gemini API reply has Gemini extras")
            .expect("the document holds the extras' shape")
    };
    let streamed = extras(raw.clone());
    let mut unary = extras(crate::cassettes::recorded_json_response(
        PROVIDER,
        "raw_capture_matrix/raw_exposes_forced_function_call",
    ));
    assert_eq!(streamed.response_id.as_deref(), terminal.response_id());
    assert_eq!(
        streamed.finish_message.as_deref(),
        Some("Model generated function call(s).")
    );
    assert_eq!(streamed.service_tier.as_deref(), Some("standard"));
    unary.response_id = streamed.response_id.clone();
    assert_eq!(streamed, unary);
}
