//! Raw provider response capture on Mistral's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the chat-completions terminal
//! record the wire's decoder assembled, serialized — and it carries it whole:
//! a dialect's own usage fields ride along beside the OpenAI-compatible ones
//! that [`chat::Terminal`] models. Capture is always on: there is
//! no flag to request it, nothing about it reaches the wire, and a
//! `Value::Null` only ever means a terminal built by hand with no provider
//! record behind it. It is the terminal record only, never the stream's frames.
//! Mistral's terminal usage carries the capacity tier (`usage.service_tier`)
//! that the normalized `Usage` has no slot for, so it is the terminal-only
//! field pinned here.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into the chat-completions terminal record and re-serializes equal, and its typed accounting normalizes to the terminal's usage; the normalized terminal reproduces the recorded terminal frame and `mistral-correlation-id` header | recorded |
//! | 2 | `stream_raw_exposes_terminal_service_tier` | terminal-only field | `raw.usage.service_tier` equals the recorded terminal frame's | recorded |
//! | 3 | `stream_tool_call_raw_round_trips_terminal_type` | forced tool call | a `tool_choice: any` stream's terminal `raw` round-trips into the terminal record and reproduces the recorded finish frame; the normalized terminal reports `ToolCalls` while the recorded frame spells `"tool_calls"`, and the stream's one tool call reassembles the fixture's `delta.tool_calls` fragments | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — Mistral's finish
//! frame, the stream's last data frame, which also carries the final content
//! delta. That is [`chat::recorded_sole_usage_frame`], the single-frame rule,
//! and each cell adds beside it the Mistral-specific half: that same frame
//! carries the `finish_reason`. So the raw terminal record's usage is
//! knowable from the bytes and a recording whose stream stopped reporting
//! usage fails loudly instead of covering nothing. Cell 3 additionally
//! re-derives that the recorded frames carry a `delta.tool_calls` fragment
//! for `lookup_city` (and that the recorded request forced the call), so a
//! recording that stopped calling fails instead of covering nothing.

use rig::message::AssistantContent;

use futures::StreamExt as _;
use rig::completion::{CompletionModel, CompletionRequest, FinishReason, ToolDefinition};
use rig::streaming::{StreamEvent, StreamFinal};
use serde_json::{Value, json};

use super::DEFAULT_MODEL;
use super::support::with_mistral_cassette_result;
use crate::cassettes::recorded_response_header;
use crate::raw_capture::{
    assert_contracted_request_id, assert_normalized_lacks, capture_terminal,
    capture_text_and_terminal, chat,
};
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "mistral";
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "mistral-correlation-id";
/// The forced-call request shape the tool-lifecycle matrix uses: a preamble
/// that forbids prose, `tool_choice: any`, and a prompt naming the one call.
const TOOL_PREAMBLE: &str =
    "Follow the user's tool-call instruction exactly. Do not answer in prose.";
const TOOL_PROMPT: &str = "Call lookup_city exactly once with city Paris.";
const TOOL_NAME: &str = "lookup_city";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

fn lookup_city_tool() -> ToolDefinition {
    ToolDefinition {
        name: TOOL_NAME.to_owned(),
        description: "Look up a city by name.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

fn tool_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .preamble(TOOL_PREAMBLE.to_owned())
        .tool(lookup_city_tool())
        .additional_params(json!({ "tool_choice": "any", "parallel_tool_calls": false }))
        .max_tokens(128)
        .build()
}

/// What a forced-call stream yields: the completed tool calls and the
/// terminal record.
struct ToolStreamObservation {
    tool_calls: Vec<rig::message::ToolCall>,
    terminal: Option<StreamFinal>,
}

async fn collect_tool_calls_and_terminal(
    mut stream: rig::streaming::StreamingCompletionResponse,
) -> ToolStreamObservation {
    let mut observation = ToolStreamObservation {
        tool_calls: Vec::new(),
        terminal: None,
    };
    while let Some(item) = stream.next().await {
        match item.expect("stream item should not be an error") {
            StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            } => {
                observation.tool_calls.push(tool_call);
            }
            StreamEvent::Final(final_record) => {
                observation.terminal = Some(final_record);
            }
            _ => {}
        }
    }
    observation
}

/// The first-choice tool call the recorded frames spell as `delta.tool_calls`
/// fragments, reassembled in wire order: `(id, name, arguments string)`.
fn recorded_tool_call(scenario: &str) -> (String, String, String) {
    let mut id = String::new();
    let mut name = String::new();
    let mut arguments = String::new();
    let mut fragments = 0usize;
    for frame in crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario) {
        for call in frame["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            fragments += 1;
            assert_eq!(
                call["index"].as_u64().unwrap_or(0),
                0,
                "the forced turn is a single call: {call}"
            );
            if let Some(fragment) = call["id"].as_str() {
                id.push_str(fragment);
            }
            if let Some(fragment) = call["function"]["name"].as_str() {
                name.push_str(fragment);
            }
            if let Some(fragment) = call["function"]["arguments"].as_str() {
                arguments.push_str(fragment);
            }
        }
    }
    assert!(
        fragments > 0,
        "a forced-call stream carries delta.tool_calls fragments"
    );
    (id, name, arguments)
}

/// The `mistral-correlation-id` the recorded SSE response carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_header(PROVIDER, scenario, 0, REQUEST_ID_HEADER)
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let observed = Observed::default();
    with_mistral_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| {
            capture_text_and_terminal(client.completion(DEFAULT_MODEL), request, observed.clone())
        },
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");

    let (text, terminal) = observed.take();
    assert!(!text.is_empty());
    // The accounting rides the record typed: `ChatUsage` folds the
    // OpenAI-compatible counters and the dialect's extras, and its mapping is
    // what the normalized terminal reports.
    chat::assert_terminal_round_trips(&terminal);

    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    assert!(
        frame["choices"][0]["finish_reason"].is_string(),
        "Mistral's terminal frame carries the finish reason: {frame}"
    );
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, "the recorded frame");
    // Mistral contracts `mistral-correlation-id`; the recorded header is the
    // premise.
    assert_contracted_request_id(
        terminal.provider_request_id.as_deref(),
        recorded_request_id(SCENARIO).as_deref(),
        REQUEST_ID_HEADER,
    );
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_service_tier() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_service_tier";
    let observed = Observed::default();
    with_mistral_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_service_tier",
        |client| capture_terminal(client.completion(DEFAULT_MODEL), request, observed.clone()),
    )
    .await
    .expect("stream_raw_exposes_terminal_service_tier should replay from its cassette");

    let terminal = observed.take();
    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    assert!(
        frame["choices"][0]["finish_reason"].is_string(),
        "Mistral's terminal frame carries the finish reason: {frame}"
    );
    let recorded_tier = frame["usage"]["service_tier"]
        .as_str()
        .expect("Mistral's terminal usage reports service_tier");
    let recorded_prompt_tokens = frame["usage"]["prompt_tokens"]
        .as_u64()
        .expect("terminal usage reports prompt_tokens");

    let raw = &terminal.raw;
    assert_eq!(raw["usage"]["service_tier"], json!(recorded_tier));
    assert_eq!(raw["usage"]["prompt_tokens"], json!(recorded_prompt_tokens));
    // The normalized terminal has no slot for the tier.
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert_normalized_lacks(&normalized_usage, &["service_tier"]);
}

// ================================================================
// 3. A forced tool call: the terminal round-trips and reports ToolCalls
// ================================================================

#[tokio::test]
async fn stream_tool_call_raw_round_trips_terminal_type() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_tool_call_raw_round_trips_terminal_type";
    let observed = Observed::default();
    let sink = observed.clone();
    with_mistral_cassette_result(
        "raw_stream_capture_matrix/stream_tool_call_raw_round_trips_terminal_type",
        |client| async move {
            // No shared capture helper collects completed tool calls beside
            // the terminal record, so this cell drives the stream itself.
            let model = client.completion(DEFAULT_MODEL);
            let stream = model.stream(tool_request(&model)).await?;
            sink.put(collect_tool_calls_and_terminal(stream).await);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_tool_call_raw_round_trips_terminal_type should replay from its cassette");

    let observation = observed.take();
    let terminal = observation
        .terminal
        .as_ref()
        .expect("the cell should observe a terminal record");
    chat::assert_terminal_round_trips(terminal);
    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    chat::assert_terminal_reproduces_frame(terminal, PROVIDER, &frame, "the recorded frame");
    assert_contracted_request_id(
        terminal.provider_request_id.as_deref(),
        recorded_request_id(SCENARIO).as_deref(),
        REQUEST_ID_HEADER,
    );
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
    assert_eq!(request_body["tool_choice"], json!("any"));
    assert_eq!(
        request_body["tools"][0]["function"]["name"],
        json!(TOOL_NAME)
    );

    // Premise, from the bytes: the finish frame spells the wire's
    // `tool_calls` — which is also this cell's "the terminal frame carries a
    // finish reason", spelled exactly — and the frames carry one call to
    // `lookup_city`.
    assert_eq!(frame["choices"][0]["finish_reason"], json!("tool_calls"));
    let (recorded_id, recorded_name, recorded_arguments) = recorded_tool_call(SCENARIO);
    assert_eq!(recorded_name, TOOL_NAME);
    let recorded_arguments: Value =
        serde_json::from_str(&recorded_arguments).expect("recorded arguments parse as JSON");
    assert_eq!(recorded_arguments["city"], json!("Paris"));

    // The normalized terminal reports ToolCalls, and raw's own finish reason
    // agrees once mapped through the terminal type.
    assert_eq!(terminal.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(terminal.raw["finish_reason"], json!("tool_calls"));
    // The stream yielded exactly the recorded call, with object arguments.
    assert_eq!(observation.tool_calls.len(), 1, "one streamed tool call");
    let call = &observation.tool_calls[0];
    assert_eq!(call.function.name, TOOL_NAME);
    assert_eq!(call.function.arguments, recorded_arguments);
    assert_matches_recorded_token(
        call.id.explicit(),
        Some(recorded_id.as_str()),
        "streamed tool call id",
    );
    // raw is the terminal record only: no frame content rides on it.
    assert!(
        terminal.raw.get("choices").is_none() && terminal.raw.get("tool_calls").is_none(),
        "the terminal raw carries no frame content: {}",
        terminal.raw
    );
}
