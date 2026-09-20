//! Edge matrix for echoed response metadata on the OpenAI Responses API
//! (rig#2483, rig#2493).
//!
//! **Bug.** The response wire struct reached the request-shaped
//! `AdditionalParameters` through `#[serde(flatten)]`, so one optional
//! metadata field disagreeing with its Rust type discarded the whole
//! response. Two ways that happened in the field: MiniMax-style endpoints
//! echo `top_p` as an object (`invalid type: map, expected f64`, killing a
//! response that carried a valid tool call — #2483), and with
//! `serde_json/arbitrary_precision` enabled anywhere in the graph every
//! buffered number becomes an internal map, so even `top_p: 0.95` failed and
//! the terminal `response.completed` frame — the one carrying usage — became a
//! hard stream error (#2493).
//!
//! **Fix.** Metadata is captured as raw JSON and projected per key; a key
//! whose value does not fit is dropped, never fatal.
//!
//! **Fixtures.** Cells 1–2 are recorded live against OpenAI. Cells 3–4 are
//! hand-derived siblings: the recorded *response* bodies were copied and every
//! `"top_p":1.0` replaced with `"top_p":{"value":1.0}` — the shape a
//! compatible endpoint emits — while the recorded *request* bodies are
//! byte-identical to their siblings so the harness still matches them. This is
//! the only way to pin the compatible-endpoint shape without a MiniMax key;
//! the module doc says so, and each derived cell asserts the fixture really
//! carries the object shape so a re-record cannot silently turn it back into
//! a plain-OpenAI cell.
//!
//! **How these cells fail on `origin/main`.** Cells 3–4 fail to decode (the
//! tool call and usage never reach the caller); cells 1–2 pass on both sides
//! and are the controls proving the request bytes did not move.
//!
//! | # | cell | transport | `top_p` on the wire | fixture |
//! |---|------|-----------|---------------------|---------|
//! | 1 | `numeric_top_p_blocking_tool_call` | blocking | `1.0` | recorded |
//! | 2 | `numeric_top_p_streaming_terminal_usage` | streaming | `1.0` | recorded |
//! | 3 | `object_top_p_blocking_tool_call` | blocking | `{"value":1.0}` | derived from 1 |
//! | 4 | `object_top_p_streaming_terminal_usage` | streaming | `{"value":1.0}` | derived from 2 |
//!
//! Unit cells for the projection itself (per-key drop, numeric decode under
//! `arbitrary_precision`, round-trip) live beside the type in
//! `crates/rig-core/src/providers/openai/responses_api/tests.rs`.

use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use rig::providers::openai;
use serde_json::Value;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, collect_raw_stream_observation, zero_arg_tool_definition,
};

const TOOL: &str = "ping";
const PREAMBLE: &str = "Follow the tool-calling instructions exactly.";

/// The `top_p` values found in every recorded response body of `scenario`
/// (non-streaming bodies and `response.completed` SSE frames alike).
fn recorded_top_p_values(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .flat_map(|(_, response)| {
            let mut found = Vec::new();
            for line in response.lines() {
                let json = line.strip_prefix("data: ").unwrap_or(line);
                let Ok(value) = serde_json::from_str::<Value>(json) else {
                    continue;
                };
                let body = value.get("response").unwrap_or(&value);
                if let Some(top_p) = body.get("top_p") {
                    found.push(top_p.clone());
                }
            }
            found
        })
        .collect()
}

fn assert_recorded_top_p_is_object(scenario: &str) {
    let values = recorded_top_p_values(scenario);
    assert!(
        !values.is_empty(),
        "{scenario}: fixture should echo top_p on at least one response body"
    );
    assert!(
        values.iter().all(Value::is_object),
        "{scenario}: this derived fixture must carry object-shaped top_p, got {values:?}"
    );
}

fn assert_recorded_top_p_is_number(scenario: &str) {
    let values = recorded_top_p_values(scenario);
    assert!(
        !values.is_empty(),
        "{scenario}: fixture should echo top_p on at least one response body"
    );
    assert!(
        values.iter().all(Value::is_number),
        "{scenario}: the recorded fixture must carry numeric top_p, got {values:?}"
    );
}

async fn assert_blocking_tool_call(client: OpenAiCassette) {
    let model = client.openai.completion(openai::GPT_4O_MINI);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .preamble(PREAMBLE.to_string())
        .tool(zero_arg_tool_definition(TOOL))
        .build();

    let response = model
        .completion(request)
        .await
        .expect("echoed metadata must never fail a response that carries a tool call");

    assert!(
        response.choice.iter().any(|content| {
            matches!(content, AssistantContent::ToolCall(call) if call.function.name == TOOL)
        }),
        "the tool call must reach the caller, got {:?}",
        response.choice
    );
    assert!(
        response.usage.total_tokens.is_some_and(|n| n > 0),
        "usage must survive, got {:?}",
        response.usage
    );
}

async fn assert_streaming_terminal_usage(client: OpenAiCassette) {
    let model = client.openai.completion(openai::GPT_4O_MINI);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .preamble(PREAMBLE.to_string())
        .tool(zero_arg_tool_definition(TOOL))
        .build();

    let stream = model
        .stream(request)
        .await
        .expect("streaming request should start");
    let observation = collect_raw_stream_observation(stream).await;

    assert!(
        observation.errors.is_empty(),
        "the terminal frame must not become a stream error: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final,
        "the terminal record must arrive; events {:?}",
        observation.events
    );
    assert!(
        observation
            .tool_calls
            .iter()
            .any(|call| call.function.name == TOOL),
        "the streamed tool call must complete, got {:?}",
        observation.tool_calls
    );
}

#[tokio::test]
async fn numeric_top_p_blocking_tool_call() {
    with_openai_cassette(
        "response_metadata_matrix/numeric_top_p_blocking_tool_call",
        assert_blocking_tool_call,
    )
    .await;
    assert_recorded_top_p_is_number("response_metadata_matrix/numeric_top_p_blocking_tool_call");
}

#[tokio::test]
async fn numeric_top_p_streaming_terminal_usage() {
    with_openai_cassette(
        "response_metadata_matrix/numeric_top_p_streaming_terminal_usage",
        assert_streaming_terminal_usage,
    )
    .await;
    assert_recorded_top_p_is_number(
        "response_metadata_matrix/numeric_top_p_streaming_terminal_usage",
    );
}

#[tokio::test]
async fn object_top_p_blocking_tool_call() {
    if crate::cassettes::skip_when_recording(
        "cell 3 is hand-derived from cell 1: the object-valued top_p is not what the API returns",
    ) {
        return;
    }
    with_openai_cassette(
        "response_metadata_matrix/object_top_p_blocking_tool_call",
        assert_blocking_tool_call,
    )
    .await;
    assert_recorded_top_p_is_object("response_metadata_matrix/object_top_p_blocking_tool_call");
}

#[tokio::test]
async fn object_top_p_streaming_terminal_usage() {
    if crate::cassettes::skip_when_recording(
        "cell 4 is hand-derived from cell 2: the object-valued top_p is not what the API returns",
    ) {
        return;
    }
    with_openai_cassette(
        "response_metadata_matrix/object_top_p_streaming_terminal_usage",
        assert_streaming_terminal_usage,
    )
    .await;
    assert_recorded_top_p_is_object(
        "response_metadata_matrix/object_top_p_streaming_terminal_usage",
    );
}
