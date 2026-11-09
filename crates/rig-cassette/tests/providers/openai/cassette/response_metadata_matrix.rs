//! Echoed response metadata on the OpenAI Responses API.
//!
//! Metadata is captured as raw JSON and projected per key, so a key whose
//! value does not fit its Rust type is dropped, never fatal to the response.
//! OpenAI-compatible endpoints echo `top_p` as an object, and
//! `serde_json/arbitrary_precision` turns every buffered number into a map;
//! neither may discard a tool call or the terminal frame's usage.
//!
//! The fixtures are derived from recorded OpenAI turns: every `"top_p":1.0`
//! in the response bodies is replaced with `"top_p":{"value":1.0}`, while
//! the request bodies stay byte-identical so the harness still matches them.
//! Each cell asserts its fixture carries the object shape, so a re-record
//! cannot silently turn it into a plain OpenAI cell.

use rig::message::AssistantContent;
use rig::providers::openai;
use serde_json::Value;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::support::{
    REQUIRED_ZERO_ARG_TOOL_PROMPT, collect_raw_stream_observation, zero_arg_tool_definition,
};
use rig::completion::CompletionRequest;

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

async fn assert_blocking_tool_call(client: OpenAiCassette) {
    let model = client.openai.completion(openai::GPT_4O_MINI);
    let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .preamble(PREAMBLE.to_string())
        .tool(zero_arg_tool_definition(TOOL));

    let response = model
        .call(request)
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
    let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .preamble(PREAMBLE.to_string())
        .tool(zero_arg_tool_definition(TOOL));

    let stream = model
        .stream(request)
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
