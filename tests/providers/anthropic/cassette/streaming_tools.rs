//! Anthropic streaming tools smoke test.

use rig::client::CompletionClient;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive, assert_mentions_expected_number,
    collect_stream_final_response, collect_stream_observation,
};

#[tokio::test]
async fn streaming_tools_smoke() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_tools_batches_multiple_tool_results_in_one_followup_message() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_batches_multiple_tool_results_in_one_followup_message",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(TWO_TOOL_STREAM_PROMPT)
                .multi_turn(8)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert!(
                observation.got_final_response,
                "stream should emit a final response"
            );
            assert!(
                observation.tool_results >= 2,
                "expected at least 2 tool-result events, got {}",
                observation.tool_results
            );
            for expected_tool in ["lookup_harbor_label", "lookup_orchard_label"] {
                assert!(
                    observation
                        .tool_calls
                        .iter()
                        .any(|tool_call| tool_call == expected_tool),
                    "expected tool call for {expected_tool}, saw {:?}",
                    observation.tool_calls
                );
            }
            assert_contains_all_case_insensitive(
                observation
                    .final_response_text
                    .as_deref()
                    .expect("stream should produce a final response string"),
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;

    assert_cassette_groups_multiple_tool_results(
        "streaming_tools/streaming_tools_batches_multiple_tool_results_in_one_followup_message",
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn assert_cassette_groups_multiple_tool_results(scenario: &str, expected_tools: &[&str]) {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    let request_bodies = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize")
                .when
                .body
        })
        .filter_map(|body| body.and_then(|body| serde_json::from_str::<Value>(&body).ok()))
        .collect::<Vec<_>>();

    let grouped = request_bodies.iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| messages_group_tool_results(messages, expected_tools))
    });

    assert!(
        grouped,
        "expected cassette {} to contain an Anthropic request where the two tool_result blocks \
         from one assistant turn are grouped into one user message",
        cassette_path.display()
    );
}

fn messages_group_tool_results(messages: &[Value], expected_tools: &[&str]) -> bool {
    messages.windows(2).any(|window| {
        let assistant = &window[0];
        let user = &window[1];

        role(assistant) == Some("assistant")
            && expected_tools
                .iter()
                .all(|tool| assistant_tool_use_names(assistant).any(|name| name == *tool))
            && role(user) == Some("user")
            && content_type_count(user, "tool_result") >= expected_tools.len()
    })
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn assistant_tool_use_names(message: &Value) -> impl Iterator<Item = &str> {
    content_items(message).filter_map(|content| {
        (content.get("type").and_then(Value::as_str) == Some("tool_use"))
            .then(|| content.get("name").and_then(Value::as_str))
            .flatten()
    })
}

fn content_type_count(message: &Value, expected_type: &str) -> usize {
    content_items(message)
        .filter(|content| content.get("type").and_then(Value::as_str) == Some(expected_type))
        .count()
}

fn content_items(message: &Value) -> impl Iterator<Item = &Value> {
    message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}
