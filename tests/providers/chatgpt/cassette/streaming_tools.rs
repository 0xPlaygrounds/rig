//! ChatGPT cassette coverage for terminal responses that omit `output`.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use super::super::support::with_chatgpt_cassette;
use crate::support::zero_arg_tool_definition;

const SCENARIO: &str = "streaming_tools/tool_call_completed_response_without_output";

/// Assert against the recorded SSE body that the terminal `response.completed`
/// event carries an empty `output` array. The normalized response no longer
/// exposes the provider-typed raw response, so this wire-shape contract is
/// checked from the cassette directly (replay only: the cassette file is
/// written after the test body in record mode).
fn assert_recorded_completed_event_output_is_empty() {
    if crate::cassettes::CassetteMode::current() != crate::cassettes::CassetteMode::Replay {
        return;
    }
    let bodies = crate::cassettes::recorded_response_bodies("chatgpt", SCENARIO);
    assert_eq!(bodies.len(), 1, "scenario should record a single interaction");
    let completed = bodies[0]
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data.trim()).ok())
        .find(|event| event["type"] == "response.completed")
        .expect("recorded SSE body should contain a response.completed event");
    assert_eq!(
        completed["response"]["output"],
        json!([]),
        "cassette should keep the terminal response.completed output empty"
    );
}

#[tokio::test]
async fn nonstreaming_tool_call_completed_response_without_output() {
    with_chatgpt_cassette(
        "streaming_tools/tool_call_completed_response_without_output",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let request = CompletionRequest {
                tools: vec![zero_arg_tool_definition("ping")],
                tool_choice: Some(ToolChoice::Required),
                ..CompletionRequest::from_prompt(
                    "Call the ping tool with no arguments. Do not write any normal text before the tool call.",
                )
            };

            let response = model
                .completion(request)
                .await
                .expect("non-streaming completion should reconstruct streamed tool call");

            assert_recorded_completed_event_output_is_empty();

            let tool_call = response.choice.iter().find_map(|content| match content {
                AssistantContent::ToolCall(tool_call) if tool_call.function.name == "ping" => {
                    Some(tool_call)
                }
                _ => None,
            });
            let tool_call = tool_call.expect("completion should include the ping tool call");
            assert_eq!(tool_call.function.arguments, json!({}));
            assert!(response.usage.input_tokens > 0, "usage should have input tokens");
            assert!(
                response.usage.output_tokens > 0,
                "usage should have output tokens"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn stream_tool_call_completed_response_without_output() {
    with_chatgpt_cassette(
        "streaming_tools/tool_call_completed_response_without_output",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let request = CompletionRequest {
                tools: vec![zero_arg_tool_definition("ping")],
                tool_choice: Some(ToolChoice::Required),
                ..CompletionRequest::from_prompt(
                    "Call the ping tool with no arguments. Do not write any normal text before the tool call.",
                )
            };

            let mut stream = model.stream(request).await.expect("stream should start");
            let mut saw_ping_tool_call = false;
            let mut final_usage = None;

            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream item should be ok") {
                    StreamedAssistantContent::ToolCall { tool_call, .. } => {
                        if tool_call.function.name == "ping" {
                            assert_eq!(tool_call.function.arguments, json!({}));
                            saw_ping_tool_call = true;
                        }
                    }
                    StreamedAssistantContent::Final(response) => {
                        final_usage = Some(response.usage);
                    }
                    _ => {}
                }
            }

            assert!(saw_ping_tool_call, "stream should emit the ping tool call");
            let usage = final_usage.expect("stream should emit terminal usage");
            assert!(usage.input_tokens > 0, "usae should have input tokens");
            assert!(usage.output_tokens > 0, "usae should have output tokens");
        },
    )
    .await;
}
