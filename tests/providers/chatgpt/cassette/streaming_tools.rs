//! ChatGPT cassette coverage for terminal responses that omit `output`.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::cassette_path;
use crate::support::zero_arg_tool_definition;

/// Assert that the recorded terminal `response.completed` event carries no
/// output items, which is the precondition this whole scenario exercises.
fn assert_terminal_response_has_no_output(scenario: &str) {
    let cassette = std::fs::read_to_string(cassette_path("chatgpt", scenario))
        .expect("cassette should be readable");
    let terminal = cassette
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data.trim()).ok())
        .find(|event| {
            event.get("type").and_then(serde_json::Value::as_str) == Some("response.completed")
        })
        .expect("cassette should contain a response.completed event");

    assert_eq!(
        terminal["response"]["output"],
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
            let request = model
                .completion_request(
                    "Call the ping tool with no arguments. Do not write any normal text before the tool call.",
                )
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .build();

            // The premise of the scenario: the terminal `response.completed`
            // event carries no output items, so the non-streaming path has to
            // rebuild the tool call from the event stream. That used to be read
            // off `response.raw_response`. The normalized response no longer
            // carries the wire payload, ChatGPT's raw wire response is exactly
            // that empty terminal record (its choice is only recoverable through
            // the crate-private SSE fallback), and the cassette records a single
            // interaction — so the premise is asserted against the recorded
            // terminal event rather than by issuing a second request.
            assert_terminal_response_has_no_output(
                "streaming_tools/tool_call_completed_response_without_output",
            );

            let response = model
                .completion(request)
                .await
                .expect("non-streaming completion should reconstruct streamed tool call");

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
            let request = model
                .completion_request(
                    "Call the ping tool with no arguments. Do not write any normal text before the tool call.",
                )
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .build();

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
