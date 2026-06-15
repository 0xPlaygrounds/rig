//! Gemini tool-choice cassette coverage.

use rig::client::CompletionClient;
use rig::completion::{AssistantContent, Chat, CompletionModel, Message};
use rig::message::ToolChoice;
use rig::providers::gemini;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;

use crate::support::{
    Adder, Subtract, assert_mentions_expected_number, collect_raw_stream_observation,
    collect_stream_observation,
};

fn specific_add_choice() -> ToolChoice {
    ToolChoice::Specific {
        function_names: vec![Adder::NAME.to_string()],
    }
}

fn assert_history_tool_calls(history: &[Message], expected: &[&str], forbidden: &[&str]) {
    let tool_names = history
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    for expected_tool in expected {
        assert!(
            tool_names.iter().any(|name| name == expected_tool),
            "expected tool call {expected_tool}, saw {tool_names:?}"
        );
    }

    for forbidden_tool in forbidden {
        assert!(
            !tool_names.iter().any(|name| name == forbidden_tool),
            "did not expect tool call {forbidden_tool}, saw {tool_names:?}"
        );
    }
}

#[tokio::test]
async fn specific_add_raw_streaming_allows_only_add() {
    super::super::support::with_gemini_cassette(
        "tool_choice/specific_add_raw_streaming",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "Use the add tool to calculate 20 + 22. Do not use subtraction.",
                )
                .temperature(0.0)
                .tool(Adder.definition(String::new()).await)
                .tool(Subtract.definition(String::new()).await)
                .tool_choice(specific_add_choice())
                .build();
            let stream = model.stream(request).await.expect("stream should start");
            let observation = collect_raw_stream_observation(stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert!(
                observation
                    .tool_calls
                    .iter()
                    .any(|tool_call| tool_call.function.name == Adder::NAME),
                "expected add tool call, saw {:?}",
                observation.tool_calls
            );
            assert!(
                !observation
                    .tool_calls
                    .iter()
                    .any(|tool_call| tool_call.function.name == Subtract::NAME),
                "did not expect subtract tool call, saw {:?}",
                observation.tool_calls
            );
            let add_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == Adder::NAME)
                .expect("expected add tool call");
            assert_eq!(
                add_call.function.arguments,
                serde_json::json!({ "x": 20, "y": 22 })
            );
        },
    )
    .await;
}

#[tokio::test]
async fn specific_add_raw_nonstreaming_allows_only_add() {
    super::super::support::with_gemini_cassette(
        "tool_choice/specific_add_raw_nonstreaming",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let response = model
                .completion_request(
                    "Use the add tool to calculate 20 + 22. Do not use subtraction.",
                )
                .temperature(0.0)
                .tool(Adder.definition(String::new()).await)
                .tool(Subtract.definition(String::new()).await)
                .tool_choice(specific_add_choice())
                .send()
                .await
                .expect("specific add raw completion should succeed");

            let tool_calls = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .collect::<Vec<_>>();

            assert!(
                tool_calls
                    .iter()
                    .any(|tool_call| tool_call.function.name == Adder::NAME),
                "expected add tool call, saw {tool_calls:?}"
            );
            assert!(
                !tool_calls
                    .iter()
                    .any(|tool_call| tool_call.function.name == Subtract::NAME),
                "did not expect subtract tool call, saw {tool_calls:?}"
            );
            let add_call = tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == Adder::NAME)
                .expect("expected add tool call");
            assert_eq!(
                add_call.function.arguments,
                serde_json::json!({ "x": 20, "y": 22 })
            );
        },
    )
    .await;
}

#[tokio::test]
async fn none_streaming_does_not_emit_tool_calls() {
    super::super::support::with_gemini_cassette(
        "tool_choice/none_streaming_no_tools",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble("You are a deterministic calculator test. Answer directly in text.")
                .temperature(0.0)
                .tool(Adder)
                .tool(Subtract)
                .tool_choice(ToolChoice::None)
                .build();

            let mut stream = agent
                .stream_prompt("Calculate 20 + 22 directly in text. Do not call tools.")
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
                observation.tool_calls.is_empty(),
                "expected no tool calls, saw {:?}",
                observation.tool_calls
            );
            assert_eq!(observation.tool_results, 0, "expected no tool results");
            assert_mentions_expected_number(
                observation
                    .final_response_text
                    .as_deref()
                    .expect("stream should produce a final response"),
                42,
            );
        },
    )
    .await;
}

#[tokio::test]
async fn none_nonstreaming_does_not_emit_tool_calls() {
    super::super::support::with_gemini_cassette(
        "tool_choice/none_nonstreaming_no_tools",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble("You are a deterministic calculator test. Answer directly in text.")
                .temperature(0.0)
                .tool(Adder)
                .tool(Subtract)
                .tool_choice(ToolChoice::None)
                .build();

            let mut chat_history = Vec::<Message>::new();
            let response = agent
                .chat(
                    "Calculate 20 + 22 directly in text. Do not call tools.",
                    &mut chat_history,
                )
                .await
                .expect("ToolChoice::None prompt should succeed");

            assert_mentions_expected_number(&response, 42);
            assert_history_tool_calls(&chat_history, &[], &[Adder::NAME, Subtract::NAME]);
        },
    )
    .await;
}
