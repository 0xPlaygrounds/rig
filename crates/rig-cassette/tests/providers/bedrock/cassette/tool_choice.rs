//! AWS Bedrock tool-choice cassette coverage ported from Gemini tests.

use rig::bedrock;
use rig::completion::AssistantContent;
use rig::message::ToolChoice;
use rig::tool::Tool;

use super::super::support::with_bedrock_cassette;
use crate::support::{Adder, Subtract, collect_raw_stream_observation};
use rig::completion::CompletionRequest;

fn specific_add_choice() -> ToolChoice {
    ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new(Adder::NAME).expect("tool name")],
    }
}

#[tokio::test]
async fn required_forces_function_call() {
    with_bedrock_cassette(
        "tool_choice/required_forces_function_call",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request = CompletionRequest::new("Use the add tool to calculate 20 + 22.")
                .temperature(0.0)
                .tool(rig::tool::tool_definition(&Adder))
                .tool_choice(ToolChoice::Required);

            let response = model
                .call(request)
                .await
                .expect("required tool choice completion should succeed");
            super::super::history::assert_recorded_history(
                bedrock::completion::AMAZON_NOVA_LITE,
                &response,
            );

            let names = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(
                !names.is_empty(),
                "required tool choice should force a tool call"
            );
            assert!(
                names.iter().all(|name| name == Adder::NAME),
                "only the provided tool can be called, saw {names:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn specific_add_raw_nonstreaming_allows_only_add() {
    with_bedrock_cassette(
        "tool_choice/specific_add_raw_nonstreaming",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let response = model
                .call(
                    CompletionRequest::new(
                        "Use the add tool to calculate 20 + 22. Do not use subtraction.",
                    )
                    .temperature(0.0)
                    .tool(rig::tool::tool_definition(&Adder))
                    .tool(rig::tool::tool_definition(&Subtract))
                    .tool_choice(specific_add_choice()),
                )
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
                add_call.function.arguments_value(),
                serde_json::json!({ "x": 20, "y": 22 })
            );
        },
    )
    .await;
}

#[tokio::test]
async fn specific_add_raw_streaming_allows_only_add() {
    with_bedrock_cassette(
        "tool_choice/specific_add_raw_streaming",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request = CompletionRequest::new(
                "Use the add tool to calculate 20 + 22. Do not use subtraction.",
            )
            .temperature(0.0)
            .tool(rig::tool::tool_definition(&Adder))
            .tool(rig::tool::tool_definition(&Subtract))
            .tool_choice(specific_add_choice());
            let stream = model.stream(request).expect("stream should start");
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
                add_call.function.arguments_value(),
                serde_json::json!({ "x": 20, "y": 22 })
            );
        },
    )
    .await;
}
