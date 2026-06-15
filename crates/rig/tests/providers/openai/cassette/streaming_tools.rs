//! OpenAI streaming tools coverage, including the migrated example path.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message};
use rig::providers::openai;
use rig::providers::openai::responses_api::streaming::StreamingCompletionChunk;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;

use serde::Deserialize;

use super::super::support::with_openai_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, assert_tool_call_precedes_later_text,
    collect_raw_stream_observation, collect_stream_final_response, collect_stream_observation,
};

#[derive(Debug, Deserialize)]
struct CassetteInteraction {
    then: CassetteResponse,
}

#[derive(Debug, Deserialize)]
struct CassetteResponse {
    body: Option<String>,
}

#[test]
fn streaming_tools_smoke_cassette_sse_events_parse() {
    let contents = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/cassettes/openai/streaming_tools/streaming_tools_smoke.yaml"
    ))
    .expect("streaming tools cassette should be readable");

    let mut event_count = 0;
    let mut function_call_delta_count = 0;
    let mut failures = Vec::new();

    for (interaction_index, document) in serde_yaml::Deserializer::from_str(&contents).enumerate() {
        let interaction = CassetteInteraction::deserialize(document)
            .expect("streaming tools cassette interaction should deserialize");
        let Some(body) = interaction.then.body else {
            continue;
        };

        for (line_index, line) in body.lines().enumerate() {
            let Some(data) = line.trim_start().strip_prefix("data:").map(str::trim) else {
                continue;
            };

            if data.is_empty() || data == "[DONE]" {
                continue;
            }

            event_count += 1;
            if data.contains(r#""type":"response.function_call_arguments.delta""#) {
                function_call_delta_count += 1;
            }

            if let Err(error) = serde_json::from_str::<StreamingCompletionChunk>(data) {
                failures.push(format!(
                    "interaction {interaction_index}, body line {line_index}: {error}\n{data}"
                ));
            }
        }
    }

    assert!(
        event_count > 0,
        "expected cassette to contain SSE data events"
    );
    assert!(
        function_call_delta_count > 0,
        "expected cassette to cover function-call argument delta events"
    );
    assert!(
        failures.is_empty(),
        "all cassette SSE data events should parse as StreamingCompletionChunk; failures:\n{}",
        failures.join("\n\n")
    );
}

#[tokio::test]
async fn streaming_tools_smoke() {
    with_openai_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
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
async fn example_streaming_with_tools() {
    with_openai_cassette("streaming_tools/example_streaming_with_tools", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(
                "You are a calculator here to help the user perform arithmetic operations. \
                 Use the tools provided to answer the user's question and answer in a full sentence.",
            )
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut stream = agent.stream_prompt("Calculate 2 - 5").await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming tools prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    }).await;
}

#[tokio::test]
async fn responses_stream_preserves_tool_result_flow() {
    with_openai_cassette(
        "streaming_tools/responses_stream_preserves_tool_result_flow",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
                .multi_turn(5)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert_tool_call_precedes_later_text(
                &observation,
                "lookup_harbor_label",
                &[ALPHA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn raw_responses_stream_preserves_tool_then_followup_text_ordering() {
    with_openai_cassette(
        "streaming_tools/raw_responses_stream_preserves_tool_then_followup_text_ordering",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(AlphaSignal.definition(String::new()).await)
                .build();

            let first_turn = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("raw responses stream should start"),
            )
            .await;

            assert_raw_stream_tool_call_precedes_text(&first_turn, "lookup_harbor_label");

            let tool_call = first_turn
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == "lookup_harbor_label")
                .cloned()
                .expect("raw responses stream should yield lookup_harbor_label");
            let assistant_message = Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
            };
            let tool_result_message =
                Message::tool_result_with_call_id(tool_call.id, tool_call.call_id, ALPHA_SIGNAL_OUTPUT);
            let followup_request = model
                .completion_request(
                    "Now reply in one short sentence using the provided tool result. Do not call any tools.",
                )
                .preamble("Use the provided tool result and answer directly.".to_string())
                .message(assistant_message)
                .message(tool_result_message)
                .build();

            let second_turn = collect_raw_stream_observation(
                model
                    .stream(followup_request)
                    .await
                    .expect("raw followup responses stream should start"),
            )
            .await;

            assert!(
                second_turn.tool_calls.is_empty(),
                "follow-up raw responses stream should not emit fresh tool calls, saw {:?}",
                second_turn
                    .tool_calls
                    .iter()
                    .map(|tool_call| tool_call.function.name.as_str())
                    .collect::<Vec<_>>()
            );
            assert_raw_stream_text_contains(&second_turn, &[ALPHA_SIGNAL_OUTPUT]);
        },
    )
    .await;
}
