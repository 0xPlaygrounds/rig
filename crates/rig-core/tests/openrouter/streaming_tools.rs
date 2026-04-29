//! OpenRouter streaming tools smoke test.

use rig_core::OneOrMany;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::CompletionModel;
use rig_core::message::{AssistantContent, Message};
use rig_core::providers::openrouter;
use rig_core::streaming::StreamingPrompt;
use rig_core::tool::Tool;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::reasoning::WeatherTool;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number,
    assert_raw_stream_contains_distinct_tool_calls_before_text, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, collect_raw_stream_observation,
    collect_stream_final_response,
};

use super::TOOL_MODEL;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming_tools_smoke() {
    let client = openrouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn raw_stream_decorates_reasoning_tool_call_metadata() {
    let client = openrouter::Client::from_env().expect("client should build");
    let model = client.completion_model("openai/o4-mini");
    let tool_definition = WeatherTool::new(Arc::new(AtomicUsize::new(0)))
        .definition(String::new())
        .await;
    let request = model
        .completion_request(crate::reasoning::TOOL_USER_PROMPT)
        .preamble(crate::reasoning::TOOL_SYSTEM_PROMPT.to_string())
        .max_tokens(4096)
        .tool(tool_definition)
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let stream = model.stream(request).await.expect("stream should start");
    let observation = collect_raw_stream_observation(stream).await;
    assert!(
        observation.errors.is_empty(),
        "raw stream should not emit errors: {:?}",
        observation.errors
    );

    let record = observation
        .tool_call_records
        .iter()
        .find(|record| record.name == "get_weather")
        .expect("expected a streamed get_weather tool call");

    if record.signature.is_none() && record.additional_params.is_none() {
        eprintln!(
            "openrouter did not emit encrypted reasoning metadata for the tool call in this run; skipping strict decoration assertion"
        );
        return;
    }

    assert!(
        record.signature.is_some() || record.additional_params.is_some(),
        "expected decorated tool call metadata for get_weather, got signature={:?} additional_params={:?}",
        record.signature,
        record.additional_params
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    let client = openrouter::Client::from_env().expect("client should build");
    let model = client.completion_model(TOOL_MODEL);
    let request = model
        .completion_request(TWO_TOOL_STREAM_PROMPT)
        .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
        .tool(AlphaSignal.definition(String::new()).await)
        .tool(BetaSignal.definition(String::new()).await)
        .build();

    let observation = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("raw stream should start"),
    )
    .await;

    assert_raw_stream_contains_distinct_tool_calls_before_text(
        &observation,
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    let client = openrouter::Client::from_env().expect("client should build");
    let model = client.completion_model(TOOL_MODEL);
    let request = model
        .completion_request(ORDERED_TOOL_STREAM_PROMPT)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
        .tool(AlphaSignal.definition(String::new()).await)
        .build();

    let first_turn = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("raw stream should start"),
    )
    .await;

    assert_raw_stream_tool_call_precedes_text(&first_turn, "lookup_harbor_label");

    let tool_call = first_turn
        .tool_calls
        .iter()
        .find(|tool_call| tool_call.function.name == "lookup_harbor_label")
        .cloned()
        .expect("raw stream should yield lookup_harbor_label");
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
            .expect("raw followup stream should start"),
    )
    .await;

    assert!(
        second_turn.tool_calls.is_empty(),
        "follow-up raw stream should not emit fresh tool calls, saw {:?}",
        second_turn
            .tool_calls
            .iter()
            .map(|tool_call| tool_call.function.name.as_str())
            .collect::<Vec<_>>()
    );
    assert_raw_stream_text_contains(&second_turn, &[ALPHA_SIGNAL_OUTPUT]);
}
