//! Cassette-backed OpenRouter streaming tools coverage.
//!
//! The two encrypted-reasoning scenarios need a model that emits
//! `reasoning_details` of type `reasoning.encrypted` (`openai/o4-mini` with
//! `reasoning.effort: high` + `include_reasoning: true`). Re-record them with:
//! `RIG_PROVIDER_TEST_MODE=record OPENROUTER_API_KEY=... cargo test -p rig --all-features --test openrouter stream_encrypted_reasoning -- --test-threads=1`
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig::prelude::*;
use rig::streaming::StreamingPrompt;
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

use super::super::{TOOL_MODEL, support::with_openrouter_cassette};

#[tokio::test]
async fn streaming_tools_smoke() {
    with_openrouter_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(TOOL_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
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

/// Model whose OpenRouter turns carry encrypted `reasoning_details`
/// (`{"type":"reasoning.encrypted"}` with `reasoning: null`). The
/// `reasoning`/`include_reasoning` parameters below are what make it emit them.
const ENCRYPTED_REASONING_MODEL: &str = "openai/o4-mini";

/// Observation of one streamed turn: the encrypted reasoning blocks the wire
/// delivered, the tool calls, and any stream errors.
struct EncryptedReasoningObservation {
    errors: Vec<String>,
    streamed_encrypted: Vec<(Option<String>, String)>,
    tool_calls: Vec<rig::message::ToolCall>,
    text: String,
}

async fn observe_stream(
    stream: &mut rig::streaming::StreamingCompletionResponse,
) -> EncryptedReasoningObservation {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    let mut observation = EncryptedReasoningObservation {
        errors: Vec::new(),
        streamed_encrypted: Vec::new(),
        tool_calls: Vec::new(),
        text: String::new(),
    };

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Text(text)) => observation.text.push_str(&text.text),
            Ok(StreamedAssistantContent::Reasoning { reasoning, .. }) => {
                observation
                    .streamed_encrypted
                    .extend(encrypted_blocks_of(&reasoning));
            }
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                observation.tool_calls.push(tool_call);
            }
            Ok(_) => {}
            Err(error) => observation.errors.push(error.to_string()),
        }
    }

    observation
}

fn encrypted_blocks_of(reasoning: &rig::message::Reasoning) -> Vec<(Option<String>, String)> {
    reasoning
        .content
        .iter()
        .filter_map(|content| match content {
            rig::message::ReasoningContent::Encrypted(data) => {
                Some((reasoning.id.clone(), data.clone()))
            }
            _ => None,
        })
        .collect()
}

fn encrypted_blocks_in_choice(choice: &[AssistantContent]) -> Vec<(Option<String>, String)> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(encrypted_blocks_of(reasoning)),
            _ => None,
        })
        .flatten()
        .collect()
}

/// OpenRouter delivers encrypted reasoning as a `reasoning_details` entry with
/// `reasoning: null` and an `rs_*` id of its own, one chunk before the `call_*`
/// tool call opens. It is the turn's own output, so it must reach the
/// aggregated choice — routing it through tool-call decoration matched nothing
/// (the two id namespaces never intersect) and dropped it on every streaming
/// turn.
#[tokio::test]
async fn stream_encrypted_reasoning_reaches_the_choice() {
    with_openrouter_cassette(
        "streaming_tools/stream_encrypted_reasoning_reaches_the_choice",
        |client| async move {
            let model = client.completion_model(ENCRYPTED_REASONING_MODEL);
            let weather_tool = WeatherTool::new(Arc::new(AtomicUsize::new(0)));
            let tool_definition = rig::tool::tool_definition(&weather_tool);
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

            let mut stream = model.stream(request).await.expect("stream should start");
            let observation = observe_stream(&mut stream).await;
            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );

            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == "get_weather")
                .expect("expected a streamed get_weather tool call");
            // The encrypted blob is not tool-call metadata: a decoration keyed
            // by the detail's own `rs_*` id could never match this `call_*` id.
            assert!(
                tool_call.signature.is_none() && tool_call.additional_params.is_none(),
                "encrypted reasoning must not ride on the tool call: signature={:?} additional_params={:?}",
                tool_call.signature,
                tool_call.additional_params
            );

            let streamed = observation.streamed_encrypted.clone();
            assert!(
                !streamed.is_empty(),
                "the recorded turn carries encrypted reasoning_details; the stream must emit them as reasoning blocks"
            );

            let aggregated = encrypted_blocks_in_choice(&stream.choice);
            assert_eq!(
                aggregated, streamed,
                "every streamed encrypted reasoning block must reach the aggregated choice"
            );
            for (id, data) in &aggregated {
                assert!(
                    id.as_deref().is_some_and(|id| id.starts_with("rs_")),
                    "the block must keep the wire's own reasoning id, got {id:?}"
                );
                assert!(!data.is_empty(), "the encrypted payload must be preserved");
            }
        },
    )
    .await;
}

/// The round trip: an encrypted reasoning block that reaches the choice is
/// replayed on the next turn's request, next to the tool call it belongs to.
///
/// Turn 2's recorded request body carries the `reasoning.encrypted` entry, and
/// the cassette matches requests on their (canonicalized) body — so this test
/// only replays if the blob really survives history conversion. Before the fix
/// the blob was dropped during the stream and turn 2 went out without it.
#[tokio::test]
async fn stream_encrypted_reasoning_survives_into_the_next_turn() {
    with_openrouter_cassette(
        "streaming_tools/stream_encrypted_reasoning_survives_into_the_next_turn",
        |client| async move {
            let model = client.completion_model(ENCRYPTED_REASONING_MODEL);
            let weather_tool = WeatherTool::new(Arc::new(AtomicUsize::new(0)));
            let tool_definition = rig::tool::tool_definition(&weather_tool);
            let reasoning_params = serde_json::json!({
                "reasoning": { "effort": "high" },
                "include_reasoning": true
            });

            let request = model
                .completion_request(crate::reasoning::TOOL_USER_PROMPT)
                .preamble(crate::reasoning::TOOL_SYSTEM_PROMPT.to_string())
                .max_tokens(4096)
                .tool(tool_definition.clone())
                .additional_params(reasoning_params.clone())
                .build();

            let mut stream = model.stream(request).await.expect("stream should start");
            let first_turn = observe_stream(&mut stream).await;
            assert!(
                first_turn.errors.is_empty(),
                "first turn should not emit errors: {:?}",
                first_turn.errors
            );

            let aggregated = encrypted_blocks_in_choice(&stream.choice);
            assert!(
                !aggregated.is_empty(),
                "first turn should aggregate the encrypted reasoning block"
            );

            let tool_call = first_turn
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == "get_weather")
                .cloned()
                .expect("first turn should call get_weather");

            // The whole choice — reasoning block included — is what a caller
            // replays as history.
            let assistant_message = Message::Assistant {
                id: stream.message_id.clone(),
                content: stream.choice.clone(),
            };
            let tool_result_message = Message::User {
        content: vec![UserContent::tool_result_for(
            tool_call.id.clone(),
            tool_call.provider.clone(),
            tool_call.function.name.clone(),
            vec![ToolResultContent::text("Weather in Tokyo, Japan: 72F (22C), sunny with light clouds, humidity 45%, wind 8 mph NW")],
        )],
    };

            let followup = model
                .completion_request("Summarize the weather using the tool result.")
                .preamble(crate::reasoning::TOOL_SYSTEM_PROMPT.to_string())
                .max_tokens(4096)
                .tool(tool_definition)
                .additional_params(reasoning_params)
                .message(assistant_message)
                .message(tool_result_message)
                .build();

            let mut followup_stream = model
                .stream(followup)
                .await
                .expect("follow-up stream should start");
            let second_turn = observe_stream(&mut followup_stream).await;

            assert!(
                second_turn.errors.is_empty(),
                "follow-up turn should not emit errors (a rejected reasoning replay surfaces here): {:?}",
                second_turn.errors
            );
            assert!(
                !second_turn.text.trim().is_empty(),
                "follow-up turn should produce a summary"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    with_openrouter_cassette(
        "streaming_tools/raw_stream_surfaces_two_distinct_tool_calls_before_text",
        |client| async move {
            let model = client.completion_model(TOOL_MODEL);
            let request = model
                .completion_request(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
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
        },
    )
    .await;
}

#[tokio::test]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    with_openrouter_cassette(
        "streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls",
        |client| async move {
            let model = client.completion_model(TOOL_MODEL);
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
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
                content: vec![AssistantContent::ToolCall(tool_call.clone())],
            };
            let tool_result_message = Message::User {
        content: vec![UserContent::tool_result_for(
            tool_call.id.clone(),
            tool_call.provider.clone(),
            tool_call.function.name.clone(),
            vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
        )],
    };
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
            let expected_words = ALPHA_SIGNAL_OUTPUT
                .split('-')
                .collect::<Vec<_>>();
            assert_raw_stream_text_contains(&second_turn, &expected_words);
        },
    )
    .await;
}
