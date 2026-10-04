//! Cassette-backed OpenRouter streaming tools coverage.
//!
//! The two encrypted-reasoning scenarios need a model that emits
//! `reasoning_details` of type `reasoning.encrypted` (`openai/o4-mini` with
//! `reasoning.effort: high` + `include_reasoning: true`). Re-record them with:
//! `RIG_PROVIDER_TEST_MODE=record OPENROUTER_API_KEY=... cargo test -p rig --all-features --test openrouter stream_encrypted_reasoning -- --test-threads=1`
use rig::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig::streaming::Item;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::reasoning::WeatherTool;
use crate::support::{
    AlphaSignal, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
    assert_raw_stream_contains_distinct_tool_calls_before_text, collect_raw_stream_observation,
};

use super::super::{TOOL_MODEL, support::with_openrouter_cassette};
use rig::completion::CompletionRequest;

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
    stream: &mut rig::streaming::CompletionStream,
) -> EncryptedReasoningObservation {
    use futures::StreamExt;
    use rig::streaming::StreamEvent;

    let mut observation = EncryptedReasoningObservation {
        errors: Vec::new(),
        streamed_encrypted: Vec::new(),
        tool_calls: Vec::new(),
        text: String::new(),
    };

    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => observation.text.push_str(&text),
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            })) => {
                observation
                    .streamed_encrypted
                    .extend(encrypted_blocks_of(&reasoning));
            }
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => {
                observation.tool_calls.push(tool_call);
            }
            Ok(_) => {}
            Err(error) => observation.errors.push(error.to_string()),
        }
    }

    observation
}

/// The encrypted `reasoning_details` entries a reasoning block holds in its
/// provider item, as `(id, data)`.
fn encrypted_blocks_of(reasoning: &rig::message::Reasoning) -> Vec<(Option<String>, String)> {
    reasoning
        .native
        .as_ref()
        .and_then(|native| native.item.get("reasoning_details"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|detail| detail["type"] == "reasoning.encrypted")
        .filter_map(|detail| {
            Some((
                detail["id"].as_str().map(str::to_owned),
                detail["data"].as_str()?.to_owned(),
            ))
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
            let model = client.completion(ENCRYPTED_REASONING_MODEL);
            let weather_tool = WeatherTool::new(Arc::new(AtomicUsize::new(0)));
            let tool_definition = rig::tool::tool_definition(&weather_tool);
            let reasoning_params = serde_json::json!({
                "reasoning": { "effort": "high" },
                "include_reasoning": true
            });

            let request = CompletionRequest::new(crate::reasoning::TOOL_USER_PROMPT)
                .preamble(crate::reasoning::TOOL_SYSTEM_PROMPT.to_string())
                .max_tokens(4096)
                .tool(tool_definition.clone())
                .additional_params(reasoning_params.clone());

            let mut stream = model.stream(request).expect("stream should start");
            let first_turn = observe_stream(&mut stream).await;
            assert!(
                first_turn.errors.is_empty(),
                "first turn should not emit errors: {:?}",
                first_turn.errors
            );

            let aggregated = encrypted_blocks_in_choice(&stream.partial().choice);
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
            let assistant_message = Message::Assistant(rig::message::AssistantMessage::new(stream.partial().choice));
            let tool_result_message = Message::User {
        content: vec![UserContent::tool_result(tool_call.id.clone(), tool_call.function.name.clone(), vec![ToolResultContent::text("Weather in Tokyo, Japan: 72F (22C), sunny with light clouds, humidity 45%, wind 8 mph NW")])],
    };

            let followup = CompletionRequest::new("Summarize the weather using the tool result.")
                .preamble(crate::reasoning::TOOL_SYSTEM_PROMPT.to_string())
                .max_tokens(4096)
                .tool(tool_definition)
                .additional_params(reasoning_params)
                .message(assistant_message)
                .message(tool_result_message);

            let mut followup_stream = model
                .stream(followup)
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
            let model = client.completion(TOOL_MODEL);
            let request = CompletionRequest::new(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal));

            let observation = collect_raw_stream_observation(
                model.stream(request).expect("raw stream should start"),
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
