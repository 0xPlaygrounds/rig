//! GPT-5.6 reasoning-control regression tests.
//!
//! Locks down the GPT-5.6 model constants and verifies that the Responses API
//! accepts `reasoning.effort = "max"`, `reasoning.mode = "pro"`, and
//! `reasoning.context`. Unit tests in the provider module cover every typed
//! context value and optional-field serialization.
//!
//! The normalized `rig::completion::CompletionResponse` no longer carries the
//! provider-typed `raw_response`, so wire-level reasoning-metadata assertions
//! deserialize the recorded cassette bodies into
//! `openai::responses_api::CompletionResponse` directly (replay mode only; in
//! record mode the cassette file is only written after the test body runs).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionResponse};
use rig::message::{AssistantContent, Message, Reasoning};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamedAssistantContent;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;
use crate::cassettes::{CassetteMode, recorded_response_bodies};

const PROMPT: &str = "Reply with exactly: OK";
const FIVE_TURN_PROMPTS: [(&str, &str); 5] = [
    (
        "Remember the codeword ALPHA-17. Reply exactly: ACK-1",
        "ACK-1",
    ),
    (
        "Remember that the shape is octagon. Reply exactly: ACK-2",
        "ACK-2",
    ),
    (
        "Remember that the city is Kyoto. Reply exactly: ACK-3",
        "ACK-3",
    ),
    (
        "Remember that the number is 8642. Reply exactly: ACK-4",
        "ACK-4",
    ),
    (
        "Reply with exactly these remembered values, including capitalization and separators: ALPHA-17 | octagon | Kyoto | 8642",
        "ALPHA-17 | octagon | Kyoto | 8642",
    ),
];

#[derive(Debug, Serialize, Deserialize)]
struct StoredTurn {
    user: Message,
    assistant: Message,
}

fn replaying() -> bool {
    CassetteMode::current() == CassetteMode::Replay
}

/// Recorded wire responses for a non-streaming scenario, in interaction order.
fn recorded_wire_responses(scenario: &str) -> Vec<openai::responses_api::CompletionResponse> {
    recorded_response_bodies("openai", scenario)
        .iter()
        .map(|body| {
            serde_json::from_str(body)
                .expect("recorded response body should deserialize as a Responses API response")
        })
        .collect()
}

/// Recorded terminal (`response.completed`) wire responses for a streaming
/// scenario, in interaction order.
fn recorded_completed_wire_responses(
    scenario: &str,
) -> Vec<openai::responses_api::CompletionResponse> {
    recorded_response_bodies("openai", scenario)
        .iter()
        .map(|body| completed_response_from_sse(body))
        .collect()
}

fn completed_response_from_sse(body: &str) -> openai::responses_api::CompletionResponse {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
        .find(|event| event["type"] == "response.completed")
        .map(|event| {
            serde_json::from_value(event["response"].clone()).expect(
                "recorded response.completed event should deserialize as a Responses API response",
            )
        })
        .expect("recorded SSE body should contain a response.completed event")
}

async fn prompt_with_reasoning<M>(model: &M, reasoning: serde_json::Value) -> CompletionResponse
where
    M: CompletionModel,
{
    let request = model
        .completion_request(PROMPT)
        .additional_params(json!({ "reasoning": reasoning }))
        .build();

    model
        .completion(request)
        .await
        .expect("completion with GPT-5.6 reasoning controls should succeed")
}

#[test]
fn model_constants() {
    assert_eq!(openai::GPT_5_6, "gpt-5.6");
    assert_eq!(openai::GPT_5_6_SOL, "gpt-5.6-sol");
    assert_eq!(openai::GPT_5_6_TERRA, "gpt-5.6-terra");
    assert_eq!(openai::GPT_5_6_LUNA, "gpt-5.6-luna");
}

fn assert_reasoning_metadata(raw: &openai::responses_api::CompletionResponse, expected: Value) {
    let expected = expected
        .as_object()
        .expect("expected reasoning metadata should be an object");
    assert_eq!(
        raw.reasoning_context.as_deref(),
        expected.get("context").and_then(Value::as_str)
    );
    assert_eq!(raw.reasoning_metadata.as_ref(), Some(expected));
    assert_eq!(
        serde_json::to_value(raw).expect("raw response should serialize")["reasoning"],
        Value::Object(expected.clone())
    );
}

fn assert_wire_roundtrip(raw: &openai::responses_api::CompletionResponse, turn: usize) {
    let raw_json = serde_json::to_value(raw).expect("raw response should serialize");
    let roundtripped: openai::responses_api::CompletionResponse =
        serde_json::from_value(raw_json.clone())
            .expect("raw response should deserialize after serialization");
    assert_eq!(
        serde_json::to_value(&roundtripped).expect("roundtripped raw response should serialize"),
        raw_json,
        "all raw response data should survive turn {turn} serialization roundtrip",
    );
}

fn assert_has_text(response: &CompletionResponse) {
    let text: String = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        !text.trim().is_empty(),
        "response should surface output text"
    );
}

#[tokio::test]
async fn effort_max() {
    const SCENARIO: &str = "gpt_5_6_reasoning/effort_max";
    with_openai_cassette("gpt_5_6_reasoning/effort_max", |client| async move {
        let model = client.completion_model(openai::GPT_5_6);
        let response = prompt_with_reasoning(&model, json!({ "effort": "max" })).await;
        assert_has_text(&response);
        if replaying() {
            let raw = recorded_wire_responses(SCENARIO);
            assert_eq!(raw.len(), 1, "scenario should record a single interaction");
            assert_reasoning_metadata(
                &raw[0],
                json!({
                    "context": "all_turns",
                    "effort": "max",
                    "mode": "standard",
                    "summary": null
                }),
            );
        }
    })
    .await;
}

#[tokio::test]
async fn mode_pro_with_independent_effort() {
    const SCENARIO: &str = "gpt_5_6_reasoning/mode_pro_with_independent_effort";
    with_openai_cassette(
        "gpt_5_6_reasoning/mode_pro_with_independent_effort",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6_SOL);
            let response =
                prompt_with_reasoning(&model, json!({ "effort": "high", "mode": "pro" })).await;
            assert_has_text(&response);
            if replaying() {
                let raw = recorded_wire_responses(SCENARIO);
                assert_eq!(raw.len(), 1, "scenario should record a single interaction");
                assert_reasoning_metadata(
                    &raw[0],
                    json!({
                        "context": "all_turns",
                        "effort": "high",
                        "mode": "pro",
                        "summary": null
                    }),
                );
            }
        },
    )
    .await;
}

#[tokio::test]
async fn context_current_turn() {
    const SCENARIO: &str = "gpt_5_6_reasoning/context_current_turn";
    with_openai_cassette(
        "gpt_5_6_reasoning/context_current_turn",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6_SOL);
            let response = prompt_with_reasoning(
                &model,
                json!({ "effort": "low", "context": "current_turn" }),
            )
            .await;
            assert_has_text(&response);
            if replaying() {
                let raw = recorded_wire_responses(SCENARIO);
                assert_eq!(raw.len(), 1, "scenario should record a single interaction");
                assert_reasoning_metadata(
                    &raw[0],
                    json!({
                        "context": "current_turn",
                        "effort": "low",
                        "mode": "standard",
                        "summary": null
                    }),
                );
            }
        },
    )
    .await;
}

#[tokio::test]
async fn five_turn_reasoning_metadata_roundtrip() {
    const SCENARIO: &str = "gpt_5_6_reasoning/five_turn_metadata_roundtrip";
    with_openai_cassette(
        "gpt_5_6_reasoning/five_turn_metadata_roundtrip",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6_SOL);
            let expected_metadata = json!({
                "context": "all_turns",
                "effort": "low",
                "mode": "pro",
                "summary": null
            });
            let mut stored_turns = Vec::<StoredTurn>::new();

            for (turn_index, (prompt, expected_text)) in FIVE_TURN_PROMPTS.into_iter().enumerate() {
                let history = stored_turns
                    .iter()
                    .flat_map(|turn| [turn.user.clone(), turn.assistant.clone()]);
                let user_message = Message::user(prompt);
                let request = model
                    .completion_request(user_message.clone())
                    .messages(history)
                    .additional_params(json!({
                        "reasoning": {
                            "context": "all_turns",
                            "effort": "low",
                            "mode": "pro"
                        }
                    }))
                    .build();
                let response = model.completion(request).await.unwrap_or_else(|error| {
                    panic!("turn {} should succeed: {error}", turn_index + 1)
                });

                assert_has_text(&response);
                let text = response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<String>();
                assert_eq!(
                    text.trim(),
                    expected_text,
                    "unexpected turn {} text",
                    turn_index + 1
                );

                stored_turns.push(StoredTurn {
                    user: user_message,
                    assistant: Message::Assistant {
                        id: response.message_id,
                        content: response.choice,
                    },
                });
                let stored_json =
                    serde_json::to_value(&stored_turns).expect("all stored turns should serialize");
                stored_turns = serde_json::from_value(stored_json.clone())
                    .expect("all stored turns should deserialize before the next request");
                assert_eq!(
                    serde_json::to_value(&stored_turns).expect("restored turns should serialize"),
                    stored_json,
                    "all session data should survive persistence after turn {}",
                    turn_index + 1
                );
                assert_eq!(stored_turns.len(), turn_index + 1);
            }

            if replaying() {
                let wire_responses = recorded_wire_responses(SCENARIO);
                assert_eq!(
                    wire_responses.len(),
                    FIVE_TURN_PROMPTS.len(),
                    "scenario should record one interaction per turn"
                );
                for (turn_index, raw) in wire_responses.iter().enumerate() {
                    assert_reasoning_metadata(raw, expected_metadata.clone());
                    assert_eq!(raw.reasoning_context.as_deref(), Some("all_turns"));
                    assert_wire_roundtrip(raw, turn_index + 1);
                }
            }
        },
    )
    .await;
}

#[tokio::test]
async fn five_turn_streaming_reasoning_metadata_roundtrip() {
    const SCENARIO: &str = "gpt_5_6_reasoning/five_turn_streaming_metadata_roundtrip";
    with_openai_cassette(
        "gpt_5_6_reasoning/five_turn_streaming_metadata_roundtrip",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6_SOL);
            let expected_metadata = json!({
                "context": "all_turns",
                "effort": "low",
                "mode": "pro",
                "summary": null
            });
            let mut stored_turns = Vec::<StoredTurn>::new();

            for (turn_index, (prompt, expected_text)) in FIVE_TURN_PROMPTS.into_iter().enumerate() {
                let history = stored_turns
                    .iter()
                    .flat_map(|turn| [turn.user.clone(), turn.assistant.clone()]);
                let user_message = Message::user(prompt);
                let request = model
                    .completion_request(user_message.clone())
                    .messages(history)
                    .additional_params(json!({
                        "reasoning": {
                            "context": "all_turns",
                            "effort": "low",
                            "mode": "pro"
                        }
                    }))
                    .build();
                let mut stream = model.stream(request).await.unwrap_or_else(|error| {
                    panic!("turn {} stream should start: {error}", turn_index + 1)
                });
                let mut text = String::new();
                let mut reasoning_blocks = Vec::new();
                let mut reasoning_delta = String::new();
                let mut final_response = None;

                while let Some(item) = stream.next().await {
                    match item.unwrap_or_else(|error| {
                        panic!("turn {} stream should succeed: {error}", turn_index + 1)
                    }) {
                        StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
                        StreamedAssistantContent::Reasoning(reasoning) => {
                            reasoning_blocks.push(AssistantContent::Reasoning(reasoning));
                        }
                        StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                            reasoning_delta.push_str(&reasoning);
                        }
                        StreamedAssistantContent::Final(response) => {
                            final_response = Some(response);
                        }
                        _ => {}
                    }
                }

                assert_eq!(
                    text.trim(),
                    expected_text,
                    "unexpected turn {} text",
                    turn_index + 1
                );
                let _final_response = final_response.unwrap_or_else(|| {
                    panic!("turn {} should yield a final response", turn_index + 1)
                });

                if reasoning_blocks.is_empty() && !reasoning_delta.is_empty() {
                    reasoning_blocks.push(AssistantContent::Reasoning(Reasoning::new(
                        &reasoning_delta,
                    )));
                }
                reasoning_blocks.push(AssistantContent::text(&text));
                stored_turns.push(StoredTurn {
                    user: user_message,
                    assistant: Message::Assistant {
                        id: stream.message_id.clone(),
                        content: rig::OneOrMany::many(reasoning_blocks)
                            .expect("streamed assistant message should not be empty"),
                    },
                });
                let stored_json = serde_json::to_value(&stored_turns)
                    .expect("all stored streaming turns should serialize");
                stored_turns = serde_json::from_value(stored_json.clone()).expect(
                    "all stored streaming turns should deserialize before the next request",
                );
                assert_eq!(
                    serde_json::to_value(&stored_turns)
                        .expect("restored streaming turns should serialize"),
                    stored_json,
                    "all streaming session data should survive persistence after turn {}",
                    turn_index + 1
                );
                assert_eq!(stored_turns.len(), turn_index + 1);
            }

            if replaying() {
                let wire_responses = recorded_completed_wire_responses(SCENARIO);
                assert_eq!(
                    wire_responses.len(),
                    FIVE_TURN_PROMPTS.len(),
                    "scenario should record one interaction per turn"
                );
                for (turn_index, raw) in wire_responses.iter().enumerate() {
                    assert_reasoning_metadata(raw, expected_metadata.clone());
                    assert_eq!(raw.reasoning_context.as_deref(), Some("all_turns"));
                    assert_wire_roundtrip(raw, turn_index + 1);
                }
            }
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_reasoning_metadata() {
    const SCENARIO: &str = "gpt_5_6_reasoning/streaming_metadata";
    with_openai_cassette(
        "gpt_5_6_reasoning/streaming_metadata",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6_SOL);
            let request = model
                .completion_request(PROMPT)
                .additional_params(json!({
                    "reasoning": {
                        "effort": "low",
                        "mode": "pro",
                        "context": "current_turn"
                    }
                }))
                .build();
            let mut stream = model
                .stream(request)
                .await
                .expect("GPT-5.6 reasoning stream should start");
            let mut saw_final = false;

            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(_) =
                    item.expect("GPT-5.6 reasoning stream should succeed")
                {
                    saw_final = true;
                }
            }

            assert!(
                saw_final,
                "GPT-5.6 reasoning stream should yield a final response"
            );

            if replaying() {
                let raw = recorded_completed_wire_responses(SCENARIO);
                assert_eq!(raw.len(), 1, "scenario should record a single interaction");
                assert_eq!(raw[0].reasoning_context.as_deref(), Some("current_turn"));
                assert_reasoning_metadata(
                    &raw[0],
                    json!({
                        "context": "current_turn",
                        "effort": "low",
                        "mode": "pro",
                        "summary": null
                    }),
                );
            }
        },
    )
    .await;
}
