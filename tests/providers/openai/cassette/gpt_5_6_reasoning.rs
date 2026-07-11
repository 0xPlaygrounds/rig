//! GPT-5.6 reasoning-control regression tests.
//!
//! Locks down the GPT-5.6 model constants and verifies that the Responses API
//! accepts `reasoning.effort = "max"`, `reasoning.mode = "pro"`, and
//! `reasoning.context`. Unit tests in the provider module cover every typed
//! context value and optional-field serialization.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, CompletionResponse};
use rig::message::{AssistantContent, Message, Reasoning};
use rig::providers::openai;
use rig::streaming::StreamedAssistantContent;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;

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
struct StoredResponseTurn {
    user: Message,
    assistant: Message,
    raw_response: openai::responses_api::CompletionResponse,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredStreamingTurn {
    user: Message,
    assistant: Message,
    final_response: openai::responses_api::streaming::StreamingCompletionResponse,
}

async fn prompt_with_reasoning<M>(
    model: &M,
    reasoning: serde_json::Value,
) -> CompletionResponse<M::Response>
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

fn assert_reasoning_metadata(
    response: &CompletionResponse<openai::responses_api::CompletionResponse>,
    expected: Value,
) {
    let expected = expected
        .as_object()
        .expect("expected reasoning metadata should be an object");
    assert_eq!(
        response.raw_response.reasoning_context.as_deref(),
        expected.get("context").and_then(Value::as_str)
    );
    assert_eq!(
        response.raw_response.reasoning_metadata.as_ref(),
        Some(expected)
    );
    assert_eq!(
        serde_json::to_value(&response.raw_response).expect("raw response should serialize")["reasoning"],
        Value::Object(expected.clone())
    );
}

fn assert_has_text(response: &CompletionResponse<openai::responses_api::CompletionResponse>) {
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
    with_openai_cassette("gpt_5_6_reasoning/effort_max", |client| async move {
        let model = client.completion_model(openai::GPT_5_6);
        let response = prompt_with_reasoning(&model, json!({ "effort": "max" })).await;
        assert_has_text(&response);
        assert_reasoning_metadata(
            &response,
            json!({
                "context": "all_turns",
                "effort": "max",
                "mode": "standard",
                "summary": null
            }),
        );
    })
    .await;
}

#[tokio::test]
async fn mode_pro_with_independent_effort() {
    with_openai_cassette(
        "gpt_5_6_reasoning/mode_pro_with_independent_effort",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6_SOL);
            let response =
                prompt_with_reasoning(&model, json!({ "effort": "high", "mode": "pro" })).await;
            assert_has_text(&response);
            assert_reasoning_metadata(
                &response,
                json!({
                    "context": "all_turns",
                    "effort": "high",
                    "mode": "pro",
                    "summary": null
                }),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn context_current_turn() {
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
            assert_reasoning_metadata(
                &response,
                json!({
                    "context": "current_turn",
                    "effort": "low",
                    "mode": "standard",
                    "summary": null
                }),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn five_turn_reasoning_metadata_roundtrip() {
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
            let mut stored_turns = Vec::<StoredResponseTurn>::new();

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
                assert_reasoning_metadata(&response, expected_metadata.clone());
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

                let raw_json = serde_json::to_value(&response.raw_response)
                    .expect("raw response should serialize");
                let roundtripped: openai::responses_api::CompletionResponse =
                    serde_json::from_value(raw_json.clone())
                        .expect("raw response should deserialize after serialization");
                assert_eq!(
                    serde_json::to_value(&roundtripped)
                        .expect("roundtripped raw response should serialize"),
                    raw_json,
                    "all raw response data should survive turn {} serialization roundtrip",
                    turn_index + 1
                );

                stored_turns.push(StoredResponseTurn {
                    user: user_message,
                    assistant: Message::Assistant {
                        id: response.message_id,
                        content: response.choice,
                    },
                    raw_response: response.raw_response,
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
                for (prior_turn, stored) in stored_turns.iter().enumerate() {
                    assert_eq!(
                        stored.raw_response.reasoning_metadata.as_ref(),
                        expected_metadata.as_object(),
                        "reasoning metadata from turn {} changed by turn {}",
                        prior_turn + 1,
                        turn_index + 1
                    );
                    assert_eq!(
                        stored.raw_response.reasoning_context.as_deref(),
                        Some("all_turns")
                    );
                }
            }
        },
    )
    .await;
}

#[tokio::test]
async fn five_turn_streaming_reasoning_metadata_roundtrip() {
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
            let mut stored_turns = Vec::<StoredStreamingTurn>::new();

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
                let final_response = final_response.unwrap_or_else(|| {
                    panic!("turn {} should yield a final response", turn_index + 1)
                });
                assert_eq!(
                    final_response.reasoning_context.as_deref(),
                    Some("all_turns")
                );
                assert_eq!(
                    final_response.reasoning_metadata.as_ref(),
                    expected_metadata.as_object()
                );
                let final_json = serde_json::to_value(&final_response)
                    .expect("final streaming response should serialize");
                let roundtripped: openai::responses_api::streaming::StreamingCompletionResponse =
                    serde_json::from_value(final_json.clone())
                        .expect("final streaming response should deserialize after serialization");
                assert_eq!(
                    serde_json::to_value(&roundtripped)
                        .expect("roundtripped streaming response should serialize"),
                    final_json,
                    "all final streaming data should survive turn {} serialization roundtrip",
                    turn_index + 1
                );

                if reasoning_blocks.is_empty() && !reasoning_delta.is_empty() {
                    reasoning_blocks.push(AssistantContent::Reasoning(Reasoning::new(
                        &reasoning_delta,
                    )));
                }
                reasoning_blocks.push(AssistantContent::text(&text));
                stored_turns.push(StoredStreamingTurn {
                    user: user_message,
                    assistant: Message::Assistant {
                        id: stream.message_id.clone(),
                        content: rig::OneOrMany::many(reasoning_blocks)
                            .expect("streamed assistant message should not be empty"),
                    },
                    final_response,
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
                for (prior_turn, stored) in stored_turns.iter().enumerate() {
                    assert_eq!(
                        stored.final_response.reasoning_metadata.as_ref(),
                        expected_metadata.as_object(),
                        "streaming reasoning metadata from turn {} changed by turn {}",
                        prior_turn + 1,
                        turn_index + 1
                    );
                    assert_eq!(
                        stored.final_response.reasoning_context.as_deref(),
                        Some("all_turns")
                    );
                }
            }
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_reasoning_metadata() {
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
            let expected = json!({
                "context": "current_turn",
                "effort": "low",
                "mode": "pro",
                "summary": null
            });

            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(response) =
                    item.expect("GPT-5.6 reasoning stream should succeed")
                {
                    assert_eq!(response.reasoning_context.as_deref(), Some("current_turn"));
                    assert_eq!(response.reasoning_metadata.as_ref(), expected.as_object());
                    assert_eq!(
                        serde_json::to_value(&response)
                            .expect("streaming response should serialize")["reasoning_metadata"],
                        expected
                    );
                    return;
                }
            }

            panic!("GPT-5.6 reasoning stream should yield a final response");
        },
    )
    .await;
}
