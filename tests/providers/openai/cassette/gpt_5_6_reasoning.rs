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
use rig::message::AssistantContent;
use rig::providers::openai;
use rig::streaming::StreamedAssistantContent;
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;

const PROMPT: &str = "Reply with exactly: OK";

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
