//! GPT-5.6 reasoning-control regression tests.
//!
//! Locks down the GPT-5.6 model constants and verifies that the Responses API
//! accepts `reasoning.effort = "max"`, `reasoning.mode = "pro"`, and
//! `reasoning.context`. Unit tests in the provider module cover every typed
//! context value and optional-field serialization.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, CompletionResponse};
use rig::message::AssistantContent;
use rig::providers::openai;
use serde_json::json;

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
            assert_eq!(
                response.raw_response.reasoning_context.as_deref(),
                Some("current_turn")
            );
        },
    )
    .await;
}
