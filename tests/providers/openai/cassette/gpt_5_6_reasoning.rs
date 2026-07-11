//! GPT-5.6 reasoning-control regression tests.
//!
//! Locks down the GPT-5.6 model constants and the reasoning controls added for
//! the family: `reasoning.effort = "max"`, `reasoning.mode = "pro"`, and
//! `reasoning.context`. The recorded request bodies pin the wire shape each
//! typed control serializes to, and the fixtures prove the real API accepts
//! them.
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
async fn context_auto() {
    with_openai_cassette("gpt_5_6_reasoning/context_auto", |client| async move {
        let model = client.completion_model(openai::GPT_5_6_TERRA);
        let response =
            prompt_with_reasoning(&model, json!({ "effort": "low", "context": "auto" })).await;

        assert_has_text(&response);
    })
    .await;
}

#[tokio::test]
async fn context_all_turns() {
    with_openai_cassette("gpt_5_6_reasoning/context_all_turns", |client| async move {
        let model = client.completion_model(openai::GPT_5_6_LUNA);
        let response =
            prompt_with_reasoning(&model, json!({ "effort": "low", "context": "all_turns" })).await;

        assert_has_text(&response);
    })
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
        },
    )
    .await;
}
