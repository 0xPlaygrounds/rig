//! Live census of the completion model families exposed by Doubleword.
//!
//! The provider uses one OpenAI-compatible adapter for model backends with
//! different response implementations. These cells keep a small blocking
//! probe for six families plus a streaming parity probe for the default Qwen
//! route. Each cell checks its own fixture's request model, response model,
//! status, choices and usage rather than treating a successful replay as
//! proof that the intended backend was exercised.
//!
//! | family | model | transport | recorded finish |
//! |---|---|---|---|
//! | Qwen 3.5 | `Qwen/Qwen3.5-9B` | blocking | `length` |
//! | Qwen 3.6 | `Qwen/Qwen3.6-35B-A3B-FP8` | blocking | `length` |
//! | GPT-OSS | `openai/gpt-oss-20b` | blocking | `stop` |
//! | DeepSeek | `deepseek-ai/DeepSeek-V4-Flash` | blocking | `stop` |
//! | Kimi | `moonshotai/Kimi-K2.6` | blocking | `stop` |
//! | GLM | `zai-org/GLM-5.2-FP8` | blocking | `length` |
//! | Qwen 3.5 parity | `Qwen/Qwen3.5-9B` | streaming | `length` |
//!
//! The 96-token cap deliberately exposes a backend split: Qwen and GLM spend
//! the whole budget in reasoning, while GPT-OSS, DeepSeek and Kimi reach the
//! requested text. This matrix checks transport integrity, not instruction
//! compliance; the finish-reason matrix tests termination semantics directly.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::doubleword;

use super::super::support::{recorded_chat_calls, with_doubleword_cassette};
use crate::support::collect_text_and_terminal;

const PROMPT: &str = "Reply with the single word: family-ok";
const CAP: u64 = 96;

async fn exercise_blocking(client: doubleword::Client, model_name: &'static str) {
    let model = client.completion_model(model_name);
    let raw = model
        .raw_completion(model.completion_request(PROMPT).max_tokens(CAP).build())
        .await
        .expect("the advertised model should answer a blocking request");

    assert!(!raw.id.is_empty());
    assert_eq!(raw.model, model_name);
    assert!(!raw.choices.is_empty());
    assert!(raw.usage.is_some(), "the live route should report usage");
}

fn assert_recorded_model(scenario: &str, requested_model: &str, streaming: bool) {
    let calls = recorded_chat_calls(scenario);
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.status, 200);
    assert_eq!(call.request["model"], requested_model);
    assert_eq!(
        call.request.get("stream").and_then(|v| v.as_bool()),
        streaming.then_some(true)
    );

    if streaming {
        assert!(!call.stream_chunks.is_empty());
        let response_models = call
            .stream_chunks
            .iter()
            .filter_map(|chunk| chunk["model"].as_str())
            .collect::<Vec<_>>();
        assert!(!response_models.is_empty());
        assert!(
            response_models
                .iter()
                .all(|model| *model == requested_model)
        );
        assert!(call.stream_chunks.iter().any(|chunk| {
            chunk["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
        }));
    } else {
        let response = call.response_json.as_ref().expect("blocking JSON response");
        assert_eq!(response["model"], requested_model);
        assert!(
            response["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
        );
        assert!(response.get("usage").is_some_and(|usage| usage.is_object()));
    }
}

#[tokio::test]
async fn qwen_3_5_family_blocking() {
    const SCENARIO: &str = "model_family_matrix/qwen_3_5_family_blocking";
    with_doubleword_cassette(
        "model_family_matrix/qwen_3_5_family_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::QWEN3_5_9B).await;
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::QWEN3_5_9B, false);
}

#[tokio::test]
async fn qwen_3_6_family_blocking() {
    const SCENARIO: &str = "model_family_matrix/qwen_3_6_family_blocking";
    with_doubleword_cassette(
        "model_family_matrix/qwen_3_6_family_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::QWEN3_6_35B_A3B).await;
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::QWEN3_6_35B_A3B, false);
}

#[tokio::test]
async fn gpt_oss_family_blocking() {
    const SCENARIO: &str = "model_family_matrix/gpt_oss_family_blocking";
    with_doubleword_cassette(
        "model_family_matrix/gpt_oss_family_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::GPT_OSS_20B).await;
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::GPT_OSS_20B, false);
}

#[tokio::test]
async fn deepseek_family_blocking() {
    const SCENARIO: &str = "model_family_matrix/deepseek_family_blocking";
    with_doubleword_cassette(
        "model_family_matrix/deepseek_family_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::DEEPSEEK_V4_FLASH).await;
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::DEEPSEEK_V4_FLASH, false);
}

#[tokio::test]
async fn kimi_family_blocking() {
    const SCENARIO: &str = "model_family_matrix/kimi_family_blocking";
    with_doubleword_cassette(
        "model_family_matrix/kimi_family_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::KIMI_K2_6).await;
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::KIMI_K2_6, false);
}

#[tokio::test]
async fn glm_family_blocking() {
    const SCENARIO: &str = "model_family_matrix/glm_family_blocking";
    with_doubleword_cassette(
        "model_family_matrix/glm_family_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::GLM_5_2).await;
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::GLM_5_2, false);
}

#[tokio::test]
async fn default_qwen_family_streaming() {
    const SCENARIO: &str = "model_family_matrix/default_qwen_family_streaming";
    with_doubleword_cassette(
        "model_family_matrix/default_qwen_family_streaming",
        |client| async move {
            let model = client.completion_model(doubleword::QWEN3_5_9B);
            let stream = model
                .stream(model.completion_request(PROMPT).max_tokens(CAP).build())
                .await
                .expect("the default model stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("the stream should end with a terminal record");
            assert!(terminal.usage.total_tokens > 0);
        },
    )
    .await;
    assert_recorded_model(SCENARIO, doubleword::QWEN3_5_9B, true);
}
