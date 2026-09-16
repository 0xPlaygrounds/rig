//! Reasoning-shape matrix across three Doubleword backend families.
//!
//! Doubleword currently emits hidden thinking in the OpenAI-compatible
//! `reasoning_content` field. Rig uses that spelling when serializing and also
//! accepts `reasoning` as an alias. The matrix crosses Qwen, GPT-OSS and DeepSeek
//! with blocking and streaming transports. It asserts both the normalized
//! reasoning events and the exact recorded wire premise, so a backend that
//! silently moves thinking into plain text cannot leave the suite green.
//!
//! | family | blocking | streaming | recorded finish |
//! |---|---|---|---|
//! | Qwen 3.5 | reasoning block | reasoning deltas | `length` / `length` |
//! | GPT-OSS | reasoning block | reasoning deltas | `stop` / `stop` |
//! | DeepSeek | reasoning block | reasoning deltas | `stop` / `stop` |

use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use rig::providers::{doubleword, openai};
use serde::Deserialize as _;

use super::super::support::{BoundDoubleword, recorded_chat_calls, with_doubleword_cassette};
use crate::support::collect_raw_stream_observation;

const PROMPT: &str = "Compute 17 * 23. Give the number only after thinking.";
const CAP: u64 = 128;

async fn exercise_blocking(client: BoundDoubleword, model_name: &'static str) {
    let model = client.completion(model_name);
    let response = model
        .completion(model.completion_request(PROMPT).max_tokens(CAP).build())
        .await
        .expect("reasoning completion should decode");

    // The wire premise, read off the captured document in the provider's own
    // vocabulary: the backend put hidden thinking in `reasoning_content`.
    let reply = openai::CompletionResponse::deserialize(&response.raw)
        .expect("raw is the shared chat-completions response");
    let has_wire_reasoning = reply.choices.iter().any(|choice| {
        matches!(
            &choice.message,
            openai::completion::Message::Assistant {
                reasoning: Some(reasoning),
                ..
            } if !reasoning.is_empty()
        )
    });
    assert!(
        has_wire_reasoning,
        "the live backend should emit hidden reasoning"
    );

    // And the decoder's one mapping turned it into a reasoning block rather
    // than gluing it onto the answer.
    assert!(response.choice.iter().any(|part| {
        matches!(part, AssistantContent::Reasoning(reasoning) if !reasoning.content.is_empty())
    }));
}

async fn exercise_streaming(client: BoundDoubleword, model_name: &'static str) {
    let model = client.completion(model_name);
    let stream = model
        .stream(model.completion_request(PROMPT).max_tokens(CAP).build())
        .await
        .expect("reasoning stream should connect");
    let observation = collect_raw_stream_observation(stream).await;

    assert!(observation.errors.is_empty(), "{:?}", observation.errors);
    assert!(observation.got_final);
    assert!(
        observation
            .events
            .iter()
            .any(|event| *event == "reasoning" || *event == "reasoning_delta"),
        "stream should surface structured reasoning: {:?}",
        observation.events
    );
}

fn assert_recorded_reasoning(scenario: &str, model: &str, streaming: bool) {
    let calls = recorded_chat_calls(scenario);
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.status, 200);
    assert_eq!(call.request["model"], model);

    if streaming {
        assert!(call.stream_chunks.iter().any(|chunk| {
            let delta = &chunk["choices"][0]["delta"];
            ["reasoning", "reasoning_content"].iter().any(|field| {
                delta[*field]
                    .as_str()
                    .is_some_and(|reasoning| !reasoning.is_empty())
            })
        }));
    } else {
        let response = call.response_json.as_ref().expect("blocking JSON response");
        let message = &response["choices"][0]["message"];
        assert!(["reasoning", "reasoning_content"].iter().any(|field| {
            message[*field]
                .as_str()
                .is_some_and(|reasoning| !reasoning.is_empty())
        }));
    }
}

#[tokio::test]
async fn qwen_reasoning_blocking() {
    const SCENARIO: &str = "reasoning_matrix/qwen_reasoning_blocking";
    with_doubleword_cassette(
        "reasoning_matrix/qwen_reasoning_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::QWEN3_5_9B).await;
        },
    )
    .await;
    assert_recorded_reasoning(SCENARIO, doubleword::QWEN3_5_9B, false);
}

#[tokio::test]
async fn qwen_reasoning_streaming() {
    const SCENARIO: &str = "reasoning_matrix/qwen_reasoning_streaming";
    with_doubleword_cassette(
        "reasoning_matrix/qwen_reasoning_streaming",
        |client| async move {
            exercise_streaming(client, doubleword::QWEN3_5_9B).await;
        },
    )
    .await;
    assert_recorded_reasoning(SCENARIO, doubleword::QWEN3_5_9B, true);
}

#[tokio::test]
async fn gpt_oss_reasoning_blocking() {
    const SCENARIO: &str = "reasoning_matrix/gpt_oss_reasoning_blocking";
    with_doubleword_cassette(
        "reasoning_matrix/gpt_oss_reasoning_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::GPT_OSS_20B).await;
        },
    )
    .await;
    assert_recorded_reasoning(SCENARIO, doubleword::GPT_OSS_20B, false);
}

#[tokio::test]
async fn gpt_oss_reasoning_streaming() {
    const SCENARIO: &str = "reasoning_matrix/gpt_oss_reasoning_streaming";
    with_doubleword_cassette(
        "reasoning_matrix/gpt_oss_reasoning_streaming",
        |client| async move {
            exercise_streaming(client, doubleword::GPT_OSS_20B).await;
        },
    )
    .await;
    assert_recorded_reasoning(SCENARIO, doubleword::GPT_OSS_20B, true);
}

#[tokio::test]
async fn deepseek_reasoning_blocking() {
    const SCENARIO: &str = "reasoning_matrix/deepseek_reasoning_blocking";
    with_doubleword_cassette(
        "reasoning_matrix/deepseek_reasoning_blocking",
        |client| async move {
            exercise_blocking(client, doubleword::DEEPSEEK_V4_FLASH).await;
        },
    )
    .await;
    assert_recorded_reasoning(SCENARIO, doubleword::DEEPSEEK_V4_FLASH, false);
}

#[tokio::test]
async fn deepseek_reasoning_streaming() {
    const SCENARIO: &str = "reasoning_matrix/deepseek_reasoning_streaming";
    with_doubleword_cassette(
        "reasoning_matrix/deepseek_reasoning_streaming",
        |client| async move {
            exercise_streaming(client, doubleword::DEEPSEEK_V4_FLASH).await;
        },
    )
    .await;
    assert_recorded_reasoning(SCENARIO, doubleword::DEEPSEEK_V4_FLASH, true);
}
