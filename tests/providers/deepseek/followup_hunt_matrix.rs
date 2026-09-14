//! Follow-up live census after the first DeepSeek bug-hunt sweep.
//!
//! These cells target documented Chat Completions fields absent from the
//! existing cassette assertions. They pin both the request premise and the
//! provider's raw response shape. A census cell that exposes a Rig defect is
//! moved into a dedicated per-bug exhaustive matrix before merge.

use rig::completion::{CompletionModel, NormalizeCompletionResponse};
use rig::prelude::*;
use rig::providers::deepseek;
use serde_json::{Value, json};

use super::support::{
    collect_raw_stream_outcome, recorded_request, recorded_response, recorded_stream_chunks,
    with_deepseek_followup_hunt_cassette_result,
};

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

fn non_thinking(extra: Value) -> Value {
    let mut params = json!({ "thinking": { "type": "disabled" } });
    params
        .as_object_mut()
        .expect("params are an object")
        .extend(
            extra
                .as_object()
                .expect("extra params are an object")
                .clone(),
        );
    params
}

fn first_blocking_choice(scenario: &str) -> Value {
    recorded_response(scenario)["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or_else(|| panic!("{scenario} should record a completion choice"))
}

fn stream_choices(scenario: &str) -> Vec<Value> {
    recorded_stream_chunks(scenario)
        .into_iter()
        .flat_map(|chunk| chunk["choices"].as_array().cloned().unwrap_or_default())
        .collect()
}

#[tokio::test]
async fn blocking_stop_sequence_reaches_the_wire_and_stops_generation() {
    const SCENARIO: &str =
        "followup_hunt_matrix/blocking_stop_sequence_reaches_the_wire_and_stops_generation";
    with_deepseek_followup_hunt_cassette_result(
        "followup_hunt_matrix/blocking_stop_sequence_reaches_the_wire_and_stops_generation",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request(
                    "Write exactly `alpha ZEBRA omega` with no punctuation or explanation.",
                )
                .additional_params(non_thinking(json!({ "stop": ["ZEBRA"] })))
                .max_tokens(24)
                .build();
            let response = model.completion(request).await?;
            assert_eq!(
                response.finish_reason(),
                Some(rig::completion::FinishReason::Stop)
            );
            Ok::<(), rig::completion::CompletionError>(())
        },
    )
    .await
    .expect("blocking stop-sequence census should replay");

    assert_eq!(recorded_request(SCENARIO)["stop"], json!(["ZEBRA"]));
    let choice = first_blocking_choice(SCENARIO);
    assert_eq!(choice["finish_reason"], "stop");
    assert!(
        !choice["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .contains("ZEBRA"),
        "the stop sequence itself must not be returned"
    );
}

#[tokio::test]
async fn streaming_stop_sequence_reaches_the_wire_and_stops_generation() {
    const SCENARIO: &str =
        "followup_hunt_matrix/streaming_stop_sequence_reaches_the_wire_and_stops_generation";
    with_deepseek_followup_hunt_cassette_result(
        "followup_hunt_matrix/streaming_stop_sequence_reaches_the_wire_and_stops_generation",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request(
                    "Write exactly `alpha ZEBRA omega` with no punctuation or explanation.",
                )
                .additional_params(non_thinking(json!({ "stop": ["ZEBRA"] })))
                .max_tokens(24)
                .build();
            let outcome = collect_raw_stream_outcome(model.stream(request).await?).await;
            assert!(
                outcome.errors.is_empty(),
                "stream errors: {:?}",
                outcome.errors
            );
            assert_eq!(
                outcome.finish_reason(),
                Some(rig::completion::FinishReason::Stop)
            );
            assert!(!outcome.text.contains("ZEBRA"));
            Ok::<(), rig::completion::CompletionError>(())
        },
    )
    .await
    .expect("streaming stop-sequence census should replay");

    assert_eq!(recorded_request(SCENARIO)["stop"], json!(["ZEBRA"]));
    assert!(
        stream_choices(SCENARIO)
            .iter()
            .any(|choice| choice["finish_reason"] == "stop")
    );
}

#[tokio::test]
async fn low_reasoning_effort_produces_a_reasoning_turn() {
    const SCENARIO: &str = "followup_hunt_matrix/low_reasoning_effort_produces_a_reasoning_turn";
    with_deepseek_followup_hunt_cassette_result(
        "followup_hunt_matrix/low_reasoning_effort_produces_a_reasoning_turn",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request("What is 17 + 25? Answer with only the number.")
                .additional_params(json!({ "reasoning_effort": "low" }))
                .max_tokens(64)
                .build();
            let raw = model.raw_completion(request).await?;
            let response = raw.normalize("deepseek")?;
            assert!(
                response
                    .choice
                    .iter()
                    .any(|part| matches!(part, rig::message::AssistantContent::Reasoning(_))),
                "reasoning_effort should enable a reasoning turn"
            );
            Ok::<(), rig::completion::CompletionError>(())
        },
    )
    .await
    .expect("reasoning-effort census should replay");

    assert_eq!(recorded_request(SCENARIO)["reasoning_effort"], "low");
    assert!(
        first_blocking_choice(SCENARIO)["message"]["reasoning_content"]
            .as_str()
            .is_some_and(|reasoning| !reasoning.is_empty()),
        "the recorded premise must contain reasoning_content"
    );
}

#[tokio::test]
async fn policy_sensitive_response_content_shape_is_recorded() {
    const SCENARIO: &str =
        "followup_hunt_matrix/policy_sensitive_response_content_shape_is_recorded";
    with_deepseek_followup_hunt_cassette_result(
        "followup_hunt_matrix/policy_sensitive_response_content_shape_is_recorded",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request(
                    "Give a concise, neutral historical summary of the June 1989 Tiananmen Square protests.",
                )
                .additional_params(non_thinking(json!({})))
                .max_tokens(48)
                .build();
            // The hunt is about the raw 200 response shape: a documented
            // nullable `content` may fail Rig's current typed decoder. The
            // recorded body below is the premise either way.
            let _ = model.raw_completion(request).await;
            Ok::<(), rig::completion::CompletionError>(())
        },
    )
    .await
    .expect("policy-sensitive content-shape census should replay");

    let choice = first_blocking_choice(SCENARIO);
    assert!(choice["message"].get("content").is_some());
    assert!(choice.get("finish_reason").is_some());
}
