//! Request-parameter pass-through matrix for Doubleword chat completions.
//!
//! The shared completion builder owns some fields while provider-specific
//! OpenAI-compatible options arrive through `additional_params`. These six
//! cells prove the final serialized body contains the requested value and
//! that Doubleword accepts it, covering both paths through the merge.
//!
//! | source | field | recorded value |
//! |---|---|---|
//! | typed builder | `temperature` | `0.0` |
//! | typed builder | `max_tokens` | `7` |
//! | additional params | `top_p` | `0.25` |
//! | additional params | `seed` | `31415` |
//! | additional params | `stop` | `["BANANA"]` |
//! | additional params | `response_format` | `{"type":"json_object"}` |

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::doubleword;
use serde_json::{Value, json};

use super::super::support::{recorded_chat_calls, with_doubleword_cassette};

const MODEL: &str = doubleword::QWEN3_5_9B;
const PROMPT: &str = "Reply briefly with the word parameter-ok.";

fn assert_recorded_parameter(scenario: &str, field: &str, expected: Value) {
    let calls = recorded_chat_calls(scenario);
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.status, 200);
    assert_eq!(call.request["model"], MODEL);
    assert_eq!(call.request[field], expected, "recorded `{field}` differs");
    assert!(call.response_json.as_ref().is_some_and(|response| {
        response["choices"]
            .as_array()
            .is_some_and(|choices| !choices.is_empty())
    }));
}

#[tokio::test]
async fn temperature_from_the_typed_builder() {
    const SCENARIO: &str = "request_parameter_matrix/temperature_from_the_typed_builder";
    with_doubleword_cassette(
        "request_parameter_matrix/temperature_from_the_typed_builder",
        |client| async move {
            let model = client.completion_model(MODEL);
            model
                .raw_completion(
                    model
                        .completion_request(PROMPT)
                        .temperature(0.0)
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect("Doubleword should accept temperature");
        },
    )
    .await;
    assert_recorded_parameter(SCENARIO, "temperature", json!(0.0));
}

#[tokio::test]
async fn max_tokens_from_the_typed_builder() {
    const SCENARIO: &str = "request_parameter_matrix/max_tokens_from_the_typed_builder";
    with_doubleword_cassette(
        "request_parameter_matrix/max_tokens_from_the_typed_builder",
        |client| async move {
            let model = client.completion_model(MODEL);
            model
                .raw_completion(model.completion_request(PROMPT).max_tokens(7).build())
                .await
                .expect("Doubleword should accept max_tokens");
        },
    )
    .await;
    assert_recorded_parameter(SCENARIO, "max_tokens", json!(7));
}

#[tokio::test]
async fn top_p_from_additional_params() {
    const SCENARIO: &str = "request_parameter_matrix/top_p_from_additional_params";
    with_doubleword_cassette(
        "request_parameter_matrix/top_p_from_additional_params",
        |client| async move {
            let model = client.completion_model(MODEL);
            model
                .raw_completion(
                    model
                        .completion_request(PROMPT)
                        .additional_params(json!({ "top_p": 0.25 }))
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect("Doubleword should accept top_p");
        },
    )
    .await;
    assert_recorded_parameter(SCENARIO, "top_p", json!(0.25));
}

#[tokio::test]
async fn seed_from_additional_params() {
    const SCENARIO: &str = "request_parameter_matrix/seed_from_additional_params";
    with_doubleword_cassette(
        "request_parameter_matrix/seed_from_additional_params",
        |client| async move {
            let model = client.completion_model(MODEL);
            model
                .raw_completion(
                    model
                        .completion_request(PROMPT)
                        .additional_params(json!({ "seed": 31_415 }))
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect("Doubleword should accept seed");
        },
    )
    .await;
    assert_recorded_parameter(SCENARIO, "seed", json!(31_415));
}

#[tokio::test]
async fn stop_sequence_from_additional_params() {
    const SCENARIO: &str = "request_parameter_matrix/stop_sequence_from_additional_params";
    with_doubleword_cassette(
        "request_parameter_matrix/stop_sequence_from_additional_params",
        |client| async move {
            let model = client.completion_model(MODEL);
            model
                .raw_completion(
                    model
                        .completion_request("Write alpha BANANA omega.")
                        .additional_params(json!({ "stop": ["BANANA"] }))
                        .max_tokens(64)
                        .build(),
                )
                .await
                .expect("Doubleword should accept stop sequences");
        },
    )
    .await;
    assert_recorded_parameter(SCENARIO, "stop", json!(["BANANA"]));
}

#[tokio::test]
async fn json_object_response_format_from_additional_params() {
    const SCENARIO: &str =
        "request_parameter_matrix/json_object_response_format_from_additional_params";
    with_doubleword_cassette(
        "request_parameter_matrix/json_object_response_format_from_additional_params",
        |client| async move {
            let model = client.completion_model(MODEL);
            model
                .raw_completion(
                    model
                        .completion_request("Return a JSON object with ok set to true.")
                        .additional_params(json!({
                            "response_format": { "type": "json_object" }
                        }))
                        .max_tokens(96)
                        .build(),
                )
                .await
                .expect("Doubleword should accept JSON-object response format");
        },
    )
    .await;
    assert_recorded_parameter(
        SCENARIO,
        "response_format",
        json!({ "type": "json_object" }),
    );
}
