//! Provider-error matrix for Doubleword chat completions.
//!
//! Three stable failure classes—unknown model, invalid credentials and an
//! out-of-range sampling parameter—are crossed with blocking and streaming
//! transports. Every cell requires a 4xx status, a preserved provider body,
//! and a recorded JSON error envelope, preventing a generic transport string
//! from replacing the actionable upstream diagnostic.
//!
//! | failure | blocking | streaming | envelope |
//! |---|---|---|---|
//! | invalid key | 403 | 403 | nested `error` |
//! | unknown model | 404 | 404 | nested `error` |
//! | `temperature: 100` | 400 | 400 | top-level `message` / `type` / `code` |
//!
//! Streaming status failures open a rig stream successfully and surface the
//! preserved `CompletionError` as its first in-band item. The matrix treats
//! that as the streaming contract and proves the body/status match blocking.

use futures::StreamExt;
use rig::completion::{CompletionError, CompletionModel};
use rig::prelude::*;
use rig::providers::doubleword;
use serde_json::json;

use super::super::support::{
    recorded_chat_calls, with_doubleword_bogus_key_cassette, with_doubleword_cassette,
};

const PROMPT: &str = "Reply with error-probe.";
const UNKNOWN_MODEL: &str = "rig/definitely-not-a-doubleword-model";

#[derive(Clone, Copy, Debug)]
enum ErrorEnvelope {
    Nested,
    Flat,
}

fn assert_error_envelope(json: &serde_json::Value, expected: ErrorEnvelope) {
    let has_nested_error = json.get("error").is_some_and(serde_json::Value::is_object);
    let has_flat_fields = ["message", "type", "code"]
        .iter()
        .all(|field| json.get(*field).is_some());

    match expected {
        ErrorEnvelope::Nested => assert!(
            has_nested_error && !has_flat_fields,
            "expected nested error envelope, got: {json}"
        ),
        ErrorEnvelope::Flat => assert!(
            has_flat_fields && !has_nested_error,
            "expected flat error envelope, got: {json}"
        ),
    }
}

fn assert_preserved_client_error(
    error: &CompletionError,
    expected_status: u16,
    expected_envelope: ErrorEnvelope,
) {
    let status = error
        .provider_response_status()
        .expect("provider status should be preserved");
    assert_eq!(status.as_u16(), expected_status, "error: {error}");
    let body = error
        .provider_response_body()
        .expect("provider response body should be preserved");
    assert!(!body.trim().is_empty());
    let json: serde_json::Value =
        serde_json::from_str(body).expect("Doubleword error body should be JSON");
    assert_error_envelope(&json, expected_envelope);
}

fn assert_recorded_error(
    scenario: &str,
    expected_status: u16,
    expected_envelope: ErrorEnvelope,
    request_field: Option<(&str, serde_json::Value)>,
) {
    let calls = recorded_chat_calls(scenario);
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.status, expected_status);
    if let Some((field, expected)) = request_field {
        assert_eq!(call.request[field], expected);
    }
    let body = call
        .response_json
        .as_ref()
        .expect("recorded JSON error body");
    assert_error_envelope(body, expected_envelope);
    assert!(call.stream_chunks.is_empty());
}

fn assert_recorded_transport_parity(blocking_scenario: &str, streaming_scenario: &str) {
    let blocking_calls = recorded_chat_calls(blocking_scenario);
    let streaming_calls = recorded_chat_calls(streaming_scenario);
    assert_eq!(blocking_calls.len(), 1);
    assert_eq!(streaming_calls.len(), 1);
    assert_eq!(blocking_calls[0].status, streaming_calls[0].status);
    assert_eq!(
        blocking_calls[0].response_json,
        streaming_calls[0].response_json
    );
}

async fn unknown_model_blocking_body(client: doubleword::Client) {
    let model = client.completion_model(UNKNOWN_MODEL);
    let error = model
        .raw_completion(model.completion_request(PROMPT).max_tokens(8).build())
        .await
        .expect_err("an unknown model should be rejected");
    assert_preserved_client_error(&error, 404, ErrorEnvelope::Nested);
}

async fn unknown_model_streaming_body(client: doubleword::Client) {
    let model = client.completion_model(UNKNOWN_MODEL);
    let result = model
        .stream(model.completion_request(PROMPT).max_tokens(8).build())
        .await;
    let mut stream = result.expect("streaming HTTP failures are delivered in-band");
    let error = loop {
        match stream.next().await {
            Some(Err(error)) => break error,
            Some(Ok(_)) => continue,
            None => panic!("unknown-model stream ended without its provider error"),
        }
    };
    assert_preserved_client_error(&error, 404, ErrorEnvelope::Nested);
}

async fn invalid_key_blocking_body(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let error = model
        .raw_completion(model.completion_request(PROMPT).max_tokens(8).build())
        .await
        .expect_err("invalid credentials should be rejected");
    assert_preserved_client_error(&error, 403, ErrorEnvelope::Nested);
}

async fn invalid_key_streaming_body(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let result = model
        .stream(model.completion_request(PROMPT).max_tokens(8).build())
        .await;
    let mut stream = result.expect("streaming HTTP failures are delivered in-band");
    let error = loop {
        match stream.next().await {
            Some(Err(error)) => break error,
            Some(Ok(_)) => continue,
            None => panic!("invalid-key stream ended without its provider error"),
        }
    };
    assert_preserved_client_error(&error, 403, ErrorEnvelope::Nested);
}

async fn invalid_temperature_blocking_body(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let error = model
        .raw_completion(
            model
                .completion_request(PROMPT)
                .additional_params(json!({ "temperature": 100 }))
                .max_tokens(8)
                .build(),
        )
        .await
        .expect_err("an out-of-range temperature should be rejected");
    assert_preserved_client_error(&error, 400, ErrorEnvelope::Flat);
}

async fn invalid_temperature_streaming_body(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let result = model
        .stream(
            model
                .completion_request(PROMPT)
                .additional_params(json!({ "temperature": 100 }))
                .max_tokens(8)
                .build(),
        )
        .await;
    let mut stream = result.expect("streaming HTTP failures are delivered in-band");
    let error = loop {
        match stream.next().await {
            Some(Err(error)) => break error,
            Some(Ok(_)) => continue,
            None => panic!("invalid-temperature stream ended without its provider error"),
        }
    };
    assert_preserved_client_error(&error, 400, ErrorEnvelope::Flat);
}

#[tokio::test]
async fn unknown_model_blocking() {
    const SCENARIO: &str = "error_matrix/unknown_model_blocking";
    with_doubleword_cassette(
        "error_matrix/unknown_model_blocking",
        unknown_model_blocking_body,
    )
    .await;
    assert_recorded_error(
        SCENARIO,
        404,
        ErrorEnvelope::Nested,
        Some(("model", json!(UNKNOWN_MODEL))),
    );
}

#[tokio::test]
async fn unknown_model_streaming() {
    const SCENARIO: &str = "error_matrix/unknown_model_streaming";
    with_doubleword_cassette(
        "error_matrix/unknown_model_streaming",
        unknown_model_streaming_body,
    )
    .await;
    assert_recorded_error(
        SCENARIO,
        404,
        ErrorEnvelope::Nested,
        Some(("model", json!(UNKNOWN_MODEL))),
    );
    assert_recorded_transport_parity(
        "error_matrix/unknown_model_blocking",
        "error_matrix/unknown_model_streaming",
    );
}

#[tokio::test]
async fn invalid_key_blocking() {
    const SCENARIO: &str = "error_matrix/invalid_key_blocking";
    with_doubleword_bogus_key_cassette(
        "error_matrix/invalid_key_blocking",
        invalid_key_blocking_body,
    )
    .await;
    assert_recorded_error(
        SCENARIO,
        403,
        ErrorEnvelope::Nested,
        Some(("model", json!(doubleword::QWEN3_5_9B))),
    );
}

#[tokio::test]
async fn invalid_key_streaming() {
    const SCENARIO: &str = "error_matrix/invalid_key_streaming";
    with_doubleword_bogus_key_cassette(
        "error_matrix/invalid_key_streaming",
        invalid_key_streaming_body,
    )
    .await;
    assert_recorded_error(
        SCENARIO,
        403,
        ErrorEnvelope::Nested,
        Some(("model", json!(doubleword::QWEN3_5_9B))),
    );
    assert_recorded_transport_parity(
        "error_matrix/invalid_key_blocking",
        "error_matrix/invalid_key_streaming",
    );
}

#[tokio::test]
async fn invalid_temperature_blocking() {
    const SCENARIO: &str = "error_matrix/invalid_temperature_blocking";
    with_doubleword_cassette(
        "error_matrix/invalid_temperature_blocking",
        invalid_temperature_blocking_body,
    )
    .await;
    assert_recorded_error(
        SCENARIO,
        400,
        ErrorEnvelope::Flat,
        Some(("temperature", json!(100))),
    );
}

#[tokio::test]
async fn invalid_temperature_streaming() {
    const SCENARIO: &str = "error_matrix/invalid_temperature_streaming";
    with_doubleword_cassette(
        "error_matrix/invalid_temperature_streaming",
        invalid_temperature_streaming_body,
    )
    .await;
    assert_recorded_error(
        SCENARIO,
        400,
        ErrorEnvelope::Flat,
        Some(("temperature", json!(100))),
    );
    assert_recorded_transport_parity(
        "error_matrix/invalid_temperature_blocking",
        "error_matrix/invalid_temperature_streaming",
    );
}
