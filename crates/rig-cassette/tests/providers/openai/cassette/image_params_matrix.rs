//! The OpenAI `/v1/images/generations` request body.
//!
//! Rig sends no `response_format`, since the endpoint rejects the field and
//! returns `data[].b64_json` by default. `additional_params` is merged into
//! the body last, so `quality`, `background`, `output_format` and the like
//! reach the API, and a caller of an OpenAI-compatible endpoint that still
//! wants `response_format` can reinstate it explicitly.
//!
//! Cells that assert a 400 are deliberate: an error naming the caller's own
//! parameter proves the parameter reached OpenAI, and rejections cost
//! nothing. Successful generations cost money, so they are spent only where
//! a rejection cannot show the same thing, on the cheapest model at its
//! lowest quality. Every cell also re-reads its own fixture: the recorded
//! request must carry the caller's parameter and must not carry
//! `response_format` unless the cell reinstates it.

use rig::error::ProviderError;
use rig::providers::openai;
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;
use crate::cassettes;
use rig::image_generation::ImageGenerationRequestBuilder;

/// The cheapest currently served image model.
const UNLISTED_MODEL: &str = "gpt-image-1-mini";
const PROMPT: &str = "a plain red circle on a white background";
/// Every image model OpenAI currently serves accepts this size.
const SIDE: u32 = 1024;

/// The provider's error text for a rejected cell, which must name the
/// caller's own parameter.
fn rejection_body(error: &ProviderError) -> String {
    error
        .provider_response_body()
        .map_or_else(|| error.to_string(), ToOwned::to_owned)
}

// ---------------------------------------------------------------------------
// The body the endpoint accepts.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retired_model_reaches_model_validation() {
    const SCENARIO: &str = "image_params_matrix/retired_model_reaches_model_validation";

    with_openai_cassette(
        "image_params_matrix/retired_model_reaches_model_validation",
        |client| async move {
            let model = client.openai.image_generation(openai::DALL_E_3);

            let error = model
                .call(
                    ImageGenerationRequestBuilder::new(PROMPT)
                        .width(SIDE)
                        .height(SIDE)
                        .build(),
                )
                .await
                .expect_err("dall-e-3 is retired");

            let body = rejection_body(&error);
            assert!(
                body.contains("dall-e-3") && !body.contains("response_format"),
                "a retired model must fail on the model, not on a parameter rig added: {body}"
            );
        },
    )
    .await;

    assert_recorded_request_lacks_response_format(SCENARIO);
}

// ---------------------------------------------------------------------------
// The caller's parameters reach the endpoint.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn additional_params_quality_reaches_the_api() {
    const SCENARIO: &str = "image_params_matrix/additional_params_quality_reaches_the_api";

    with_openai_cassette(
        "image_params_matrix/additional_params_quality_reaches_the_api",
        |client| async move {
            let model = client.openai.image_generation(UNLISTED_MODEL);

            let response = model
                .call(
                    ImageGenerationRequestBuilder::new(PROMPT)
                        .width(SIDE)
                        .height(SIDE)
                        .additional_params(json!({ "quality": "low", "background": "opaque" }))
                        .build(),
                )
                .await
                .expect("generation with caller parameters");

            crate::support::assert_image_bytes(&response.image);
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "background", &json!("opaque"));
    assert_recorded_response_echoes(SCENARIO, "background", "opaque");
}

#[tokio::test]
async fn additional_params_output_format_reaches_the_api() {
    const SCENARIO: &str = "image_params_matrix/additional_params_output_format_reaches_the_api";

    with_openai_cassette(
        "image_params_matrix/additional_params_output_format_reaches_the_api",
        |client| async move {
            let model = client.openai.image_generation(UNLISTED_MODEL);

            let response = model
                .call(
                    ImageGenerationRequestBuilder::new(PROMPT)
                        .width(SIDE)
                        .height(SIDE)
                        .additional_params(json!({ "quality": "low", "output_format": "jpeg" }))
                        .build(),
                )
                .await
                .expect("generation with caller parameters");

            assert_eq!(
                crate::support::assert_image_bytes(&response.image),
                crate::support::ImageContainer::Jpeg,
                "the requested output_format reaches the bytes"
            );
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "output_format", &json!("jpeg"));
    assert_recorded_response_echoes(SCENARIO, "output_format", "jpeg");
}

#[tokio::test]
async fn additional_params_invalid_background_is_rejected() {
    const SCENARIO: &str = "image_params_matrix/additional_params_invalid_background_is_rejected";

    with_openai_cassette(
        "image_params_matrix/additional_params_invalid_background_is_rejected",
        |client| async move {
            let model = client.openai.image_generation(UNLISTED_MODEL);

            let error = model
                .call(
                    ImageGenerationRequestBuilder::new(PROMPT)
                        .width(SIDE)
                        .height(SIDE)
                        .additional_params(json!({ "background": "rig-invalid" }))
                        .build(),
                )
                .await
                .expect_err("an invalid caller parameter must be rejected by OpenAI");

            assert!(rejection_body(&error).contains("background"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "background", &json!("rig-invalid"));
}

#[tokio::test]
async fn additional_params_invalid_output_format_is_rejected() {
    const SCENARIO: &str =
        "image_params_matrix/additional_params_invalid_output_format_is_rejected";

    with_openai_cassette(
        "image_params_matrix/additional_params_invalid_output_format_is_rejected",
        |client| async move {
            let model = client.openai.image_generation(UNLISTED_MODEL);

            let error = model
                .call(
                    ImageGenerationRequestBuilder::new(PROMPT)
                        .width(SIDE)
                        .height(SIDE)
                        .additional_params(json!({ "output_format": "rig-invalid" }))
                        .build(),
                )
                .await
                .expect_err("rejected parameter");

            assert!(rejection_body(&error).contains("output_format"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "output_format", &json!("rig-invalid"));
}

/// Rig sends no `response_format`, and a caller who needs it for a
/// compatible endpoint can put it back. OpenAI rejecting the reinstated
/// field proves it was sent.
#[tokio::test]
async fn caller_can_reinstate_response_format() {
    const SCENARIO: &str = "image_params_matrix/caller_can_reinstate_response_format";

    with_openai_cassette(
        "image_params_matrix/caller_can_reinstate_response_format",
        |client| async move {
            let model = client.openai.image_generation(UNLISTED_MODEL);

            let error = model
                .call(
                    ImageGenerationRequestBuilder::new(PROMPT)
                        .width(SIDE)
                        .height(SIDE)
                        .additional_params(json!({ "response_format": "b64_json" }))
                        .build(),
                )
                .await
                .expect_err("OpenAI rejects the reinstated field, which proves it was sent");

            assert!(rejection_body(&error).contains("response_format"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "response_format", &json!("b64_json"));
}

// ---------------------------------------------------------------------------
// Fixture-premise checks.
// ---------------------------------------------------------------------------

fn recorded_bodies(scenario: &str, side: &str) -> Vec<Value> {
    cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .filter_map(|(request, response)| {
            let body = if side == "when" { request } else { response };
            serde_json::from_str(&body).ok()
        })
        .collect()
}

fn assert_recorded_request_has(scenario: &str, key: &str, value: &Value) {
    assert!(
        recorded_bodies(scenario, "when")
            .iter()
            .any(|body| body.get(key) == Some(value)),
        "cassette {scenario} does not record {key}={value} in its request body"
    );
}

fn assert_recorded_request_lacks_response_format(scenario: &str) {
    assert!(
        recorded_bodies(scenario, "when")
            .iter()
            .all(|body| body.get("response_format").is_none()),
        "cassette {scenario} still records the `response_format` field OpenAI rejects"
    );
}

fn assert_recorded_response_echoes(scenario: &str, key: &str, value: &str) {
    assert!(
        recorded_bodies(scenario, "then")
            .iter()
            .any(|body| body.get(key).and_then(Value::as_str) == Some(value)),
        "cassette {scenario} response does not echo {key}={value}, so the cell no longer \
         demonstrates that the parameter took effect"
    );
}
