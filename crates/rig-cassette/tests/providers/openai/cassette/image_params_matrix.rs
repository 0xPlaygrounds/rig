//! Edge matrix for the OpenAI `/v1/images/generations` request body.
//!
//! Two defects lived in the same twelve-line body builder
//! (`providers::openai::image_generation::build_request`):
//!
//! **D1 — a parameter the endpoint rejects.** Rig added
//! `"response_format": "b64_json"` for every model *outside* a hardcoded
//! `gpt-image-1` / `gpt-image-1.5` / `gpt-image-2` allowlist. The endpoint no
//! longer accepts that field at all:
//!
//! ```text
//! 400 Unknown parameter: 'response_format'.
//! ```
//!
//! so every other image model OpenAI currently serves — `gpt-image-1-mini`,
//! `chatgpt-image-latest`, and any *dated snapshot* of an allowlisted model
//! such as `gpt-image-2-2026-04-21` — could not generate an image at all. The
//! field is also unnecessary: the endpoint returns `data[].b64_json` by
//! default, which is what rig already decodes.
//!
//! **D2 — a parameter rig dropped.**
//! `ImageGenerationRequestBuilder::additional_params` was never merged into the
//! body, so `quality`, `background`, `output_format`, `user`, `n` … were
//! silently inert for OpenAI — while xAI's and Gemini's image bodies honor the
//! same field. Merging it last also gives a caller back the escape hatch D1
//! removes: an OpenAI-*compatible* images endpoint that still wants
//! `response_format` can be handed it explicitly (cell 12).
//!
//! **How these cells fail on `origin/main`.** The harness matches the recorded
//! request body, so every cell is a mock miss on `main`: the D1 cells because
//! `main` adds `response_format`, the D2 cells because `main` omits the
//! caller's parameters.
//!
//! Cells that assert a **400** are deliberate and free to record: an error
//! naming the caller's own parameter is direct evidence that the parameter
//! reached OpenAI, which is exactly what D2 was about. Successful generations
//! cost money, so the matrix spends them only where a rejection cannot show
//! the same thing, and uses the cheapest served model at its lowest quality.
//!
//! | # | cell | model | params | outcome | status |
//! |---|------|-------|--------|---------|--------|
//! | 3 | `additional_params_quality_reaches_the_api` | gpt-image-1-mini | quality | 200 (echoed) | recorded |
//! | 4 | `additional_params_output_format_reaches_the_api` | gpt-image-1-mini | quality+output_format | 200 (echoed) | recorded |
//! | 6 | `additional_params_invalid_background_is_rejected` | gpt-image-1-mini | background | 400 background | recorded |
//! | 7 | `additional_params_invalid_output_format_is_rejected` | gpt-image-1-mini | output_format | 400 output_format | recorded |
//! | 12 | `caller_can_reinstate_response_format` | gpt-image-1-mini | response_format | 400 response_format | recorded |
//! | 15 | `retired_model_reaches_model_validation` | dall-e-3 | none | 400 model | recorded |
//!
//! Unit cells for the body shape itself (`build_request_*`, beside the fix)
//! cover: no `response_format` for any model, allowlisted and unlisted models
//! producing identical keys, `additional_params` merged last, overriding each
//! derived key, and a non-object payload being ignored.
//!
//! Every cell also re-reads its own fixture: the recorded request must carry
//! the caller's parameter and must not carry `response_format` unless the cell
//! is the one that reinstates it.

use rig::error::ProviderError;
use rig::providers::openai;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_openai_image_params_cassette;
use crate::cassettes;
use rig::image_generation::ImageGenerationRequestBuilder;

/// The cheapest currently-served image model — and the one `main` cannot use
/// at all, because it falls outside the old `response_format` allowlist.
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
// D1: the body the endpoint actually accepts.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retired_model_reaches_model_validation() {
    const SCENARIO: &str = "image_params_matrix/retired_model_reaches_model_validation";

    with_openai_image_params_cassette(
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
// D2: the caller's parameters reach the endpoint.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn additional_params_quality_reaches_the_api() {
    const SCENARIO: &str = "image_params_matrix/additional_params_quality_reaches_the_api";

    with_openai_image_params_cassette(
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

    with_openai_image_params_cassette(
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

    with_openai_image_params_cassette(
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

    with_openai_image_params_cassette(
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

/// The compose story for D1: rig no longer sends `response_format`, and a
/// caller who needs it for a compatible endpoint can put it back — proven by
/// OpenAI itself rejecting the reinstated field.
#[tokio::test]
async fn caller_can_reinstate_response_format() {
    const SCENARIO: &str = "image_params_matrix/caller_can_reinstate_response_format";

    with_openai_image_params_cassette(
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
    let path = cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
        .filter_map(|interaction| {
            interaction
                .get(side)
                .and_then(|side| side.get("body"))
                .and_then(serde_yaml::Value::as_str)
                .and_then(|body| serde_json::from_str::<Value>(body).ok())
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
