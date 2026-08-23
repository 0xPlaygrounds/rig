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
//! | 1 | `unlisted_model_generates_without_response_format` | gpt-image-1-mini | none | 200 | recorded |
//! | 2 | `allowlisted_model_still_generates` | gpt-image-1 | none | 200 | recorded |
//! | 3 | `additional_params_quality_reaches_the_api` | gpt-image-1-mini | quality | 200 (echoed) | recorded |
//! | 4 | `additional_params_output_format_reaches_the_api` | gpt-image-1-mini | quality+output_format | 200 (echoed) | recorded |
//! | 5 | `completions_client_shares_the_fixed_body` | gpt-image-1-mini | quality | 200 | recorded |
//! | 6 | `additional_params_invalid_background_is_rejected` | gpt-image-1-mini | background | 400 background | recorded |
//! | 7 | `additional_params_invalid_output_format_is_rejected` | gpt-image-1-mini | output_format | 400 output_format | recorded |
//! | 8 | `additional_params_invalid_quality_is_rejected` | gpt-image-1-mini | quality | 400 quality | recorded |
//! | 9 | `additional_params_override_size` | gpt-image-1-mini | size | 400 size | recorded |
//! | 10 | `additional_params_override_model` | gpt-image-1-mini | model | 400 model | recorded |
//! | 11 | `additional_params_override_prompt` | gpt-image-1-mini | prompt | 400 prompt | recorded |
//! | 12 | `caller_can_reinstate_response_format` | gpt-image-1-mini | response_format | 400 response_format | recorded |
//! | 13 | `unlisted_dated_snapshot_reaches_its_own_validation` | gpt-image-2-2026-04-21 | size | 400 size | recorded |
//! | 14 | `chatgpt_image_latest_reaches_its_own_validation` | chatgpt-image-latest | quality | 400 quality | recorded |
//! | 15 | `retired_model_reaches_model_validation` | dall-e-3 | none | 400 model | recorded |
//! | 16 | `non_object_additional_params_are_a_no_op` | gpt-image-1-mini | `"not-an-object"` | 400 prompt | recorded |
//! | 17 | `response_format_is_rejected_before_the_model_is_looked_at` | rig-nonexistent | response_format | 400 response_format | recorded |
//!
//! Unit cells for the body shape itself (`build_request_*`, beside the fix)
//! cover: no `response_format` for any model, allowlisted and unlisted models
//! producing identical keys, `additional_params` merged last, overriding each
//! derived key, and a non-object payload being ignored.
//!
//! Every cell also re-reads its own fixture: the recorded request must carry
//! the caller's parameter and must not carry `response_format` unless the cell
//! is the one that reinstates it.

use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::{ImageGenerationError, ImageGenerationModel};
use rig::providers::openai;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_openai_image_params_cassette;
use crate::cassettes;

/// The cheapest currently-served image model — and the one `main` cannot use
/// at all, because it falls outside the old `response_format` allowlist.
const UNLISTED_MODEL: &str = "gpt-image-1-mini";
const PROMPT: &str = "a plain red circle on a white background";
/// Every image model OpenAI currently serves accepts this size.
const SIDE: u32 = 1024;

/// The provider's error text for a rejected cell, which must name the
/// caller's own parameter.
fn rejection_body(error: &ImageGenerationError) -> String {
    error
        .provider_response_body()
        .map_or_else(|| error.to_string(), ToOwned::to_owned)
}

// ---------------------------------------------------------------------------
// D1: the body the endpoint actually accepts.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unlisted_model_generates_without_response_format() {
    const SCENARIO: &str = "image_params_matrix/unlisted_model_generates_without_response_format";

    with_openai_image_params_cassette(
        "image_params_matrix/unlisted_model_generates_without_response_format",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let response = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "low" }))
                .send()
                .await
                .expect("a model outside the old allowlist must be able to generate at all");

            assert!(!response.image.is_empty());
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "quality", &json!("low"));
    assert_recorded_request_lacks_response_format(SCENARIO);
    assert_recorded_response_echoes(SCENARIO, "quality", "low");
}

#[tokio::test]
async fn allowlisted_model_still_generates() {
    const SCENARIO: &str = "image_params_matrix/allowlisted_model_still_generates";

    with_openai_image_params_cassette(
        "image_params_matrix/allowlisted_model_still_generates",
        |client| async move {
            let model = client.image_generation_model(openai::GPT_IMAGE_1);

            let response = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "low" }))
                .send()
                .await
                .expect("the previously-allowlisted path must be unchanged");

            assert!(!response.image.is_empty());
        },
    )
    .await;

    assert_recorded_request_lacks_response_format(SCENARIO);
}

#[tokio::test]
async fn retired_model_reaches_model_validation() {
    const SCENARIO: &str = "image_params_matrix/retired_model_reaches_model_validation";

    with_openai_image_params_cassette(
        "image_params_matrix/retired_model_reaches_model_validation",
        |client| async move {
            let model = client.image_generation_model(openai::DALL_E_3);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .send()
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
            let model = client.image_generation_model(UNLISTED_MODEL);

            let response = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "low", "background": "opaque" }))
                .send()
                .await
                .expect("generation with caller parameters");

            assert!(!response.image.is_empty());
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
            let model = client.image_generation_model(UNLISTED_MODEL);

            let response = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "low", "output_format": "jpeg" }))
                .send()
                .await
                .expect("generation with caller parameters");

            assert!(!response.image.is_empty());
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "output_format", &json!("jpeg"));
    assert_recorded_response_echoes(SCENARIO, "output_format", "jpeg");
}

/// The Chat Completions client builds the same body through the same helper.
#[tokio::test]
async fn completions_client_shares_the_fixed_body() {
    const SCENARIO: &str = "image_params_matrix/completions_client_shares_the_fixed_body";

    with_openai_image_params_cassette(
        "image_params_matrix/completions_client_shares_the_fixed_body",
        |client| async move {
            let model = client
                .completions_api()
                .image_generation_model(UNLISTED_MODEL);

            let response = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "low" }))
                .send()
                .await
                .expect("the completions-client image model must behave identically");

            assert!(!response.image.is_empty());
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "quality", &json!("low"));
    assert_recorded_request_lacks_response_format(SCENARIO);
}

#[tokio::test]
async fn additional_params_invalid_background_is_rejected() {
    const SCENARIO: &str = "image_params_matrix/additional_params_invalid_background_is_rejected";

    with_openai_image_params_cassette(
        "image_params_matrix/additional_params_invalid_background_is_rejected",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "background": "rig-invalid" }))
                .send()
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
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "output_format": "rig-invalid" }))
                .send()
                .await
                .expect_err("rejected parameter");

            assert!(rejection_body(&error).contains("output_format"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "output_format", &json!("rig-invalid"));
}

#[tokio::test]
async fn additional_params_invalid_quality_is_rejected() {
    const SCENARIO: &str = "image_params_matrix/additional_params_invalid_quality_is_rejected";

    with_openai_image_params_cassette(
        "image_params_matrix/additional_params_invalid_quality_is_rejected",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "rig-invalid" }))
                .send()
                .await
                .expect_err("rejected parameter");

            assert!(rejection_body(&error).contains("quality"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "quality", &json!("rig-invalid"));
}

/// `additional_params` is merged *last*, so it can override a key rig derives.
#[tokio::test]
async fn additional_params_override_size() {
    const SCENARIO: &str = "image_params_matrix/additional_params_override_size";

    with_openai_image_params_cassette(
        "image_params_matrix/additional_params_override_size",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "size": "3x3" }))
                .send()
                .await
                .expect_err("the overridden size must reach OpenAI and be rejected");

            assert!(rejection_body(&error).contains("size"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "size", &json!("3x3"));
}

#[tokio::test]
async fn additional_params_override_model() {
    const SCENARIO: &str = "image_params_matrix/additional_params_override_model";

    with_openai_image_params_cassette(
        "image_params_matrix/additional_params_override_model",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "model": "rig-nonexistent-image-model" }))
                .send()
                .await
                .expect_err("the overridden model must reach OpenAI and be rejected");

            assert!(rejection_body(&error).contains("rig-nonexistent-image-model"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "model", &json!("rig-nonexistent-image-model"));
}

#[tokio::test]
async fn additional_params_override_prompt() {
    const SCENARIO: &str = "image_params_matrix/additional_params_override_prompt";

    with_openai_image_params_cassette(
        "image_params_matrix/additional_params_override_prompt",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "prompt": "" }))
                .send()
                .await
                .expect_err("the overridden prompt must reach OpenAI and be rejected");

            assert!(rejection_body(&error).contains("prompt"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "prompt", &json!(""));
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
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "response_format": "b64_json" }))
                .send()
                .await
                .expect_err("OpenAI rejects the reinstated field, which proves it was sent");

            assert!(rejection_body(&error).contains("response_format"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "response_format", &json!("b64_json"));
}

/// A *dated snapshot* of an allowlisted model: the exact shape the hardcoded
/// `matches!` allowlist could never keep up with.
#[tokio::test]
async fn unlisted_dated_snapshot_reaches_its_own_validation() {
    const SCENARIO: &str = "image_params_matrix/unlisted_dated_snapshot_reaches_its_own_validation";

    with_openai_image_params_cassette(
        "image_params_matrix/unlisted_dated_snapshot_reaches_its_own_validation",
        |client| async move {
            let model = client.image_generation_model("gpt-image-2-2026-04-21");

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "size": "3x3" }))
                .send()
                .await
                .expect_err("rejected on its own parameter, not on one rig added");

            let body = rejection_body(&error);
            assert!(
                body.contains("size") && !body.contains("response_format"),
                "{body}"
            );
        },
    )
    .await;

    assert_recorded_request_lacks_response_format(SCENARIO);
}

#[tokio::test]
async fn chatgpt_image_latest_reaches_its_own_validation() {
    const SCENARIO: &str = "image_params_matrix/chatgpt_image_latest_reaches_its_own_validation";

    with_openai_image_params_cassette(
        "image_params_matrix/chatgpt_image_latest_reaches_its_own_validation",
        |client| async move {
            let model = client.image_generation_model("chatgpt-image-latest");

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "quality": "rig-invalid" }))
                .send()
                .await
                .expect_err("rejected on its own parameter");

            let body = rejection_body(&error);
            assert!(
                body.contains("quality") && !body.contains("response_format"),
                "{body}"
            );
        },
    )
    .await;

    assert_recorded_request_lacks_response_format(SCENARIO);
}

/// The evidence for why the field had to go for *every* model, not just the
/// ones outside the old allowlist: it is not in the endpoint's request schema
/// at all, so a request carrying it fails on the field even when the model
/// named does not exist. That ordering is what made the old allowlist fatal —
/// an unlisted model never reached its own validation.
#[tokio::test]
async fn response_format_is_rejected_before_the_model_is_looked_at() {
    const SCENARIO: &str =
        "image_params_matrix/response_format_is_rejected_before_the_model_is_looked_at";

    with_openai_image_params_cassette(
        "image_params_matrix/response_format_is_rejected_before_the_model_is_looked_at",
        |client| async move {
            let model = client.image_generation_model("rig-nonexistent-image-model");

            let error = model
                .image_generation_request()
                .prompt(PROMPT)
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!({ "response_format": "b64_json" }))
                .send()
                .await
                .expect_err("both the field and the model are invalid");

            let body = rejection_body(&error);
            assert!(
                body.contains("response_format") && !body.contains("rig-nonexistent-image-model"),
                "the parameter is rejected before the model is even looked at: {body}"
            );
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "response_format", &json!("b64_json"));
}

/// A non-object `additional_params` payload merges nothing and leaves the
/// derived body exactly as it was.
#[tokio::test]
async fn non_object_additional_params_are_a_no_op() {
    const SCENARIO: &str = "image_params_matrix/non_object_additional_params_are_a_no_op";

    with_openai_image_params_cassette(
        "image_params_matrix/non_object_additional_params_are_a_no_op",
        |client| async move {
            let model = client.image_generation_model(UNLISTED_MODEL);

            let error = model
                .image_generation_request()
                .prompt("")
                .width(SIDE)
                .height(SIDE)
                .additional_params(json!("not-an-object"))
                .send()
                .await
                .expect_err("the empty prompt still reaches OpenAI");

            assert!(rejection_body(&error).contains("prompt"));
        },
    )
    .await;

    assert_recorded_request_has(SCENARIO, "prompt", &json!(""));
    assert_recorded_request_lacks_response_format(SCENARIO);
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
