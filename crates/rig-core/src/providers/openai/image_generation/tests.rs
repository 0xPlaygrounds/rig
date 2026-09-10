use super::*;
use crate::client::image_generation::ImageGenerationClient;
use crate::image_generation::ImageGenerationModel as _;
use crate::providers::openai::Client;
use crate::test_utils::RecordingHttpClient;

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "draw a cat".to_string(),
        width: 256,
        height: 256,
        additional_params: None,
    }
}

fn body(model: &str, additional_params: Option<serde_json::Value>) -> serde_json::Value {
    build_request(
        model,
        ImageGenerationRequest {
            additional_params,
            ..request()
        },
    )
}

/// The field is not in the endpoint's request schema, so no model may be
/// sent it — including the ones the old hardcoded allowlist happened to
/// cover, and the retired `dall-e` names rig still exports.
#[test]
fn build_request_never_sends_response_format() {
    for model in [
        DALL_E_2,
        DALL_E_3,
        GPT_IMAGE_1,
        GPT_IMAGE_1_5,
        GPT_IMAGE_2,
        "gpt-image-1-mini",
        "gpt-image-2-2026-04-21",
        "chatgpt-image-latest",
    ] {
        assert!(
            body(model, None).get("response_format").is_none(),
            "{model} must not be sent a field outside the endpoint's schema"
        );
    }
}

/// Allowlisted and unlisted models now build the *same* body — the split
/// the allowlist created is what made unlisted models unusable.
#[test]
fn build_request_is_model_independent_apart_from_the_model_field() {
    let listed = body(GPT_IMAGE_1, None);
    let unlisted = body("gpt-image-1-mini", None);

    assert_eq!(listed["model"], json!(GPT_IMAGE_1));
    assert_eq!(unlisted["model"], json!("gpt-image-1-mini"));
    assert_eq!(
        listed.as_object().map(|body| body.len()),
        unlisted.as_object().map(|body| body.len())
    );
    assert_eq!(listed["prompt"], unlisted["prompt"]);
    assert_eq!(listed["size"], unlisted["size"]);
}

#[test]
fn build_request_derives_prompt_and_size() {
    let body = body(GPT_IMAGE_1, None);

    assert_eq!(body["prompt"], json!("draw a cat"));
    assert_eq!(body["size"], json!("256x256"));
}

#[test]
fn build_request_merges_additional_params() {
    let body = body(
        GPT_IMAGE_1,
        Some(json!({ "quality": "low", "background": "opaque" })),
    );

    assert_eq!(body["quality"], json!("low"));
    assert_eq!(body["background"], json!("opaque"));
}

/// Merged last, so a caller can override each derived key.
#[test]
fn build_request_lets_additional_params_override_derived_keys() {
    let body = body(
        GPT_IMAGE_1,
        Some(json!({ "model": "other", "prompt": "other prompt", "size": "1024x1024" })),
    );

    assert_eq!(body["model"], json!("other"));
    assert_eq!(body["prompt"], json!("other prompt"));
    assert_eq!(body["size"], json!("1024x1024"));
}

/// The escape hatch: a compatible endpoint that still wants
/// `response_format` can be handed it explicitly.
#[test]
fn build_request_lets_a_caller_reinstate_response_format() {
    let body = body(GPT_IMAGE_1, Some(json!({ "response_format": "b64_json" })));

    assert_eq!(body["response_format"], json!("b64_json"));
}

#[test]
fn build_request_ignores_non_object_additional_params() {
    assert_eq!(
        body(GPT_IMAGE_1, Some(json!("not-an-object"))),
        body(GPT_IMAGE_1, None)
    );
    assert_eq!(
        body(GPT_IMAGE_1, Some(json!(null))),
        body(GPT_IMAGE_1, None)
    );
}

#[tokio::test]
async fn image_generation_non_success_response_preserves_status_and_body() {
    let body = r#"{"error":{"message":"invalid image","type":"invalid_request_error"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(DALL_E_3);

    let error = model
        .image_generation(request())
        .await
        .expect_err("image generation should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn image_generation_preserves_raw_provider_error_json_on_api_error_envelope() {
    let body = r#"{"message":"quota exceeded","type":"insufficient_quota"}"#;
    let http_client = RecordingHttpClient::new(body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(DALL_E_3);

    let error = model
        .image_generation(request())
        .await
        .expect_err("image generation should fail with provider error envelope");

    match &error {
        ImageGenerationError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
            assert_eq!(error.provider_response_body(), Some(body));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}
