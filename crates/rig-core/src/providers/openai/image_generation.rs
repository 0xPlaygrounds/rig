use super::{OpenAICompletionsExt, OpenAIResponsesExt};
use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::providers::internal::image_generation::{
    GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
};
use serde::Deserialize;
use serde_json::json;

// ================================================================
// OpenAI Image Generation API
// ================================================================
pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";
pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";

#[derive(Debug, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i32,
    pub data: Vec<ImageGenerationData>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        decode_base64_image(
            value,
            |response| response.data.first().map(|image| image.b64_json.as_str()),
            "missing image data",
            None,
        )
    }
}

/// OpenAI image generation model.
pub type ImageGenerationModel<T = reqwest::Client> =
    GenericImageGenerationModel<OpenAIResponsesExt, T>;

/// OpenAI image generation model for a client using Chat Completions.
pub type CompletionsImageGenerationModel<T = reqwest::Client> =
    GenericImageGenerationModel<OpenAICompletionsExt, T>;

/// Build the `/v1/images/generations` body.
///
/// `response_format` is deliberately absent: it is no longer part of this
/// endpoint's request schema, which rejects it before it even looks at the
/// model — a request naming a model that does not exist still fails on
/// `400 Unknown parameter: 'response_format'` first. Rig used to add it for
/// every model outside a hardcoded `gpt-image-1`/`1.5`/`2` allowlist, so every
/// other image model — `gpt-image-1-mini`, `chatgpt-image-latest`, and any
/// dated snapshot of an allowlisted model such as `gpt-image-2-2026-04-21` —
/// could not generate an image at all. The models this endpoint currently
/// serves answer with `data[].b64_json`, which is what
/// [`decode_base64_image`] reads.
///
/// This is a statement about *this* endpoint. An OpenAI-**compatible** images
/// endpoint reached through the same client may still take the field, and may
/// need it to answer with base64 rather than a URL; such a caller passes it
/// explicitly through `additional_params`, which the merge below now honors.
fn build_request(
    model: &str,
    generation_request: ImageGenerationRequest,
) -> Result<serde_json::Value, ImageGenerationError> {
    let mut request = json!({
        "model": model,
        "prompt": generation_request.prompt,
        "size": format!("{}x{}", generation_request.width, generation_request.height),
    });

    // Last, so a caller can reach the endpoint's other parameters (`quality`,
    // `background`, `output_format`, `user`, …) and override what is derived
    // above. xAI's and Gemini's image bodies already honor this field;
    // dropping it here made `ImageGenerationRequestBuilder::additional_params`
    // silently inert for OpenAI.
    //
    // Azure OpenAI's image body (`providers::azure`) has both defects and in a
    // worse combination: it hardcodes `response_format` *and* drops
    // `additional_params`, so an Azure caller cannot even work around the
    // former. Left alone here because a fix that cannot be recorded against
    // Azure would be a guess, which is what this change set is trying not to
    // ship.
    if let Some(additional_params) = generation_request.additional_params {
        merge_inplace(&mut request, additional_params);
    }

    Ok(request)
}

impl JsonImageGenerationProvider for OpenAIResponsesExt {
    const IMAGE_GENERATION_PATH: &'static str = "/images/generations";
    type Response = ImageGenerationResponse;

    fn image_generation_request_body(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError> {
        build_request(model, request)
    }
}

impl JsonImageGenerationProvider for OpenAICompletionsExt {
    const IMAGE_GENERATION_PATH: &'static str = "/images/generations";
    type Response = ImageGenerationResponse;

    fn image_generation_request_body(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError> {
        build_request(model, request)
    }
}

#[cfg(test)]
mod tests {
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
        .expect("body should build")
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
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
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
}
