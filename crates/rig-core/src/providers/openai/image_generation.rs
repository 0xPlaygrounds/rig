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

fn build_request(
    model: &str,
    generation_request: ImageGenerationRequest,
) -> Result<serde_json::Value, ImageGenerationError> {
    let mut request = json!({
        "model": model,
        "prompt": generation_request.prompt,
        "size": format!("{}x{}", generation_request.width, generation_request.height),
    });

    if !matches!(model, GPT_IMAGE_1 | GPT_IMAGE_1_5 | GPT_IMAGE_2) {
        merge_inplace(
            &mut request,
            json!({
                "response_format": "b64_json"
            }),
        );
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
