use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::providers::internal::image_generation::{
    GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
};
use serde::Deserialize;
use serde_json::json;

// ================================================================
// xAI Image Generation API
// ================================================================
pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";

#[derive(Debug, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
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
            "No image data returned",
            Some("Base64 decode error: "),
        )
    }
}

/// xAI image generation model.
pub type ImageGenerationModel<T = reqwest::Client> =
    GenericImageGenerationModel<super::client::XAiExt, T>;

impl JsonImageGenerationProvider for super::client::XAiExt {
    const IMAGE_GENERATION_PATH: &'static str = "/v1/images/generations";
    type Response = ImageGenerationResponse;

    fn image_generation_request_body(
        model: &str,
        generation_request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError> {
        let mut request = json!({
            "model": model,
            "prompt": generation_request.prompt,
            "response_format": "b64_json",
            "aspect_ratio": "1:1",
        });

        if let Some(additional_params) = generation_request.additional_params {
            merge_inplace(&mut request, additional_params);
        }

        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;

    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        }
    }

    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(GROK_IMAGINE_IMAGE);

        let error = model
            .image_generation(request())
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn image_generation_2xx_error_envelope_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Err(ApiErrorResponse)` on a 200 OK.
        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(GROK_IMAGINE_IMAGE);

        let error = model
            .image_generation(request())
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            ImageGenerationError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
