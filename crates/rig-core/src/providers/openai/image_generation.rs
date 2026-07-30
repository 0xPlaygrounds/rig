use crate::image_generation;
use crate::image_generation::ImageGenerationError;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;

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
        let b64_json = value
            .data
            .first()
            .ok_or_else(|| ImageGenerationError::ResponseError("missing image data".into()))?
            .b64_json
            .clone();

        let bytes = BASE64_STANDARD
            .decode(&b64_json)
            .map_err(|err| ImageGenerationError::ResponseError(err.to_string()))?;

        Ok(image_generation::ImageGenerationResponse {
            image: bytes,
            response: value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_runtime::HttpRuntime;
    use crate::image_generation::ImageGenerationRequest;
    use crate::providers::openai::functions;
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
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));
        let cfg = functions::Config::new(DALL_E_3).with_api_key("test-key");

        let error = functions::generate_image(&cfg, &rt, request())
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
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body));
        let cfg = functions::Config::new(DALL_E_3).with_api_key("test-key");

        let error = functions::generate_image(&cfg, &rt, request())
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

    #[test]
    fn base64_payload_decodes_into_the_normalized_response() {
        let raw = ImageGenerationResponse {
            created: 1,
            data: vec![ImageGenerationData {
                b64_json: BASE64_STANDARD.encode(b"png-bytes"),
            }],
        };

        let normalized: image_generation::ImageGenerationResponse<ImageGenerationResponse> =
            raw.try_into().expect("response should convert");
        assert_eq!(normalized.image, b"png-bytes".to_vec());
    }

    #[test]
    fn empty_data_array_is_a_response_error() {
        let raw = ImageGenerationResponse {
            created: 1,
            data: Vec::new(),
        };

        let error =
            <image_generation::ImageGenerationResponse<ImageGenerationResponse>>::try_from(raw)
                .expect_err("missing image data should error");
        assert!(matches!(error, ImageGenerationError::ResponseError(_)));
    }
}
