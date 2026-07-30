use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use serde_json::json;

#[allow(non_upper_case_globals)]
pub mod image_generation_models {
    pub const Flux1: &str = "black-forest-labs/FLUX.1-dev";
    pub const Kolors: &str = "Kwai-Kolors/Kolors";
    pub const StableDiffusion3: &str = "stabilityai/stable-diffusion-3-medium-diffusers";
}
pub use image_generation_models::*;

#[derive(Debug)]
pub struct ImageGenerationResponse {
    data: Vec<u8>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        Ok(image_generation::ImageGenerationResponse {
            image: value.data.clone(),
            response: value,
        })
    }
}

/// Build the serialized image-generation request body. Pure.
pub(crate) fn build_image_generation_body(
    request: &ImageGenerationRequest,
) -> Result<Vec<u8>, ImageGenerationError> {
    Ok(serde_json::to_vec(&json!({
        "inputs": request.prompt,
        "parameters": {
            "width": request.width,
            "height": request.height
        }
    }))?)
}

/// Parse an image-generation response: success bodies are raw image bytes.
/// Pure.
pub(crate) fn parse_image_generation_response(
    status: http::StatusCode,
    body: Vec<u8>,
) -> Result<image_generation::ImageGenerationResponse<ImageGenerationResponse>, ImageGenerationError>
{
    if !status.is_success() {
        return Err(ImageGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&body),
        ));
    }
    ImageGenerationResponse { data: body }.try_into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_runtime::HttpRuntime;
    use crate::providers::huggingface::functions;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let rt = HttpRuntime::recording(http_client);
        let cfg = functions::Config::new(Flux1).with_api_key("test-key");

        let error = functions::generate_image(&cfg, &rt, ImageGenerationRequest::new("draw a cat"))
            .await
            .expect_err("should fail with non-success status");

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
