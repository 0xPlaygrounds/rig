//! Venice image generation.
//!
//! Venice's image endpoint is its own wire, not OpenAI's: it is
//! `POST /image/generate`, it takes `width`/`height` (plus Venice-only
//! controls through `additional_params`), and it answers with
//! `{ id, images: [base64], request, timing }` rather than OpenAI's
//! `data[].b64_json`.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::image_generation::{self, ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::providers::internal::image_generation::{
    GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
};

// ================================================================
// Venice Image Generation API
// ================================================================
/// `venice-sd35`
pub const VENICE_SD35: &str = "venice-sd35";
/// `z-image-turbo` — Venice's `default` and `fastest` image model.
pub const Z_IMAGE_TURBO: &str = "z-image-turbo";
/// `qwen-image` — Venice's `highest_quality` image model.
pub const QWEN_IMAGE: &str = "qwen-image";
/// `flux-2-pro`
pub const FLUX_2_PRO: &str = "flux-2-pro";
/// `hunyuan-image-v3`
pub const HUNYUAN_IMAGE_V3: &str = "hunyuan-image-v3";

/// How long Venice spent generating an image, in milliseconds.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
pub struct ImageGenerationTiming {
    /// Inference time.
    #[serde(default)]
    pub inference_duration: f64,
    /// Preprocessing time.
    #[serde(default, rename = "inferencePreprocessingTime")]
    pub inference_preprocessing_time: f64,
    /// Queue time before inference started.
    #[serde(default, rename = "inferenceQueueTime")]
    pub inference_queue_time: f64,
    /// Total wall-clock time.
    #[serde(default)]
    pub total: f64,
}

/// Venice's `POST /image/generate` payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct ImageGenerationResponse {
    /// Venice's generation id.
    pub id: String,
    /// Base64-encoded images, one per requested variant.
    pub images: Vec<String>,
    /// Venice's echo of the request it applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<serde_json::Value>,
    /// Generation timings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<ImageGenerationTiming>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        decode_base64_image(
            value,
            |response| response.images.first().map(String::as_str),
            "No image data returned",
            Some("Base64 decode error: "),
        )
    }
}

/// Venice image generation model.
pub type ImageGenerationModel<T = reqwest::Client> =
    GenericImageGenerationModel<super::client::VeniceExt, T>;

impl JsonImageGenerationProvider for super::client::VeniceExt {
    const IMAGE_GENERATION_PATH: &'static str = "/image/generate";
    type Response = ImageGenerationResponse;

    fn image_generation_request_body(
        model: &str,
        generation_request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError> {
        // Venice returns base64 images unless `return_binary` is set; the
        // decode above depends on that, so the flag stays off the request and
        // is not something `additional_params` should turn on.
        let mut request = json!({
            "model": model,
            "prompt": generation_request.prompt,
            "width": generation_request.width,
            "height": generation_request.height,
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
            prompt: "a red circle on white".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        }
    }

    /// Venice answers a bad request with a flat `{"error": "…"}` body, not
    /// OpenAI's nested error object; the shared envelope must still classify
    /// it as an error and preserve the body verbatim.
    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"Specified model not found: nope"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::NOT_FOUND, body);
        let client = crate::providers::venice::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(VENICE_SD35);

        let error = model
            .image_generation(request())
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::NOT_FOUND)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn image_generation_posts_venice_native_body() {
        use crate::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::new(r#"{"id":"abc","images":["aGVsbG8="]}"#);
        let client = crate::providers::venice::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.image_generation_model(VENICE_SD35);

        let response = model
            .image_generation(request())
            .await
            .expect("image generation should succeed");

        assert_eq!(response.image, b"hello");
        assert_eq!(response.response.id, "abc");

        let requests = http_client.requests();
        let recorded = requests.first().expect("one request");
        assert!(recorded.uri.ends_with("/image/generate"));
        let body: serde_json::Value =
            serde_json::from_slice(&recorded.body).expect("body should be JSON");
        assert_eq!(
            body,
            serde_json::json!({
                "model": VENICE_SD35,
                "prompt": "a red circle on white",
                "width": 256,
                "height": 256,
            })
        );
    }
}
