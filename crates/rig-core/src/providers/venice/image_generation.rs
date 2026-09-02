//! Venice image generation.
//!
//! Venice's image endpoint is its own wire, not OpenAI's: it is
//! `POST /image/generate`, it takes `width`/`height` (plus Venice-only
//! controls through `additional_params`), and it answers with
//! `{ id, images: [base64], request, timing }` rather than OpenAI's
//! `data[].b64_json`.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::image_generation::{
    self, ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
};
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

impl NormalizeImageGenerationResponse for ImageGenerationResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        let image = decode_base64_image(
            &self,
            |response| response.images.first().map(String::as_str),
            "No image data returned",
            Some("Base64 decode error: "),
        )?;
        Ok(image_generation::ImageGenerationResponse::new(
            image, provider,
        ))
    }
}

/// Venice image generation model.
pub type ImageGenerationModel<T = crate::http_client::BoxedHttpClient> =
    GenericImageGenerationModel<super::client::Venice, T>;

impl JsonImageGenerationProvider for super::client::Venice {
    const IMAGE_GENERATION_PATH: &'static str = "/image/generate";
    const PROVIDER_NAME: &'static str = "venice";
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
mod tests;
