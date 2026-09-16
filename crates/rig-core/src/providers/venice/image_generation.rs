//! Venice's image-generation model identifiers and its own view of a
//! generated image.
//!
//! Venice's image endpoint is its own wire, not OpenAI's: it is
//! `POST /image/generate` (the path
//! [`openai::wire::VENICE`](crate::providers::openai::wire::VENICE) carries),
//! it takes `width`/`height` plus Venice-only controls through
//! `additional_params`, and it answers with
//! `{ id, images: [base64], request, timing }` rather than OpenAI's
//! `data[].b64_json` — which is what [`ImageGenerationResponse`] models, for
//! a caller reading the raw document the normalized image does not name.

use serde::{Deserialize, Serialize};

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
