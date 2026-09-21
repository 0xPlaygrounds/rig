//! Venice image-generation model identifiers and typed response metadata.
//!
//! ```
//! use rig_core::providers::venice::image_generation::ImageGenerationResponse;
//! let response: ImageGenerationResponse = serde_json::from_value(
//!     serde_json::json!({"id": "generation-1", "images": []})
//! )?;
//! assert!(response.images.is_empty());
//! # Ok::<(), serde_json::Error>(())
//! ```

use serde::{Deserialize, Serialize};

/// `venice-sd35`
pub const VENICE_SD35: &str = "venice-sd35";
/// Identifier for `z-image-turbo`.
pub const Z_IMAGE_TURBO: &str = "z-image-turbo";
/// Identifier for `qwen-image`.
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
