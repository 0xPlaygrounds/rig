//! xAI's image-generation model identifiers and its own view of a generated
//! image.
//!
//! The request runs on the shared OpenAI image wire, whose
//! [`XAI`](crate::providers::openai::wire::XAI) dialect carries the
//! `/v1/images/generations` path and xAI's body — no `size`, an explicit
//! `response_format: "b64_json"`, and an `aspect_ratio`.

use serde::{Deserialize, Serialize};

// ================================================================
// xAI Image Generation API
// ================================================================
pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub data: Vec<ImageGenerationData>,
}
