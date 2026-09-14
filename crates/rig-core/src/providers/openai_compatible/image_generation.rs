//! Image response types shared by OpenAI-compatible endpoints.
use crate::image_generation;
use crate::image_generation::{ImageGenerationError, NormalizeImageGenerationResponse};
use crate::providers::internal::image_generation::decode_base64_image;
use serde::{Deserialize, Serialize};

// ================================================================
// OpenAI Image Generation API
// ================================================================
pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";
pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i32,
    pub data: Vec<ImageGenerationData>,
}

impl NormalizeImageGenerationResponse for ImageGenerationResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        let image = decode_base64_image(
            &self,
            |response| response.data.first().map(|image| image.b64_json.as_str()),
            "missing image data",
            None,
        )?;
        Ok(image_generation::ImageGenerationResponse::new(
            image, provider,
        ))
    }
}
