use crate::image_generation;
use crate::image_generation::{
    ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
};
use crate::json_utils::merge_inplace;
use crate::providers::internal::image_generation::{
    GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

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

impl NormalizeImageGenerationResponse for ImageGenerationResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        let image = decode_base64_image(
            &self,
            |response| response.data.first().map(|image| image.b64_json.as_str()),
            "No image data returned",
            Some("Base64 decode error: "),
        )?;
        Ok(image_generation::ImageGenerationResponse::new(
            image, provider,
        ))
    }
}

/// xAI image generation model.
pub type ImageGenerationModel<T = crate::http_client::BoxedHttpClient> =
    GenericImageGenerationModel<super::client::XAi, T>;

impl JsonImageGenerationProvider for super::client::XAi {
    const IMAGE_GENERATION_PATH: &'static str = "/v1/images/generations";
    const PROVIDER_NAME: &'static str = "xai";
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
mod tests;
