use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use rig_core::error::ProviderError;
use rig_core::image_generation;
use rig_core::image_generation::NormalizeImageGenerationResponse;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageQuality {
    Standard,
    Premium,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerationConfig {
    /// Image quality. Defaults to standard.
    pub quality: Option<ImageQuality>,
    /// Requested image count, defaulting to one. Provider limits are not validated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number_of_images: Option<u32>,
    /// Image height in pixels, defaulting to 512.
    pub height: Option<u32>,
    /// Image width in pixels, defaulting to 512.
    pub width: Option<u32>,
    /// Prompt adherence strength. Omitted by default for the provider to choose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    /// Initial noise seed for reproducible generation. Omitted by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        ImageGenerationConfig {
            quality: Some(ImageQuality::Standard),
            number_of_images: Some(1),
            height: Some(512),
            width: Some(512),
            cfg_scale: None,
            seed: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextToImageParams {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_text: Option<String>,
}

impl TextToImageParams {
    pub fn new(text: String) -> Self {
        Self {
            text,
            negative_text: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextToImageGeneration {
    pub task_type: &'static str,
    pub text_to_image_params: TextToImageParams,
    pub image_generation_config: ImageGenerationConfig,
}

impl TextToImageGeneration {
    pub(crate) fn new(text: String) -> TextToImageGeneration {
        TextToImageGeneration {
            task_type: "TEXT_IMAGE",
            text_to_image_params: TextToImageParams::new(text),
            image_generation_config: Default::default(),
        }
    }

    pub fn height(mut self, height: u32) -> Self {
        self.image_generation_config.height = Some(height);
        self
    }

    pub fn width(mut self, width: u32) -> Self {
        self.image_generation_config.width = Some(width);
        self
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TextToImageResponse {
    pub images: Option<Vec<String>>,
    pub error: Option<String>,
}

impl NormalizeImageGenerationResponse for TextToImageResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<image_generation::ImageGenerationResponse, ProviderError> {
        if let Some(error) = self.error {
            return Err(ProviderError::Response(error));
        }

        if let Some(images) = self.images {
            let image = images.first().ok_or_else(|| {
                ProviderError::Response("Bedrock image response was empty".into())
            })?;
            let data = BASE64_STANDARD
                .decode(image)
                .map_err(|err| ProviderError::Response(err.to_string()))?;

            return Ok(image_generation::ImageGenerationResponse::new(
                data, provider,
            ));
        }

        Err(ProviderError::Response(
            "Malformed response from model".to_string(),
        ))
    }
}
