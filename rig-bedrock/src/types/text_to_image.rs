use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use rig::image_generation;
use rig::image_generation::ImageGenerationError;
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
    // The quality of the image.
    // Default: standard
    pub quality: Option<ImageQuality>,
    // The number of images to generate.
    // Default: 1, Minimum: 1, Maximum: 5
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number_of_images: Option<u32>,
    // The height of the image in pixels.
    pub height: Option<u32>,
    // The width of the image in pixels.
    pub width: Option<u32>,
    // Specifies how strongly the generated image should adhere to the prompt. Use a lower value to introduce more randomness in the generation.
    // Default: 8.0. Minimum: 1.1, Maximum: 10.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    // Use to control and reproduce results. Determines the initial noise setting.
    // Use the same seed and the same settings as a previous run to allow inference to create a similar image.
    // Default: 42, Minimum: 0, Maximum: 2147483646
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

    pub fn height(&mut self, height: u32) -> &Self {
        self.image_generation_config.height = Some(height);
        self
    }

    pub fn width(&mut self, width: u32) -> &Self {
        self.image_generation_config.width = Some(width);
        self
    }
}

#[derive(Clone, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TextToImageResponse {
    pub images: Option<Vec<String>>,
    pub error: Option<String>,
}

impl TryFrom<TextToImageResponse>
    for image_generation::ImageGenerationResponse<TextToImageResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: TextToImageResponse) -> Result<Self, Self::Error> {
        if let Some(error) = value.error {
            return Err(ImageGenerationError::ResponseError(error));
        }

        if let Some(images) = value.to_owned().images {
            let data = BASE64_STANDARD
                .decode(&images[0])
                .expect("Could not decode image.");

            return Ok(Self {
                image: data,
                response: value,
            });
        }

        Err(ImageGenerationError::ResponseError(
            "Malformed response from model".to_string(),
        ))
    }
}
