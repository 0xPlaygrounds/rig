use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::providers::openai::{ApiResponse, Client};
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// OpenAI Image Generation API
// ================================================================
pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";

pub const GPT_IMAGE_1: &str = "gpt-image-1";

#[derive(Debug, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i32,
    pub data: Vec<ImageGenerationData>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        let b64_json = value.data[0].b64_json.clone();

        let bytes = BASE64_STANDARD
            .decode(&b64_json)
            .expect("Failed to decode b64");

        Ok(image_generation::ImageGenerationResponse {
            image: bytes,
            response: value,
        })
    }
}

#[derive(Clone)]
pub struct ImageGenerationModel {
    client: Client,
    /// Name of the model (e.g.: dall-e-2)
    pub model: String,
}

impl ImageGenerationModel {
    pub(crate) fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl image_generation::ImageGenerationModel for ImageGenerationModel {
    type Response = ImageGenerationResponse;

    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let request = json!({
            "model": self.model,
            "prompt": generation_request.prompt,
            "size": format!("{}x{}", generation_request.width, generation_request.height),
            "response_format": "b64_json"
        });

        let response = self
            .client
            .post("/images/generations")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        let t = response.text().await?;

        match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&t)? {
            ApiResponse::Ok(response) => response.try_into(),
            ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message)),
        }
    }
}
