use super::Client;
use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use serde_json::json;

pub const FLUX_1: &str = "black-forest-labs/FLUX.1-dev";
pub const KOLORS: &str = "Kwai-Kolors/Kolors";
pub const STABLE_DIFFUSION_3: &str = "stabilityai/stable-diffusion-3-medium-diffusers";

#[derive(Debug)]
pub struct ImageGenerationResponse {
    data: Vec<u8>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        Ok(image_generation::ImageGenerationResponse {
            image: value.data.clone(),
            response: value,
        })
    }
}

#[derive(Clone)]
pub struct ImageGenerationModel {
    client: Client,
    pub model: String,
}

impl ImageGenerationModel {
    pub fn new(client: Client, model: &str) -> Self {
        ImageGenerationModel {
            client,
            model: model.to_string(),
        }
    }
}

impl image_generation::ImageGenerationModel for ImageGenerationModel {
    type Response = ImageGenerationResponse;

    async fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let request = json!({
            "inputs": request.prompt,
            "parameters": {
                "width": request.width,
                "height": request.height
            }
        });

        let route = self
            .client
            .sub_provider
            .image_generation_endpoint(&self.model)?;

        let response = self.client.post(&route).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        let data = response.bytes().await?.to_vec();

        ImageGenerationResponse { data }.try_into()
    }
}
