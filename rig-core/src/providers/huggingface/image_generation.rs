use super::Client;
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::{image_generation, models};
use serde_json::json;

models! {
    pub enum ImageGenerationModels {
        Flux1 => "black-forest-labs/FLUX.1-dev",
        Kolors => "Kwai-Kolors/Kolors",
        StableDiffusion3 => "stabilityai/stable-diffusion-3-medium-diffusers"
    }
}
pub use ImageGenerationModels::*;

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
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> ImageGenerationModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        ImageGenerationModel {
            client,
            model: model.to_string(),
        }
    }
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Send + Clone + 'static,
{
    type Response = ImageGenerationResponse;

    type Client = Client<T>;
    type Models = ImageGenerationModels;

    fn make(client: &Self::Client, model: Self::Models) -> Self {
        ImageGenerationModel::new(client.clone(), &model.to_string())
    }

    #[cfg_attr(feature = "worker", worker::send)]
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
            .subprovider()
            .image_generation_endpoint(&self.model)?;

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post(&route)?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let status = response.status();
            let text: Vec<u8> = response.into_body().await?;
            let text: String = String::from_utf8_lossy(&text).into();

            return Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                status, text
            )));
        }

        let data: Vec<u8> = response.into_body().await?;

        ImageGenerationResponse { data }.try_into()
    }
}
