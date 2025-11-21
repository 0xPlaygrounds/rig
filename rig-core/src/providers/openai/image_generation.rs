use super::{Client, client::ApiResponse};
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::models;
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// OpenAI Image Generation API
// ================================================================

models! {
    pub enum ImageGenerationModels {
        DallE2 => "dall-e-2",
        DallE3 => "dall-e-3",
        GPTImage1 => "gpt-image-1",
    }
}
pub use ImageGenerationModels::*;

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
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: dall-e-2)
    pub model: String,
}

impl<T> ImageGenerationModel<T> {
    pub(crate) fn new(client: Client<T>, model: ImageGenerationModels) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = ImageGenerationResponse;

    type Client = Client<T>;
    type Models = ImageGenerationModels;

    fn make(client: &Self::Client, model: Self::Models) -> Self {
        Self::new(client.clone(), model)
    }

    fn make_custom(client: &Self::Client, model: &str) -> Self {
        Self::with_model(client.clone(), model)
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let mut request = json!({
            "model": self.model,
            "prompt": generation_request.prompt,
            "size": format!("{}x{}", generation_request.width, generation_request.height),
        });

        if self.model.as_str()
            != <ImageGenerationModels as Into<&str>>::into(ImageGenerationModels::GPTImage1)
        {
            merge_inplace(
                &mut request,
                json!({
                    "response_format": "b64_json"
                }),
            );
        }

        let body = serde_json::to_vec(&request)?;

        let request = self
            .client
            .post("/images/generations")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = http_client::text(response).await?;

            return Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                status, text,
            )));
        }

        let text = http_client::text(response).await?;

        match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&text)? {
            ApiResponse::Ok(response) => response.try_into(),
            ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message)),
        }
    }
}
