use super::client::Client;
use crate::http_client::HttpClientExt;
use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use serde_json::json;

#[allow(non_upper_case_globals)]
pub mod image_generation_models {
    pub const Flux1: &str = "black-forest-labs/FLUX.1-dev";
    pub const Kolors: &str = "Kwai-Kolors/Kolors";
    pub const StableDiffusion3: &str = "stabilityai/stable-diffusion-3-medium-diffusers";
}
pub use image_generation_models::*;

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
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        ImageGenerationModel {
            client,
            model: model.into(),
        }
    }
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Send + Clone + 'static,
{
    type Response = ImageGenerationResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

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

            return Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(&text),
            ));
        }

        let data: Vec<u8> = response.into_body().await?;

        ImageGenerationResponse { data }.try_into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(Flux1);

        let request = model
            .image_generation_request()
            .prompt("draw a cat")
            .build();

        let error = model
            .image_generation(request)
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
