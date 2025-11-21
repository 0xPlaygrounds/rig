use crate::client::Client;
use crate::types::errors::AwsSdkInvokeModelError;
use crate::types::text_to_image::{TextToImageGeneration, TextToImageResponse};
use aws_smithy_types::Blob;
use rig::image_generation::{
    self, ImageGenerationError, ImageGenerationRequest, ImageGenerationResponse,
};
use rig::models;

models! {
    pub enum ImageGenerationModels {
        /// `amazon.titan-image-generator-v1`
        AmazonTitanImageGenerator1 => "amazon.titan-image-generator-v1",
        /// `amazon.titan-image-generator-v2:0`
        AmazonTitanImageGenerator2 => "amazon.titan-image-generator-v2:0",
        /// `amazon.nova-canvas-v1:0`
        AmazonNovaCanvas => "amazon.nova-canvas-v1:0",
    }
}
pub use ImageGenerationModels::*;

#[derive(Clone)]
pub struct ImageGenerationModel {
    pub(crate) client: Client,
    pub model: String,
}

impl ImageGenerationModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub fn with_model(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl image_generation::ImageGenerationModel for ImageGenerationModel {
    type Response = TextToImageResponse;

    type Client = Client;
    type Models = String;

    fn make(client: &Self::Client, model: Self::Models) -> Self {
        Self::new(client.clone(), model.as_str())
    }

    fn make_custom(client: &Self::Client, model: &str) -> Self {
        Self::with_model(client.clone(), model)
    }

    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse<Self::Response>, ImageGenerationError> {
        let mut request = TextToImageGeneration::new(generation_request.prompt);
        request.width(generation_request.width);
        request.height(generation_request.height);

        let body = serde_json::to_string(&request)?;
        let model_response = self
            .client
            .get_inner()
            .await
            .invoke_model()
            .model_id(self.model.as_str())
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|sdk_error| {
                Into::<ImageGenerationError>::into(AwsSdkInvokeModelError(sdk_error))
            })?;

        let response_str = String::from_utf8(model_response.body.into_inner())
            .map_err(|e| ImageGenerationError::ResponseError(e.to_string()))?;

        let result: TextToImageResponse = serde_json::from_str(&response_str)
            .map_err(|e| ImageGenerationError::ResponseError(e.to_string()))?;

        result.try_into()
    }
}
