use crate::client::Client;
use crate::types::errors::AwsSdkInvokeModelError;
use crate::types::text_to_image::{TextToImageGeneration, TextToImageResponse};
use aws_smithy_types::Blob;
use rig_core::image_generation::{
    self, ImageGenerationError, ImageGenerationRequest, ImageGenerationResponse,
};

// The model-id string values are canonically defined in `crate::completion`.
// The Titan image generators are gone: `amazon.titan-image-generator-v1` and
// `-v2:0` are absent from `ListFoundationModels` in every region checked
// (us-east-1, us-west-2, eu-central-1, ap-northeast-1), so their aliases are
// removed rather than left pointing at identifiers Bedrock rejects.
pub use crate::completion::{
    AMAZON_NOVA_CANVAS, STABILITY_SD3_5_LARGE, STABILITY_STABLE_IMAGE_CORE_1_0,
    STABILITY_STABLE_IMAGE_ULTRA_1_0,
};

#[derive(Clone)]
pub struct ImageGenerationModel {
    pub(crate) client: Client,
    pub model: String,
}

impl ImageGenerationModel {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl image_generation::ImageGenerationModel for ImageGenerationModel {
    type Response = TextToImageResponse;

    type Client = Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
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
