use crate::client::Client;
use crate::types::assistant_content::PROVIDER_NAME;
use crate::types::errors::AwsSdkInvokeModelError;
use crate::types::text_to_image::{TextToImageGeneration, TextToImageResponse};
use aws_smithy_types::Blob;
use rig_core::image_generation::{
    self, ImageGenerationError, ImageGenerationRequest, ImageGenerationResponse,
    NormalizeImageGenerationResponse,
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

impl ImageGenerationModel {
    /// Perform the generation and return Bedrock's native
    /// [`TextToImageResponse`] instead of the normalized
    /// [`ImageGenerationResponse`]. Same request, transport, parser, and error
    /// path as [`image_generation::ImageGenerationModel::image_generation`].
    pub async fn raw_image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<TextToImageResponse, ImageGenerationError> {
        self.raw_image_generation_with_request_id(generation_request)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_image_generation`] plus the AWS request id
    /// (`x-amzn-RequestId`) from the SDK's response metadata, when present.
    pub async fn raw_image_generation_with_request_id(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<(TextToImageResponse, Option<String>), ImageGenerationError> {
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

        let provider_request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&model_response)
                .map(str::to_string);

        let response_str = String::from_utf8(model_response.body.into_inner())
            .map_err(|e| ImageGenerationError::ResponseError(e.to_string()))?;

        let result: TextToImageResponse = serde_json::from_str(&response_str)
            .map_err(|e| ImageGenerationError::ResponseError(e.to_string()))?;

        Ok((result, provider_request_id))
    }
}

impl image_generation::ImageGenerationModel for ImageGenerationModel {
    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, ImageGenerationError> {
        let (response, provider_request_id) = self
            .raw_image_generation_with_request_id(generation_request)
            .await?;
        let captured = serde_json::to_value(&response)?;
        Ok(response
            .normalize(PROVIDER_NAME)?
            .with_optional_provider_request_id(provider_request_id)
            .with_raw(captured))
    }
}

impl rig_core::client::ConstructImageGenerationModel<Client> for ImageGenerationModel {
    fn construct(client: &Client, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}
