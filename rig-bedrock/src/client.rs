use crate::image::ImageGenerationModel;
use crate::{completion::CompletionModel, embedding::EmbeddingModel};
use aws_config::{BehaviorVersion, Region};
use rig::client::ProviderValue;
use rig::impl_conversion_traits;
use rig::prelude::*;

pub const DEFAULT_AWS_REGION: &str = "us-east-1";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    region: &'a str,
}

/// Create a new Bedrock client using the builder <br>
impl<'a> ClientBuilder<'a> {
    pub fn new() -> Self {
        Self {
            region: DEFAULT_AWS_REGION,
        }
    }

    /// Make sure to verify model and region [compatibility]
    ///
    /// [compatibility]: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html
    pub fn region(mut self, region: &'a str) -> Self {
        self.region = region;
        self
    }

    /// Make sure you have permissions to access [Amazon Bedrock foundation model]
    ///
    /// [ Amazon Bedrock foundation model]: <https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html>
    pub async fn build(self) -> Client {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(String::from(self.region)))
            .load()
            .await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
        Client { aws_client: client }
    }
}

impl Default for ClientBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    pub(crate) aws_client: aws_sdk_bedrockruntime::Client,
}

impl From<aws_sdk_bedrockruntime::Client> for Client {
    fn from(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Client { aws_client }
    }
}

impl Client {}

impl ProviderClient for Client {
    fn from_env() -> Self
    where
        Self: Sized,
    {
        panic!("You should not call from_env to build a Bedrock client");
    }

    fn from_val(_: ProviderValue) -> Self
    where
        Self: Sized,
    {
        panic!("Unimplemented due to lack of use. Please reach out if you need to use this!");
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;

    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, None)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }
}

impl ImageGenerationClient for Client {
    type ImageGenerationModel = ImageGenerationModel;

    fn image_generation_model(&self, model: &str) -> ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }
}
impl_conversion_traits!(
    AsTranscription,
    AsAudioGeneration for Client
);
