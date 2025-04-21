use crate::image::ImageGenerationModel;
use crate::{completion::CompletionModel, embedding::EmbeddingModel};
use aws_config::{BehaviorVersion, Region};
use rig::{agent::AgentBuilder, embeddings, extractor::ExtractorBuilder, Embed};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

#[derive(Clone)]
pub struct Client {
    pub(crate) aws_client: aws_sdk_bedrockruntime::Client,
}

impl From<aws_sdk_bedrockruntime::Client> for Client {
    fn from(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Client { aws_client }
    }
}

impl Client {
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn embedding_model(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }

    pub fn embeddings<D: Embed>(
        &self,
        model: &str,
        ndims: usize,
    ) -> embeddings::EmbeddingsBuilder<EmbeddingModel, D> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model, ndims))
    }

    pub fn image_generation_model(&self, model: &str) -> ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }
}
