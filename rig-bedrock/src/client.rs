use aws_config::{BehaviorVersion, Region};
use rig::{agent::AgentBuilder, completion, embeddings, extractor::ExtractorBuilder, Embed};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{completion::BedrockProvider, embedding::EmbeddingModel};

// Important: make sure to verify model and region compatibility: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html
pub const DEFAULT_AWS_REGION: &str = "us-east-1";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    region: &'a str,
    pub(crate) additional_fields: Vec<serde_json::Value>,
    pub(crate) deletable_fields: Vec<String>,
}

/// Create a new Bedrock client using the builder
///
/// #(Make sure you have permissions to access Amazon Bedrock foundation model)
/// [https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html]
impl<'a> ClientBuilder<'a> {
    pub fn new() -> Self {
        Self {
            region: DEFAULT_AWS_REGION,
            additional_fields: vec![],
            deletable_fields: vec![],
        }
    }

    pub fn region(mut self, region: &'a str) -> Self {
        self.region = region;
        self
    }

    pub fn additional_fields(mut self, additional_fields: Vec<serde_json::Value>) -> Self {
        self.additional_fields = additional_fields;
        self
    }

    pub fn deletable_fields(mut self, deletable_fields: Vec<String>) -> Self {
        self.deletable_fields = deletable_fields;
        self
    }

    pub async fn build(self) -> Client {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(String::from(self.region)))
            .load()
            .await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
        Client {
            aws_client: client,
            additional_fields: self.additional_fields,
            deletable_fields: self.deletable_fields,
        }
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
    pub(crate) additional_fields: Vec<serde_json::Value>,
    pub(crate) deletable_fields: Vec<String>,
}

impl Client {
    pub fn completion_model<T: completion::CompletionModel>(
        &self,
        completion_model: T,
        model: &str,
    ) -> BedrockProvider<T> {
        BedrockProvider::new(completion_model, self.clone(), model)
    }

    pub fn agent<T: completion::CompletionModel + Clone>(
        &self,
        completion_model: T,
        model: &str,
    ) -> AgentBuilder<BedrockProvider<T>>
    where
        <T as completion::CompletionModel>::Response: serde::de::DeserializeOwned,
        <T as completion::CompletionModel>::Response:
            TryInto<completion::CompletionResponse<<T as completion::CompletionModel>::Response>>,
        <<T as completion::CompletionModel>::Response as TryInto<
            completion::CompletionResponse<<T as completion::CompletionModel>::Response>,
        >>::Error: std::error::Error + Send + Sync + 'static,
    {
        AgentBuilder::new(self.completion_model(completion_model, model))
    }

    pub fn embedding_model(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }

    pub fn extractor<
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
        C: completion::CompletionModel + Clone,
    >(
        &self,
        completion_model: C,
        model: &str,
    ) -> ExtractorBuilder<T, BedrockProvider<C>>
    where
        C::Response: serde::de::DeserializeOwned,
        C::Response: TryInto<completion::CompletionResponse<C::Response>>,
        <C::Response as TryInto<completion::CompletionResponse<C::Response>>>::Error:
            std::error::Error + Send + Sync + 'static,
    {
        ExtractorBuilder::new(self.completion_model(completion_model, model))
    }

    pub fn embeddings<D: Embed>(
        &self,
        model: &str,
        ndims: usize,
    ) -> embeddings::EmbeddingsBuilder<EmbeddingModel, D> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model, ndims))
    }
}
