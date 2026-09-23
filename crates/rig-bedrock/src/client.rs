use crate::image::ImageGenerationModel;
use crate::{completion::CompletionModel, embedding::EmbeddingModel};
use aws_config::{BehaviorVersion, Region};
use rig_core::driver::CompletionProvider;
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::error::ProviderError;
use std::sync::Arc;
use tokio::sync::OnceCell;

pub const DEFAULT_AWS_REGION: &str = "us-east-1";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    region: &'a str,
}

impl<'a> ClientBuilder<'a> {
    /// Sets the AWS region. The selected model must be
    /// [available there](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html).
    pub fn region(mut self, region: &'a str) -> Self {
        self.region = region;
        self
    }

    /// Loads AWS SDK configuration and constructs a client for the selected region.
    /// Requests require permission to access the selected Bedrock model.
    pub async fn build(self) -> Client {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(String::from(self.region)))
            .load()
            .await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
        Client {
            profile_name: None,
            aws_client: Arc::new(OnceCell::from(client)),
        }
    }
}

impl Default for ClientBuilder<'_> {
    fn default() -> Self {
        Self {
            region: DEFAULT_AWS_REGION,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    profile_name: Option<String>,
    pub(crate) aws_client: Arc<OnceCell<aws_sdk_bedrockruntime::Client>>,
}

impl From<aws_sdk_bedrockruntime::Client> for Client {
    fn from(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Client {
            profile_name: None,
            aws_client: Arc::new(OnceCell::from(aws_client)),
        }
    }
}

impl Client {
    fn new() -> Self {
        Self {
            profile_name: None,
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// Create an AWS Bedrock client using AWS profile name
    pub fn with_profile_name(profile_name: &str) -> Self {
        Self {
            profile_name: Some(profile_name.into()),
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    pub async fn inner(&self) -> &aws_sdk_bedrockruntime::Client {
        self.aws_client
            .get_or_init(|| async {
                let config = if let Some(profile_name) = &self.profile_name {
                    aws_config::defaults(BehaviorVersion::latest())
                        .profile_name(profile_name)
                        .load()
                        .await
                } else {
                    aws_config::load_from_env().await
                };
                aws_sdk_bedrockruntime::Client::new(&config)
            })
            .await
    }
}

impl Client {
    /// Creates a client that loads AWS SDK configuration on first use.
    /// Construction does not validate credentials and always succeeds.
    pub fn from_env() -> Result<Self, rig_core::client::ProviderClientError> {
        Ok(Client::new())
    }

    /// This provider's embedding model for `model`, at `ndims` dimensions
    /// when the caller named one rather than taking the model's default.
    pub fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// An embedding builder over this provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embedding builder over this provider's `model` at `ndims`
    /// dimensions.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }

    /// This provider's image-generation model for `model`.
    pub fn image_generation(&self, model: impl Into<String>) -> ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }

    /// Returns success without making a request or validating credentials.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}

impl CompletionProvider for Client {
    type Model = CompletionModel;

    fn completion(&self, model: impl Into<String>) -> Self::Model {
        CompletionModel::new(self.clone(), model)
    }
}
