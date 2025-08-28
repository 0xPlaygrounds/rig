use crate::image::ImageGenerationModel;
use crate::{completion::CompletionModel, embedding::EmbeddingModel};
use aws_config::{BehaviorVersion, Region};
use rig::client::ProviderValue;
use rig::impl_conversion_traits;
use rig::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

pub const DEFAULT_AWS_REGION: &str = "us-east-1";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    region: &'a str,
}

impl<'a> ClientBuilder<'a> {
    #[deprecated(
        since = "0.2.6",
        note = "Use `Client::from_env` or `Client::with_profile_name(\"aws_profile\")` instead"
    )]
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
        Client {
            profile_name: None,
            aws_client: Arc::new(OnceCell::from(client)),
        }
    }
}

impl Default for ClientBuilder<'_> {
    fn default() -> Self {
        #[allow(deprecated)]
        Self::new()
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

    pub async fn get_inner(&self) -> &aws_sdk_bedrockruntime::Client {
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

impl ProviderClient for Client {
    fn from_env() -> Self
    where
        Self: Sized,
    {
        Client::new()
    }

    fn from_val(_: ProviderValue) -> Self
    where
        Self: Sized,
    {
        panic!(
            "Please use `Client::from_env` or `Client::with_profile_name(\"aws_profile\")` instead"
        );
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

impl VerifyClient for Client {
    async fn verify(&self) -> Result<(), VerifyError> {
        // No API endpoint to verify the API key
        Ok(())
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsAudioGeneration for Client
);
