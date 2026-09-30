//! The Bedrock runtime transport: the AWS SDK client every Bedrock wire is
//! sent through.
//!
//! ```no_run
//! use rig_bedrock::client::BedrockRuntime;
//! use rig_bedrock::completion::{AMAZON_NOVA_LITE, Converse};
//! use rig_core::Model;
//!
//! let model = BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE);
//! # let _ = model;
//! ```

use aws_config::{BehaviorVersion, Region};
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::completion::Converse;
use crate::embedding::Embeddings;
use crate::image::Images;
use rig_core::Model;

pub const DEFAULT_AWS_REGION: &str = "us-east-1";

#[derive(Clone)]
pub struct Builder<'a> {
    region: &'a str,
}

impl<'a> Builder<'a> {
    /// Sets the AWS region. The selected model must be
    /// [available there](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html).
    pub fn region(mut self, region: &'a str) -> Self {
        self.region = region;
        self
    }

    /// Loads AWS SDK configuration and constructs a runtime for the selected region.
    /// Requests require permission to access the selected Bedrock model.
    pub async fn build(self) -> BedrockRuntime {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(String::from(self.region)))
            .load()
            .await;
        BedrockRuntime::from(aws_sdk_bedrockruntime::Client::new(&sdk_config))
    }
}

impl Default for Builder<'_> {
    fn default() -> Self {
        Self {
            region: DEFAULT_AWS_REGION,
        }
    }
}

/// The Bedrock runtime client every Bedrock wire is sent through. Clones
/// share one client, which loads its AWS configuration on first use unless
/// it was built from a configured SDK client.
#[derive(Clone, Debug)]
pub struct BedrockRuntime {
    profile_name: Option<String>,
    aws_client: Arc<OnceCell<aws_sdk_bedrockruntime::Client>>,
}

impl From<aws_sdk_bedrockruntime::Client> for BedrockRuntime {
    fn from(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Self {
            profile_name: None,
            aws_client: Arc::new(OnceCell::from(aws_client)),
        }
    }
}

impl BedrockRuntime {
    /// A builder that loads AWS SDK configuration for a chosen region.
    pub fn builder<'a>() -> Builder<'a> {
        Builder::default()
    }

    /// A runtime that loads AWS SDK configuration from the environment on
    /// first use. Construction does not validate credentials.
    pub fn from_env() -> Self {
        Self {
            profile_name: None,
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// A runtime that loads the named AWS profile on first use.
    pub fn with_profile_name(profile_name: &str) -> Self {
        Self {
            profile_name: Some(profile_name.into()),
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// The Converse model for `model`. Wire options such as a guardrail go
    /// on the wire: `Model::new(Converse::new(model).with_guardrail(..), runtime)`.
    pub fn completion(&self, model: impl Into<String>) -> Model<Converse, Self> {
        Model::new(Converse::new(model), self.clone())
    }

    /// The embedding model for `model`, at its default width.
    /// [`Embeddings::with_ndims`] asks for another.
    pub fn embedding(&self, model: impl Into<String>) -> Model<Embeddings, Self> {
        Model::new(Embeddings::new(model), self.clone())
    }

    /// The image-generation model for `model`.
    pub fn image_generation(&self, model: impl Into<String>) -> Model<Images, Self> {
        Model::new(Images::new(model), self.clone())
    }

    /// The AWS SDK client, loading its configuration on first use.
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
