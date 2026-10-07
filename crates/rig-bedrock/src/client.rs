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
use rig_core::Model;

/// Settings for a [`BedrockRuntime`]. Unset values fall back to the AWS SDK's
/// default provider chain (environment, shared config files, instance
/// metadata) when the runtime first loads its configuration.
#[derive(Clone, Debug, Default)]
pub struct Builder {
    region: Option<String>,
    profile_name: Option<String>,
}

impl Builder {
    /// Sets the AWS region. The selected model must be
    /// [available there](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html).
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Loads credentials and settings from the named profile in the shared
    /// AWS config files. An explicit [`region`](Self::region) overrides the
    /// profile's region.
    pub fn profile_name(mut self, profile_name: impl Into<String>) -> Self {
        self.profile_name = Some(profile_name.into());
        self
    }

    /// A runtime that loads its AWS configuration on first use.
    /// Construction does not validate credentials.
    pub fn build(self) -> BedrockRuntime {
        BedrockRuntime {
            settings: self,
            aws_client: Arc::new(OnceCell::new()),
        }
    }
}

/// The Bedrock runtime client every Bedrock wire is sent through. Clones
/// share one client, which loads its AWS configuration on first use unless
/// it was built from a configured SDK client.
#[derive(Clone, Debug)]
pub struct BedrockRuntime {
    settings: Builder,
    aws_client: Arc<OnceCell<aws_sdk_bedrockruntime::Client>>,
}

impl From<aws_sdk_bedrockruntime::Client> for BedrockRuntime {
    fn from(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Self {
            settings: Builder::default(),
            aws_client: Arc::new(OnceCell::from(aws_client)),
        }
    }
}

impl BedrockRuntime {
    /// A builder for a runtime with an explicit region or profile.
    ///
    /// ```no_run
    /// use rig_bedrock::client::BedrockRuntime;
    ///
    /// let runtime = BedrockRuntime::builder()
    ///     .profile_name("bedrock")
    ///     .region("eu-west-1")
    ///     .build();
    /// # let _ = runtime;
    /// ```
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// A runtime that loads AWS SDK configuration from the environment on
    /// first use. Construction does not validate credentials.
    pub fn from_env() -> Self {
        Builder::default().build()
    }

    /// The Converse model for `model`. Request options such as a guardrail
    /// go in the request's provider options
    /// ([`BedrockOptions`](crate::extension::BedrockOptions)).
    pub fn completion(&self, model: impl Into<String>) -> Model<Converse, Self> {
        Model::new(Converse::new(model), self.clone())
    }

    /// The AWS SDK client, loading its configuration on first use.
    pub async fn inner(&self) -> &aws_sdk_bedrockruntime::Client {
        self.aws_client
            .get_or_init(|| async {
                let mut loader = aws_config::defaults(BehaviorVersion::latest());
                if let Some(profile_name) = &self.settings.profile_name {
                    loader = loader.profile_name(profile_name);
                }
                if let Some(region) = self.settings.region.clone() {
                    loader = loader.region(Region::new(region));
                }
                aws_sdk_bedrockruntime::Client::new(&loader.load().await)
            })
            .await
    }
}

#[cfg(test)]
mod tests;
