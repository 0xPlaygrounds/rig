//! Concrete AWS Bedrock connection client.

use aws_sdk_bedrockruntime::Client as AwsClient;
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::functions::{Config, ConnectionConfig, EmbeddingConfig, ImageConfig};

/// Default region retained for the explicit builder path.
pub const DEFAULT_AWS_REGION: &str = "us-east-1";

/// Reusable Bedrock connection data plus an optional live SDK handle.
#[derive(Clone, Debug)]
pub struct Client {
    connection: ConnectionConfig,
    aws_client: Arc<OnceCell<AwsClient>>,
}

/// Owned, monomorphic Bedrock client builder.
#[derive(Clone, Debug)]
pub struct ClientBuilder {
    connection: ConnectionConfig,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            connection: ConnectionConfig {
                region: Some(DEFAULT_AWS_REGION.to_string()),
                profile: None,
                endpoint_url: None,
            },
        }
    }
}

impl ClientBuilder {
    /// Pin the AWS region.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.connection.region = Some(region.into());
        self
    }

    /// Use the SDK's default region resolution.
    pub fn default_region(mut self) -> Self {
        self.connection.region = None;
        self
    }

    /// Load credentials from an AWS profile.
    pub fn profile(mut self, profile: impl Into<String>) -> Self {
        self.connection.profile = Some(profile.into());
        self
    }

    /// Override the Bedrock endpoint URL.
    pub fn endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.connection.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Build and retain the live AWS SDK client for runtime reuse.
    pub async fn build(self) -> Client {
        let aws_client = crate::functions::client_from_connection(&self.connection).await;
        Client {
            connection: self.connection,
            aws_client: Arc::new(OnceCell::new_with(Some(aws_client))),
        }
    }
}

impl From<AwsClient> for Client {
    fn from(aws_client: AwsClient) -> Self {
        Self {
            connection: ConnectionConfig::default(),
            aws_client: Arc::new(OnceCell::new_with(Some(aws_client))),
        }
    }
}

impl Client {
    /// A lazy client using the AWS SDK's default credential and region chains.
    pub fn from_env() -> Self {
        Self {
            connection: ConnectionConfig::default(),
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// Start an owned Bedrock client builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// A lazy client using a named AWS profile.
    pub fn with_profile_name(profile: impl Into<String>) -> Self {
        Self {
            connection: ConnectionConfig {
                region: None,
                profile: Some(profile.into()),
                endpoint_url: None,
            },
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// Materialize completion configuration for `model`.
    pub fn config(&self, model: impl Into<String>) -> Config {
        Config {
            connection: self.connection.clone(),
            model: model.into(),
            prompt_caching: false,
        }
    }

    /// Materialize embedding configuration for `model`.
    pub fn embedding_config(&self, model: impl Into<String>) -> EmbeddingConfig {
        let mut config = EmbeddingConfig::new(model);
        config.connection = self.connection.clone();
        config
    }

    /// Materialize image-generation configuration for `model`.
    pub fn image_config(&self, model: impl Into<String>) -> ImageConfig {
        let mut config = ImageConfig::new(model);
        config.connection = self.connection.clone();
        config
    }

    /// Canonical AWS connection data.
    pub fn connection_config(&self) -> &ConnectionConfig {
        &self.connection
    }

    /// The already-initialized SDK client, when one is available.
    pub fn seeded_aws_client(&self) -> Option<AwsClient> {
        self.aws_client.get().cloned()
    }

    /// Resolve the live AWS SDK client, reusing a seeded handle when present.
    pub async fn get_inner(&self) -> AwsClient {
        self.aws_client
            .get_or_init(|| crate::functions::client_from_connection(&self.connection))
            .await
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::config::{BehaviorVersion, Region};

    fn test_sdk_client() -> AwsClient {
        AwsClient::from_conf(
            aws_sdk_bedrockruntime::config::Builder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(Region::new("shared-cache-marker"))
                .endpoint_url("http://bedrock-cache.invalid")
                .build(),
        )
    }

    #[test]
    fn lazy_client_clones_share_one_uninitialized_cache() {
        let client = Client::from_env();
        let clone = client.clone();

        assert!(Arc::ptr_eq(&client.aws_client, &clone.aws_client));
        assert!(client.seeded_aws_client().is_none());
        assert!(clone.seeded_aws_client().is_none());
    }

    #[tokio::test]
    async fn repeated_direct_access_reuses_the_same_sdk_client() {
        let client = Client::from(test_sdk_client());
        let clone = client.clone();

        let first = client.get_inner().await;
        let second = client.get_inner().await;
        let through_clone = clone.get_inner().await;

        assert!(std::ptr::eq(first.config(), second.config()));
        assert!(std::ptr::eq(first.config(), through_clone.config()));
    }
}
