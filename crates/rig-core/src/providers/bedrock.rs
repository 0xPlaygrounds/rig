//! Transport-independent configuration for the AWS Bedrock provider.
//!
//! The AWS SDK transport remains in `rig-bedrock`. These records live in
//! `rig-core` so closed provider vocabularies can retain the same serde shape
//! whether or not SDK fulfillment is compiled into the current binary.

use serde::{Deserialize, Serialize};

use super::descriptor::ProviderDescriptor;

/// Bedrock's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("aws_bedrock")
    .with_tools(true)
    .with_response_format(true)
    .with_composes_native_output_with_tools(true)
    .with_max_embedding_documents(1024);

/// Plain-data description of how to build a Bedrock runtime client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct ConnectionConfig {
    /// AWS region (`None` defers to the SDK's default region resolution).
    pub region: Option<String>,
    /// Named AWS profile to load credentials from.
    pub profile: Option<String>,
    /// Custom endpoint URL override (local stacks, VPC endpoints).
    pub endpoint_url: Option<String>,
}

/// Plain-data Bedrock completion configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Reusable AWS client-construction data.
    #[serde(flatten)]
    pub connection: ConnectionConfig,
    /// Model identifier requests are built for.
    pub model: String,
    /// Enable Bedrock prompt caching for completion requests.
    #[serde(default)]
    pub prompt_caching: bool,
}

impl Config {
    /// Config for `model` using the SDK's default credential and region chain.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: ConnectionConfig::default(),
            model: model.into(),
            prompt_caching: false,
        }
    }

    /// Pin the AWS region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.connection.region = Some(region.into());
        self
    }

    /// Load credentials from a named AWS profile.
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.connection.profile = Some(profile.into());
        self
    }

    /// Override the endpoint URL.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.connection.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Enable Bedrock prompt caching.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }
}

impl std::ops::Deref for Config {
    type Target = ConnectionConfig;

    fn deref(&self) -> &Self::Target {
        &self.connection
    }
}

impl std::ops::DerefMut for Config {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.connection
    }
}

/// Plain-data Bedrock embeddings configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// Reusable AWS client-construction data.
    #[serde(flatten)]
    pub connection: ConnectionConfig,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions.
    pub ndims: Option<usize>,
}

impl EmbeddingConfig {
    /// Config for `model` using the SDK's default credential and region chain.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: ConnectionConfig::default(),
            model: model.into(),
            ndims: None,
        }
    }

    /// Pin the AWS region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.connection.region = Some(region.into());
        self
    }

    /// Load credentials from a named AWS profile.
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.connection.profile = Some(profile.into());
        self
    }

    /// Override the endpoint URL.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.connection.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Request `ndims`-sized embeddings.
    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }

    /// Project this embedding config to the completion-shaped client config.
    pub fn client_config(&self) -> Config {
        Config {
            connection: self.connection.clone(),
            model: self.model.clone(),
            prompt_caching: false,
        }
    }
}

impl std::ops::Deref for EmbeddingConfig {
    type Target = ConnectionConfig;

    fn deref(&self) -> &Self::Target {
        &self.connection
    }
}

impl std::ops::DerefMut for EmbeddingConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.connection
    }
}
