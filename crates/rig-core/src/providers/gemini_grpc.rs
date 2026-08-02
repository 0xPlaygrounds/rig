//! Transport-independent configuration for Gemini's gRPC provider.
//!
//! The tonic transport remains in `rig-gemini-grpc`. These records live in
//! `rig-core` so closed provider vocabularies remain feature-stable.

use serde::{Deserialize, Serialize};

use super::descriptor::{ApiKeyLocation, ProviderDescriptor};

/// Default Gemini gRPC endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

/// Gemini gRPC's capability sheet.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("gemini-grpc")
    .with_tools(true)
    .with_max_embedding_documents(100);

/// Plain-data Gemini gRPC connection configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ConnectionConfig {
    /// gRPC endpoint URL (`None` uses [`DEFAULT_ENDPOINT`]).
    pub endpoint: Option<String>,
    /// Credential location.
    pub api_key: ApiKeyLocation,
}

/// Plain-data Gemini gRPC completion configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Reusable channel-construction data.
    #[serde(flatten)]
    pub connection: ConnectionConfig,
    /// Model identifier requests are built for.
    pub model: String,
}

impl Config {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: ConnectionConfig {
                endpoint: None,
                api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            },
            model: model.into(),
        }
    }

    /// Config for `model` with an explicit API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.connection.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the gRPC endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.connection.endpoint = Some(endpoint.into());
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

/// Plain-data Gemini gRPC embeddings configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// Reusable channel-construction data.
    #[serde(flatten)]
    pub connection: ConnectionConfig,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested `output_dimensionality`.
    pub ndims: Option<usize>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            connection: ConnectionConfig {
                endpoint: None,
                api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            },
            model: model.into(),
            ndims: None,
        }
    }

    /// Config for `model` with an explicit API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.connection.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the gRPC endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.connection.endpoint = Some(endpoint.into());
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
