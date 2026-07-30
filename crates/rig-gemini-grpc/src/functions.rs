//! Gemini gRPC as config + free functions — the crate's entry point.
//!
//! gRPC is a non-HTTP transport: instead of a pure request-builder there is a
//! serde [`Config`] describing how to *build* the connected tonic channel and
//! API-key interceptor (never holding them), an async [`client_from_config`]
//! producing the live [`Client`] handle, and free
//! [`complete`]/[`open_stream`]/[`embed`]/[`embed_batches`] functions taking
//! that handle.

use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
use rig_core::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use tonic::transport::{ClientTlsConfig, Endpoint};

use crate::Client;

/// Default Gemini gRPC endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

/// Gemini gRPC's capability sheet.
///
/// `supports_response_format` is false because `create_grpc_request` drops
/// `output_schema` (and `tool_choice`/`additional_params`) during request
/// conversion. The streaming flags are OpenAI-wire concepts that do not
/// apply to the gRPC transport.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("gemini-grpc")
    .with_tools(true)
    .with_max_embedding_documents(100);

/// Plain-data Gemini gRPC provider configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// gRPC endpoint URL (`None` uses [`DEFAULT_ENDPOINT`] with the exact
    /// TLS setup of [`Client::new`]).
    pub endpoint: Option<String>,
    /// Credential location.
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
}

impl Config {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            endpoint: None,
            api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            model: model.into(),
        }
    }

    /// Config for `model` with an explicit API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the gRPC endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }
}

/// Build a connected [`Client`] (tonic channel + API-key interceptor) from
/// `cfg`.
///
/// With no endpoint override this is exactly [`Client::new`]: webpki roots
/// and the `generativelanguage.googleapis.com` TLS domain. A custom endpoint
/// keeps webpki roots but takes its TLS domain from the URL.
pub async fn client_from_config(
    cfg: &Config,
) -> Result<Client, Box<dyn std::error::Error + Send + Sync>> {
    let api_key = cfg
        .api_key
        .resolve()?
        .ok_or("Gemini gRPC requires an API key (ApiKeyLocation::None is not supported)")?;

    match &cfg.endpoint {
        None => Client::new(api_key).await,
        Some(endpoint) => {
            let endpoint = Endpoint::from_shared(endpoint.clone())?
                .tls_config(ClientTlsConfig::new().with_webpki_roots())?;
            let channel = endpoint.connect().await?;
            Ok(Client::from_parts(api_key, channel))
        }
    }
}

/// Send `request` over the unary `GenerateContent` RPC and return the
/// normalized response.
pub async fn complete(
    client: &Client,
    model: &str,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let request = crate::completion::create_grpc_request(model.to_string(), request)?;

    let mut grpc_client = client
        .grpc_client()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let response = grpc_client
        .generate_content(request)
        .await
        .map_err(crate::completion::rpc_error)?
        .into_inner();

    response.try_into()
}

/// Open a `StreamGenerateContent` streaming completion for `request`.
pub async fn open_stream(
    client: &Client,
    model: &str,
    request: CompletionRequest,
) -> Result<StreamingCompletionResponse, CompletionError> {
    crate::streaming::stream(client.clone(), model.to_string(), request).await
}

// ================================================================
// Embeddings
// ================================================================

/// Plain-data Gemini gRPC embeddings configuration.
///
/// A sibling of [`Config`]: the same channel-construction fields plus the
/// embedding model identifier and its optional `output_dimensionality`.
/// [`Self::client_config`] projects the connection half so the host runtime
/// can share its cached channel machinery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// gRPC endpoint URL (`None` uses [`DEFAULT_ENDPOINT`]).
    pub endpoint: Option<String>,
    /// Credential location.
    pub api_key: ApiKeyLocation,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested `output_dimensionality` (`None` sends none; the classic
    /// model defaults to 768).
    pub ndims: Option<usize>,
}

impl EmbeddingConfig {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            endpoint: None,
            api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            model: model.into(),
            ndims: None,
        }
    }

    /// Config for `model` with an explicit API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the gRPC endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Request `ndims`-sized embeddings.
    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }

    /// The channel-construction half of this config as a [`Config`], for
    /// sharing a host runtime's cached [`client_from_config`] machinery.
    pub fn client_config(&self) -> Config {
        Config {
            endpoint: self.endpoint.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
        }
    }
}

/// Embed `texts` with `model`, one `EmbedContent` RPC per document (the
/// gRPC embedding API is single-document). Gemini gRPC reports no
/// embedding usage. The first RPC failure aborts the batch.
pub async fn embed(
    client: &Client,
    model: &str,
    ndims: Option<usize>,
    texts: Vec<String>,
) -> Result<rig_core::embeddings::EmbeddingResponse, rig_core::embeddings::EmbeddingError> {
    let mut embeddings = Vec::with_capacity(texts.len());
    for doc in texts {
        embeddings.push(crate::embedding::embed_one(client, model, ndims, doc).await?);
    }
    Ok(rig_core::embeddings::EmbeddingResponse {
        embeddings,
        usage: rig_core::completion::Usage::new(),
    })
}

/// Embed caller-defined batches, returning one order-aligned
/// [`rig_core::OneOrMany`] group per input batch plus summed usage.
pub async fn embed_batches(
    client: &Client,
    model: &str,
    ndims: Option<usize>,
    texts: Vec<Vec<String>>,
) -> Result<
    (
        Vec<rig_core::OneOrMany<rig_core::embeddings::Embedding>>,
        rig_core::completion::Usage,
    ),
    rig_core::embeddings::EmbeddingError,
> {
    let counts: Vec<usize> = texts.iter().map(Vec::len).collect();
    let flat: Vec<String> = texts.into_iter().flatten().collect();
    let response = embed(client, model, ndims, flat).await?;
    let groups = rig_core::embeddings::batching::group_batches(&counts, response.embeddings)?;
    Ok((groups, response.usage))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_env_key_and_default_endpoint() {
        let cfg = Config::new("gemini-2.5-flash");
        assert_eq!(cfg.endpoint, None);
        assert_eq!(cfg.api_key, ApiKeyLocation::Env("GEMINI_API_KEY".into()));
        assert_eq!(cfg.model, "gemini-2.5-flash");
    }

    #[test]
    fn config_serde_round_trip() {
        let cfg = Config::new("gemini-2.0-flash")
            .with_api_key("k")
            .with_endpoint("https://example.test");
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }

    #[test]
    // Pinning const capability values is the point of this test.
    #[allow(clippy::assertions_on_constants)]
    fn descriptor_is_honest() {
        assert_eq!(DESCRIPTOR.name, "gemini-grpc");
        assert!(DESCRIPTOR.supports_tools);
        // create_grpc_request drops output_schema, so no native structured output.
        assert!(!DESCRIPTOR.supports_response_format);
        assert_eq!(DESCRIPTOR.max_embedding_documents, Some(100));
    }
}
