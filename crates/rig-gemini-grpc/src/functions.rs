//! Gemini gRPC as config + free functions (data-oriented face).
//!
//! gRPC is a non-HTTP transport: instead of a pure request-builder there is a
//! serde [`Config`] describing how to *build* the connected tonic channel and
//! API-key interceptor (never holding them), an async [`client_from_config`],
//! and free [`complete`]/[`open_stream`] functions the existing
//! [`crate::completion::CompletionModel`] trait impl is rewired through.

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
///
/// Extracted from the `CompletionModel::completion` trait impl, which is
/// rewired through this function; behavior is unchanged.
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
///
/// Returns the concrete [`StreamingCompletionResponse`]; the
/// `CompletionModel::stream` trait impl is rewired through this function.
pub async fn open_stream(
    client: &Client,
    model: &str,
    request: CompletionRequest,
) -> Result<StreamingCompletionResponse, CompletionError> {
    crate::streaming::stream(client.clone(), model.to_string(), request).await
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
