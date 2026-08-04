//! Gemini gRPC as config + free functions — the crate's entry point.
//!
//! gRPC is a non-HTTP transport: instead of a pure request-builder there is a
//! serde [`Config`] describing how to *build* the connected tonic channel and
//! API-key interceptor (never holding them), an async [`client_from_config`]
//! producing the live [`Client`] handle, and free
//! [`complete`]/[`open_stream`]/[`embed`]/[`embed_batches`] functions taking
//! that handle.

use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::streaming::CompletionStream;
use tonic::transport::{ClientTlsConfig, Endpoint};

use crate::Client;

pub use rig_core::providers::gemini_grpc::{
    Config, ConnectionConfig, DEFAULT_ENDPOINT, DESCRIPTOR, EmbeddingConfig,
};

/// Build a connected [`Client`] (tonic channel + API-key interceptor) from
/// `cfg`.
///
/// With no endpoint override this is exactly [`Client::new`]: webpki roots
/// and the `generativelanguage.googleapis.com` TLS domain. A custom endpoint
/// keeps webpki roots but takes its TLS domain from the URL.
pub async fn client_from_config(
    cfg: &Config,
) -> Result<Client, Box<dyn std::error::Error + Send + Sync>> {
    client_from_connection(&cfg.connection).await
}

/// Build a connected client from reusable channel-construction data.
pub async fn client_from_connection(
    connection: &ConnectionConfig,
) -> Result<Client, Box<dyn std::error::Error + Send + Sync>> {
    let api_key = connection
        .api_key
        .resolve()?
        .ok_or("Gemini gRPC requires an API key (ApiKeyLocation::None is not supported)")?;

    match &connection.endpoint {
        None => Client::new(api_key).await,
        Some(endpoint) => {
            let endpoint = Endpoint::from_shared(endpoint.clone())?
                .tls_config(ClientTlsConfig::new().with_webpki_roots())?;
            let channel = endpoint.connect().await?;
            Ok(Client::from_parts(connection.clone(), api_key, channel))
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
) -> Result<CompletionStream, CompletionError> {
    crate::streaming::stream(client.clone(), model.to_string(), request).await
}

// ================================================================
// Embeddings
// ================================================================

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
    use rig_core::providers::ApiKeyLocation;

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
