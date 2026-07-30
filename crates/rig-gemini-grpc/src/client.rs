//! The live gRPC handle.
//!
//! gRPC is a non-HTTP transport, so a connected tonic [`Channel`] cannot be
//! plain configuration. [`Client`] is that live handle; build it from data
//! with [`crate::functions::client_from_config`] (or directly with
//! [`Client::new`]) and pass it to the free functions in
//! [`crate::functions`].

use std::fmt::Debug;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use super::GenerativeServiceClient;

// ================================================================
// Google Gemini gRPC Client
// ================================================================
const GEMINI_GRPC_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

/// User agent identifier for API tracking
const RIG_GRPC_CLIENT_IDENTIFIER: &str = "rig-grpc/0.1.0";

#[derive(Clone)]
pub struct Client {
    api_key: String,
    channel: Channel,
}

impl Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("api_key", &"******")
            .field("channel", &"Channel")
            .finish()
    }
}

// Interceptor to add API key and client identification to metadata
#[derive(Clone)]
pub struct ApiKeyInterceptor {
    api_key: MetadataValue<tonic::metadata::Ascii>,
    client_id: MetadataValue<tonic::metadata::Ascii>,
}

impl Interceptor for ApiKeyInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert("x-goog-api-key", self.api_key.clone());
        request
            .metadata_mut()
            .insert("x-goog-api-client", self.client_id.clone());
        Ok(request)
    }
}

impl Client {
    /// Create a gRPC client with the given API key
    pub async fn new(
        api_key: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = api_key.into();
        let endpoint = Endpoint::from_static(GEMINI_GRPC_ENDPOINT).tls_config(
            tonic::transport::ClientTlsConfig::new()
                .with_webpki_roots()
                .domain_name("generativelanguage.googleapis.com"),
        )?;

        let channel = endpoint.connect().await?;

        Ok(Self { api_key, channel })
    }

    /// Assemble a client from an already-resolved API key and connected
    /// channel (used by [`crate::functions::client_from_config`]).
    pub(crate) fn from_parts(api_key: String, channel: Channel) -> Self {
        Self { api_key, channel }
    }

    /// Get a gRPC client with API key interceptor
    pub(crate) fn grpc_client(
        &self,
    ) -> Result<
        GenerativeServiceClient<
            tonic::service::interceptor::InterceptedService<Channel, ApiKeyInterceptor>,
        >,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let api_key = MetadataValue::try_from(&self.api_key)?;
        let client_id = MetadataValue::try_from(RIG_GRPC_CLIENT_IDENTIFIER)?;
        let interceptor = ApiKeyInterceptor { api_key, client_id };

        Ok(GenerativeServiceClient::with_interceptor(
            self.channel.clone(),
            interceptor,
        ))
    }

    /// Connect using the `GEMINI_API_KEY` environment variable.
    ///
    /// A shortcut for the zero-configuration case; prefer
    /// [`crate::functions::client_from_config`] when you already hold a
    /// [`Config`](crate::functions::Config).
    pub async fn from_env() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new(std::env::var("GEMINI_API_KEY")?).await
    }
}
