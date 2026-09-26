//! The Gemini gRPC transport: a tonic channel with the API-key interceptor.
//!
//! ```no_run
//! use rig_gemini_grpc::GeminiGrpc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let transport = GeminiGrpc::new("API_KEY").await?;
//! # let _ = transport;
//! # Ok(())
//! # }
//! ```

use std::fmt::Debug;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use super::GenerativeServiceClient;

const GEMINI_GRPC_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

/// User agent identifier for API tracking
const RIG_GRPC_CLIENT_IDENTIFIER: &str = "rig-grpc/0.1.0";

/// The transport every Gemini gRPC wire is sent through. Clones share one
/// channel.
#[derive(Clone)]
pub struct GeminiGrpc {
    api_key: String,
    channel: Channel,
}

impl Debug for GeminiGrpc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiGrpc")
            .field("api_key", &"******")
            .field("channel", &"Channel")
            .finish()
    }
}

/// Adds API-key and client-identification metadata to outgoing requests.
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

impl GeminiGrpc {
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
}

impl GeminiGrpc {
    /// Create a new Google Gemini gRPC client from the `GEMINI_API_KEY` environment variable.
    ///
    /// Returns environment, TLS, or connection errors.
    ///
    /// # Panics
    /// Panics outside a Tokio runtime or inside a current-thread runtime.
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = std::env::var("GEMINI_API_KEY")?;
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(Self::new(api_key))
        })
    }

    /// Connects using an explicit API key. Returns TLS or connection errors.
    ///
    /// # Panics
    /// Panics outside a Tokio runtime or inside a current-thread runtime.
    pub fn from_val(api_key: String) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(Self::new(api_key))
        })
    }
}
