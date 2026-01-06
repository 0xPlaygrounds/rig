use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Provider, ProviderBuilder, ProviderClient,
};
use crate::http_client;
use std::fmt::Debug;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use super::proto::generative_service_client::GenerativeServiceClient;

// ================================================================
// Google Gemini gRPC Client
// ================================================================
const GEMINI_GRPC_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

/// User agent identifier for API tracking
const RIG_GRPC_CLIENT_IDENTIFIER: &str = "rig-grpc/0.28.0";

#[derive(Clone)]
pub struct GeminiGrpcExt {
    api_key: String,
    channel: Channel,
}

impl Debug for GeminiGrpcExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiGrpcExt")
            .field("api_key", &"******")
            .field("channel", &"Channel")
            .finish()
    }
}

#[derive(Debug, Default, Clone)]
pub struct GeminiGrpcBuilder;

pub struct GeminiGrpcApiKey(String);

impl<S> From<S> for GeminiGrpcApiKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

pub type Client = client::Client<GeminiGrpcExt, reqwest::Client>;
pub type ClientBuilder =
    client::ClientBuilder<GeminiGrpcBuilder, GeminiGrpcApiKey, reqwest::Client>;

impl ApiKey for GeminiGrpcApiKey {}

impl DebugExt for GeminiGrpcExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
        std::iter::once(("api_key", (&"******") as &dyn Debug))
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

impl GeminiGrpcExt {
    /// Create a gRPC client with the given API key
    pub async fn new(api_key: String) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let endpoint = Endpoint::from_static(GEMINI_GRPC_ENDPOINT)
            .tls_config(tonic::transport::ClientTlsConfig::new())?;

        let channel = endpoint.connect().await?;

        Ok(Self { api_key, channel })
    }

    /// Get a gRPC client with API key interceptor
    pub fn grpc_client(
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

impl Provider for GeminiGrpcExt {
    type Builder = GeminiGrpcBuilder;

    const VERIFY_PATH: &'static str = ""; // gRPC doesn't use HTTP paths

    fn build<H>(
        builder: &client::ClientBuilder<Self::Builder, GeminiGrpcApiKey, H>,
    ) -> http_client::Result<Self> {
        let api_key = builder.get_api_key().0.clone();

        // Use tokio's block_in_place to avoid creating a new runtime
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(Self::new(api_key))
                .map_err(|e| {
                    http_client::Error::Instance(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    )))
                })
        })
    }
}

impl<H> Capabilities<H> for GeminiGrpcExt {
    type Completion = Capable<super::completion::CompletionModel>;
    type Embeddings = Capable<super::embedding::EmbeddingModel>;
    type Transcription = crate::client::Nothing;

    #[cfg(feature = "image")]
    type ImageGeneration = crate::client::Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = crate::client::Nothing;
}

impl ProviderBuilder for GeminiGrpcBuilder {
    type Output = GeminiGrpcExt;
    type ApiKey = GeminiGrpcApiKey;

    const BASE_URL: &'static str = GEMINI_GRPC_ENDPOINT;
}

impl ProviderClient for Client {
    type Input = GeminiGrpcApiKey;

    /// Create a new Google Gemini gRPC client from the `GEMINI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        Self::new(api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}
