//! The Gemini gRPC transport: a tonic channel with the API-key interceptor.
//!
//! ```no_run
//! use rig_gemini_grpc::{GeminiGrpc, GeminiGrpcError};
//!
//! # async fn example() -> Result<(), GeminiGrpcError> {
//! let transport = GeminiGrpc::from_env().await?;
//! # let _ = transport;
//! # Ok(())
//! # }
//! ```

use std::fmt::Debug;
use tonic::metadata::errors::InvalidMetadataValue;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

use super::GenerativeServiceClient;
pub use crate::auth::ApiKeyInterceptor;
use crate::completion::GenerateContent;
use crate::embedding::Embeddings;
use rig_core::Model;
use rig_core::client::env::{self, EnvError};
use rig_core::wire::Secret;

const GEMINI_GRPC_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

/// The variable [`GeminiGrpc::from_env`] reads the API key from.
const GEMINI_API_KEY_ENV: &str = "GEMINI_API_KEY";

/// User agent identifier for API tracking
const RIG_GRPC_CLIENT_IDENTIFIER: &str = "rig-grpc/0.1.0";

/// A [`GeminiGrpc`] transport could not be constructed.
#[derive(Debug, thiserror::Error)]
pub enum GeminiGrpcError {
    /// The API key could not be read from the environment.
    #[error(transparent)]
    Env(#[from] EnvError),
    /// The API key cannot be sent as gRPC metadata, for example because it
    /// contains a control character.
    #[error("the Gemini API key is not a valid gRPC metadata value")]
    InvalidApiKey(#[source] InvalidMetadataValue),
    /// TLS setup or the connection to the Gemini endpoint failed.
    #[error("failed to connect to the Gemini gRPC endpoint")]
    Transport(#[from] tonic::transport::Error),
}

/// The transport every Gemini gRPC wire is sent through. Clones share one
/// channel.
#[derive(Clone)]
pub struct GeminiGrpc {
    interceptor: ApiKeyInterceptor,
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

impl GeminiGrpc {
    /// Connect to the Gemini gRPC endpoint with `api_key`.
    ///
    /// Returns [`GeminiGrpcError::InvalidApiKey`] when the key cannot be sent
    /// as gRPC metadata, before connecting, and [`GeminiGrpcError::Transport`]
    /// when TLS setup or the connection fails.
    pub async fn new(api_key: impl Into<Secret>) -> Result<Self, GeminiGrpcError> {
        let mut api_key = MetadataValue::<Ascii>::try_from(api_key.into().expose())
            .map_err(GeminiGrpcError::InvalidApiKey)?;
        api_key.set_sensitive(true);
        let interceptor = ApiKeyInterceptor {
            api_key,
            client_id: MetadataValue::from_static(RIG_GRPC_CLIENT_IDENTIFIER),
        };
        let channel = Endpoint::from_static(GEMINI_GRPC_ENDPOINT)
            .tls_config(
                ClientTlsConfig::new()
                    .with_webpki_roots()
                    .domain_name("generativelanguage.googleapis.com"),
            )?
            .connect()
            .await?;
        Ok(Self {
            interceptor,
            channel,
        })
    }

    /// Connect with the API key in `GEMINI_API_KEY`.
    ///
    /// Returns [`GeminiGrpcError::Env`] when the variable is unset or not
    /// Unicode, and otherwise the errors of [`Self::new`].
    pub async fn from_env() -> Result<Self, GeminiGrpcError> {
        Self::new(env::required(GEMINI_API_KEY_ENV)?).await
    }

    /// A service client that sends through this transport's channel.
    pub(crate) fn grpc_client(
        &self,
    ) -> GenerativeServiceClient<
        tonic::service::interceptor::InterceptedService<Channel, ApiKeyInterceptor>,
    > {
        GenerativeServiceClient::with_interceptor(self.channel.clone(), self.interceptor.clone())
    }
}

impl GeminiGrpc {
    /// The `GenerateContent` model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<GenerateContent, Self> {
        Model::new(GenerateContent::new(model), self.clone())
    }

    /// The embedding model for `model`, `dims` wide when set.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        dims: Option<usize>,
    ) -> Model<Embeddings, Self> {
        Model::new(Embeddings::new(model, dims), self.clone())
    }
}

#[cfg(test)]
mod tests;
