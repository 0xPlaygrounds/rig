//! The Gemini gRPC transport: the Gemini API's [`GeminiConfig`] on a tonic
//! channel, with the API key sent as request metadata.
//!
//! ```no_run
//! use rig_core::providers::gemini::GeminiConfig;
//! use rig_gemini_grpc::{GeminiGrpc, GeminiGrpcError};
//!
//! # async fn example() -> Result<(), GeminiGrpcError> {
//! let transport = GeminiGrpc::connect(&GeminiConfig::from_env()?).await?;
//! # let _ = transport;
//! # Ok(())
//! # }
//! ```

use std::fmt::Debug;
use std::time::Duration;
use tonic::metadata::errors::InvalidMetadataValue;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

use super::GenerativeServiceClient;
use crate::auth::ApiKeyInterceptor;
use crate::completion::GenerateContent;
use crate::embedding::Embeddings;
use rig_core::Model;

/// The largest reply a [`GeminiGrpc`] decodes: 256 MiB, far above any
/// generated image or long completion.
pub const MAX_REPLY_BYTES: usize = 256 * 1024 * 1024;
use rig_core::client::env::EnvError;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::wire::Secret;

/// The client identification every request carries.
const CLIENT_IDENTIFIER: &str = concat!("rig-gemini-grpc/", env!("CARGO_PKG_VERSION"));

/// How long [`GeminiGrpc::connect`] waits for each connection to open.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

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
    /// The base URL is not a URI, or TLS setup or the connection failed.
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
    /// Connect to `config`'s base URL with its API key, over TLS with the
    /// WebPKI roots for an `https` URL. Opening a connection, the first or a
    /// reconnection, fails after 30 seconds; build the channel yourself and
    /// pass it to [`Self::with_channel`] for other timeouts.
    ///
    /// Returns [`GeminiGrpcError::InvalidApiKey`] when the key cannot be sent
    /// as gRPC metadata, before connecting, and [`GeminiGrpcError::Transport`]
    /// when the base URL is not a URI or TLS setup or the connection fails.
    pub async fn connect(config: &GeminiConfig) -> Result<Self, GeminiGrpcError> {
        let interceptor = interceptor(&config.api_key)?;
        let mut endpoint =
            Endpoint::from_shared(config.base_url.clone())?.connect_timeout(CONNECT_TIMEOUT);
        if endpoint.uri().scheme_str() == Some("https") {
            endpoint = endpoint.tls_config(ClientTlsConfig::new().with_webpki_roots())?;
        }
        Ok(Self {
            interceptor,
            channel: endpoint.connect().await?,
        })
    }

    /// Send with `config`'s API key through `channel`, a channel the host
    /// built with its own endpoint, TLS and timeouts. `config`'s base URL is
    /// not read.
    ///
    /// Returns [`GeminiGrpcError::InvalidApiKey`] when the key cannot be sent
    /// as gRPC metadata.
    pub fn with_channel(config: &GeminiConfig, channel: Channel) -> Result<Self, GeminiGrpcError> {
        Ok(Self {
            interceptor: interceptor(&config.api_key)?,
            channel,
        })
    }

    /// Connect to the public endpoint with `api_key`. See [`Self::connect`].
    pub async fn new(api_key: impl Into<Secret>) -> Result<Self, GeminiGrpcError> {
        Self::connect(&GeminiConfig::new(api_key)).await
    }

    /// Connect to the public endpoint with the API key in `GEMINI_API_KEY`.
    ///
    /// Returns [`GeminiGrpcError::Env`] when the variable is unset or not
    /// Unicode, and otherwise the errors of [`Self::connect`].
    pub async fn from_env() -> Result<Self, GeminiGrpcError> {
        Self::connect(&GeminiConfig::from_env()?).await
    }

    /// A service client that sends through this transport's channel. It
    /// decodes replies up to [`MAX_REPLY_BYTES`]: tonic's default 4 MB
    /// would fail a generated image, and tonic reserves a reply's declared
    /// length before it arrives, so a capped size keeps a misbehaving
    /// endpoint from forcing a 4 GiB allocation.
    pub(crate) fn grpc_client(
        &self,
    ) -> GenerativeServiceClient<
        tonic::service::interceptor::InterceptedService<Channel, ApiKeyInterceptor>,
    > {
        GenerativeServiceClient::with_interceptor(self.channel.clone(), self.interceptor.clone())
            .max_decoding_message_size(MAX_REPLY_BYTES)
    }

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

/// The interceptor that sends `api_key`, marked sensitive so tonic never
/// logs it.
fn interceptor(api_key: &Secret) -> Result<ApiKeyInterceptor, GeminiGrpcError> {
    let mut api_key = MetadataValue::<Ascii>::try_from(api_key.expose())
        .map_err(GeminiGrpcError::InvalidApiKey)?;
    api_key.set_sensitive(true);
    Ok(ApiKeyInterceptor {
        api_key,
        client_id: MetadataValue::from_static(CLIENT_IDENTIFIER),
    })
}

#[cfg(test)]
mod tests;
