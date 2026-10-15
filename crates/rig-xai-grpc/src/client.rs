//! The xAI gRPC transport: a tonic channel to `api.x.ai` with the API-key
//! interceptor.
//!
//! ```no_run
//! use rig_xai_grpc::{XaiGrpc, XaiGrpcError};
//!
//! # async fn example() -> Result<(), XaiGrpcError> {
//! let transport = XaiGrpc::from_env().await?;
//! # let _ = transport;
//! # Ok(())
//! # }
//! ```

use std::fmt::Debug;
use tonic::metadata::errors::InvalidMetadataValue;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

use crate::ChatClient;
pub use crate::auth::ApiKeyInterceptor;
use crate::completion::GetCompletion;
use rig_core::Model;
use rig_core::client::env::{self, EnvError};
use rig_core::wire::Secret;

const XAI_GRPC_ENDPOINT: &str = "https://api.x.ai";

/// The variable [`XaiGrpc::from_env`] reads the API key from.
const XAI_API_KEY_ENV: &str = "XAI_API_KEY";

/// The largest reply message accepted, as xAI's own SDK sets it.
const MAX_MESSAGE_BYTES: usize = 20 * 1024 * 1024;

/// An [`XaiGrpc`] transport could not be constructed.
#[derive(Debug, thiserror::Error)]
pub enum XaiGrpcError {
    /// The API key could not be read from the environment.
    #[error(transparent)]
    Env(#[from] EnvError),
    /// The API key cannot be sent as gRPC metadata, for example because it
    /// contains a control character.
    #[error("the xAI API key is not a valid gRPC metadata value")]
    InvalidApiKey(#[source] InvalidMetadataValue),
    /// TLS setup or the connection to the xAI endpoint failed.
    #[error("failed to connect to the xAI gRPC endpoint")]
    Transport(#[from] tonic::transport::Error),
}

/// The transport the xAI gRPC wire is sent through. Clones share one
/// channel.
#[derive(Clone)]
pub struct XaiGrpc {
    interceptor: ApiKeyInterceptor,
    channel: Channel,
}

impl Debug for XaiGrpc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XaiGrpc")
            .field("api_key", &"******")
            .field("channel", &"Channel")
            .finish()
    }
}

impl XaiGrpc {
    /// Connect to the xAI gRPC endpoint with `api_key`.
    ///
    /// Returns [`XaiGrpcError::InvalidApiKey`] when the key cannot be sent
    /// as gRPC metadata, before connecting, and [`XaiGrpcError::Transport`]
    /// when TLS setup or the connection fails.
    pub async fn new(api_key: impl Into<Secret>) -> Result<Self, XaiGrpcError> {
        let bearer = format!("Bearer {}", api_key.into().expose());
        let mut authorization =
            MetadataValue::<Ascii>::try_from(bearer).map_err(XaiGrpcError::InvalidApiKey)?;
        authorization.set_sensitive(true);
        let channel = Endpoint::from_static(XAI_GRPC_ENDPOINT)
            .tls_config(
                ClientTlsConfig::new()
                    .with_webpki_roots()
                    .domain_name("api.x.ai"),
            )?
            .connect()
            .await?;
        Ok(Self {
            interceptor: ApiKeyInterceptor { authorization },
            channel,
        })
    }

    /// Connect with the API key in `XAI_API_KEY`.
    ///
    /// Returns [`XaiGrpcError::Env`] when the variable is unset or not
    /// Unicode, and otherwise the errors of [`Self::new`].
    pub async fn from_env() -> Result<Self, XaiGrpcError> {
        Self::new(env::required(XAI_API_KEY_ENV)?).await
    }

    /// A `Chat` service client that sends through this transport's channel.
    pub(crate) fn grpc_client(&self) -> ChatClient<InterceptedService<Channel, ApiKeyInterceptor>> {
        ChatClient::with_interceptor(self.channel.clone(), self.interceptor.clone())
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES)
    }

    /// The completion model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<GetCompletion, Self> {
        Model::new(GetCompletion::new(model), self.clone())
    }
}

#[cfg(test)]
mod tests;
