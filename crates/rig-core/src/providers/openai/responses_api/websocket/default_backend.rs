//! Provider websocket session constructors using the bundled backend.
//!
//! ```no_run
//! use rig_core::providers::openai::responses_api::websocket::DefaultWebSocketBuilder;
//! use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketSessionBuilder;
//!
//! async fn connect(builder: ResponsesWebSocketSessionBuilder)
//!     -> Result<(), rig_core::error::ProviderError>
//! {
//!     let session = builder.connect().await?;
//!     Ok(())
//! }
//! ```

use crate::driver::Bound;
use crate::error::ProviderError;
use crate::providers::openai::responses_api::websocket::{
    ResponsesWebSocketExt, ResponsesWebSocketSession, ResponsesWebSocketSessionBuilder,
};
use crate::providers::openai::responses_api::wire::Responses;
use crate::wasm_compat::{MaybeSend, MaybeSync};
use rig_tungstenite::TungsteniteClient;

/// Open a provider websocket session over the bundled backend.
pub trait DefaultWebSocketClient {
    /// Open an OpenAI Responses websocket session for this wire's model, with
    /// default options, over the bundled backend. Returns an error if connection
    /// setup fails.
    fn responses_websocket(
        &self,
    ) -> impl Future<Output = Result<ResponsesWebSocketSession, ProviderError>> + Send
    where
        Self: Sync;
}

impl<H> DefaultWebSocketClient for Bound<Responses, H>
where
    H: MaybeSend + MaybeSync,
{
    fn responses_websocket(
        &self,
    ) -> impl Future<Output = Result<ResponsesWebSocketSession, ProviderError>> + Send
    where
        Self: Sync,
    {
        self.responses_websocket_with(&TungsteniteClient)
    }
}

/// Connect a configured session using the bundled [`TungsteniteClient`].
pub trait DefaultWebSocketBuilder {
    /// Open the session over the bundled backend, returning an error if
    /// connection setup fails.
    fn connect(
        self,
    ) -> impl Future<Output = Result<ResponsesWebSocketSession, ProviderError>> + Send;
}

impl DefaultWebSocketBuilder for ResponsesWebSocketSessionBuilder {
    async fn connect(self) -> Result<ResponsesWebSocketSession, ProviderError> {
        self.connect_with(&TungsteniteClient).await
    }
}
