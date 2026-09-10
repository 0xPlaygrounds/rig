//! Default-backend conveniences: open a provider websocket session over the
//! bundled [`TungsteniteClient`] without naming a backend.
//!
//! rig-core deliberately names no websocket backend — every connect there takes
//! a `W: WebSocketClientExt` — for the same reason it names no HTTP transport.
//! These traits are implemented once, over the bundled backend, which is what
//! lets `client.responses_websocket("gpt-5.4")` resolve with nothing named at
//! the call site. They are the websocket twin of `rig-reqwest`'s
//! `DefaultTransportClient` / `DefaultTransportBuilder`; bring them into scope
//! with `use rig::prelude::*` or `use rig_tungstenite::prelude::*`.

use crate::TungsteniteClient;
use rig_core::completion::CompletionError;
use rig_core::http_client::HttpClientExt;
use rig_core::providers::openai::Client as OpenAIClient;
use rig_core::providers::openai::responses_api::websocket::{
    ResponsesWebSocketExt, ResponsesWebSocketSession, ResponsesWebSocketSessionBuilder,
};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Open a provider websocket session over the bundled backend.
pub trait DefaultWebSocketClient<H> {
    /// Open an OpenAI Responses websocket session for `model`, with default
    /// options, over the bundled backend.
    fn responses_websocket(
        &self,
        model: impl Into<String>,
    ) -> impl Future<Output = Result<ResponsesWebSocketSession<H>, CompletionError>> + Send
    where
        Self: Sync;
}

impl<H> DefaultWebSocketClient<H> for OpenAIClient<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn responses_websocket(
        &self,
        model: impl Into<String>,
    ) -> impl Future<Output = Result<ResponsesWebSocketSession<H>, CompletionError>> + Send
    where
        Self: Sync,
    {
        self.responses_websocket_with(model, &TungsteniteClient)
    }
}

/// `connect()` for a session builder with no backend named: substitutes the
/// bundled [`TungsteniteClient`].
///
/// rig-core's own `connect_with(..)` always takes a backend, so this trait is
/// what makes `client.responses_websocket_builder("gpt-5.4").event_timeout(..).connect()`
/// resolve.
pub trait DefaultWebSocketBuilder<H> {
    /// Open the session over the bundled backend.
    fn connect(
        self,
    ) -> impl Future<Output = Result<ResponsesWebSocketSession<H>, CompletionError>> + Send;
}

impl<H> DefaultWebSocketBuilder<H> for ResponsesWebSocketSessionBuilder<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn connect(self) -> Result<ResponsesWebSocketSession<H>, CompletionError> {
        self.connect_with(&TungsteniteClient).await
    }
}
