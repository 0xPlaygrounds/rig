//! WebSocket session support for the OpenAI Responses API.
//!
//! This module implements OpenAI's `/v1/responses` WebSocket mode as a stateful,
//! sequential session. Each connection supports a single in-flight response at a
//! time, which matches OpenAI's current protocol constraints.

use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::providers::openai::responses_api::streaming::{
    ItemChunk, ResponseChunk, ResponseChunkKind, StreamingCompletionChunk,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{self, Message, client::IntoClientRequest},
};
use tracing::Level;
use url::Url;

use super::{CompletionResponse, ResponseError, ResponseStatus, ResponsesCompletionModel};

type OpenAIWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Options for a `response.create` message sent over OpenAI WebSocket mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesWebSocketCreateOptions {
    /// When set to `false`, OpenAI prepares request state without generating a model output.
    ///
    /// This is the "warmup" mode described in the OpenAI WebSocket mode guide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<bool>,
}

impl ResponsesWebSocketCreateOptions {
    /// Creates warmup options equivalent to `generate: false`.
    #[must_use]
    pub fn warmup() -> Self {
        Self {
            generate: Some(false),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ResponsesWebSocketClientEvent {
    #[serde(rename = "type")]
    kind: ResponsesWebSocketClientEventKind,
    #[serde(flatten)]
    request: super::CompletionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
enum ResponsesWebSocketClientEventKind {
    #[serde(rename = "response.create")]
    ResponseCreate,
}

/// A protocol error event emitted by OpenAI WebSocket mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSocketErrorEvent {
    /// The event type.
    #[serde(rename = "type")]
    pub kind: ResponsesWebSocketErrorEventKind,
    /// The provider error payload.
    pub error: ResponsesWebSocketErrorPayload,
}

impl std::fmt::Display for ResponsesWebSocketErrorEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

/// The event kind for an OpenAI WebSocket protocol error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponsesWebSocketErrorEventKind {
    #[serde(rename = "error")]
    Error,
}

/// The payload carried by an OpenAI WebSocket protocol error event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesWebSocketErrorPayload {
    /// Provider-specific error code when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable error message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Any extra fields supplied by the provider.
    #[serde(flatten, default)]
    pub extra: Map<String, Value>,
}

impl std::fmt::Display for ResponsesWebSocketErrorPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.code, &self.message) {
            (Some(code), Some(message)) => write!(f, "{code}: {message}"),
            (None, Some(message)) => f.write_str(message),
            (Some(code), None) => f.write_str(code),
            (None, None) => f.write_str("OpenAI websocket error"),
        }
    }
}

/// The optional `response.done` event emitted by OpenAI WebSocket mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSocketDoneEvent {
    /// The event type.
    #[serde(rename = "type")]
    pub kind: ResponsesWebSocketDoneEventKind,
    /// The provider payload for the finished response.
    pub response: Value,
}

impl ResponsesWebSocketDoneEvent {
    /// Returns the response ID if the payload includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        self.response.get("id").and_then(Value::as_str)
    }

    fn status(&self) -> Option<ResponseStatus> {
        self.response
            .get("status")
            .cloned()
            .and_then(|status| serde_json::from_value(status).ok())
    }

    fn as_completion_response(&self) -> Option<CompletionResponse> {
        serde_json::from_value(self.response.clone()).ok()
    }
}

/// The event kind for the terminal websocket event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponsesWebSocketDoneEventKind {
    #[serde(rename = "response.done")]
    ResponseDone,
}

/// A server event emitted by OpenAI WebSocket mode.
#[derive(Debug, Clone)]
pub enum ResponsesWebSocketEvent {
    /// A response lifecycle event such as `response.created` or `response.completed`.
    Response(Box<ResponseChunk>),
    /// A streaming item/delta event such as `response.output_text.delta`.
    Item(ItemChunk),
    /// A protocol-level websocket error event.
    Error(ResponsesWebSocketErrorEvent),
    /// An optional `response.done` event emitted by OpenAI over WebSockets.
    Done(ResponsesWebSocketDoneEvent),
}

impl ResponsesWebSocketEvent {
    /// Returns the response ID when the event includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        match self {
            Self::Response(chunk) => Some(&chunk.response.id),
            Self::Done(done) => done.response_id(),
            Self::Item(_) | Self::Error(_) => None,
        }
    }

    /// Returns `true` when this event ends the current in-flight websocket turn.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Response(chunk) => matches!(
                chunk.kind,
                ResponseChunkKind::ResponseCompleted
                    | ResponseChunkKind::ResponseFailed
                    | ResponseChunkKind::ResponseIncomplete
            ),
            Self::Error(_) | Self::Done(_) => true,
            Self::Item(_) => false,
        }
    }
}

/// A builder for an OpenAI Responses WebSocket session.
///
/// The default builder applies a 30 second connection timeout and leaves the
/// per-event timeout disabled.
pub struct ResponsesWebSocketSessionBuilder<H = reqwest::Client> {
    model: ResponsesCompletionModel<H>,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
}

impl<H> ResponsesWebSocketSessionBuilder<H> {
    pub(crate) fn new(model: ResponsesCompletionModel<H>) -> Self {
        Self {
            model,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
        }
    }

    /// Sets the timeout for establishing the websocket connection.
    #[must_use]
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Disables the websocket connection timeout.
    #[must_use]
    pub fn without_connect_timeout(mut self) -> Self {
        self.connect_timeout = None;
        self
    }

    /// Sets the timeout for waiting on the next websocket event.
    #[must_use]
    pub fn event_timeout(mut self, timeout: Duration) -> Self {
        self.event_timeout = Some(timeout);
        self
    }

    /// Disables the websocket event timeout.
    #[must_use]
    pub fn without_event_timeout(mut self) -> Self {
        self.event_timeout = None;
        self
    }
}

impl<H> ResponsesWebSocketSessionBuilder<H>
where
    H: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    /// Opens the websocket session using the configured builder options.
    pub async fn connect(self) -> Result<ResponsesWebSocketSession<H>, CompletionError> {
        ResponsesWebSocketSession::connect_with_timeouts(
            self.model,
            self.connect_timeout,
            self.event_timeout,
        )
        .await
    }
}

/// A stateful OpenAI Responses WebSocket session.
///
/// This session keeps track of the most recent successful `response.id` so later
/// turns can automatically chain via `previous_response_id` unless the request
/// explicitly sets a different one.
///
/// Call [`ResponsesWebSocketSession::close`] when you are finished with the
/// session so the websocket can complete a close handshake cleanly.
pub struct ResponsesWebSocketSession<H = reqwest::Client> {
    model: ResponsesCompletionModel<H>,
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    socket: OpenAIWebSocket,
    in_flight: bool,
    event_timeout: Option<Duration>,
    closed: bool,
    failed: bool,
}

impl<H> ResponsesWebSocketSession<H>
where
    H: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    async fn connect_with_timeouts(
        model: ResponsesCompletionModel<H>,
        connect_timeout: Option<Duration>,
        event_timeout: Option<Duration>,
    ) -> Result<Self, CompletionError> {
        let url = websocket_url(model.client.base_url())?;
        let request = websocket_request(&url, model.client.headers())?;
        let socket = connect_websocket(request, connect_timeout).await?;

        Ok(Self {
            model,
            previous_response_id: None,
            pending_done_response_id: None,
            socket,
            in_flight: false,
            event_timeout,
            closed: false,
            failed: false,
        })
    }

    /// Returns the most recent successful `response.id` tracked by this session.
    #[must_use]
    pub fn previous_response_id(&self) -> Option<&str> {
        self.previous_response_id.as_deref()
    }

    /// Clears the cached `previous_response_id` so the next turn starts a fresh chain.
    pub fn clear_previous_response_id(&mut self) {
        self.previous_response_id = None;
    }

    /// Sends a `response.create` event for a Rig completion request.
    pub async fn send(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::default(),
        )
        .await
    }

    /// Sends a `response.create` event with explicit websocket-mode options.
    pub async fn send_with_options(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
        options: ResponsesWebSocketCreateOptions,
    ) -> Result<(), CompletionError> {
        self.ensure_open()?;

        if self.in_flight {
            return Err(CompletionError::ProviderError(
                "An OpenAI websocket response is already in flight on this session".to_string(),
            ));
        }

        let payload = ResponsesWebSocketClientEvent {
            kind: ResponsesWebSocketClientEventKind::ResponseCreate,
            request: self.prepare_request(completion_request)?,
            generate: options.generate,
        };

        if tracing::enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI websocket request: {}",
                serde_json::to_string_pretty(&payload)?
            );
        }

        let payload = serde_json::to_string(&payload)?;

        if let Err(error) = self.socket.send(Message::text(payload)).await {
            return Err(self.fail_session(websocket_provider_error(error)));
        }
        self.in_flight = true;

        Ok(())
    }

    /// Reads the next server event for the current in-flight turn.
    pub async fn next_event(&mut self) -> Result<ResponsesWebSocketEvent, CompletionError> {
        self.ensure_open()?;

        if !self.in_flight {
            return Err(CompletionError::ProviderError(
                "No OpenAI websocket response is currently in flight on this session".to_string(),
            ));
        }

        loop {
            let message = match self.read_next_message().await {
                Ok(message) => message,
                Err(error) => return Err(error),
            };

            let Some(message) = message else {
                self.mark_closed();
                return Err(CompletionError::ProviderError(
                    "The OpenAI websocket connection closed before the turn finished".to_string(),
                ));
            };

            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(self.fail_session(websocket_provider_error(error))),
            };
            let payload = match websocket_message_to_text(message) {
                Ok(Some(payload)) => payload,
                Ok(None) => continue,
                Err(error) => return Err(self.fail_session(error)),
            };
            let event = match parse_server_event(&payload) {
                Ok(Some(event)) => event,
                Ok(None) => continue,
                Err(error) => return Err(self.fail_session(error)),
            };
            if let ResponsesWebSocketEvent::Done(done) = &event {
                // OpenAI may emit `response.done` after the turn has already ended at
                // `response.completed`. Ignore that trailing event on the next turn.
                if self.pending_done_response_id.as_deref() == done.response_id() {
                    self.pending_done_response_id = None;
                    continue;
                }
            }
            self.update_state_for_event(&event);
            return Ok(event);
        }
    }

    /// Sends a warmup turn (`generate: false`) and returns the resulting response ID.
    pub async fn warmup(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<String, CompletionError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::warmup(),
        )
        .await?;
        let response = self.wait_for_completed_response().await?;
        Ok(response.id)
    }

    /// Sends a completion turn and collects the final OpenAI response.
    pub async fn completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        self.send(completion_request).await?;
        let response = self.wait_for_completed_response().await?;
        response.try_into()
    }

    /// Closes the websocket connection.
    ///
    /// Call this when you are finished with the session so the websocket can
    /// terminate with a clean close handshake.
    pub async fn close(&mut self) -> Result<(), CompletionError> {
        if self.closed {
            return Ok(());
        }

        let result = self
            .socket
            .close(None)
            .await
            .map_err(websocket_provider_error);
        self.mark_closed();
        result
    }

    fn prepare_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<super::CompletionRequest, CompletionError> {
        let mut request = self.model.create_completion_request(completion_request)?;

        // WebSocket mode is always event-driven, so these HTTP/SSE-specific flags
        // are ignored by the provider and only add noise to the payload.
        request.stream = None;
        request.additional_parameters.background = None;

        if request.additional_parameters.previous_response_id.is_none() {
            request.additional_parameters.previous_response_id = self.previous_response_id.clone();
        }

        Ok(request)
    }

    async fn wait_for_completed_response(&mut self) -> Result<CompletionResponse, CompletionError> {
        loop {
            match self.next_event().await? {
                ResponsesWebSocketEvent::Response(chunk) => {
                    if matches!(
                        chunk.kind,
                        ResponseChunkKind::ResponseCompleted
                            | ResponseChunkKind::ResponseFailed
                            | ResponseChunkKind::ResponseIncomplete
                    ) {
                        return terminal_response_result(chunk.response);
                    }
                }
                ResponsesWebSocketEvent::Done(done) => {
                    if let Some(response) = done.as_completion_response() {
                        return terminal_response_result(response);
                    }

                    let message = if let Some(response_id) = done.response_id() {
                        format!(
                            "OpenAI websocket turn ended with response.done before a terminal response body was available (response_id={response_id})"
                        )
                    } else {
                        "OpenAI websocket turn ended with response.done before a terminal response body was available"
                            .to_string()
                    };

                    return Err(CompletionError::ProviderError(message));
                }
                ResponsesWebSocketEvent::Error(error) => {
                    return Err(CompletionError::ProviderError(error.to_string()));
                }
                ResponsesWebSocketEvent::Item(_) => {}
            }
        }
    }

    fn update_state_for_event(&mut self, event: &ResponsesWebSocketEvent) {
        match event {
            ResponsesWebSocketEvent::Response(chunk) => match chunk.kind {
                ResponseChunkKind::ResponseCompleted => {
                    let response_id = chunk.response.id.clone();
                    self.previous_response_id = Some(response_id.clone());
                    self.pending_done_response_id = Some(response_id);
                    self.in_flight = false;
                }
                ResponseChunkKind::ResponseFailed | ResponseChunkKind::ResponseIncomplete => {
                    self.pending_done_response_id = Some(chunk.response.id.clone());
                    self.previous_response_id = None;
                    self.in_flight = false;
                }
                ResponseChunkKind::ResponseCreated | ResponseChunkKind::ResponseInProgress => {}
            },
            ResponsesWebSocketEvent::Done(done) => {
                match done.status() {
                    Some(ResponseStatus::Completed) => {
                        if let Some(response_id) = done.response_id() {
                            self.previous_response_id = Some(response_id.to_string());
                        }
                    }
                    Some(ResponseStatus::Failed)
                    | Some(ResponseStatus::Incomplete)
                    | Some(ResponseStatus::Cancelled) => {
                        self.previous_response_id = None;
                    }
                    Some(ResponseStatus::InProgress | ResponseStatus::Queued) | None => {}
                }
                self.pending_done_response_id = None;
                self.in_flight = false;
            }
            ResponsesWebSocketEvent::Error(_) => {
                self.previous_response_id = None;
                self.pending_done_response_id = None;
                self.in_flight = false;
            }
            ResponsesWebSocketEvent::Item(_) => {}
        }
    }

    fn abort_turn(&mut self) {
        self.previous_response_id = None;
        self.pending_done_response_id = None;
        self.in_flight = false;
    }

    fn mark_closed(&mut self) {
        self.abort_turn();
        self.closed = true;
        self.failed = false;
    }

    fn mark_failed(&mut self) {
        self.abort_turn();
        self.failed = true;
    }

    fn ensure_open(&self) -> Result<(), CompletionError> {
        if self.closed || self.failed {
            return Err(CompletionError::ProviderError(
                "The OpenAI websocket session is closed".to_string(),
            ));
        }

        Ok(())
    }

    fn fail_session(&mut self, error: CompletionError) -> CompletionError {
        self.mark_failed();
        error
    }

    async fn read_next_message(
        &mut self,
    ) -> Result<Option<Result<Message, tungstenite::Error>>, CompletionError> {
        if let Some(timeout_duration) = self.event_timeout {
            match tokio::time::timeout(timeout_duration, self.socket.next()).await {
                Ok(message) => Ok(message),
                Err(_) => Err(self.fail_session(event_timeout_error(timeout_duration))),
            }
        } else {
            Ok(self.socket.next().await)
        }
    }
}

impl<H> Drop for ResponsesWebSocketSession<H> {
    fn drop(&mut self) {
        if !self.closed {
            tracing::warn!(
                target: "rig::completions",
                in_flight = self.in_flight,
                "Dropping an OpenAI websocket session without calling close(); the connection will end without a close handshake"
            );
        }
    }
}

fn terminal_response_result(
    response: CompletionResponse,
) -> Result<CompletionResponse, CompletionError> {
    match response.status {
        ResponseStatus::Completed => Ok(response),
        ResponseStatus::Failed => Err(CompletionError::ProviderError(response_error_message(
            response.error.as_ref(),
            "failed response",
        ))),
        ResponseStatus::Incomplete => {
            let reason = response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str())
                .unwrap_or("unknown reason");
            Err(CompletionError::ProviderError(format!(
                "OpenAI websocket response was incomplete: {reason}"
            )))
        }
        status => Err(CompletionError::ProviderError(format!(
            "OpenAI websocket response ended with status {:?}",
            status
        ))),
    }
}

fn response_error_message(error: Option<&ResponseError>, fallback: &str) -> String {
    if let Some(error) = error {
        if error.code.is_empty() {
            error.message.clone()
        } else {
            format!("{}: {}", error.code, error.message)
        }
    } else {
        format!("OpenAI websocket returned a {fallback}")
    }
}

fn is_known_streaming_event(kind: &str) -> bool {
    matches!(
        kind,
        "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.delta"
            | "response.output_text.done"
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
    )
}

fn parse_server_event(payload: &str) -> Result<Option<ResponsesWebSocketEvent>, CompletionError> {
    #[derive(Deserialize)]
    struct EventType {
        #[serde(rename = "type")]
        kind: String,
    }

    let event_type = serde_json::from_str::<EventType>(payload)?;
    match event_type.kind.as_str() {
        "error" => serde_json::from_str(payload)
            .map(|e| Some(ResponsesWebSocketEvent::Error(e)))
            .map_err(CompletionError::from),
        "response.done" => serde_json::from_str(payload)
            .map(|d| Some(ResponsesWebSocketEvent::Done(d)))
            .map_err(CompletionError::from),
        kind if is_known_streaming_event(kind) => match serde_json::from_str(payload)? {
            StreamingCompletionChunk::Response(response) => {
                Ok(Some(ResponsesWebSocketEvent::Response(response)))
            }
            StreamingCompletionChunk::Delta(item) => Ok(Some(ResponsesWebSocketEvent::Item(item))),
        },
        _ => {
            tracing::debug!(
                target: "rig::completions",
                event_type = event_type.kind.as_str(),
                "Skipping unrecognised OpenAI websocket event"
            );
            Ok(None)
        }
    }
}

fn websocket_message_to_text(message: Message) -> Result<Option<String>, CompletionError> {
    match message {
        Message::Text(text) => Ok(Some(text.to_string())),
        Message::Binary(bytes) => String::from_utf8(bytes.to_vec())
            .map(Some)
            .map_err(|error| CompletionError::ResponseError(error.to_string())),
        Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => Ok(None),
        Message::Close(frame) => {
            let reason = frame
                .map(|frame| frame.reason.to_string())
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "without a close reason".to_string());
            Err(CompletionError::ProviderError(format!(
                "The OpenAI websocket connection closed {reason}"
            )))
        }
    }
}

fn websocket_url(base_url: &str) -> Result<String, CompletionError> {
    let mut url = Url::parse(base_url)?;
    match url.scheme() {
        "https" => {
            url.set_scheme("wss").map_err(|_| {
                CompletionError::ProviderError("Failed to convert https URL to wss".to_string())
            })?;
        }
        "http" => {
            url.set_scheme("ws").map_err(|_| {
                CompletionError::ProviderError("Failed to convert http URL to ws".to_string())
            })?;
        }
        scheme => {
            return Err(CompletionError::ProviderError(format!(
                "Unsupported base URL scheme for OpenAI websocket mode: {scheme}"
            )));
        }
    }

    let path = format!("{}/responses", url.path().trim_end_matches('/'));
    url.set_path(&path);
    Ok(url.to_string())
}

fn websocket_request(
    url: &str,
    headers: &http::HeaderMap,
) -> Result<http::Request<()>, CompletionError> {
    let mut request = url.into_client_request().map_err(|error| {
        CompletionError::ProviderError(format!("Failed to build OpenAI websocket request: {error}"))
    })?;

    for (name, value) in headers {
        request.headers_mut().insert(name, value.clone());
    }

    Ok(request)
}

async fn connect_websocket(
    request: http::Request<()>,
    connect_timeout: Option<Duration>,
) -> Result<OpenAIWebSocket, CompletionError> {
    if let Some(timeout_duration) = connect_timeout {
        match tokio::time::timeout(timeout_duration, connect_async(request)).await {
            Ok(result) => result
                .map(|(socket, _)| socket)
                .map_err(websocket_provider_error),
            Err(_) => Err(connect_timeout_error(timeout_duration)),
        }
    } else {
        connect_async(request)
            .await
            .map(|(socket, _)| socket)
            .map_err(websocket_provider_error)
    }
}

fn connect_timeout_error(timeout: Duration) -> CompletionError {
    CompletionError::ProviderError(format!(
        "Timed out connecting to the OpenAI websocket after {timeout:?}"
    ))
}

fn event_timeout_error(timeout: Duration) -> CompletionError {
    CompletionError::ProviderError(format!(
        "Timed out waiting for the next OpenAI websocket event after {timeout:?}"
    ))
}

fn websocket_provider_error(error: tungstenite::Error) -> CompletionError {
    CompletionError::ProviderError(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        ResponsesWebSocketCreateOptions, ResponsesWebSocketDoneEvent, ResponsesWebSocketEvent,
        parse_server_event, terminal_response_result, websocket_url,
    };
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::providers::openai::responses_api::{
        CompletionResponse, ResponseObject, ResponseStatus, ResponsesUsage,
    };
    use futures::{SinkExt, StreamExt};
    use serde_json::json;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::time::sleep;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    fn sample_response(status: ResponseStatus) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            created_at: 0,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-5.4".to_string(),
            usage: Some(ResponsesUsage {
                input_tokens: 1,
                input_tokens_details: None,
                output_tokens: 2,
                output_tokens_details:
                    crate::providers::openai::responses_api::OutputTokensDetails {
                        reasoning_tokens: 0,
                    },
                total_tokens: 3,
            }),
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: Default::default(),
        }
    }

    #[test]
    fn warmup_options_serialize_generate_false() {
        let options = ResponsesWebSocketCreateOptions::warmup();
        let json = serde_json::to_value(options).expect("options should serialize");

        assert_eq!(json, json!({ "generate": false }));
    }

    #[test]
    fn websocket_url_converts_https_to_wss() {
        let url = websocket_url("https://api.openai.com/v1").expect("url should convert");
        assert_eq!(url, "wss://api.openai.com/v1/responses");
    }

    #[test]
    fn parse_done_event_exposes_response_id() {
        let payload = json!({
            "type": "response.done",
            "response": {
                "id": "resp_done_1",
                "status": "completed"
            }
        });

        let event = parse_server_event(&payload.to_string())
            .expect("done event should deserialize")
            .expect("done event should not be skipped");

        assert!(matches!(
            event,
            ResponsesWebSocketEvent::Done(ResponsesWebSocketDoneEvent { .. })
        ));
        assert_eq!(event.response_id(), Some("resp_done_1"));
        assert!(event.is_terminal());
    }

    #[test]
    fn parse_response_completed_event_is_terminal() {
        let payload = json!({
            "type": "response.completed",
            "sequence_number": 12,
            "response": {
                "id": "resp_completed_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.4",
                "usage": null,
                "output": [],
                "tools": []
            }
        });

        let event = parse_server_event(&payload.to_string())
            .expect("response event should deserialize")
            .expect("response event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Response(_)));
        assert!(event.is_terminal());
        assert_eq!(event.response_id(), Some("resp_completed_1"));
    }

    #[test]
    fn parse_live_output_item_added_event() {
        let payload = json!({
            "type": "response.output_item.added",
            "item": {
                "id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant"
            },
            "output_index": 0,
            "sequence_number": 2
        });

        let event = parse_server_event(&payload.to_string())
            .expect("output item event should parse")
            .expect("output item event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    }

    #[test]
    fn parse_live_content_part_added_event() {
        let payload = json!({
            "type": "response.content_part.added",
            "content_index": 0,
            "item_id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
            "output_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": ""
            },
            "sequence_number": 3
        });

        let event = parse_server_event(&payload.to_string())
            .expect("content part event should parse")
            .expect("content part event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    }

    #[test]
    fn parse_live_output_text_delta_event() {
        let payload = json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "Web",
            "item_id": "msg_023af0f0a91bc2a90069ae788612e881958345bb156915ba29",
            "logprobs": [],
            "obfuscation": "2YYErYq7jkqqM",
            "output_index": 0,
            "sequence_number": 4
        });

        let event = parse_server_event(&payload.to_string())
            .expect("output text delta event should parse")
            .expect("output text delta event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    }

    #[test]
    fn terminal_response_requires_completed_status() {
        let completed = terminal_response_result(sample_response(ResponseStatus::Completed))
            .expect("completed response should succeed");
        assert_eq!(completed.id, "resp_123");

        let failed = terminal_response_result(sample_response(ResponseStatus::Failed))
            .expect_err("failed response should error");
        assert!(failed.to_string().contains("failed response"));
    }

    #[tokio::test]
    async fn malformed_known_event_rejects_reuse_and_allows_close() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed"
                    })
                    .to_string(),
                ))
                .await
                .expect("malformed known event should send");

            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");

        let error = session
            .next_event()
            .await
            .expect_err("malformed known event should fail");
        assert!(
            error.to_string().contains("StreamingCompletionChunk"),
            "expected strict decode failure, got {error}"
        );

        let closed = session
            .send(model.completion_request("retry").build())
            .await
            .expect_err("session should close after fatal parse error");
        assert!(
            closed.to_string().contains("session is closed"),
            "expected closed-session error, got {closed}"
        );

        session
            .close()
            .await
            .expect("explicit close after fatal parse error should succeed");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn event_timeout_rejects_reuse_and_allows_close() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            sleep(Duration::from_millis(60)).await;
            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket_builder("gpt-4o")
            .event_timeout(Duration::from_millis(20))
            .connect()
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");

        let error = session
            .next_event()
            .await
            .expect_err("next_event should time out");
        assert!(
            error
                .to_string()
                .contains("Timed out waiting for the next OpenAI websocket event"),
            "expected timeout error, got {error}"
        );

        let closed = session
            .send(model.completion_request("retry").build())
            .await
            .expect_err("timed-out session should close");
        assert!(
            closed.to_string().contains("session is closed"),
            "expected closed-session error, got {closed}"
        );

        session
            .close()
            .await
            .expect("explicit close after timeout should succeed");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn late_response_done_is_ignored_on_next_turn() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            for (index, response_id) in ["resp_1", "resp_2"].iter().enumerate() {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create payload, got {payload}"
                );

                let response = sample_response(ResponseStatus::Completed);
                let response = serde_json::to_value(CompletionResponse {
                    id: (*response_id).to_string(),
                    ..response
                })
                .expect("response should serialize");

                socket
                    .send(Message::text(
                        json!({
                            "type": "response.completed",
                            "sequence_number": (index * 2) + 1,
                            "response": response,
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("completed event should send");
                socket
                    .send(Message::text(
                        json!({
                            "type": "response.done",
                            "response": {
                                "id": response_id,
                                "status": "completed",
                            },
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("done event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn clearing_previous_response_id_does_not_disable_late_done_filter() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            for response_id in ["resp_1", "resp_2"] {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create payload, got {payload}"
                );

                let response = sample_response(ResponseStatus::Completed);
                let response = serde_json::to_value(CompletionResponse {
                    id: response_id.to_string(),
                    ..response
                })
                .expect("response should serialize");

                socket
                    .send(Message::text(
                        json!({
                            "type": "response.completed",
                            "sequence_number": 1,
                            "response": response,
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("completed event should send");
                socket
                    .send(Message::text(
                        json!({
                            "type": "response.done",
                            "response": {
                                "id": response_id,
                                "status": "completed",
                            },
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("done event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");

        session.clear_previous_response_id();
        assert_eq!(session.previous_response_id(), None);

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn failed_turn_keeps_late_done_out_of_next_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let first_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = first_request
                .into_text()
                .expect("failed request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            let failed_response = serde_json::to_value(CompletionResponse {
                id: "resp_failed".to_string(),
                status: ResponseStatus::Failed,
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("failed response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.failed",
                        "sequence_number": 1,
                        "response": failed_response,
                    })
                    .to_string(),
                ))
                .await
                .expect("failed event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": {
                            "id": "resp_failed",
                            "status": "failed",
                        },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");

            let second_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = second_request
                .into_text()
                .expect("second request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            let response = sample_response(ResponseStatus::Completed);
            let response = serde_json::to_value(CompletionResponse {
                id: "resp_2".to_string(),
                ..response
            })
            .expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": {
                            "id": "resp_2",
                            "status": "completed",
                        },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let error = session
            .wait_for_completed_response()
            .await
            .expect_err("failed response should error");
        assert!(error.to_string().contains("failed response"));
        assert_eq!(session.previous_response_id(), None);

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn done_first_completed_turn_updates_previous_response_id() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            for response_id in ["resp_1", "resp_2"] {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create payload, got {payload}"
                );

                if response_id == "resp_2" {
                    assert!(
                        payload.contains("\"previous_response_id\":\"resp_1\""),
                        "expected chained previous_response_id in payload, got {payload}"
                    );
                }

                let response = serde_json::to_value(CompletionResponse {
                    id: response_id.to_string(),
                    ..sample_response(ResponseStatus::Completed)
                })
                .expect("response should serialize");

                socket
                    .send(Message::text(
                        json!({
                            "type": "response.done",
                            "response": response,
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("done event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn done_first_failed_turn_does_not_chain_next_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let first_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = first_request
                .into_text()
                .expect("first request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );
            assert!(
                !payload.contains("\"previous_response_id\""),
                "did not expect previous_response_id in first payload, got {payload}"
            );

            let failed_response = serde_json::to_value(CompletionResponse {
                id: "resp_failed".to_string(),
                status: ResponseStatus::Failed,
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("failed response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": failed_response,
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");

            let second_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = second_request
                .into_text()
                .expect("second request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );
            assert!(
                !payload.contains("\"previous_response_id\""),
                "did not expect chained previous_response_id in payload, got {payload}"
            );

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_2".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let error = session
            .wait_for_completed_response()
            .await
            .expect_err("failed response should error");
        assert!(error.to_string().contains("failed response"));
        assert_eq!(session.previous_response_id(), None);

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[test]
    fn websocket_url_converts_http_to_ws() {
        let url = websocket_url("http://localhost:8080/v1").expect("url should convert");
        assert_eq!(url, "ws://localhost:8080/v1/responses");
    }

    #[test]
    fn websocket_url_rejects_unsupported_scheme() {
        let result = websocket_url("ftp://example.com/v1");
        assert!(result.is_err());
    }

    #[test]
    fn websocket_url_trims_trailing_slash() {
        let url = websocket_url("https://api.openai.com/v1/").expect("url should convert");
        assert_eq!(url, "wss://api.openai.com/v1/responses");
    }

    #[test]
    fn unknown_event_type_is_skipped() {
        let payload = json!({
            "type": "response.some_future_event",
            "data": "hello"
        });

        let result =
            parse_server_event(&payload.to_string()).expect("unknown event should not error");
        assert!(result.is_none(), "unknown event should be skipped");
    }

    #[test]
    fn malformed_known_event_returns_error() {
        let payload = json!({
            "type": "response.completed"
        });

        let error = parse_server_event(&payload.to_string())
            .expect_err("malformed known event should error");
        assert!(
            error.to_string().contains("StreamingCompletionChunk"),
            "expected strict decode failure, got {error}"
        );
    }

    #[tokio::test]
    async fn close_is_idempotent() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session.close().await.expect("first close should succeed");
        session.close().await.expect("second close should succeed");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn send_while_in_flight_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // Read the first request but don't respond — keep it in-flight
            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            // Wait for client to finish its test
            sleep(Duration::from_millis(100)).await;
            let _ = socket.close(None).await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");

        let error = session
            .send(model.completion_request("second").build())
            .await
            .expect_err("second send while in-flight should error");
        assert!(
            error.to_string().contains("already in flight"),
            "expected in-flight error, got {error}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn send_after_close_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let _socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");
            sleep(Duration::from_millis(100)).await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session.close().await.expect("close should succeed");

        let error = session
            .send(model.completion_request("after close").build())
            .await
            .expect_err("send after close should error");
        assert!(
            error.to_string().contains("session is closed"),
            "expected closed-session error, got {error}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn next_event_without_send_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let _socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");
            sleep(Duration::from_millis(100)).await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let error = session
            .next_event()
            .await
            .expect_err("next_event without send should error");
        assert!(
            error
                .to_string()
                .contains("No OpenAI websocket response is currently in flight"),
            "expected not-in-flight error, got {error}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn unknown_event_is_skipped_during_session() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            // Send an unknown event type first
            socket
                .send(Message::text(
                    json!({
                        "type": "response.some_future_event",
                        "data": "should be skipped"
                    })
                    .to_string(),
                ))
                .await
                .expect("unknown event should send");

            // Then send the real completed response
            let response = serde_json::to_value(CompletionResponse {
                id: "resp_after_unknown".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("send should succeed");
        let response = session
            .wait_for_completed_response()
            .await
            .expect("response should complete despite unknown event");
        assert_eq!(response.id, "resp_after_unknown");

        server.await.expect("server task should finish");
    }
}
