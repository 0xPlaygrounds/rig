//! WebSocket session support for the OpenAI Responses API.
//!
//! This module implements OpenAI's `/v1/responses` WebSocket mode as a stateful,
//! sequential session. Each connection supports a single in-flight response at a
//! time, which matches OpenAI's current protocol constraints.

use crate::completion::NormalizeCompletionResponse;
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::providers::internal::adapter::{TriagedFrame, triage_frame};
use crate::providers::openai::responses_api::streaming::{
    ItemChunk, RawChoiceAccumulator, ResponseChunk, ResponseChunkKind, ResponsesStreamOptions,
    StreamingCompletionChunk, classify_responses_frame, completion_response_from_raw_choices,
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
use url::Url;

use super::{CompletionResponse, ResponseStatus, ResponsesCompletionModel, ResponsesUsage};

type OpenAIWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WebSocketRawChoice = crate::streaming::RawStreamingChoice<
    crate::providers::openai::responses_api::streaming::StreamingCompletionResponse,
>;
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// The transport request-id header this endpoint reports, shared with the
/// HTTP twins through [`super::ResponsesProviderExt::REQUEST_ID_HEADER`] — the
/// websocket upgrade is answered by the same service and reports the same id.
const REQUEST_ID_HEADER: Option<&'static str> =
    <crate::providers::openai::OpenAIResponsesExt as super::ResponsesProviderExt>::REQUEST_ID_HEADER;

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
    /// An unrecognized event's raw payload — warned and skipped on the
    /// semantic path, forwarded verbatim so the streaming surface can carry
    /// it on the `RawStreamingChoice::Unknown` passthrough channel.
    Unknown(crate::streaming::UnknownPayload),
}

impl ResponsesWebSocketEvent {
    /// Returns the response ID when the event includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        match self {
            Self::Response(chunk) => Some(&chunk.response.id),
            Self::Done(done) => done.response_id(),
            Self::Item(_) | Self::Error(_) | Self::Unknown(_) => None,
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
            Self::Item(_) | Self::Unknown(_) => false,
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

        // The session takes a raw `CompletionRequest`, bypassing the builder's
        // `send`/`stream` — so this is a direct-to-model surface and validates
        // here, per `validate_message_content`'s own contract. Every session
        // entry point (`send`, `warmup`, `completion`, `raw_completion`)
        // funnels through this method.
        completion_request.validate_message_content()?;

        let payload = ResponsesWebSocketClientEvent {
            kind: ResponsesWebSocketClientEventKind::ResponseCreate,
            request: self.prepare_request(completion_request)?,
            generate: options.generate,
        };

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI websocket request",
            &payload,
        );

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

    /// Sends a completion turn and collects the final OpenAI response,
    /// normalized.
    ///
    /// Use [`ResponsesWebSocketSession::raw_completion`] when the provider's own
    /// wire response is needed.
    pub async fn completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let provider = self.model.provider_name();
        self.send(completion_request).await?;
        let (response, raw_choices) = self.wait_for_terminal_response().await?;
        // Replay the accumulated deltas through the shared normalization
        // pipeline so streamed partial output survives even when the terminal
        // body's `output` is empty (e.g. an incomplete turn). A turn that
        // carried no deltas (e.g. a `response.done`-only turn) falls back to
        // normalizing the terminal body itself.
        match completion_response_from_raw_choices(provider, raw_choices, &response).await? {
            Some(normalized) => Ok(normalized),
            None => response.normalize(provider),
        }
    }

    /// Sends a completion turn and returns the provider's own wire response.
    ///
    /// Shares the send/receive path with
    /// [`ResponsesWebSocketSession::completion`], which calls it and then
    /// applies the provider-local mapping — one websocket turn either way.
    pub async fn raw_completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.send(completion_request).await?;
        self.wait_for_completed_response().await
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
        Ok(self.wait_for_terminal_response().await?.0)
    }

    /// Drives the shared [`RawChoiceAccumulator`] over the websocket events —
    /// the same decode state machine the SSE path uses, fed by a different
    /// transport — so streamed deltas survive alongside the terminal body.
    ///
    /// **A failed turn discards the choices collected so far, deliberately
    /// (#2258 G3).** Every error exit below — the `?` on `next_event()`, the
    /// `response.done`-without-a-body branch, and the provider `error` event —
    /// returns `Err` and drops `accumulator`/`raw_choices` with whatever text,
    /// reasoning and tool calls had already arrived.
    ///
    /// That is not a divergence from the SSE side: the right comparison is the
    /// *buffered* SSE path, `run_wire_buffered`, which likewise fails the whole
    /// operation on the first `Err` rather than returning partial content plus
    /// an error. Only the *live* SSE surface can do better, and only because it
    /// is a `Stream`: it yields the partial items first and the `Err` as a
    /// later element. This session exposes a unary surface —
    /// [`completion()`](Self::wait_for_completed_response) /
    /// `raw_completion()` return one `Result<CompletionResponse, _>` — and a
    /// unary return type cannot express partial-content-plus-error without
    /// inventing a second channel. Keeping the failed turn's fragments would
    /// mean returning a `CompletionResponse` that never completed, which is the
    /// exact fabrication the terminal-record rules exist to prevent.
    ///
    /// If a caller needs the partial content of a failed websocket turn, the
    /// fix is a streaming websocket surface, not a partial unary response.
    async fn wait_for_terminal_response(
        &mut self,
    ) -> Result<(CompletionResponse, Vec<WebSocketRawChoice>), CompletionError> {
        let mut accumulator = RawChoiceAccumulator::new(ResponsesUsage::new());
        let mut raw_choices = Vec::new();
        loop {
            match self.next_event().await? {
                ResponsesWebSocketEvent::Response(chunk) => {
                    if matches!(
                        chunk.kind,
                        ResponseChunkKind::ResponseCompleted
                            | ResponseChunkKind::ResponseFailed
                            | ResponseChunkKind::ResponseIncomplete
                    ) {
                        return finish_terminal_response(accumulator, chunk.response, raw_choices);
                    }
                }
                ResponsesWebSocketEvent::Done(done) => {
                    if let Some(response) = done.as_completion_response() {
                        return finish_terminal_response(accumulator, response, raw_choices);
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
                    // Genuine provider error event: preserve the serialized payload
                    // (code + message + any extra fields) so provider_response_json()
                    // parses it, matching the response.failed path. No HTTP status on
                    // the websocket stream, so status: None.
                    return Err(provider_error_from_event(error));
                }
                ResponsesWebSocketEvent::Item(chunk) => {
                    raw_choices.extend(
                        accumulator.decode_item_chunk(chunk, ResponsesStreamOptions::strict()),
                    );
                }
                ResponsesWebSocketEvent::Unknown(value) => {
                    // Semantic skip, raw passthrough: the accumulator never
                    // sees the frame, but the streaming surface still yields
                    // it verbatim.
                    raw_choices.push(crate::streaming::RawStreamingChoice::Unknown(value));
                }
            }
        }
    }

    fn update_state_for_event(&mut self, event: &ResponsesWebSocketEvent) {
        match event {
            ResponsesWebSocketEvent::Response(chunk) => match chunk.kind {
                // An incomplete turn still produced a response the next turn
                // can chain from, so it keeps `previous_response_id` like a
                // completed one.
                ResponseChunkKind::ResponseCompleted | ResponseChunkKind::ResponseIncomplete => {
                    let response_id = chunk.response.id.clone();
                    self.previous_response_id = Some(response_id.clone());
                    self.pending_done_response_id = Some(response_id);
                    self.in_flight = false;
                }
                ResponseChunkKind::ResponseFailed => {
                    self.pending_done_response_id = Some(chunk.response.id.clone());
                    self.previous_response_id = None;
                    self.in_flight = false;
                }
                ResponseChunkKind::ResponseCreated | ResponseChunkKind::ResponseInProgress => {}
            },
            ResponsesWebSocketEvent::Done(done) => {
                match done.status() {
                    Some(ResponseStatus::Completed) | Some(ResponseStatus::Incomplete) => {
                        if let Some(response_id) = done.response_id() {
                            self.previous_response_id = Some(response_id.to_string());
                        }
                    }
                    Some(ResponseStatus::Failed)
                    | Some(ResponseStatus::Cancelled)
                    | Some(ResponseStatus::Other(_)) => {
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
            // An unknown frame carries no turn-lifecycle signal.
            ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => {}
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

/// Records the terminal event into the accumulator and drains it, so the raw
/// choices end with the terminal record exactly as the SSE path produces them.
fn finish_terminal_response(
    mut accumulator: RawChoiceAccumulator,
    response: CompletionResponse,
    mut raw_choices: Vec<WebSocketRawChoice>,
) -> Result<(CompletionResponse, Vec<WebSocketRawChoice>), CompletionError> {
    let response = terminal_response_result(response)?;
    // Only completed/incomplete get through `terminal_response_result`, so the
    // accumulator's failed-event error mapping (which needs the raw event
    // bytes this path no longer has) is unreachable here.
    let kind = if matches!(response.status, ResponseStatus::Incomplete) {
        ResponseChunkKind::ResponseIncomplete
    } else {
        ResponseChunkKind::ResponseCompleted
    };
    accumulator.record_response_chunk(kind, response.clone(), "")?;
    raw_choices.extend(accumulator.finish());
    Ok((response, raw_choices))
}

fn terminal_response_result(
    response: CompletionResponse,
) -> Result<CompletionResponse, CompletionError> {
    match response.status {
        ResponseStatus::Completed => Ok(response),
        // Deliberate two-tier behaviour: when the provider supplies its own error
        // object we preserve the full failed-response envelope through
        // `from_provider_body` (status: None, no HTTP status on the websocket
        // stream) so `provider_response_json()` parses it — consistent with the
        // `error` event and the streaming paths. The body is re-serialized from
        // the parsed response (not byte-identical to the wire bytes, which aren't
        // retained past parsing) — semantically the provider's payload. When the
        // object is absent we have nothing provider-authored to surface, so we
        // emit a Rig-authored `ProviderError` diagnostic (provider_response_body()
        // is None).
        ResponseStatus::Failed => match response.error.as_ref() {
            Some(error) => Err(CompletionError::from_provider_body(
                serde_json::to_string(&response).unwrap_or_else(|_| error.message.clone()),
            )),
            None => Err(CompletionError::ProviderError(response_error_message(
                "failed response",
            ))),
        },
        // An incomplete response (e.g. hitting `max_output_tokens`) is a
        // genuine terminal: the partial output and usage are kept, and the
        // normalization path maps the status/incomplete_details to a finish
        // reason via `map_finish_reason`, matching the unary and SSE paths.
        ResponseStatus::Incomplete => Ok(response),
        other => Err(CompletionError::ProviderError(format!(
            "OpenAI websocket response ended in state {other:?}"
        ))),
    }
}

fn response_error_message(fallback: &str) -> String {
    format!("OpenAI websocket returned a {fallback}")
}

/// Maps a provider `error` event into a [`CompletionError`] that preserves the
/// raw error payload as JSON (code + message + any extra provider fields) so the
/// `provider_response_*` helpers can inspect it. The websocket stream carries no
/// HTTP status, so `status` is `None`. The body is the event re-serialized from
/// the parsed representation (not byte-identical to the original wire bytes,
/// which are not retained past parsing) — semantically the provider's payload.
fn provider_error_from_event(error: ResponsesWebSocketErrorEvent) -> CompletionError {
    CompletionError::from_provider_body(
        serde_json::to_string(&error).unwrap_or_else(|_| error.to_string()),
    )
}

/// Parses one websocket JSON payload into a server event.
///
/// Only the websocket-only envelope types (`error`, `response.done`) are
/// dispatched here; every other frame classifies through the same
/// [`classify_responses_frame`] interpreter the SSE paths use, so the modeled
/// Responses event set — and its strict decode policy — is stated once for the
/// wire family rather than duplicated per transport.
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
        // Shared per-frame triage (`Unknown` is warned and forwarded raw for
        // the passthrough channel, `Corrupt` fails the turn — this surface
        // has no stream to carry `Err` items).
        _ => Ok(Some(
            match triage_frame(classify_responses_frame(payload))? {
                TriagedFrame::Event(StreamingCompletionChunk::Response(response)) => {
                    ResponsesWebSocketEvent::Response(response)
                }
                TriagedFrame::Event(StreamingCompletionChunk::Delta(item)) => {
                    ResponsesWebSocketEvent::Item(item)
                }
                TriagedFrame::Unknown(value) => ResponsesWebSocketEvent::Unknown(value),
            },
        )),
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

/// Map a transport failure onto rig's error model, preserving the provider's
/// own response when the failure carried one.
///
/// A websocket upgrade that the provider *rejects* never becomes a websocket:
/// it is an ordinary HTTP response, and this endpoint answers it exactly as
/// the HTTP twin answers a bad request — a status, an `x-request-id`, and a
/// JSON error body naming the cause. A live handshake with an invalid key
/// returns `401` with `x-request-id` and
/// `{"error":{"code":"invalid_api_key",…}}`. `tungstenite` hands all of it back
/// on [`tungstenite::Error::Http`] (its body is filled in from the read tail),
/// so flattening it to `error.to_string()` — `"HTTP error: 401 Unauthorized"` —
/// discarded the status, the body and the request id, leaving
/// `provider_response_status()`, `provider_response_body()` and
/// `provider_request_id()` all `None`.
///
/// That is the contract the crate's other two completion transports keep
/// (rig#2314, rig#2315): the blocking path through `send_completion` and the
/// SSE path through `sse_transport` both classify a connect failure as
/// [`CompletionError::ProviderResponse`] with the body and id attached. This
/// makes the websocket the third.
///
/// The rejection's **headers** ride along too, by the same rule and for the
/// same reason (rig#2210): a `429` upgrade carries `Retry-After`, and a caller
/// that has to back off needs it from whichever transport it was refused on.
/// This mirrors `sse_transport`, which attaches its handshake's headers to the
/// error it builds.
///
/// Failures that never reached the provider — TLS, DNS, a protocol violation —
/// have no response to preserve and stay [`CompletionError::ProviderError`].
fn websocket_provider_error(error: tungstenite::Error) -> CompletionError {
    let tungstenite::Error::Http(response) = error else {
        return CompletionError::ProviderError(error.to_string());
    };

    let (parts, body) = (*response).into_parts();
    let provider_request_id = REQUEST_ID_HEADER
        .and_then(|header| parts.headers.get(header))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    // The body is the provider's own error envelope; an upgrade rejected
    // without one still carries its status, which is more than the string form
    // preserved.
    let body = body
        .map(|body| String::from_utf8_lossy(&body).into_owned())
        .unwrap_or_default();

    CompletionError::from_http_response_with_request_id(parts.status, body, provider_request_id)
        // Read after the id, since this consumes the map.
        .with_response_headers(Some(Box::new(parts.headers)))
}

#[cfg(test)]
mod tests {
    use super::{CompletionError, tungstenite};
    use super::{
        ResponsesWebSocketCreateOptions, ResponsesWebSocketDoneEvent, ResponsesWebSocketEvent,
        parse_server_event, terminal_response_result, websocket_provider_error, websocket_url,
    };
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::providers::openai::responses_api::{
        CompletionResponse, IncompleteDetailsReason, Output, ResponseError, ResponseObject,
        ResponseStatus, ResponsesUsage,
    };
    use futures::{SinkExt, StreamExt};
    use serde_json::json;

    /// Build the `tungstenite::Error` a rejected upgrade produces: the status,
    /// the headers the endpoint set, and the body read off the tail.
    fn handshake_rejection(
        status: u16,
        request_id: Option<&str>,
        body: Option<&str>,
    ) -> tungstenite::Error {
        handshake_rejection_with_headers(status, request_id, body, &[])
    }

    /// [`handshake_rejection`] plus arbitrary response headers, for the
    /// rate-limit metadata a `429` upgrade carries.
    fn handshake_rejection_with_headers(
        status: u16,
        request_id: Option<&str>,
        body: Option<&str>,
        headers: &[(&str, &str)],
    ) -> tungstenite::Error {
        let mut response = http::Response::builder().status(status);
        if let Some(request_id) = request_id {
            response = response.header("x-request-id", request_id);
        }
        for (name, value) in headers {
            response = response.header(*name, *value);
        }
        tungstenite::Error::Http(Box::new(
            response
                .body(body.map(|body| body.as_bytes().to_vec()))
                .expect("response should build"),
        ))
    }

    /// The live shape, recorded in
    /// `websocket_error_identity_matrix/handshake_rejection_carries_status_body_and_request_id`.
    const REJECTION_BODY: &str = r#"{"error":{"message":"Incorrect API key provided: sk-inval***-key.","type":"invalid_request_error","code":"invalid_api_key","param":null},"status":401}"#;

    #[test]
    fn websocket_provider_error_preserves_status_body_and_request_id() {
        let error = websocket_provider_error(handshake_rejection(
            401,
            Some("req_websocket_1"),
            Some(REJECTION_BODY),
        ));

        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
        assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
        assert_eq!(error.provider_request_id(), Some("req_websocket_1"));
        assert_eq!(
            error
                .provider_response_json()
                .expect("body should be valid JSON")
                .expect("parsed JSON should be present")["error"]["code"],
            "invalid_api_key"
        );
    }

    /// The id is optional everywhere else in this crate and is optional here:
    /// its absence must not cost the status or the body.
    #[test]
    fn websocket_provider_error_without_a_request_id_keeps_the_rest() {
        let error = websocket_provider_error(handshake_rejection(429, None, Some("slow down")));

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(error.provider_response_body(), Some("slow down"));
        assert_eq!(error.provider_request_id(), None);
    }

    /// An empty `x-request-id` is absence, not an id — the same rule
    /// `sse_transport` applies.
    #[test]
    fn websocket_provider_error_treats_an_empty_request_id_as_absent() {
        let error = websocket_provider_error(handshake_rejection(401, Some(""), Some("nope")));

        assert_eq!(error.provider_request_id(), None);
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
    }

    /// A rejection whose body never arrived still carries more than the string
    /// form did: the status.
    #[test]
    fn websocket_provider_error_without_a_body_keeps_the_status() {
        let error = websocket_provider_error(handshake_rejection(403, Some("req_2"), None));

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::FORBIDDEN)
        );
        assert_eq!(error.provider_response_body(), Some(""));
        assert_eq!(error.provider_request_id(), Some("req_2"));
    }

    /// Every status an upgrade can be answered with survives — the mapper keys
    /// on the error carrying a response, never on the status. `tungstenite`
    /// raises `Error::Http` for *any* non-101 status, and `connect_async` does
    /// not follow redirects, so a 2xx or a proxy's 3xx reaches this mapper too.
    #[test]
    fn websocket_provider_error_preserves_every_rejection_status() {
        for status in [200u16, 302, 400, 401, 403, 404, 429, 500, 503] {
            let error = websocket_provider_error(handshake_rejection(status, None, Some("body")));
            assert_eq!(
                error.provider_response_status().map(|s| s.as_u16()),
                Some(status),
                "status {status} must survive"
            );
        }
    }

    /// The other half of the error space, enumerated rather than sampled:
    /// **every** non-`Http` variant of [`tungstenite::Error`] must stay a
    /// [`CompletionError::ProviderError`] carrying its own text, because none
    /// of them reached the provider and so none has a response to preserve.
    ///
    /// `Error::Http` is the only variant that carries one, which is what makes
    /// the mapper's `let-else` correct for the rest by construction. The list
    /// below is the crate's full variant set minus `Http`; `Tls` is the one
    /// exclusion, since its inner `TlsError` is `#[non_exhaustive]`-shaped per
    /// TLS backend and cannot be constructed portably in a test — it takes the
    /// same branch as its ten siblings.
    #[test]
    fn websocket_provider_error_leaves_every_transport_failure_alone() {
        let cases: Vec<tungstenite::Error> = vec![
            tungstenite::Error::ConnectionClosed,
            tungstenite::Error::AlreadyClosed,
            tungstenite::Error::Io(std::io::Error::other("connection reset")),
            tungstenite::Error::Capacity(tungstenite::error::CapacityError::TooManyHeaders),
            tungstenite::Error::Protocol(tungstenite::error::ProtocolError::HandshakeIncomplete),
            tungstenite::Error::WriteBufferFull(Box::new(tungstenite::Message::Text(
                "queued".into(),
            ))),
            tungstenite::Error::AttackAttempt,
            tungstenite::Error::Url(tungstenite::error::UrlError::NoPathOrQuery),
            tungstenite::Error::HttpFormat(
                http::header::HeaderName::from_bytes(b"not a header")
                    .expect_err("an invalid header name should not parse")
                    .into(),
            ),
        ];

        for error in cases {
            let expected = error.to_string();
            let mapped = websocket_provider_error(error);

            assert!(
                matches!(mapped, CompletionError::ProviderError(_)),
                "a failure with no provider response must stay a ProviderError: {mapped:?}"
            );
            assert_eq!(mapped.to_string(), format!("ProviderError: {expected}"));
            assert_eq!(mapped.provider_response_status(), None);
            assert_eq!(mapped.provider_response_body(), None);
            assert_eq!(mapped.provider_request_id(), None);
        }
    }

    /// rig#2210's contract, on this transport: a rejected upgrade's headers
    /// survive onto the error, so a caller refused with `429` can read
    /// `Retry-After` no matter which transport carried the refusal. The SSE
    /// path attaches its handshake's headers the same way.
    #[test]
    fn websocket_provider_error_preserves_the_rejections_headers() {
        let error = websocket_provider_error(handshake_rejection_with_headers(
            429,
            Some("req_rate_limited"),
            Some(r#"{"error":{"code":"rate_limit_exceeded"}}"#),
            &[("retry-after", "20"), ("x-ratelimit-remaining", "0")],
        ));

        let headers = error
            .provider_response_headers()
            .expect("a rejection's headers must survive onto the error");
        assert_eq!(
            headers
                .get(http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("20")
        );
        assert_eq!(
            headers
                .get("x-ratelimit-remaining")
                .and_then(|value| value.to_str().ok()),
            Some("0")
        );
        // The rest of the identity is untouched by the header capture.
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(error.provider_request_id(), Some("req_rate_limited"));
    }

    /// A failure that never reached the provider has no headers to report,
    /// just as it has no status or body.
    #[test]
    fn websocket_provider_error_reports_no_headers_for_a_transport_failure() {
        let error = websocket_provider_error(tungstenite::Error::ConnectionClosed);

        assert!(error.provider_response_headers().is_none());
    }

    /// A canary, not matrix coverage: it pins the exact string the deleted
    /// mapper produced, so restoring that line fails loudly rather than
    /// quietly. The behavior it protects is asserted positively above.
    #[test]
    fn websocket_provider_error_no_longer_flattens_a_rejection_to_a_string() {
        let error = websocket_provider_error(handshake_rejection(
            401,
            Some("req_websocket_1"),
            Some(REJECTION_BODY),
        ));

        assert_ne!(
            error.to_string(),
            "ProviderError: HTTP error: 401 Unauthorized",
            "the pre-fix behavior discarded the status, body and request id"
        );
    }
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::time::sleep;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    #[test]
    fn websocket_error_event_preserves_provider_payload_as_json() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "type".to_string(),
            serde_json::Value::String("invalid_request_error".to_string()),
        );
        let event = super::ResponsesWebSocketErrorEvent {
            kind: super::ResponsesWebSocketErrorEventKind::Error,
            error: super::ResponsesWebSocketErrorPayload {
                code: Some("rate_limit_exceeded".to_string()),
                message: Some("slow down".to_string()),
                extra,
            },
        };

        let err = super::provider_error_from_event(event);

        // No HTTP status on the websocket stream, and the raw payload round-trips
        // through provider_response_json() (code + message + extra all preserved).
        assert_eq!(err.provider_response_status(), None);
        let json = err
            .provider_response_json()
            .expect("preserved body should be valid JSON")
            .expect("provider response body should be present");
        assert_eq!(json["error"]["code"], "rate_limit_exceeded");
        assert_eq!(json["error"]["message"], "slow down");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    fn sample_response(status: ResponseStatus) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            provider_request_id: None,
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
                output_tokens_details: Some(
                    crate::providers::openai::responses_api::OutputTokensDetails {
                        reasoning_tokens: 0,
                    },
                ),
                total_tokens: 3,
            }),
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: Default::default(),
            provider_reasoning: None,
            reasoning_metadata: None,
            reasoning_context: None,
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
    async fn incomplete_turn_keeps_streamed_partial_output() {
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

            // The content exists ONLY in the delta events; the terminal
            // `response.incomplete` body has an empty `output`, which is a
            // sequence the wire protocol permits.
            socket
                .send(Message::text(
                    json!({
                        "type": "response.output_text.delta",
                        "content_index": 0,
                        "delta": "partial",
                        "item_id": "msg_incomplete_1",
                        "logprobs": [],
                        "output_index": 0,
                        "sequence_number": 1
                    })
                    .to_string(),
                ))
                .await
                .expect("delta event should send");

            let mut response = sample_response(ResponseStatus::Incomplete);
            response.incomplete_details = Some(IncompleteDetailsReason {
                reason: "max_output_tokens".to_string(),
            });
            let response = serde_json::to_value(response).expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.incomplete",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("incomplete event should send");
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

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("incomplete turn should be a successful terminal");

        // The streamed partial text survives, and normalization maps the
        // incomplete status to the same finish reason as the unary path.
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(normalized.usage.input_tokens, 1);
        assert_eq!(normalized.usage.output_tokens, 2);
        assert_eq!(normalized.usage.total_tokens, 3);
        assert!(matches!(
            normalized.choice.first(),
            Some(crate::completion::AssistantContent::Text(text)) if text.text == "partial"
        ));

        server.await.expect("server task should finish");
    }

    /// #2258 P2: the websocket session shares `decode_item_chunk`, so text for
    /// one message item interleaved with reasoning must aggregate as one text
    /// part here too.
    #[tokio::test]
    async fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            let events = [
                json!({
                    "type": "response.output_text.delta",
                    "content_index": 0,
                    "delta": "hello ",
                    "item_id": "msg_1",
                    "logprobs": [],
                    "output_index": 0,
                    "sequence_number": 1
                }),
                json!({
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "because",
                    "item_id": "rs_2",
                    "output_index": 1,
                    "summary_index": 0,
                    "sequence_number": 2
                }),
                json!({
                    "type": "response.output_text.delta",
                    "content_index": 0,
                    "delta": "world",
                    "item_id": "msg_1",
                    "logprobs": [],
                    "output_index": 0,
                    "sequence_number": 3
                }),
                json!({
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
                        .expect("response should serialize"),
                }),
            ];
            for event in events {
                socket
                    .send(Message::text(event.to_string()))
                    .await
                    .expect("event should send");
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

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("interleaved turn should normalize");

        let texts: Vec<_> = normalized
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            ["hello world"],
            "same-item text must aggregate as one part around the reasoning"
        );
        assert!(
            normalized.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Reasoning(_)
            )),
            "the interleaved reasoning must survive"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn completed_turn_without_deltas_falls_back_to_terminal_body() {
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

            // No delta events at all: the terminal body carries the full
            // output, so normalization must fall back to it.
            let mut response = sample_response(ResponseStatus::Completed);
            response.output = vec![
                serde_json::from_value::<Output>(json!({
                    "type": "message",
                    "id": "msg_terminal_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "annotations": [], "text": "hello there" }]
                }))
                .expect("output message should deserialize"),
            ];
            let response = serde_json::to_value(response).expect("response should serialize");

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

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("completed turn should normalize");

        assert!(matches!(
            normalized.choice.first(),
            Some(crate::completion::AssistantContent::Text(text)) if text.text == "hello there"
        ));
        assert_eq!(normalized.message_id.as_deref(), Some("msg_terminal_1"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn incomplete_turn_without_deltas_normalizes_terminal_body_output() {
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

            // No delta events at all AND an incomplete terminal whose body
            // carries the partial output: the body must be normalized rather
            // than the turn reading as empty.
            let mut response = sample_response(ResponseStatus::Incomplete);
            response.incomplete_details = Some(IncompleteDetailsReason {
                reason: "max_output_tokens".to_string(),
            });
            response.output = vec![
                serde_json::from_value::<Output>(json!({
                    "type": "message",
                    "id": "msg_body_only_1",
                    "status": "incomplete",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "annotations": [], "text": "partial from body" }]
                }))
                .expect("output message should deserialize"),
            ];
            let response = serde_json::to_value(response).expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.incomplete",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("incomplete event should send");
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

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("incomplete turn with body output should normalize");

        assert!(matches!(
            normalized.choice.first(),
            Some(crate::completion::AssistantContent::Text(text)) if text.text == "partial from body"
        ));
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(normalized.message_id.as_deref(), Some("msg_body_only_1"));

        server.await.expect("server task should finish");
    }

    #[test]
    fn terminal_failed_response_with_error_preserves_raw_payload() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "server_error".to_string(),
            message: "the model failed to generate a response".to_string(),
        });

        let err = match terminal_response_result(response) {
            Ok(_) => panic!("failed response with an error object should fail"),
            Err(e) => e,
        };

        // The full failed-response envelope is preserved as a ProviderResponse with
        // no HTTP status (the websocket stream carries none), so the raw JSON parses
        // back with the provider error nested under `error` — proving the whole
        // envelope is kept, not just the error object.
        assert_eq!(err.provider_response_status(), None);

        let json = err
            .provider_response_json()
            .expect("preserved body should parse as JSON")
            .expect("preserved body should not be empty");
        assert_eq!(
            json["error"]["message"],
            "the model failed to generate a response"
        );
        assert_eq!(json["error"]["code"], "server_error");
    }

    #[test]
    fn terminal_failed_response_without_error_is_rig_diagnostic() {
        let err = match terminal_response_result(sample_response(ResponseStatus::Failed)) {
            Ok(_) => panic!("failed response should fail"),
            Err(e) => e,
        };

        // No provider error object, so this is a Rig-authored diagnostic and exposes
        // no preserved provider response body.
        assert_eq!(err.provider_response_body(), None);
        assert!(err.to_string().contains("failed response"));
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
    fn unknown_event_type_is_forwarded_raw() {
        let payload = json!({
            "type": "response.some_future_event",
            "data": "hello"
        });

        let result =
            parse_server_event(&payload.to_string()).expect("unknown event should not error");
        // Semantically skipped, but carried verbatim so the streaming surface
        // can yield it on the `RawStreamingChoice::Unknown` passthrough.
        match result {
            Some(ResponsesWebSocketEvent::Unknown(value)) => assert_eq!(value, payload.into()),
            other => panic!("expected the raw Unknown passthrough event, got {other:?}"),
        }
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
    async fn unknown_event_is_skipped_and_reasoning_metadata_is_preserved() {
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

            // Then send the real completed response, including reasoning
            // metadata to verify that the WebSocket path preserves it.
            let mut response = sample_response(ResponseStatus::Completed);
            response.id = "resp_after_unknown".to_string();
            response.reasoning_metadata = Some(
                json!({
                    "context": "all_turns",
                    "effort": "ultra",
                    "summary": null,
                    "future_control": true
                })
                .as_object()
                .expect("reasoning metadata should be an object")
                .clone(),
            );
            response.reasoning_context = Some("all_turns".to_string());
            let response = serde_json::to_value(response).expect("response should serialize");

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
        assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
        assert_eq!(
            response.reasoning_metadata.as_ref(),
            json!({
                "context": "all_turns",
                "effort": "ultra",
                "summary": null,
                "future_control": true
            })
            .as_object()
        );

        server.await.expect("server task should finish");
    }

    /// Re-wraps SSE conformance fixture frames as websocket text payloads: the
    /// wire events are identical across the two transports, only the framing
    /// (`data:` lines vs. one JSON message per ws frame) differs.
    fn ws_messages_from_sse_frames<'a>(
        frames: impl IntoIterator<Item = &'a bytes::Bytes>,
    ) -> Vec<String> {
        frames
            .into_iter()
            .flat_map(|frame| {
                std::str::from_utf8(frame)
                    .expect("SSE fixture frames should be UTF-8")
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                    .filter(|data| !data.is_empty() && *data != "[DONE]")
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn spawn_ws_server_with_messages(
        listener: TcpListener,
        messages: Vec<String>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
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

            for message in messages {
                socket
                    .send(Message::text(message))
                    .await
                    .expect("event should send");
            }
        })
    }

    /// Websocket conformance invocation over the shared Responses fixture:
    /// the SAME frames the SSE conformance suite streams, re-wrapped as ws
    /// messages, must yield the same content through the shared
    /// `classify_responses_frame` + accumulator interpretation — text and
    /// tool-call deltas delivered, the unknown event skipped, usage and finish
    /// reason taken from the terminal.
    #[tokio::test]
    async fn websocket_conformance_replays_sse_fixture_frames() {
        let fixture =
            crate::test_utils::streaming_conformance::fixtures::openai_responses::fixture();
        // The shared fixture scripts byte frames; re-wrap them as ws messages.
        let byte_frame = |frame: &crate::test_utils::streaming_conformance::WireInput| {
            frame
                .as_bytes()
                .cloned()
                .expect("the Responses fixture scripts byte frames")
        };
        let mut frames: Vec<bytes::Bytes> = Vec::new();
        frames.extend(fixture.text_frames.iter().map(byte_frame));
        frames.extend(fixture.tool_call_frames.iter().map(byte_frame));
        frames.extend(fixture.unknown_event_frame.iter().map(byte_frame));
        frames.extend(fixture.terminal_frames.iter().map(byte_frame));
        let messages = ws_messages_from_sse_frames(frames.iter());

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let server = spawn_ws_server_with_messages(listener, messages);

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

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("fixture turn should normalize");

        let texts: Vec<&str> = normalized
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, fixture.expected_texts);
        let tool_names: Vec<&str> = normalized
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::ToolCall(call) => {
                    Some(call.function.name.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(tool_names, vec![fixture.expected_tool_name]);
        assert_eq!(normalized.usage.total_tokens, fixture.expected_usage_total);
        // The fixture's expected finish reason applies to its text-only
        // sequences; this combined replay carries a tool call, which the
        // shared normalization maps to `ToolCalls` on every transport.
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::ToolCalls)
        );

        server.await.expect("server task should finish");
    }

    /// Regression for the diverged websocket dispatch: `response.reasoning_text.delta`
    /// was absent from the ws-private known-event list and silently dropped,
    /// while the SSE path delivered it. Routed through the shared classifier,
    /// the reasoning delta must survive to the normalized response.
    #[tokio::test]
    async fn reasoning_text_delta_arrives_over_websocket() {
        let messages = vec![
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "delta": "thinking hard",
            })
            .to_string(),
            json!({
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "answer",
                "item_id": "msg_1",
                "output_index": 0,
                "sequence_number": 2,
            })
            .to_string(),
            json!({
                "type": "response.completed",
                "sequence_number": 3,
                "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
                    .expect("response should serialize"),
            })
            .to_string(),
        ];

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let server = spawn_ws_server_with_messages(listener, messages);

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

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("turn with reasoning deltas should normalize");

        assert!(
            normalized.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Reasoning(reasoning)
                    if reasoning.content.iter().any(|block| matches!(
                        block,
                        crate::message::ReasoningContent::Text { text, .. }
                            if text.contains("thinking hard")
                    ))
            )),
            "reasoning delta should survive over websocket, got {:?}",
            normalized.choice
        );
        assert!(
            normalized.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Text(text) if text.text == "answer"
            )),
            "text delta should survive alongside reasoning, got {:?}",
            normalized.choice
        );

        server.await.expect("server task should finish");
    }

    #[test]
    fn parse_reasoning_text_delta_event_is_item() {
        let payload = json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        });

        let event = parse_server_event(&payload.to_string())
            .expect("reasoning delta should parse")
            .expect("reasoning delta should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
        assert!(!event.is_terminal());
    }
}
