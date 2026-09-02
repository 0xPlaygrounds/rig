//! WebSocket session support for the OpenAI Responses API.
//!
//! This module implements OpenAI's `/v1/responses` WebSocket mode as a stateful,
//! sequential session. Each connection supports a single in-flight response at a
//! time, which matches OpenAI's current protocol constraints.
//!
//! The session is transport-agnostic: it drives a
//! [`crate::ws_client::WebSocketConnection`] supplied by a
//! backend such as `rig-tungstenite`, exactly as the rest of this provider
//! drives an [`HttpClientExt`]. The protocol — the event envelopes, the
//! `previous_response_id` chaining, the terminal-record rules — lives here with
//! the provider rather than in whichever crate owns the socket library.

use crate::completion::NormalizeCompletionResponse;
use crate::completion::{self, CompletionError};
use crate::http_client::{self, HttpClientExt, NoBody};
use crate::providers::internal::adapter::{TriagedFrame, triage_frame};
use crate::providers::openai::Client as OpenAIClient;
use crate::providers::openai::responses_api::streaming::{
    ItemChunk, RawChoiceAccumulator, ResponseChunk, ResponseChunkKind, ResponsesStreamOptions,
    StreamingCompletionChunk, classify_responses_frame, completion_response_from_raw_choices,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt, WebSocketConnection,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;

use crate::providers::openai::responses_api::{
    CompletionResponse, ResponseStatus, ResponsesCompletionModel, ResponsesUsage,
};

type WebSocketRawChoice = crate::streaming::RawStreamingChoice<
    crate::providers::openai::responses_api::streaming::StreamingCompletionResponse,
>;

/// The websocket endpoint's path, appended to the client's configured base URL.
const WEBSOCKET_PATH: &str = "responses";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The transport request-id header this endpoint reports, shared with the
/// HTTP twins through [`ResponsesProviderExt::REQUEST_ID_HEADER`](crate::providers::openai::responses_api::ResponsesProviderExt::REQUEST_ID_HEADER) — the
/// websocket upgrade is answered by the same service and reports the same id.
const REQUEST_ID_HEADER: Option<&'static str> =
    <crate::providers::openai::OpenAIResponsesExt as crate::providers::openai::responses_api::ResponsesProviderExt>::REQUEST_ID_HEADER;

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
    request: crate::providers::openai::responses_api::CompletionRequest,
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
pub struct ResponsesWebSocketSessionBuilder<H = crate::http_client::BoxedHttpClient> {
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
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Opens the websocket session over `backend`, using the configured
    /// builder options.
    ///
    /// rig-core names no websocket backend, exactly as it names no HTTP
    /// transport. A caller using the bundled one reaches for the
    /// `connect()` convenience the backend crate supplies instead of naming it
    /// here.
    pub async fn connect_with<W>(
        self,
        backend: &W,
    ) -> Result<ResponsesWebSocketSession<H>, CompletionError>
    where
        W: WebSocketClientExt,
    {
        ResponsesWebSocketSession::connect_with_timeouts(
            backend,
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
pub struct ResponsesWebSocketSession<H = crate::http_client::BoxedHttpClient> {
    model: ResponsesCompletionModel<H>,
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    socket: BoxedWebSocketConnection,
    in_flight: bool,
    event_timeout: Option<Duration>,
    closed: bool,
    failed: bool,
}

impl<H> ResponsesWebSocketSession<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn connect_with_timeouts<W>(
        backend: &W,
        model: ResponsesCompletionModel<H>,
        connect_timeout: Option<Duration>,
        event_timeout: Option<Duration>,
    ) -> Result<Self, CompletionError>
    where
        W: WebSocketClientExt,
    {
        let request = websocket_request(model.client().base_url(), model.client().headers())?;
        let socket = backend
            .connect(request, ConnectOptions::new().with_timeout(connect_timeout))
            .await
            .map_err(websocket_provider_error)?;

        Ok(Self::from_connection(model, socket, event_timeout))
    }

    /// Build a session over an already-open connection.
    ///
    /// The entry point for a backend that opens its socket some other way —
    /// a pre-authenticated connection handed in by a host, or an in-memory
    /// connection in a test. `event_timeout` matches
    /// [`ResponsesWebSocketSessionBuilder::event_timeout`]; `None` waits
    /// indefinitely for each event.
    pub fn from_connection(
        model: ResponsesCompletionModel<H>,
        connection: BoxedWebSocketConnection,
        event_timeout: Option<Duration>,
    ) -> Self {
        Self {
            model,
            previous_response_id: None,
            pending_done_response_id: None,
            socket: connection,
            in_flight: false,
            event_timeout,
            closed: false,
            failed: false,
        }
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

        if let Err(error) = self.socket.send(Frame::Text(payload)).await {
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
            let message = match self.read_next_frame().await? {
                Ok(message) => message,
                Err(error) => return Err(self.fail_session(websocket_provider_error(error))),
            };

            let Some(message) = message else {
                self.mark_closed();
                return Err(CompletionError::ProviderError(
                    "The OpenAI websocket connection closed before the turn finished".to_string(),
                ));
            };

            let payload = match websocket_frame_to_text(message) {
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
    ) -> Result<crate::providers::openai::responses_api::CompletionRequest, CompletionError> {
        let mut request = self.model.create_completion_request(completion_request)?;

        // WebSocket mode is always event-driven, so these HTTP/SSE-specific flags
        // are ignored by the provider and only add noise to the payload.
        request.stream = None;
        request.additional_parameters.background = None;

        if request.additional_parameters.previous_response_id.is_none() {
            request
                .additional_parameters
                .previous_response_id
                .clone_from(&self.previous_response_id);
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
                    return Err(provider_error_from_event(&error));
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

    /// Reads the next frame, honoring the session's event timeout.
    ///
    /// The timeout is [`crate::wasm_compat::timeout`], not `tokio::time`: this
    /// session is transport-agnostic and builds on wasm, where `tokio::time`
    /// does not function (and rig's tokio is built without its `time` feature
    /// regardless).
    async fn read_next_frame(
        &mut self,
    ) -> Result<http_client::Result<Option<Frame>>, CompletionError> {
        let Some(timeout_duration) = self.event_timeout else {
            return Ok(self.socket.recv().await);
        };

        match crate::wasm_compat::timeout(timeout_duration, self.socket.recv()).await {
            Ok(message) => Ok(message),
            Err(_) => Err(self.fail_session(event_timeout_error(timeout_duration))),
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
fn provider_error_from_event(error: &ResponsesWebSocketErrorEvent) -> CompletionError {
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

/// Lower one websocket frame onto the JSON payload the protocol carries.
///
/// `Ok(None)` is a frame with no protocol payload (a keepalive), which the
/// session skips; a close frame mid-turn is an error naming the peer's reason.
fn websocket_frame_to_text(frame: Frame) -> Result<Option<String>, CompletionError> {
    match frame {
        Frame::Text(text) => Ok(Some(text)),
        Frame::Binary(bytes) => String::from_utf8(bytes.to_vec())
            .map(Some)
            .map_err(|error| CompletionError::ResponseError(error.to_string())),
        Frame::Ping(_) | Frame::Pong(_) => Ok(None),
        Frame::Close(frame) => {
            let reason = frame
                .map(|frame| frame.reason)
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "without a close reason".to_string());
            Err(CompletionError::ProviderError(format!(
                "The OpenAI websocket connection closed {reason}"
            )))
        }
    }
}

/// Build the handshake request: the websocket URL derived from the client's
/// base URL, carrying the client's own auth headers.
///
/// The backend supplies the websocket-specific handshake headers; this only
/// states where to connect and who is connecting.
fn websocket_request(
    base_url: &str,
    headers: &http::HeaderMap,
) -> Result<http_client::Request<NoBody>, CompletionError> {
    let url = crate::ws_client::websocket_url(base_url, WEBSOCKET_PATH)
        .map_err(CompletionError::HttpError)?;

    let mut request = http_client::Request::builder()
        .method(http::Method::GET)
        .uri(url);
    if let Some(request_headers) = request.headers_mut() {
        *request_headers = headers.clone();
    }

    request.body(NoBody).map_err(|error| {
        CompletionError::ProviderError(format!("Failed to build OpenAI websocket request: {error}"))
    })
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
/// `{"error":{"code":"invalid_api_key",…}}`. Flattening that to a display
/// string — `"HTTP error: 401 Unauthorized"` — discards the status, the body
/// and the request id, leaving `provider_response_status()`,
/// `provider_response_body()` and `provider_request_id()` all `None`.
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
/// The backend's job is to report the rejection as
/// [`http_client::Error::non_success_with_details`]; reading OpenAI's own
/// request-id header off it is provider knowledge and belongs here.
///
/// Failures that never reached the provider — TLS, DNS, a protocol violation —
/// have no response to preserve and stay [`CompletionError::ProviderError`].
fn websocket_provider_error(error: http_client::Error) -> CompletionError {
    let Some(status) = error.non_success_status() else {
        return CompletionError::ProviderError(error.to_string());
    };

    let provider_request_id = REQUEST_ID_HEADER
        .and_then(|header| error.non_success_headers()?.get(header))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    // The body is the provider's own error envelope; an upgrade rejected
    // without one still carries its status, which is more than the string form
    // preserved.
    let body = error.non_success_body().unwrap_or_default().to_string();
    let headers = error.non_success_headers().cloned().map(Box::new);

    CompletionError::from_http_response_with_request_id(status, body, provider_request_id)
        .with_response_headers(headers)
}

/// OpenAI Responses websocket mode on an OpenAI client.
///
/// `H` is the client's HTTP transport, used for the completion model the
/// session wraps; the websocket itself comes from the `W` backend passed at
/// connect time. A caller using the bundled backend gets a no-argument
/// `responses_websocket(model)` from that crate's own extension trait, the
/// way `DefaultTransportClient` supplies `from_env()` over the bundled HTTP
/// transport. Bring this trait into scope with `use rig::prelude::*`.
pub trait ResponsesWebSocketExt<H> {
    /// Start configuring a websocket session for `model`.
    fn responses_websocket_builder(
        &self,
        model: impl Into<String>,
    ) -> ResponsesWebSocketSessionBuilder<H>;

    /// Open a websocket session for `model` over `backend`, with default
    /// options.
    fn responses_websocket_with<W>(
        &self,
        model: impl Into<String>,
        backend: &W,
    ) -> impl std::future::Future<Output = Result<ResponsesWebSocketSession<H>, CompletionError>>
    + WasmCompatSend
    where
        W: WebSocketClientExt + WasmCompatSync,
        Self: WasmCompatSync;
}

impl<H> ResponsesWebSocketExt<H> for OpenAIClient<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn responses_websocket_builder(
        &self,
        model: impl Into<String>,
    ) -> ResponsesWebSocketSessionBuilder<H> {
        use crate::client::CompletionClient as _;
        ResponsesWebSocketSessionBuilder::new(self.completion_model(model))
    }

    fn responses_websocket_with<W>(
        &self,
        model: impl Into<String>,
        backend: &W,
    ) -> impl std::future::Future<Output = Result<ResponsesWebSocketSession<H>, CompletionError>>
    + WasmCompatSend
    where
        W: WebSocketClientExt + WasmCompatSync,
        Self: WasmCompatSync,
    {
        let builder = self.responses_websocket_builder(model);
        async move { builder.connect_with(backend).await }
    }
}

/// Compile-time API contract: a session is `Send + Sync`, as it was before the
/// connection became an erased trait object. Hosts embed sessions in types that
/// carry those bounds, so losing one is a breaking change that no runtime test
/// would catch.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    fn probe<H>()
    where
        H: HttpClientExt + Clone + Send + Sync + 'static,
    {
        assert_send_sync::<ResponsesWebSocketSession<H>>();
        assert_send_sync::<ResponsesWebSocketSessionBuilder<H>>();
    }
    let _ = probe::<crate::http_client::BoxedHttpClient>;
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests;
