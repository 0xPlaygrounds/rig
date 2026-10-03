//! Stateful Responses WebSocket sessions over caller-supplied connections.
//! Sessions permit one in-flight turn and chain completed or incomplete response IDs.
//!
//! ```
//! use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketCreateOptions;
//! let options = ResponsesWebSocketCreateOptions::warmup();
//! assert_eq!(options.generate, Some(false));
//! ```

use crate::completion;
use crate::driver::Model;
use crate::error::{EncodeError, ProviderError};
use crate::http_client::{self, NoBody};
use crate::operation::{Completion, Turn};
use crate::providers::openai::responses_api::streaming::{
    ResponsesDecoder, ResponsesEvent, classify_responses_payload, is_lifecycle_event,
};
use crate::providers::openai::responses_api::wire::Responses;
use crate::streaming::Item;
use crate::wire::{Call, Flow, Mode, Operation, Reply, Shared, Wire, WireFrame};
use crate::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt, WebSocketConnection,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;

/// The websocket endpoint's path, appended to the client's configured base URL.
const WEBSOCKET_PATH: &str = "responses";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Request-ID header read from rejected WebSocket upgrades.
const REQUEST_ID_HEADER: Option<&'static str> =
    crate::providers::openai::wire::OPENAI.request_id_header;

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
    request: Value,
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

    fn status(&self) -> Option<&str> {
        self.response.get("status").and_then(Value::as_str)
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
    /// A response lifecycle event such as `response.created` or
    /// `response.completed`, with the response object it carries.
    Response {
        /// The event's `type`.
        kind: String,
        /// The response object, as the provider sent it.
        response: Value,
    },
    /// A streaming item or delta event such as `response.output_text.delta`,
    /// as the provider sent it.
    Item(Value),
    /// A protocol-level websocket error event.
    Error(ResponsesWebSocketErrorEvent),
    /// An optional `response.done` event emitted by OpenAI over WebSockets.
    Done(ResponsesWebSocketDoneEvent),
    /// Unrecognized event retained for [`Item::Unknown`] passthrough.
    Unknown(crate::streaming::UnknownPayload),
}

impl ResponsesWebSocketEvent {
    /// Returns the response ID when the event includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        match self {
            Self::Response { response, .. } => response.get("id").and_then(Value::as_str),
            Self::Done(done) => done.response_id(),
            Self::Item(_) | Self::Error(_) | Self::Unknown(_) => None,
        }
    }

    /// Returns `true` when this event ends the current in-flight websocket turn.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Response { kind, .. } => is_terminal_kind(kind),
            Self::Error(_) | Self::Done(_) => true,
            Self::Item(_) | Self::Unknown(_) => false,
        }
    }
}

/// Whether a lifecycle event of `kind` ends the turn.
fn is_terminal_kind(kind: &str) -> bool {
    matches!(
        kind,
        "response.completed" | "response.failed" | "response.incomplete"
    )
}

/// A builder for an OpenAI Responses WebSocket session.
///
/// The default builder applies a 30 second connection timeout and leaves the
/// per-event timeout disabled.
pub struct ResponsesWebSocketSessionBuilder {
    wire: Responses,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
}

impl ResponsesWebSocketSessionBuilder {
    pub(crate) fn new(wire: Responses) -> Self {
        Self {
            wire,
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

impl ResponsesWebSocketSessionBuilder {
    /// Open a session over the bundled tungstenite backend with the configured
    /// timeouts. Return handshake construction, transport, or provider errors.
    #[cfg(all(feature = "tungstenite", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "tungstenite")))]
    pub async fn connect(self) -> Result<ResponsesWebSocketSession, ProviderError> {
        self.connect_with(&rig_tungstenite::TungsteniteClient::new())
            .await
    }

    /// Open a session over `backend` with the configured timeouts.
    /// Return handshake construction, transport, or provider errors.
    pub async fn connect_with<W>(
        self,
        backend: &W,
    ) -> Result<ResponsesWebSocketSession, ProviderError>
    where
        W: WebSocketClientExt,
    {
        ResponsesWebSocketSession::connect_with_timeouts(
            backend,
            self.wire,
            self.connect_timeout,
            self.event_timeout,
        )
        .await
    }
}

/// Sequential Responses session with automatic response-ID chaining.
/// Completed and incomplete responses update the chain unless a request supplies
/// its own `previous_response_id`. Call [`Self::close`] to perform a close handshake.
pub struct ResponsesWebSocketSession {
    wire: Responses,
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    socket: BoxedWebSocketConnection,
    in_flight: bool,
    /// The fold of the turn in flight, opened when it was sent.
    turn: Option<Turn>,
    event_timeout: Option<Duration>,
    closed: bool,
    failed: bool,
}

impl ResponsesWebSocketSession {
    async fn connect_with_timeouts<W>(
        backend: &W,
        wire: Responses,
        connect_timeout: Option<Duration>,
        event_timeout: Option<Duration>,
    ) -> Result<Self, ProviderError>
    where
        W: WebSocketClientExt,
    {
        let request = websocket_request(&wire)?;
        let socket = backend
            .connect(request, ConnectOptions::new().with_timeout(connect_timeout))
            .await
            .map_err(websocket_provider_error)?;

        Ok(Self::from_connection(wire, socket, event_timeout))
    }

    /// Build a session over an already-open, authenticated connection.
    /// `event_timeout: None` waits indefinitely for each event.
    pub fn from_connection(
        wire: Responses,
        connection: BoxedWebSocketConnection,
        event_timeout: Option<Duration>,
    ) -> Self {
        Self {
            wire,
            previous_response_id: None,
            pending_done_response_id: None,
            socket: connection,
            in_flight: false,
            turn: None,
            event_timeout,
            closed: false,
            failed: false,
        }
    }

    /// Return the response ID retained for automatic chaining, if any.
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
    ) -> Result<(), ProviderError> {
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
    ) -> Result<(), ProviderError> {
        self.ensure_open()?;

        if self.in_flight {
            return Err(ProviderError::Provider(
                "An OpenAI websocket response is already in flight on this session".to_string(),
            ));
        }

        // A request that names no `previous_response_id` continues the
        // session's chain, so the history is shaped as a stored
        // continuation's.
        let mut completion_request = completion_request;
        if let Some(previous) = &self.previous_response_id {
            let params = completion_request
                .additional_params
                .get_or_insert_with(|| Value::Object(Map::new()));
            if let Some(params) = params.as_object_mut()
                && params
                    .get("previous_response_id")
                    .is_none_or(Value::is_null)
            {
                params.insert(
                    "previous_response_id".to_owned(),
                    Value::String(previous.clone()),
                );
            }
        }
        // The session sends without the driver, so it shapes the history
        // for the model the way the driver does.
        let describe = self.wire.describe();
        let completion_request = Completion::prepare(completion_request, &describe)?;
        let turn = Completion::fold(
            &completion_request,
            &mut Call::new(&describe, Mode::Streaming),
        );
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
        self.turn = Some(turn);

        Ok(())
    }

    /// Reads the next server event for the current in-flight turn.
    pub async fn next_event(&mut self) -> Result<ResponsesWebSocketEvent, ProviderError> {
        self.next_event_with_payload().await.map(|(event, _)| event)
    }

    /// Reads the next lifecycle event and retains its payload for content decoding
    /// by [`ResponsesDecoder`]. Returns the same session errors as [`Self::next_event`].
    async fn next_event_with_payload(
        &mut self,
    ) -> Result<(ResponsesWebSocketEvent, String), ProviderError> {
        self.ensure_open()?;

        if !self.in_flight {
            return Err(ProviderError::Provider(
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
                return Err(ProviderError::Provider(
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
            return Ok((event, payload));
        }
    }

    /// Sends a warmup turn (`generate: false`) and returns the resulting response ID.
    pub async fn warmup(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<String, ProviderError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::warmup(),
        )
        .await?;
        let response = self.wait_for_terminal_response().await?.0;
        Ok(response
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned())
    }

    /// Sends a completion turn and collects the final OpenAI response,
    /// normalized; its `raw` is the provider's own terminal response object.
    pub async fn completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, ProviderError> {
        self.send(completion_request).await?;
        Ok(self.wait_for_terminal_response().await?.1)
    }

    /// Closes the websocket connection.
    ///
    /// Call this when you are finished with the session so the websocket can
    /// terminate with a clean close handshake.
    pub async fn close(&mut self) -> Result<(), ProviderError> {
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
    ) -> Result<Value, ProviderError> {
        let mut request = self.wire.responses_request(completion_request, false)?;

        // WebSocket mode is always event-driven, so these HTTP/SSE-specific flags
        // are ignored by the provider and only add noise to the payload.
        if let Some(fields) = request.as_object_mut() {
            fields.shift_remove("stream");
            fields.shift_remove("background");
        }
        Ok(request)
    }

    /// Decode the turn's events as they arrive, and return the provider's
    /// completed or incomplete response with the reply they fold into.
    /// Transport, protocol, and decoder failures return an error. A terminal
    /// event without a response body is an error.
    async fn wait_for_terminal_response(
        &mut self,
    ) -> Result<(Value, completion::CompletionResponse), ProviderError> {
        let provider = self.wire.describe().name.to_owned();
        let Some(turn) = self.turn.take() else {
            return Err(ProviderError::Provider(
                "No OpenAI websocket response is currently in flight on this session".to_string(),
            ));
        };
        // The reply's state and its decoder live for this turn only.
        let reply = std::sync::Mutex::new(Shared::new(turn));
        let mut decoder = self.wire.decoder();
        loop {
            let (event, payload) = self.next_event_with_payload().await?;
            match event {
                ResponsesWebSocketEvent::Response { kind, response } => {
                    if !is_terminal_kind(&kind) {
                        feed(&mut decoder, &reply, payload)?;
                        continue;
                    }
                    // A failed turn is reported from its own envelope; only a
                    // completed or incomplete one reaches the decoder, whose
                    // end closes the turn.
                    let response = terminal_response_result(response)?;
                    let ended = feed(&mut decoder, &reply, payload)?;
                    let folded = fold_reply(reply, ended, &provider, &response)?;
                    return Ok((response, folded));
                }
                ResponsesWebSocketEvent::Done(done) => {
                    if done.response.is_object() {
                        // A failed turn is reported from its own envelope, as
                        // on the `response.failed` path.
                        let response = terminal_response_result(done.response)?;
                        // `response.done` carries the response object itself,
                        // which is the decoder's whole-body shape: hand it
                        // over as the frame it is.
                        let body = serde_json::to_string(&response)?;
                        let ended = feed(&mut decoder, &reply, body)?;
                        let folded = fold_reply(reply, ended, &provider, &response)?;
                        return Ok((response, folded));
                    }

                    let message = if let Some(response_id) = done.response_id() {
                        format!(
                            "OpenAI websocket turn ended with response.done before a terminal response body was available (response_id={response_id})"
                        )
                    } else {
                        "OpenAI websocket turn ended with response.done before a terminal response body was available"
                            .to_string()
                    };

                    return Err(ProviderError::Provider(message));
                }
                ResponsesWebSocketEvent::Error(error) => {
                    // Genuine provider error event: preserve the serialized payload
                    // (code + message + any extra fields) so provider_response_json()
                    // parses it, matching the response.failed path. No HTTP status on
                    // the websocket stream, so status: None.
                    return Err(provider_error_from_event(&error));
                }
                // Unknown frames keep their raw payload.
                ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => {
                    feed(&mut decoder, &reply, payload)?;
                }
            }
        }
    }

    fn update_state_for_event(&mut self, event: &ResponsesWebSocketEvent) {
        match event {
            ResponsesWebSocketEvent::Response { kind, response } => {
                let response_id = response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                match kind.as_str() {
                    // An incomplete turn still produced a response the next
                    // turn can chain from, so it keeps
                    // `previous_response_id` like a completed one.
                    "response.completed" | "response.incomplete" => {
                        self.previous_response_id.clone_from(&response_id);
                        self.pending_done_response_id = response_id;
                        self.in_flight = false;
                    }
                    "response.failed" => {
                        self.pending_done_response_id = response_id;
                        self.previous_response_id = None;
                        self.in_flight = false;
                    }
                    _ => {}
                }
            }
            ResponsesWebSocketEvent::Done(done) => {
                match done.status() {
                    Some("completed" | "incomplete") => {
                        if let Some(response_id) = done.response_id() {
                            self.previous_response_id = Some(response_id.to_string());
                        }
                    }
                    Some("in_progress" | "queued") | None => {}
                    Some(_) => {
                        self.previous_response_id = None;
                    }
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
        self.turn = None;
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

    fn ensure_open(&self) -> Result<(), ProviderError> {
        if self.closed || self.failed {
            return Err(ProviderError::Provider(
                "The OpenAI websocket session is closed".to_string(),
            ));
        }

        Ok(())
    }

    fn fail_session(&mut self, error: ProviderError) -> ProviderError {
        self.mark_failed();
        error
    }

    /// Read a frame with a WASM-compatible timeout.
    /// Timeout failure marks the session failed; transport results remain nested.
    async fn read_next_frame(
        &mut self,
    ) -> Result<http_client::Result<Option<Frame>>, ProviderError> {
        let Some(timeout_duration) = self.event_timeout else {
            return Ok(self.socket.recv().await);
        };

        match crate::wasm_compat::timeout(timeout_duration, self.socket.recv()).await {
            Ok(message) => Ok(message),
            Err(_) => Err(self.fail_session(event_timeout_error(timeout_duration))),
        }
    }
}

impl Drop for ResponsesWebSocketSession {
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

/// Feed one message to the turn's decoder. Returns whether it ended the
/// turn.
fn feed(
    decoder: &mut ResponsesDecoder,
    reply: &std::sync::Mutex<Shared<Completion>>,
    payload: String,
) -> Result<bool, ProviderError> {
    crate::driver::step(decoder, reply, WireFrame::Text(payload), None)
        .map(|step| matches!(step, Flow::Ended(_)))
}

/// Fold the turn's events and end into the normalized response, retaining
/// the terminal body as raw JSON.
fn fold_reply(
    reply: std::sync::Mutex<Shared<Completion>>,
    ended: bool,
    provider: &str,
    response: &Value,
) -> Result<completion::CompletionResponse, ProviderError> {
    let fed = if ended {
        Ok(())
    } else {
        Err(ProviderError::Truncated)
    };
    let reply_of = Reply {
        provider: provider.to_owned(),
        raw: response.clone(),
        // The websocket carries no reply headers past the handshake.
        provider_request_id: None,
    };
    crate::driver::settle(reply, fed, reply_of).outcome
}

fn terminal_response_result(response: Value) -> Result<Value, ProviderError> {
    match response.get("status").and_then(Value::as_str) {
        // An incomplete response (e.g. hitting `max_output_tokens`) is a
        // genuine terminal: the partial output and usage are kept, and the
        // decoder maps its status and reason to the finish.
        Some("completed" | "incomplete") | None => Ok(response),
        // Preserve provider error envelopes as reserialized JSON without an HTTP status.
        // Without an error object, return a local diagnostic instead.
        Some("failed") => match response.get("error").filter(|error| !error.is_null()) {
            Some(_) => Err(ProviderError::from_provider_body(response.to_string())),
            None => Err(ProviderError::Provider(response_error_message(
                "failed response",
            ))),
        },
        Some(other) => Err(ProviderError::Provider(format!(
            "OpenAI websocket response ended in state `{other}`"
        ))),
    }
}

fn response_error_message(fallback: &str) -> String {
    format!("OpenAI websocket returned a {fallback}")
}

/// Preserve an error event as reserialized provider JSON without an HTTP status.
/// Fall back to its display text if serialization fails.
fn provider_error_from_event(error: &ResponsesWebSocketErrorEvent) -> ProviderError {
    ProviderError::from_provider_body(
        serde_json::to_string(&error).unwrap_or_else(|_| error.to_string()),
    )
}

/// Decode WebSocket error and done events or delegate to Responses classification.
/// Return parsing and triage errors; preserve unknown payloads.
fn parse_server_event(payload: &str) -> Result<Option<ResponsesWebSocketEvent>, ProviderError> {
    #[derive(Deserialize)]
    struct EventType {
        #[serde(rename = "type")]
        kind: String,
    }

    let event_type = serde_json::from_str::<EventType>(payload)?;
    match event_type.kind.as_str() {
        "error" => serde_json::from_str(payload)
            .map(|e| Some(ResponsesWebSocketEvent::Error(e)))
            .map_err(ProviderError::from),
        "response.done" => serde_json::from_str(payload)
            .map(|d| Some(ResponsesWebSocketEvent::Done(d)))
            .map_err(ProviderError::from),
        _ => Ok(
            match crate::driver::triage(classify_responses_payload(payload))? {
                Item::Event(ResponsesEvent::Frame {
                    kind, mut frame, ..
                }) => {
                    if is_lifecycle_event(&kind) {
                        let response = frame
                            .as_object_mut()
                            .and_then(|frame| frame.shift_remove("response"))
                            .unwrap_or_else(|| Value::Object(Map::new()));
                        Some(ResponsesWebSocketEvent::Response { kind, response })
                    } else {
                        Some(ResponsesWebSocketEvent::Item(frame))
                    }
                }
                // Only the event types above reach here: a frame without a
                // `type` is not a websocket event.
                Item::Event(ResponsesEvent::Whole(value)) => {
                    Some(ResponsesWebSocketEvent::Unknown(value.into()))
                }
                Item::Event(ResponsesEvent::Failure(body)) => {
                    return Err(ProviderError::from_provider_body(body));
                }
                Item::Event(ResponsesEvent::Sentinel) => None,
                Item::Unknown(value) => Some(ResponsesWebSocketEvent::Unknown(value)),
            },
        ),
    }
}

/// Lower one websocket frame onto the JSON payload the protocol carries.
///
/// `Ok(None)` is a frame with no protocol payload (a keepalive), which the
/// session skips; a close frame mid-turn is an error naming the peer's reason.
fn websocket_frame_to_text(frame: Frame) -> Result<Option<String>, ProviderError> {
    match frame {
        Frame::Text(text) => Ok(Some(text)),
        Frame::Binary(bytes) => String::from_utf8(bytes.to_vec())
            .map(Some)
            .map_err(|error| ProviderError::Response(error.to_string())),
        Frame::Ping(_) | Frame::Pong(_) => Ok(None),
        Frame::Close(frame) => {
            let reason = frame
                .map(|frame| frame.reason)
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "without a close reason".to_string());
            Err(ProviderError::Provider(format!(
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
fn websocket_request(wire: &Responses) -> Result<http_client::Request<NoBody>, EncodeError> {
    let url = crate::ws_client::websocket_url(&wire.provider.base_url, WEBSOCKET_PATH)
        .map_err(EncodeError::request)?;

    let request = wire.provider.headers(
        http_client::Request::builder()
            .method(http::Method::GET)
            .uri(url),
    );

    request.body(NoBody).map_err(|error| {
        EncodeError::request(format!("Failed to build OpenAI websocket request: {error}"))
    })
}

fn event_timeout_error(timeout: Duration) -> ProviderError {
    ProviderError::Provider(format!(
        "Timed out waiting for the next OpenAI websocket event after {timeout:?}"
    ))
}

/// Convert transport errors, retaining rejected-upgrade status, body, and request ID.
/// Failures without a provider response retain transport error classification.
fn websocket_provider_error(error: http_client::Error) -> ProviderError {
    let provider_request_id = error.non_success_headers().and_then(|headers| {
        crate::driver::http_transport::request_id_from(headers, REQUEST_ID_HEADER)
    });
    ProviderError::from_transport_error(error).with_provider_request_id(provider_request_id)
}

impl<T> Model<Responses, T> {
    /// Start configuring a Responses WebSocket session for this model. Open
    /// it with [`connect_with`](ResponsesWebSocketSessionBuilder::connect_with)
    /// and a backend, or, under the `tungstenite` feature, with
    /// [`connect`](ResponsesWebSocketSessionBuilder::connect).
    pub fn responses_websocket(&self) -> ResponsesWebSocketSessionBuilder {
        ResponsesWebSocketSessionBuilder::new(self.wire.clone())
    }
}

/// Native sessions and builders satisfy Send and Sync.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ResponsesWebSocketSession>();
    assert_send_sync::<ResponsesWebSocketSessionBuilder>();
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
