//! OpenAI's Responses API in WebSocket mode, as a wire and a transport. A
//! [`ResponsesSocket`] encodes each request as a `response.create` event,
//! and a [`ResponsesWebSocket`] sends it over one connection and delivers
//! that turn's events as frames, so a turn is an ordinary reply:
//! [`Model::stream`] yields its events and [`Model::call`] folds them.
//!
//! [`Model::responses_websocket`] configures the connection and opens it as a
//! model. Turns queue on the connection, and nothing is chained unless the
//! caller asks: a caller that sends only the new input turns on chaining, one
//! that sends the whole conversation leaves it off.
//!
//! ```no_run
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # async fn run(backend: impl rig_core::ws_client::WebSocketClientExt) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.responses(openai::GPT_5_2);
//! let socket = model.responses_websocket().connect_with(&backend).await?;
//! let response = socket.call("Hello").await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use crate::completion;
use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::http_client::{self, NoBody};
use crate::observe::{AdapterErrorBoundary, AdapterSlot};
use crate::operation::Completion;
use crate::providers::openai::responses_api::streaming::{
    ResponseChunk, ResponseChunkKind, ResponsesDecoder, StreamingCompletionChunk,
    classify_responses_frame,
};
use crate::providers::openai::responses_api::wire::Responses;
use crate::providers::openai::responses_api::{CompletionResponse, ResponseStatus};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Descriptor, Mode, Wire, WireEvent, WireFrame};
use crate::ws_client::{BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::time::Duration;

/// The websocket endpoint's path, appended to the client's configured base URL.
const WEBSOCKET_PATH: &str = "responses";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Request-ID header read from rejected WebSocket upgrades.
const REQUEST_ID_HEADER: Option<&'static str> =
    crate::providers::openai::wire::OPENAI.request_id_header;

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

/// The Responses API over WebSocket mode: the [`Responses`] request, sent as
/// a `response.create` event, and the [`Responses`] decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesSocket {
    /// The wire whose requests and decoder this one uses.
    pub responses: Responses,
    /// `Some(false)` prepares each request's state without generating
    /// output: a warmup turn.
    pub generate: Option<bool>,
}

impl ResponsesSocket {
    /// Send `responses`' requests over WebSocket mode.
    pub fn new(responses: Responses) -> Self {
        Self {
            responses,
            generate: None,
        }
    }

    /// Prepare each request's state without generating output.
    pub fn warmup(mut self) -> Self {
        self.generate = Some(false);
        self
    }
}

/// The `response.create` event a [`ResponsesSocket`] sends for one turn.
/// It serializes to the event itself.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseCreate {
    #[serde(rename = "type")]
    kind: ResponseCreateKind,
    #[serde(flatten)]
    request: super::CompletionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate: Option<bool>,
    /// The endpoint template observation groups the turn under.
    #[serde(skip)]
    route: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize)]
enum ResponseCreateKind {
    #[serde(rename = "response.create")]
    ResponseCreate,
}

impl ResponseCreate {
    /// The response this turn continues, when it names one.
    pub fn previous_response_id(&self) -> Option<&str> {
        self.request
            .additional_parameters
            .previous_response_id
            .as_deref()
    }

    /// Whether the turn generates output.
    pub fn generate(&self) -> Option<bool> {
        self.generate
    }
}

impl Wire for ResponsesSocket {
    type Op = Completion;
    type Payload = ResponseCreate;
    type Frame = WireFrame;
    type Decoder<'id> = ResponsesDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        self.responses.describe()
    }

    /// WebSocket mode is always event-driven, so `mode` changes nothing.
    fn encode(
        &self,
        request: completion::CompletionRequest,
        _mode: Mode,
    ) -> Result<ResponseCreate, EncodeError> {
        let (request, issuers) = crate::providers::openai::wire::scope_reasoning(
            &self.responses.provider.dialect,
            &self.responses.model,
            request,
        )?;
        // Not streamed: `stream` is an HTTP flag. So is `background`.
        let mut request = self.responses.responses_request(request, issuers, false)?;
        request.additional_parameters.background = None;
        Ok(ResponseCreate {
            kind: ResponseCreateKind::ResponseCreate,
            request,
            generate: self.generate,
            route: self.responses.provider.dialect.quirks.responses.path,
        })
    }

    fn decoder<'id>(&self) -> ResponsesDecoder<'id> {
        self.responses.decoder()
    }
}

/// How long reading a dropped turn to its end may take by default.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// One WebSocket-mode connection as a [`Transport`] for [`ResponsesSocket`].
///
/// Clones share the connection, which carries one turn at a time: a turn
/// sent while another is in flight waits for it. A turn whose stream was
/// dropped before its end is read to its end, within the drain timeout,
/// before the next turn is sent. A timeout, a transport failure or the peer
/// closing fails the connection, and later turns fail.
#[derive(Clone)]
pub struct ResponsesWebSocket {
    session: Arc<futures::lock::Mutex<Session>>,
    /// The credentials the handshake carried, scrubbed from observations.
    secrets: Arc<[String]>,
    event_timeout: Option<Duration>,
    drain_timeout: Duration,
    chaining: bool,
}

impl std::fmt::Debug for ResponsesWebSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesWebSocket")
            .field("event_timeout", &self.event_timeout)
            .field("drain_timeout", &self.drain_timeout)
            .field("chaining", &self.chaining)
            .finish_non_exhaustive()
    }
}

/// The connection and what it knows across turns.
struct Session {
    socket: BoxedWebSocketConnection,
    chain: Chain,
    /// A turn was sent and its end not yet read: its stream was dropped.
    dirty: bool,
    closed: bool,
    failed: bool,
}

impl ResponsesWebSocket {
    /// A transport over an open, authenticated connection, with no event
    /// timeout, the default drain timeout, and no chaining.
    pub fn from_connection(connection: BoxedWebSocketConnection) -> Self {
        Self {
            session: Arc::new(futures::lock::Mutex::new(Session {
                socket: connection,
                chain: Chain::default(),
                dirty: false,
                closed: false,
                failed: false,
            })),
            secrets: Arc::from([]),
            event_timeout: None,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
            chaining: false,
        }
    }

    /// Scrub `secrets`, the credentials the handshake carried, from the
    /// observations of every turn.
    pub(crate) fn scrubbing(mut self, secrets: Arc<[String]>) -> Self {
        self.secrets = secrets;
        self
    }

    /// Fail a turn when no event arrives for `timeout`. `None` waits
    /// indefinitely.
    pub fn event_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.event_timeout = timeout;
        self
    }

    /// Fail the connection when reading a dropped turn to its end takes
    /// longer than `timeout`.
    pub fn drain_timeout(mut self, timeout: Duration) -> Self {
        self.drain_timeout = timeout;
        self
    }

    /// Chain each turn to the response the last completed or incomplete
    /// turn produced, unless its request names its own
    /// `previous_response_id`. For callers that send only the new input;
    /// a caller that sends the whole conversation must not chain.
    pub fn chaining(mut self) -> Self {
        self.chaining = true;
        self
    }

    /// The response the last completed or incomplete turn produced, until
    /// a turn fails or [`Self::clear_chain`].
    pub async fn last_response_id(&self) -> Option<String> {
        self.session.lock().await.chain.previous_response_id.clone()
    }

    /// Start the next chained turn on a fresh chain.
    pub async fn clear_chain(&self) {
        self.session.lock().await.chain.previous_response_id = None;
    }

    /// Close the connection with a close handshake, after the turn in
    /// flight. Later turns fail. Closing again does nothing.
    pub async fn close(&self) -> Result<(), ProviderError> {
        let mut session = self.session.lock().await;
        if session.closed {
            return Ok(());
        }
        session.closed = true;
        session
            .socket
            .close(None)
            .await
            .map_err(websocket_provider_error)
    }
}

impl Transport<ResponsesSocket> for ResponsesWebSocket {
    fn send(&self, payload: ResponseCreate, exchange: Exchange) -> Opening<WireFrame> {
        let session = Arc::clone(&self.session);
        let secrets = Arc::clone(&self.secrets);
        let timeouts = (self.event_timeout, self.drain_timeout);
        let chaining = self.chaining;
        let observation = exchange.observation;
        Opening::new(async move {
            // Turns queue here: the connection carries one at a time.
            let mut session = session.lock_owned().await;
            // The attempt begins when the turn is next, never while queued
            // or unpolled.
            let slot = observation.map(|context| {
                let slot = AdapterSlot::default();
                slot.install(context.attempt_with_secrets(
                    &http::Method::GET,
                    payload.route,
                    secrets,
                ));
                slot
            });
            let mut opened = match session
                .open(payload, chaining, timeouts, slot.as_ref())
                .await
            {
                Ok(()) => {
                    // The turn's answer is the connection's upgrade.
                    if let Some(slot) = &slot {
                        slot.response(http::StatusCode::SWITCHING_PROTOCOLS);
                    }
                    Opened::new(turn(session, timeouts.0, slot.clone()))
                }
                Err(error) => Opened::failed(error),
            };
            opened.slot = slot;
            Ok(opened)
        })
    }
}

/// One turn's frames, ending at its terminal event. Each payload the turn
/// reads is projected onto its attempt. The connection is released before
/// the last item is yielded: the driver stops reading at the reply's end,
/// so code after that yield would not run.
fn turn(
    mut session: futures::lock::OwnedMutexGuard<Session>,
    event_timeout: Option<Duration>,
    slot: Option<AdapterSlot>,
) -> impl futures::Stream<Item = Result<WireFrame, ProviderError>> + WasmCompatSend + 'static {
    async_stream::stream! {
        loop {
            let text = match session.next_text(event_timeout, slot.as_ref()).await {
                Ok(text) => text,
                Err(error) => {
                    drop(session);
                    yield Err(error);
                    return;
                }
            };
            let lifecycle = session.chain.read(&text);
            if !matches!(lifecycle, Lifecycle::Skip)
                && let Some(slot) = &slot
            {
                slot.project(|sink| super::wire::project_payload(text.as_bytes(), sink));
            }
            match lifecycle {
                Lifecycle::Skip => {}
                Lifecycle::Frame => yield Ok(WireFrame::Text(text)),
                Lifecycle::Last(lowered) => {
                    session.dirty = false;
                    drop(session);
                    yield Ok(WireFrame::Text(lowered.unwrap_or(text)));
                    return;
                }
                Lifecycle::Fail(error) => {
                    session.dirty = false;
                    drop(session);
                    yield Err(error);
                    return;
                }
            }
        }
    }
}

impl Session {
    fn ensure_open(&self) -> Result<(), ProviderError> {
        if self.closed || self.failed {
            return Err(ProviderError::Provider(
                "The OpenAI websocket session is closed".to_string(),
            ));
        }
        Ok(())
    }

    fn mark_failed(&mut self) {
        self.chain = Chain::default();
        self.dirty = false;
        self.failed = true;
    }

    /// Ready the connection for a turn and send its event: drain a dropped
    /// turn, chain when asked, and write.
    async fn open(
        &mut self,
        mut payload: ResponseCreate,
        chaining: bool,
        (event_timeout, drain_timeout): (Option<Duration>, Duration),
        slot: Option<&AdapterSlot>,
    ) -> Result<(), ProviderError> {
        self.ensure_open()?;
        if self.dirty {
            self.drain(event_timeout, drain_timeout, slot).await?;
        }
        let chained = &mut payload.request.additional_parameters.previous_response_id;
        if chaining && chained.is_none() {
            chained.clone_from(&self.chain.previous_response_id);
        }
        let text = serde_json::to_string(&payload)?;
        // Set before the write: a write cut off midway may still have
        // reached the provider, whose reply must not reach the next turn.
        self.dirty = true;
        self.chain.streamed = false;
        if let Err(error) = self.socket.send(Frame::Text(text)).await {
            self.mark_failed();
            transport_failed(slot);
            return Err(websocket_provider_error(error));
        }
        Ok(())
    }

    /// Read the dropped turn to its end, within `drain_timeout`.
    async fn drain(
        &mut self,
        event_timeout: Option<Duration>,
        drain_timeout: Duration,
        slot: Option<&AdapterSlot>,
    ) -> Result<(), ProviderError> {
        let drained = crate::wasm_compat::timeout(drain_timeout, async {
            loop {
                let text = self.next_text(event_timeout, slot).await?;
                if let Lifecycle::Last(_) | Lifecycle::Fail(_) = self.chain.read(&text) {
                    return Ok(());
                }
            }
        })
        .await;
        match drained {
            Ok(Ok(())) => {
                self.dirty = false;
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(_) => {
                self.mark_failed();
                transport_failed(slot);
                Err(ProviderError::Provider(format!(
                    "Timed out reading a dropped OpenAI websocket turn to its end after \
                     {drain_timeout:?}"
                )))
            }
        }
    }

    /// The next text payload. A timeout, a transport failure, a close frame
    /// or the peer hanging up fails the connection.
    async fn next_text(
        &mut self,
        event_timeout: Option<Duration>,
        slot: Option<&AdapterSlot>,
    ) -> Result<String, ProviderError> {
        let read = self.read_text(event_timeout).await;
        if read.is_err() {
            transport_failed(slot);
        }
        read
    }

    async fn read_text(
        &mut self,
        event_timeout: Option<Duration>,
    ) -> Result<String, ProviderError> {
        loop {
            let received = match event_timeout {
                None => self.socket.recv().await,
                Some(timeout) => {
                    match crate::wasm_compat::timeout(timeout, self.socket.recv()).await {
                        Ok(received) => received,
                        Err(_) => {
                            self.mark_failed();
                            return Err(event_timeout_error(timeout));
                        }
                    }
                }
            };
            let frame = match received {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    self.mark_failed();
                    self.closed = true;
                    return Err(ProviderError::Provider(
                        "The OpenAI websocket connection closed before the turn finished"
                            .to_string(),
                    ));
                }
                Err(error) => {
                    self.mark_failed();
                    return Err(websocket_provider_error(error));
                }
            };
            match websocket_frame_to_text(frame) {
                Ok(Some(text)) => return Ok(text),
                Ok(None) => {}
                Err(error) => {
                    self.mark_failed();
                    return Err(error);
                }
            }
        }
    }
}

/// What one server event means for the turn in flight.
#[derive(Debug)]
enum Lifecycle {
    /// The previous turn's trailing `response.done`.
    Skip,
    /// A frame of the turn, for the decoder as it arrived.
    Frame,
    /// The turn's last frame, for the decoder: as it arrived, or lowered to
    /// the body it carries.
    Last(Option<String>),
    /// The turn failed.
    Fail(ProviderError),
}

/// The response chain across turns, the trailing `response.done` to skip,
/// and whether the turn in flight has streamed any output item.
#[derive(Debug, Default)]
struct Chain {
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    streamed: bool,
}

/// The fields of a server event the connection reads: its type, and the
/// response it carries.
#[derive(Deserialize)]
struct Probe {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    response: Option<Value>,
}

impl Probe {
    fn response_id(&self) -> Option<String> {
        self.response
            .as_ref()?
            .get("id")?
            .as_str()
            .map(str::to_owned)
    }

    fn status(&self) -> Option<&str> {
        self.response.as_ref()?.get("status")?.as_str()
    }
}

impl Chain {
    /// Read one server event by its type: update the chain and say what the
    /// event is for the turn. Content is left to the decoder.
    fn read(&mut self, text: &str) -> Lifecycle {
        let lifecycle = self.lifecycle(text);
        if !matches!(lifecycle, Lifecycle::Frame | Lifecycle::Skip) {
            self.streamed = false;
        }
        lifecycle
    }

    fn lifecycle(&mut self, text: &str) -> Lifecycle {
        let Ok(probe) = serde_json::from_str::<Probe>(text) else {
            return Lifecycle::Frame;
        };
        match probe.kind.as_str() {
            "response.completed" | "response.incomplete" => {
                self.previous_response_id = probe.response_id();
                self.pending_done_response_id = probe.response_id();
                match probe.status() {
                    Some("completed" | "incomplete") | None if self.streamed => {
                        Lifecycle::Last(None)
                    }
                    // A turn that streamed no item states its output only in
                    // the terminal body: the decoder reads that body whole.
                    Some("completed" | "incomplete") | None => match probe
                        .response
                        .map(serde_json::from_value::<CompletionResponse>)
                    {
                        Some(Ok(response)) => match serde_json::to_string(&response) {
                            Ok(body) => Lifecycle::Last(Some(body)),
                            Err(error) => Lifecycle::Fail(error.into()),
                        },
                        // A body that does not decode reaches the decoder
                        // as it arrived, which reports it.
                        Some(Err(_)) | None => Lifecycle::Last(None),
                    },
                    Some(_) => Lifecycle::Fail(terminal_failure(&probe)),
                }
            }
            "response.failed" => {
                self.previous_response_id = None;
                self.pending_done_response_id = probe.response_id();
                Lifecycle::Fail(terminal_failure(&probe))
            }
            "error" => {
                self.previous_response_id = None;
                self.pending_done_response_id = None;
                match serde_json::from_str::<ResponsesWebSocketErrorEvent>(text) {
                    Ok(error) => Lifecycle::Fail(provider_error_from_event(&error)),
                    Err(error) => Lifecycle::Fail(error.into()),
                }
            }
            "response.done" => {
                let id = probe.response_id();
                if id.is_some() && self.pending_done_response_id == id {
                    self.pending_done_response_id = None;
                    return Lifecycle::Skip;
                }
                self.pending_done_response_id = None;
                match probe.status() {
                    Some("completed" | "incomplete") => self.previous_response_id = id,
                    Some("in_progress" | "queued") | None => {}
                    Some(_) => self.previous_response_id = None,
                }
                let body = probe
                    .response
                    .clone()
                    .and_then(|body| serde_json::from_value::<CompletionResponse>(body).ok());
                // `response.done` carries the response object itself. After
                // streamed items it ends the turn as `response.completed`
                // would; otherwise the decoder reads it whole.
                match body.map(terminal_response_result) {
                    Some(Ok(response)) => {
                        let lowered = if self.streamed {
                            let kind = match response.status {
                                ResponseStatus::Incomplete => ResponseChunkKind::ResponseIncomplete,
                                _ => ResponseChunkKind::ResponseCompleted,
                            };
                            serde_json::to_string(&ResponseChunk {
                                kind,
                                response,
                                sequence_number: 0,
                            })
                        } else {
                            serde_json::to_string(&response)
                        };
                        match lowered {
                            Ok(frame) => Lifecycle::Last(Some(frame)),
                            Err(error) => Lifecycle::Fail(error.into()),
                        }
                    }
                    Some(Err(error)) => Lifecycle::Fail(error),
                    None => Lifecycle::Fail(ProviderError::Provider(
                        "OpenAI websocket turn ended with response.done before a terminal \
                         response body was available"
                            .to_string(),
                    )),
                }
            }
            _ => {
                if matches!(
                    classify_responses_frame(text),
                    WireEvent::Known(StreamingCompletionChunk::Delta(_))
                ) {
                    self.streamed = true;
                }
                Lifecycle::Frame
            }
        }
    }
}

/// Mark the attempt's failure as the connection's, not the provider's.
fn transport_failed(slot: Option<&AdapterSlot>) {
    if let Some(slot) = slot {
        slot.error_boundary(AdapterErrorBoundary::Transport);
    }
}

/// The error a failed, or unfinished, terminal response reports.
fn terminal_failure(probe: &Probe) -> ProviderError {
    match probe
        .response
        .clone()
        .map(serde_json::from_value::<CompletionResponse>)
    {
        Some(Ok(response)) => match terminal_response_result(response) {
            Ok(_) => ProviderError::Provider(
                "OpenAI websocket terminal event disagrees with its response status".to_string(),
            ),
            Err(error) => error,
        },
        Some(Err(error)) => error.into(),
        None => ProviderError::Provider(
            "OpenAI websocket terminal event carried no response".to_string(),
        ),
    }
}

/// Configures the WebSocket-mode connection of one model, and opens it as a
/// [`Model`] of [`ResponsesSocket`] over [`ResponsesWebSocket`].
///
/// By default the connection times out after 30 seconds, events have no
/// timeout, a dropped turn is drained for up to 30 seconds, and nothing is
/// chained.
pub struct ResponsesWebSocketBuilder {
    wire: Responses,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
    drain_timeout: Duration,
    chaining: bool,
}

impl ResponsesWebSocketBuilder {
    pub(crate) fn new(wire: Responses) -> Self {
        Self {
            wire,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
            chaining: false,
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

    /// Fails a turn when no event arrives for `timeout`.
    #[must_use]
    pub fn event_timeout(mut self, timeout: Duration) -> Self {
        self.event_timeout = Some(timeout);
        self
    }

    /// Disables the event timeout.
    #[must_use]
    pub fn without_event_timeout(mut self) -> Self {
        self.event_timeout = None;
        self
    }

    /// Sets how long reading a dropped turn to its end may take before the
    /// connection fails. See [`ResponsesWebSocket::drain_timeout`].
    #[must_use]
    pub fn drain_timeout(mut self, timeout: Duration) -> Self {
        self.drain_timeout = timeout;
        self
    }

    /// Chains each turn to the last response. See
    /// [`ResponsesWebSocket::chaining`].
    #[must_use]
    pub fn chaining(mut self) -> Self {
        self.chaining = true;
        self
    }

    /// Connects over the bundled tungstenite backend. Returns handshake
    /// construction, transport, or provider errors.
    #[cfg(all(feature = "tungstenite", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "tungstenite")))]
    pub async fn connect(
        self,
    ) -> Result<Model<ResponsesSocket, ResponsesWebSocket>, ProviderError> {
        self.connect_with(&rig_tungstenite::TungsteniteClient::new())
            .await
    }

    /// Connects over `backend`. Returns handshake construction, transport,
    /// or provider errors; a rejected upgrade keeps its status, headers and
    /// body.
    pub async fn connect_with<W>(
        self,
        backend: &W,
    ) -> Result<Model<ResponsesSocket, ResponsesWebSocket>, ProviderError>
    where
        W: WebSocketClientExt,
    {
        let request = websocket_request(&self.wire)?;
        let secrets = crate::observe::handshake_secrets(&request);
        let socket = backend
            .connect(
                request,
                ConnectOptions::new().with_timeout(self.connect_timeout),
            )
            .await
            .map_err(websocket_provider_error)?;
        let mut transport = ResponsesWebSocket::from_connection(socket)
            .scrubbing(secrets)
            .event_timeout(self.event_timeout)
            .drain_timeout(self.drain_timeout);
        if self.chaining {
            transport = transport.chaining();
        }
        Ok(Model::new(ResponsesSocket::new(self.wire), transport))
    }
}

fn terminal_response_result(
    response: CompletionResponse,
) -> Result<CompletionResponse, ProviderError> {
    match response.status {
        ResponseStatus::Completed => Ok(response),
        // Preserve provider error envelopes as reserialized JSON without an HTTP status.
        // Without an error object, return a local diagnostic instead.
        ResponseStatus::Failed => match response.error.as_ref() {
            Some(error) => Err(ProviderError::from_provider_body(
                serde_json::to_string(&response).unwrap_or_else(|_| error.message.clone()),
            )),
            None => Err(ProviderError::Provider(response_error_message(
                "failed response",
            ))),
        },
        // An incomplete response (e.g. hitting `max_output_tokens`) is a
        // genuine terminal: the partial output and usage are kept, and the
        // normalization path maps the status/incomplete_details to a finish
        // reason via `map_finish_reason`, matching the unary and SSE paths.
        ResponseStatus::Incomplete => Ok(response),
        other => Err(ProviderError::Provider(format!(
            "OpenAI websocket response ended in state {other:?}"
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
        crate::providers::internal::request_id_from_headers(headers, REQUEST_ID_HEADER)
    });
    ProviderError::from_transport_error(error).with_provider_request_id(provider_request_id)
}

impl<T> Model<Responses, T> {
    /// Configures a WebSocket-mode connection for this model's wire. Open it
    /// with [`connect_with`](ResponsesWebSocketBuilder::connect_with) and a
    /// backend, or, under the `tungstenite` feature, with
    /// [`connect`](ResponsesWebSocketBuilder::connect).
    pub fn responses_websocket(&self) -> ResponsesWebSocketBuilder {
        ResponsesWebSocketBuilder::new(self.wire.clone())
    }
}

/// The transport and the builder satisfy Send and Sync on native targets.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ResponsesWebSocket>();
    assert_send_sync::<ResponsesWebSocketBuilder>();
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
