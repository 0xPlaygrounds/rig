//! OpenAI's Responses API in WebSocket mode, as a wire and a transport. A
//! [`ResponsesSocket`] encodes each request as a `response.create` event, and
//! a [`ResponsesWebSocket`] sends it over one connection and delivers that
//! turn's events as frames, so a turn is an ordinary reply:
//! [`Model::stream`] yields its events and [`Model::call`] folds them.
//!
//! [`Model::responses_websocket`] configures the connection and opens it as a
//! model. Turns queue on the connection. Nothing is chained unless the caller
//! asks: a caller that sends only the new input turns on chaining, one that
//! sends the whole conversation leaves it off.
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

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::completion::options::FinalBody;
use crate::completion::{self, Accepts, LaterSystem, Media, Pairing, ReplayTarget};
use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::http_client::{self, NoBody};
use crate::message::{Api, AssistantContent, Origin};
use crate::observe::{AdapterErrorBoundary, AdapterSlot};
use crate::operation::Completion;
use crate::providers::openai::responses_api::streaming::{ResponsesDecoder, document};
use crate::providers::openai::responses_api::wire::{Responses, project_payload};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Descriptor, Mode, Wire, WireFrame};
use crate::ws_client::{BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt};

/// The websocket endpoint's path, appended to the client's configured base URL.
const WEBSOCKET_PATH: &str = "responses";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How long reading a dropped turn to its end may take by default.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Request-ID header read from rejected WebSocket upgrades.
const REQUEST_ID_HEADER: Option<&'static str> =
    crate::providers::openai::wire::OPENAI.request_id_header;

/// The Responses API over WebSocket mode: the [`Responses`] request, sent as
/// a `response.create` event, and the [`Responses`] decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesSocket {
    /// The wire whose requests and decoder this one uses.
    pub responses: Responses,
    /// `Some(false)` prepares each request's state without generating
    /// output: a warmup turn.
    #[serde(default)]
    pub generate: Option<bool>,
    /// Whether each turn continues the response the connection's last
    /// completed or incomplete turn produced. See [`Self::chaining`].
    #[serde(default)]
    pub chaining: bool,
}

impl ResponsesSocket {
    /// Send `responses`' requests over WebSocket mode, unchained.
    pub fn new(responses: Responses) -> Self {
        Self {
            responses,
            generate: None,
            chaining: false,
        }
    }

    /// Prepare each request's state without generating output.
    #[must_use]
    pub fn warmup(mut self) -> Self {
        self.generate = Some(false);
        self
    }

    /// Continue the connection's last completed or incomplete response with
    /// each turn, unless its request names its own `previous_response_id`.
    /// For callers that send only the new input, whose history is then
    /// shaped as a stored continuation's. A caller that sends the whole
    /// conversation must not chain.
    #[must_use]
    pub fn chaining(mut self) -> Self {
        self.chaining = true;
        self
    }
}

/// The `response.create` event a [`ResponsesSocket`] sends for one turn:
/// opaque, it serializes to the event itself.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseCreate {
    #[serde(rename = "type")]
    kind: ResponseCreateKind,
    #[serde(flatten)]
    body: FinalBody,
    /// The response the connection chained this turn to.
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate: Option<bool>,
    /// Whether the transport chains the turn.
    #[serde(skip)]
    chain: bool,
    /// The endpoint template observation groups the turn under.
    #[serde(skip)]
    route: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize)]
enum ResponseCreateKind {
    #[serde(rename = "response.create")]
    ResponseCreate,
}

impl Wire for ResponsesSocket {
    type Op = Completion;
    type Payload = ResponseCreate;
    type Frame = WireFrame;
    type Decoder<'id> = ResponsesDecoder;
    type Reassembler = document::Response;

    fn describe(&self) -> Descriptor<'_> {
        self.responses.describe().replay(self)
    }

    /// WebSocket mode is always event-driven, so `mode` changes nothing.
    fn encode(
        &self,
        request: completion::CompletionRequest,
        _mode: Mode,
    ) -> Result<ResponseCreate, EncodeError> {
        let body = self
            .responses
            .responses_request(&request, super::Delivery::WebSocket)?;
        Ok(ResponseCreate {
            kind: ResponseCreateKind::ResponseCreate,
            body,
            previous_response_id: None,
            generate: self.generate,
            chain: self.chaining,
            route: self.responses.provider.dialect.quirks.responses.path,
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        self.responses.decoder()
    }
}

/// The [`Responses`] replay rules, with a chaining wire's turns continuing
/// stored state.
impl ReplayTarget for ResponsesSocket {
    fn api(&self) -> Api {
        self.responses.api()
    }

    fn provider(&self) -> &str {
        ReplayTarget::provider(&self.responses)
    }

    fn model(&self) -> &str {
        ReplayTarget::model(&self.responses)
    }

    fn accepts(&self, model: &str) -> Accepts {
        self.responses.accepts(model)
    }

    fn map_options(
        &self,
        request: &completion::CompletionRequest,
        fields: completion::options::OptionFields<'_>,
    ) -> completion::options::OptionMap {
        self.responses.map_options(request, fields)
    }

    fn facts(&self) -> Option<&crate::catalog::ModelFacts> {
        self.responses.facts()
    }

    fn encodes(&self, model: &str, media: Media<'_>) -> bool {
        self.responses.encodes(model, media)
    }

    fn normalize_tool_call_id(&self, id: &str, model: &str, source: Option<&Origin>) -> String {
        self.responses.normalize_tool_call_id(id, model, source)
    }

    fn continues_stored(&self, request: &completion::CompletionRequest) -> bool {
        self.chaining || self.responses.continues_stored(request)
    }

    fn declares_tools(&self, request: &completion::CompletionRequest) -> bool {
        self.responses.declares_tools(request)
    }

    fn identity(&self, item: &Value) -> Map<String, Value> {
        self.responses.identity(item)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        self.responses.call_id_slot()
    }

    fn states_finish_reason(&self) -> bool {
        self.responses.states_finish_reason()
    }

    fn binds_context(&self, model: &str) -> bool {
        self.responses.binds_context(model)
    }

    fn drops_unbound_items(&self, request: &completion::CompletionRequest) -> bool {
        self.responses.drops_unbound_items(request)
    }

    fn takes_documents(&self) -> bool {
        self.responses.takes_documents()
    }

    fn needs_next(&self, item: &Value) -> bool {
        self.responses.needs_next(item)
    }

    fn hosted_pair(&self, item: &Value) -> Option<(Pairing, String)> {
        self.responses.hosted_pair(item)
    }

    fn starts_with_user(&self) -> bool {
        self.responses.starts_with_user()
    }

    fn later_system(&self, model: &str) -> LaterSystem {
        self.responses.later_system(model)
    }

    fn alternates_roles(&self) -> bool {
        self.responses.alternates_roles()
    }

    fn hosted_needs_tools(&self) -> bool {
        self.responses.hosted_needs_tools()
    }

    fn result_parts(&self, model: &str) -> bool {
        self.responses.result_parts(model)
    }

    fn sends_alone(&self, block: &AssistantContent) -> bool {
        self.responses.sends_alone(block)
    }
}

/// One WebSocket-mode connection as a [`Transport`] for [`ResponsesSocket`].
///
/// Clones share the connection, which carries one turn at a time: a turn
/// sent while another is in flight waits for it, in no guaranteed order, so
/// a chaining caller awaits each turn before sending the next. A stream
/// holds the connection until its turn ends, so the task holding a half-read
/// stream must not await another turn or [`Self::close`] before it reads or
/// drops it. [`Self::last_response_id`] and [`Self::clear_chain`] never wait.
///
/// A turn whose stream was dropped before its end is read to its end, within
/// the drain timeout, before the next turn is sent; past it the connection
/// fails. So is a turn that failed on an `error` after its response opened,
/// or on an event for another response. A timeout, a transport failure or
/// the peer closing fails the connection, and later turns fail: there is no
/// reconnect.
#[derive(Clone)]
pub struct ResponsesWebSocket {
    session: Arc<futures::lock::Mutex<Session>>,
    /// The chain's response, readable while a turn holds the connection.
    previous: Previous,
    /// The credentials the handshake carried, scrubbed from observations.
    secrets: Arc<[String]>,
    event_timeout: Option<Duration>,
    drain_timeout: Duration,
}

impl std::fmt::Debug for ResponsesWebSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesWebSocket")
            .field("event_timeout", &self.event_timeout)
            .field("drain_timeout", &self.drain_timeout)
            .finish_non_exhaustive()
    }
}

impl ResponsesWebSocket {
    /// A transport over an open, authenticated connection, with no event
    /// timeout and the default drain timeout. Its credentials never passed
    /// through rig, so observations cannot scrub them; the builder's
    /// `connect_with` does.
    pub fn from_connection(connection: BoxedWebSocketConnection) -> Self {
        let chain = Chain::default();
        Self {
            previous: chain.previous.clone(),
            session: Arc::new(futures::lock::Mutex::new(Session {
                socket: connection,
                chain,
                dirty: false,
                closed: false,
                failed: false,
            })),
            secrets: Arc::from([]),
            event_timeout: None,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
        }
    }

    /// Fail a turn when no event arrives for `timeout`. `None` waits
    /// indefinitely.
    #[must_use]
    pub fn event_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.event_timeout = timeout;
        self
    }

    /// Fail the connection when reading a dropped turn to its end takes
    /// longer than `timeout`. A dropped turn that outlives it, such as long
    /// reasoning, costs the connection.
    #[must_use]
    pub fn drain_timeout(mut self, timeout: Duration) -> Self {
        self.drain_timeout = timeout;
        self
    }

    /// The response the last completed or incomplete turn produced, until a
    /// turn fails, a dropped turn is drained, or [`Self::clear_chain`].
    /// Never waits for a turn.
    pub fn last_response_id(&self) -> Option<String> {
        self.previous.get()
    }

    /// Start the next chained turn on a fresh chain. Never waits for a turn.
    pub fn clear_chain(&self) {
        self.previous.set(None);
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
        let observation = exchange.observation;
        let route = payload.route;
        Opening::new(async move {
            // Turns queue here: the connection carries one at a time.
            let mut session = session.lock_owned().await;
            // The attempt begins when the turn is next, never while queued
            // or unpolled.
            let slot = observation.map(|context| {
                let slot = AdapterSlot::default();
                slot.install(context.attempt_with_secrets(
                    &http::Method::GET,
                    route,
                    secrets.to_vec(),
                ));
                slot
            });
            let mut opened = match session.open(payload, timeouts, slot.as_ref()).await {
                Ok(()) => {
                    // The turn's answer is the connection's upgrade.
                    if let Some(slot) = &slot {
                        slot.response(http::StatusCode::SWITCHING_PROTOCOLS);
                    }
                    Opened::new(turn(session, timeouts.0, slot.clone()))
                }
                Err(error) => Opened::failed(error),
            }
            .with_route(route);
            opened.slot = slot;
            Ok(opened)
        })
    }
}

/// One turn's frames, ending at its last event. Each payload the turn reads
/// is projected onto its attempt. The connection is released before the
/// last item is yielded: the driver stops reading at the reply's end, so
/// code after that yield would not run.
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
                slot.project(|sink| project_payload(text.as_bytes(), sink));
            }
            match lifecycle {
                Lifecycle::Skip => {}
                Lifecycle::Frame => yield Ok(WireFrame::Text(text)),
                Lifecycle::Last { frame, settled } => {
                    session.dirty = !settled;
                    drop(session);
                    yield Ok(WireFrame::Text(frame.unwrap_or(text)));
                    return;
                }
                Lifecycle::Fail(error) => {
                    session.dirty = false;
                    drop(session);
                    yield Err(error);
                    return;
                }
                Lifecycle::Corrupt(error) => {
                    drop(session);
                    yield Err(error);
                    return;
                }
            }
        }
    }
}

/// The connection and what it knows across turns.
struct Session {
    socket: BoxedWebSocketConnection,
    chain: Chain,
    /// A turn was sent and its end not yet read: its stream was dropped, or
    /// it failed while its response may still be sending.
    dirty: bool,
    closed: bool,
    failed: bool,
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
        self.chain.reset();
        self.dirty = false;
        self.failed = true;
    }

    /// Ready the connection for a turn and send its event: drain the turn
    /// before it, chain when asked, and write.
    async fn open(
        &mut self,
        mut payload: ResponseCreate,
        (event_timeout, drain_timeout): (Option<Duration>, Duration),
        slot: Option<&AdapterSlot>,
    ) -> Result<(), ProviderError> {
        if let Err(error) = self.ensure_open() {
            transport_failed(slot);
            return Err(error);
        }
        if self.dirty {
            self.drain(event_timeout, drain_timeout, slot).await?;
        }
        if payload.chain && payload.body.get("previous_response_id").is_none() {
            payload.previous_response_id = self.chain.previous.get();
        }
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI websocket request",
            &payload,
        );
        let text = serde_json::to_string(&payload)?;
        // Set before the write: a write cut off midway may still have
        // reached the provider, whose reply must not reach the next turn.
        self.dirty = true;
        self.chain.end_turn();
        if let Err(error) = self.socket.send(Frame::Text(text)).await {
            self.mark_failed();
            transport_failed(slot);
            return Err(websocket_provider_error(error));
        }
        Ok(())
    }

    /// Read the turn before this one to its end, within `drain_timeout`.
    async fn drain(
        &mut self,
        event_timeout: Option<Duration>,
        drain_timeout: Duration,
        slot: Option<&AdapterSlot>,
    ) -> Result<(), ProviderError> {
        let drained = crate::wasm_compat::timeout(drain_timeout, async {
            loop {
                let text = self.next_text(event_timeout, slot).await?;
                match self.chain.read(&text) {
                    Lifecycle::Last { settled: true, .. } | Lifecycle::Fail(_) => return Ok(()),
                    _ => {}
                }
            }
        })
        .await;
        match drained {
            Ok(Ok(())) => {
                self.dirty = false;
                // The caller never saw the drained turn's response, so a
                // chained turn must not continue it.
                self.chain.previous.set(None);
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(_) => {
                self.mark_failed();
                transport_failed(slot);
                Err(ProviderError::Provider(format!(
                    "Timed out reading an earlier OpenAI websocket turn to its end after \
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
            self.mark_failed();
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
                Some(timeout) => crate::wasm_compat::timeout(timeout, self.socket.recv())
                    .await
                    .map_err(|_| event_timeout_error(timeout))?,
            };
            let frame = match received {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    self.closed = true;
                    return Err(ProviderError::Provider(
                        "The OpenAI websocket connection closed before the turn finished"
                            .to_string(),
                    ));
                }
                Err(error) => return Err(websocket_provider_error(error)),
            };
            if let Some(text) = websocket_frame_to_text(frame)? {
                return Ok(text);
            }
        }
    }
}

/// What one server event means for the turn in flight.
#[derive(Debug)]
enum Lifecycle {
    /// Not this turn's: the previous turn's trailing `response.done`, or one
    /// for another response.
    Skip,
    /// A frame of the turn, for the decoder as it arrived.
    Frame,
    /// The turn's last frame, for the decoder: as it arrived, or lowered to
    /// the terminal event it stands for. `settled` is false when the
    /// turn's response may still be sending.
    Last {
        frame: Option<String>,
        settled: bool,
    },
    /// The turn failed, and its response sends nothing more.
    Fail(ProviderError),
    /// A frame that is no event of this turn: the turn fails, and the
    /// connection is left to drain the rest of its response.
    Corrupt(ProviderError),
}

/// The response chain across turns, the trailing `response.done` to skip,
/// and the response the turn in flight opened.
#[derive(Debug, Default)]
struct Chain {
    previous: Previous,
    pending_done_response_id: Option<String>,
    current_response_id: Option<String>,
}

/// The response a chained turn continues, shared by the connection and its
/// handles: a plain lock, so reading it never waits for a turn.
#[derive(Debug, Default, Clone)]
struct Previous(Arc<std::sync::Mutex<Option<String>>>);

impl Previous {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn get(&self) -> Option<String> {
        self.lock().clone()
    }

    fn set(&self, id: Option<String>) {
        *self.lock() = id;
    }
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
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
    }

    fn status(&self) -> Option<&str> {
        self.response.as_ref()?.get("status")?.as_str()
    }
}

impl Chain {
    /// Read one server event by its type: update the chain and say what the
    /// event is for the turn. Content is left to the decoder. Only a turn
    /// that ended in a completed or incomplete response continues the
    /// chain; any failure ends it.
    fn read(&mut self, text: &str) -> Lifecycle {
        let lifecycle = self.lifecycle(text);
        match &lifecycle {
            Lifecycle::Frame | Lifecycle::Skip => {}
            Lifecycle::Last { .. } => self.end_turn(),
            Lifecycle::Fail(_) => {
                self.previous.set(None);
                self.end_turn();
            }
            Lifecycle::Corrupt(_) => self.previous.set(None),
        }
        lifecycle
    }

    /// Forget everything but the shared handle: the connection failed.
    fn reset(&mut self) {
        self.previous.set(None);
        self.pending_done_response_id = None;
        self.end_turn();
    }

    /// Forget which response the turn in flight opened.
    fn end_turn(&mut self) {
        self.current_response_id = None;
    }

    /// Whether `probe` names a response other than the one this turn opened.
    fn foreign(&self, probe: &Probe) -> bool {
        let id = probe.response_id();
        id.is_some() && self.current_response_id.is_some() && self.current_response_id != id
    }

    fn lifecycle(&mut self, text: &str) -> Lifecycle {
        let probe = match serde_json::from_str::<Probe>(text) {
            Ok(probe) => probe,
            // Every websocket event names its type: anything else is not
            // part of a turn, and must not end one as a whole body.
            Err(error) => return Lifecycle::Corrupt(error.into()),
        };
        match probe.kind.as_str() {
            "response.created" | "response.queued" | "response.in_progress" => {
                if let Some(id) = probe.response_id() {
                    self.current_response_id = Some(id);
                }
                Lifecycle::Frame
            }
            "response.completed" | "response.incomplete" | "response.failed"
                if self.foreign(&probe) =>
            {
                Lifecycle::Corrupt(ProviderError::Provider(format!(
                    "OpenAI websocket turn read a {} for another response ({}); the \
                     connection is out of step with its turns",
                    probe.kind,
                    probe.response_id().unwrap_or_default()
                )))
            }
            kind @ ("response.completed" | "response.incomplete" | "response.failed") => {
                self.pending_done_response_id = probe.response_id();
                let chained = (kind != "response.failed")
                    .then(|| probe.response_id())
                    .flatten();
                self.previous.set(chained);
                Lifecycle::Last {
                    frame: None,
                    settled: true,
                }
            }
            "error" => {
                self.previous.set(None);
                // An error after the turn's response opened may leave that
                // response still sending: its rest is drained, not read as
                // the next turn's.
                Lifecycle::Last {
                    frame: None,
                    settled: self.current_response_id.is_none(),
                }
            }
            "response.done" => {
                let id = probe.response_id();
                if id.is_some() && self.pending_done_response_id == id {
                    self.pending_done_response_id = None;
                    return Lifecycle::Skip;
                }
                if self.foreign(&probe) {
                    return Lifecycle::Skip;
                }
                self.pending_done_response_id = None;
                self.done(&probe)
            }
            _ => Lifecycle::Frame,
        }
    }

    /// A `response.done` that ends the turn, lowered to the terminal event
    /// its response's status names, which the decoder reads.
    fn done(&mut self, probe: &Probe) -> Lifecycle {
        let terminal = match probe.status() {
            Some("completed") => "response.completed",
            Some("incomplete") => "response.incomplete",
            Some("failed") => "response.failed",
            Some(status) => {
                return Lifecycle::Fail(ProviderError::Provider(format!(
                    "OpenAI websocket response ended in state `{status}`"
                )));
            }
            None => {
                return Lifecycle::Fail(ProviderError::Provider(
                    "OpenAI websocket turn ended with response.done before a terminal \
                     response body was available"
                        .to_string(),
                ));
            }
        };
        let chained = (terminal != "response.failed")
            .then(|| probe.response_id())
            .flatten();
        self.previous.set(chained);
        let frame = json!({ "type": terminal, "response": probe.response });
        Lifecycle::Last {
            frame: Some(frame.to_string()),
            settled: true,
        }
    }
}

/// Mark the attempt's failure as the connection's, not the provider's.
fn transport_failed(slot: Option<&AdapterSlot>) {
    if let Some(slot) = slot {
        slot.error_boundary(AdapterErrorBoundary::Transport);
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
    /// [`ResponsesSocket::chaining`].
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
        let secrets = crate::observe::request_secrets(&request);
        let socket = backend
            .connect(
                request,
                ConnectOptions::new().with_timeout(self.connect_timeout),
            )
            .await
            .map_err(websocket_provider_error)?;
        let mut transport = ResponsesWebSocket::from_connection(socket)
            .event_timeout(self.event_timeout)
            .drain_timeout(self.drain_timeout);
        transport.secrets = Arc::from(secrets);
        let mut wire = ResponsesSocket::new(self.wire);
        wire.chaining = self.chaining;
        Ok(Model::new(wire, transport))
    }
}

/// Lower one websocket frame onto the JSON payload the protocol carries.
///
/// `Ok(None)` is a frame with no protocol payload (a keepalive), which the
/// connection skips; a close frame mid-turn is an error naming the peer's
/// reason.
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

impl Model<ResponsesSocket, ResponsesWebSocket> {
    /// Prepare `request`'s state on the connection without generating
    /// output, and return the id of the response it prepared. With
    /// [`chaining`](ResponsesSocket::chaining) on, the next turn continues
    /// it; otherwise pass the id as the next request's
    /// `previous_response_id`. Returns the turn's error, or a response error
    /// when the provider named no response.
    pub async fn warmup(
        &self,
        request: impl Into<completion::CompletionRequest>,
    ) -> Result<String, ProviderError> {
        let warm = Model::new(self.wire.clone().warmup(), self.transport.clone());
        warm.call(request).await?.origin.response_id.ok_or_else(|| {
            ProviderError::Response("the OpenAI websocket warmup named no response".to_string())
        })
    }
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
