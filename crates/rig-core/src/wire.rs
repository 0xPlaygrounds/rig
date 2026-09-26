//! Transport-independent provider encoding and reply decoding. A [`Wire`]
//! produces requests and a fresh [`Decoder`]; the driver supplies transport and
//! framing. Buffered and streamed replies use the same decoder and event fold.
//!
//! ```
//! use rig_core::wire::WireFrame;
//!
//! let frame = WireFrame::Bytes(b"response".to_vec());
//! assert_eq!(frame.as_str(), "response");
//! ```

use std::borrow::Cow;

use crate::error::{EncodeError, ProviderError};
use crate::http_client::MultipartForm;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub use crate::http_client::framing::Framing;
pub use crate::observe::{AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict};
pub use crate::providers::internal::wire::WireEvent;

pub(crate) mod secret;

pub use secret::Secret;

/// One transport frame, after framing but before decoding.
///
/// The transport layer (SSE framer, NDJSON splitter, websocket reader) owns
/// byte splitting and yields these; a decoder never splits bytes.
#[derive(Debug, Clone)]
pub enum WireFrame {
    /// Text payload, such as an SSE `data:` field or WebSocket message.
    Text(String),
    /// Byte payload, such as an NDJSON line or binary frame.
    Bytes(Vec<u8>),
}

impl WireFrame {
    /// The frame payload as text (lossy for byte frames).
    pub fn as_str(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text),
            Self::Bytes(bytes) => String::from_utf8_lossy(bytes),
        }
    }
}

/// The request a wire sends, and how its reply is framed.
///
/// Data only: built by [`Wire::encode`] from the wire and the request, and
/// read by the driver. A wire never touches a socket or a request extension.
///
/// `Debug` shows request methods and URI paths, not schemes, authorities,
/// queries, header values or bodies. Paths are not scrubbed: callers must
/// still avoid placing sensitive data in them.
pub struct Encoded {
    /// HTTP requests in dispatch order. Buffered calls fold all replies into
    /// one response; streaming requires exactly one request.
    pub requests: Vec<http::Request<Body>>,
    /// How the reply's bytes split into frames.
    pub framing: Framing,
    /// The reply header carrying the provider's transport request id
    /// (Anthropic `request-id`, OpenAI `x-request-id`), when the provider
    /// reports one. `None` is "does not report one", never an error.
    pub request_id_header: Option<&'static str>,
    /// Whether a streamed reply may omit `Content-Type` (one gateway
    /// replays Responses bodies without it). A *wrong* content type is
    /// still rejected.
    pub relaxed_content_type: bool,
    /// Stable endpoint template for observation grouping, without base-URL
    /// prefixes or interpolated values. `None` uses the concrete request path.
    pub route: Option<&'static str>,
}

impl Encoded {
    /// One request whose provider reports no transport request id.
    pub fn new(request: http::Request<Body>, framing: Framing) -> Self {
        Self::batch(vec![request], framing)
    }

    /// Several requests whose replies fold into one response, in order.
    pub fn batch(requests: Vec<http::Request<Body>>, framing: Framing) -> Self {
        Self {
            requests,
            framing,
            request_id_header: None,
            relaxed_content_type: false,
            route: None,
        }
    }

    /// Name the reply header carrying the provider's transport request id.
    pub fn with_request_id_header(mut self, header: Option<&'static str>) -> Self {
        self.request_id_header = header;
        self
    }

    /// Accept a streamed reply that names no content type.
    pub fn with_relaxed_content_type(mut self) -> Self {
        self.relaxed_content_type = true;
        self
    }

    /// Name the endpoint template observation groups attempts under.
    pub fn with_route(mut self, route: Option<&'static str>) -> Self {
        self.route = route;
        self
    }
}

/// A request body: bytes, or a multipart form for the upload endpoints.
///
/// `Debug` prints the body's shape and size, never its bytes: a request
/// body carries prompts, documents and uploaded files.
pub enum Body {
    /// A serialized body (JSON for every wire in this crate, or empty).
    Bytes(Vec<u8>),
    /// A multipart form (audio transcription, image edits).
    Multipart(MultipartForm),
}

impl Body {
    /// An empty body, for a `GET`.
    pub fn empty() -> Self {
        Self::Bytes(Vec::new())
    }
}

impl std::fmt::Debug for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(bytes) => write!(f, "Bytes({} bytes)", bytes.len()),
            Self::Multipart(_) => f.write_str("Multipart"),
        }
    }
}

impl std::fmt::Debug for Encoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Diagnostics omit query, authority, headers and bodies. A caller's
        // path can still contain sensitive data; it is not scrubbed here.
        struct Requests<'a>(&'a [http::Request<Body>]);

        impl std::fmt::Debug for Requests<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list()
                    .entries(
                        self.0
                            .iter()
                            .map(|request| (request.method(), request.uri().path())),
                    )
                    .finish()
            }
        }
        f.debug_struct("Encoded")
            .field("requests", &Requests(&self.requests))
            .field("framing", &self.framing)
            .field("request_id_header", &self.request_id_header)
            .field("relaxed_content_type", &self.relaxed_content_type)
            .field("route", &self.route)
            .finish()
    }
}

/// Reply mode used by the wire to select request encoding, framing, and decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One whole reply ([`Model::call`](crate::driver::Model::call)).
    Unary,
    /// A streamed reply ([`Model::stream`](crate::driver::Model::stream)).
    Streaming,
}

/// An operation: what goes in, what comes out event by event, and how those
/// events fold into one response.
///
/// Implemented once per operation in [`crate::operation`], never per
/// provider. For a unary operation `Event` is `Response` and the fold takes
/// the one event.
pub trait Operation: Sized + 'static {
    /// The normalized request this operation accepts.
    type Request: WasmCompatSend + 'static;
    /// One decoded step of a reply.
    type Event: WasmCompatSend + 'static;
    /// The normalized response the events fold into.
    type Response: WasmCompatSend + 'static;
    /// What a runtime accounts for. `()` for operations with nothing to
    /// declare.
    type Capabilities: Default;
    /// Where a decoder writes the events of one `interpret` step.
    type Output: Sink<Self> + WasmCompatSend;
    /// The fold from events to the response.
    type Fold: Fold<Self> + WasmCompatSend;
    /// The canonical telemetry operation a wire performs. `()` for
    /// operations that open no span.
    type Telemetry: Copy;

    /// The operation's name, as telemetry and records spell it.
    const NAME: &'static str;

    /// Whether this event signals provider completion and stops driver consumption.
    fn is_terminal(event: &Self::Event) -> bool;

    /// Creates a response fold, optionally retaining request data needed to
    /// associate output with input. Defaults to an empty fold.
    fn fold(_request: &Self::Request) -> Self::Fold {
        Self::Fold::default()
    }

    /// Scope a request to the wire about to encode it: drop request content
    /// that no issuer the wire replays for the request's model can
    /// interpret. Operations with no such content do nothing.
    fn scope_to_wire<W: Wire<Op = Self>>(_request: &mut Self::Request, _wire: &W) {}

    /// Stamp what the driver learned about a unary reply beyond its events.
    fn stamp_reply(_response: &mut Self::Response, _reply: Reply) {}

    /// The canonical telemetry operation for a unary (`false`) or streaming
    /// (`true`) call. A wire whose endpoint has its own canonical name
    /// overrides [`Wire::telemetry`].
    fn telemetry(streaming: bool) -> Self::Telemetry;

    /// The operation's telemetry span. The default is no span: an operation
    /// with nothing to record (verification, model listing) opens none.
    fn span(
        _provider: &str,
        _model: Option<&str>,
        _telemetry: Self::Telemetry,
        _request: &Self::Request,
    ) -> tracing::Span {
        tracing::Span::none()
    }

    /// Record the folded response onto the operation's span.
    fn record(_span: &tracing::Span, _response: &Self::Response) {}

    /// Adds the provider and request path to a failed reply's error. The
    /// default adds nothing.
    fn with_route(error: ProviderError, _provider: &str, _path: &str) -> ProviderError {
        error
    }

    /// Refuse a folded response that contradicts what the wire declared,
    /// such as an embedding width the caller asked for. The default accepts.
    fn accept(
        _capabilities: &Self::Capabilities,
        _provider: &str,
        _response: &Self::Response,
    ) -> Result<(), ProviderError> {
        Ok(())
    }
}

/// What the driver learned about a unary reply beyond its events: the
/// provider's name, the body as JSON (a completion's `raw`) and the
/// transport request id.
pub struct Reply {
    /// The provider descriptor name, for the response's `provider` field.
    pub provider: String,
    /// The reply body parsed as JSON, `Null` when it is not JSON.
    pub raw: serde_json::Value,
    /// The provider's transport request id from the reply headers.
    pub provider_request_id: Option<String>,
}

/// Where a decoder writes the events of one `interpret` step.
///
/// The driver drains the sink after every frame, so a decoder never has to
/// know whether it is feeding a stream or a buffered fold.
pub trait Sink<Op: Operation>: Default {
    /// Debug-mode sequence laws checked against what the decoder actually
    /// emitted. `()` for operations with no sequence to check.
    type Laws: Default + WasmCompatSend;

    /// Push one event or one in-band error.
    fn push(&mut self, item: Result<Op::Event, ProviderError>);

    /// Take everything pushed since the last drain.
    fn drain(&mut self) -> std::vec::Drain<'_, Result<Op::Event, ProviderError>>;

    /// What this sink holds, without taking it.
    fn items(&self) -> &[Result<Op::Event, ProviderError>];

    /// An unmodeled payload the decoder classified but cannot interpret.
    /// The default skips it; a sink with a passthrough channel forwards it.
    fn unknown(&mut self, _payload: crate::streaming::UnknownPayload) {}

    /// Check the operation's sequence laws over this batch.
    fn check_laws(&self, _laws: &mut Self::Laws) {}
}

/// The fold from a reply's events to its response.
pub trait Fold<Op: Operation>: Default {
    /// Absorb one event. An error fails the whole operation: a buffered
    /// reply has no stream to carry an in-band defect.
    fn absorb(&mut self, event: Op::Event) -> Result<(), ProviderError>;

    /// The folded response.
    fn finish(self, reply: Reply) -> Result<Op::Response, ProviderError>;
}

/// Where a decoder's observation projection writes its facts.
///
/// The projector reads verdicts, usage, ids and error envelopes off a raw
/// payload before normalization discards them. Text it forwards must go
/// through [`Self::scrub`]: a payload can echo credentials.
pub trait ObservationSink {
    /// Record one boundary fact.
    fn emit(&mut self, event: AdapterEvent);

    /// Record the provider's verdict, and the response id it named.
    fn provider(&mut self, verdict: AdapterVerdict, response_id: Option<String>);

    /// Bound and redact diagnostic text from the payload.
    fn scrub(&self, value: &str) -> String;
}

/// Where a decoder writes one `interpret` step's events.
pub type Output<Op> = <Op as Operation>::Output;

/// Synchronous state machine for one reply. Classifies frames and interprets
/// known events without transport access. HTTP wires read [`WireFrame`]s;
/// other transports name their own frame type.
pub trait Decoder<Op: Operation, Frame = WireFrame> {
    /// The wire's typed event, produced by this decoder's classifier.
    type Event;

    /// Decode and classify one frame. A JSON wire MUST delegate to a
    /// classifier in [`crate::providers::internal::wire`], so the
    /// decode-then-validate policy is not re-derived per provider.
    fn classify(&self, frame: Frame) -> WireEvent<Self::Event>;

    /// Map one `Known` event onto the operation's events. Stateful: index
    /// maps, open-block state and wire-quirk quarantine live here.
    fn interpret(&mut self, event: Self::Event, out: &mut Output<Op>);

    /// End-of-reply flush without a terminal (close open blocks). Must not
    /// synthesize a terminal: EOF without the provider's end event is
    /// truncation.
    fn finish(&mut self, _out: &mut Output<Op>) {}

    /// Flush content the provider fully delivered before a terminal error
    /// reaches the consumer. Must not push a terminal.
    fn flush_before_terminal_error(&mut self, _out: &mut Output<Op>) {}

    /// The observation projection: verdicts, usage, ids and error envelopes
    /// read off a raw payload before normalization discards them. The
    /// default projects nothing.
    fn project(&self, _payload: &[u8], _sink: &mut dyn ObservationSink) {}

    /// Returns the reassembled provider document for a buffered reply delivered
    /// as frames. The driver uses this when the body is not a JSON document.
    fn document(&self) -> Option<serde_json::Value> {
        None
    }

    /// The cursor of a paged operation's next page, if the reply named one.
    /// [`Wire::page`] builds its payload.
    fn cursor(&self) -> Option<String> {
        None
    }

    /// Whether this frame carries only analysis metadata: it still decodes,
    /// but does not advance observation's EOF/corruption positions.
    fn is_analysis_only(&self, _frame: &Frame) -> bool {
        false
    }

    /// Whether `interpret` consumed the wire's own in-band terminal failure
    /// and already pushed the flush-then-error sequence itself.
    fn is_finished(&self) -> bool {
        false
    }
}

/// A provider: data, an encoder, and a decoder.
///
/// No transport, no future, no type parameter. Implementations are plain
/// data (`Clone + PartialEq + Debug + Serialize + Deserialize`, with
/// credentials held in [`Secret`]), so a host can store one in a scene, a
/// component, or a config file.
pub trait Wire: WasmCompatSend + WasmCompatSync + 'static {
    /// The operation this wire performs.
    type Op: Operation;
    /// What [`Self::encode`] produces for the transport to send.
    type Payload: WasmCompatSend + 'static;
    /// One unit of a reply, as the transport delivers it.
    type Frame: WasmCompatSend + 'static;
    /// The decoder for one of its replies.
    type Decoder: Decoder<Self::Op, Self::Frame> + WasmCompatSend + 'static;

    /// The provider descriptor name (`"anthropic"`), as records and
    /// telemetry name it.
    fn name(&self) -> &str;

    /// The request to send. Pure: it may read `self`, `request` and `mode`,
    /// and nothing else. A request that cannot be built is an
    /// [`EncodeError`], which always reports as a request failure.
    fn encode(&self, request: Request<Self>, mode: Mode) -> Result<Self::Payload, EncodeError>;

    /// The payload of the page at `cursor`, for a paged operation whose
    /// decoder named one ([`Decoder::cursor`]). The default refuses: an
    /// operation that does not page never names a cursor.
    fn page(&self, cursor: &str) -> Result<Self::Payload, EncodeError> {
        let _ = cursor;
        Err(EncodeError::request(format!(
            "{} does not page its replies",
            self.name()
        )))
    }

    /// A fresh decoder for one reply, in the mode [`Self::encode`] was
    /// given.
    ///
    /// The mode is what this reply's EOF will mean: a whole reply that
    /// named no terminal is the provider answering with nothing, while a
    /// stream that ends the same way stopped early and reports truncation
    /// by carrying no terminal record. A decoder whose terminal is
    /// deferred to EOF needs that difference, and it is known before the
    /// first frame rather than stated afterwards.
    fn decoder(&self, mode: Mode) -> Self::Decoder;

    /// What a runtime accounts for.
    fn capabilities(&self) -> Capabilities<Self> {
        Capabilities::<Self>::default()
    }

    /// The issuer of the reasoning a reply for `model` carries when a
    /// deployment serves another provider's models, known before the reply's
    /// terminal record names it. `None` is the wire itself.
    fn reasoning_issuer(&self, _model: Option<&str>) -> Option<&str> {
        None
    }

    /// The model this wire addresses, for telemetry. `None` for operations
    /// that address no model.
    fn model(&self) -> Option<&str> {
        None
    }

    /// The issuers whose provider state (reasoning signatures, ciphertext,
    /// ids) a request to `model` may replay. The default is this wire alone;
    /// a gateway whose state depends on the upstream model narrows it.
    /// `None` scopes nothing: the backend checks the history it is given.
    fn replay_issuers(&self, _model: Option<&str>) -> Option<Vec<String>> {
        Some(vec![self.name().to_owned()])
    }

    /// The canonical telemetry operation this wire performs. Override when
    /// the endpoint has its own name (Gemini `generate_content`).
    fn telemetry(&self, streaming: bool) -> Telemetry<Self> {
        <Self::Op as Operation>::telemetry(streaming)
    }
}

/// A wire's request type.
pub type Request<W> = <<W as Wire>::Op as Operation>::Request;
/// A wire's response type.
pub type Response<W> = <<W as Wire>::Op as Operation>::Response;
/// A wire's event type.
pub type Event<W> = <<W as Wire>::Op as Operation>::Event;
/// A wire's capability type.
pub type Capabilities<W> = <<W as Wire>::Op as Operation>::Capabilities;
/// A wire's telemetry operation type.
pub type Telemetry<W> = <<W as Wire>::Op as Operation>::Telemetry;
