//! Operations, provider wires and their decoders. An [`Operation`] names what
//! goes in and what comes out event by event, and its [`Fold`] is one reply's
//! state. A [`Wire`] describes a provider endpoint as plain data, encodes
//! requests without transport access, and names a fresh [`Decoder`] for each
//! reply. The driver supplies transport and framing, so buffered and streamed
//! replies use the same decoder and fold.
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
pub use crate::observe::{
    AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, ObservationSink,
};

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
    /// Reads observation facts off each raw reply payload before
    /// normalization discards them. `None` projects nothing.
    pub project: Option<Projector>,
    /// Whether a frame carries only analysis metadata: it still decodes, but
    /// does not advance observation's EOF and corruption positions.
    pub analysis_only: Option<fn(&WireFrame) -> bool>,
}

/// Reads one raw payload's observation facts (verdicts, usage, ids, error
/// envelopes) through the sink. Text it forwards goes through
/// [`ObservationSink::scrub`]: a payload can echo credentials.
pub type Projector = fn(&[u8], &mut ObservationSink<'_>);

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
            project: None,
            analysis_only: None,
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

    /// Read observation facts off each reply payload through `project`.
    pub fn with_projection(mut self, project: Projector) -> Self {
        self.project = Some(project);
        self
    }

    /// Exempt frames `analysis_only` accepts from observation's frame
    /// positions.
    pub fn with_analysis_only(mut self, analysis_only: fn(&WireFrame) -> bool) -> Self {
        self.analysis_only = Some(analysis_only);
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
            .field("project", &self.project.is_some())
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

/// How a reply ended without the provider's terminal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    /// The frames ran out. A decoder must not synthesize a terminal here:
    /// EOF without the provider's end event is truncation.
    Eof,
    /// A transport failure is about to reach the consumer. A decoder flushes
    /// content the provider fully delivered, and pushes no terminal.
    Failed,
}

/// An operation: what goes in, what comes out event by event, and how those
/// events fold into one response.
///
/// Implemented once per operation, never per provider, so every provider of
/// an operation is interchangeable behind a [`DynModel`](crate::DynModel).
pub trait Operation: Sized + 'static {
    /// The normalized request this operation accepts.
    type Request: WasmCompatSend + 'static;
    /// One decoded step of a reply.
    type Event: WasmCompatSend + 'static;
    /// The normalized response the events fold into.
    type Response: WasmCompatSend + 'static;
    /// One reply's state: what decoders write through, and what folds the
    /// events the consumer sees into the response.
    type Fold: Fold<Self> + WasmCompatSend + 'static;

    /// Whether this event is the provider's end of the reply. The driver
    /// stops reading at it.
    fn is_terminal(event: &Self::Event) -> bool;

    /// The state for one reply to `request`, before its first frame. The
    /// call says who answers and in which mode; an operation that records
    /// telemetry opens its span here and hands it to
    /// [`Call::instrument`].
    fn fold(request: &Self::Request, call: &mut Call<'_>) -> Self::Fold;
}

/// What the driver knows about a call before its first frame.
pub struct Call<'a> {
    /// What the wire says about itself.
    pub wire: &'a Descriptor<'a>,
    /// How the reply arrives.
    pub mode: Mode,
    pub(crate) span: tracing::Span,
}

impl<'a> Call<'a> {
    /// A call to the wire `wire` describes, in `mode`, under no span: what
    /// a caller that drives a decoder by hand opens a fold with.
    pub fn new(wire: &'a Descriptor<'a>, mode: Mode) -> Self {
        Self {
            wire,
            mode,
            span: tracing::Span::none(),
        }
    }

    /// Run the call under `span`: a unary call sends under it and a stream
    /// decodes under it.
    pub fn instrument(&mut self, span: tracing::Span) {
        self.span = span;
    }
}

/// One reply's state for an operation.
///
/// Every item a decoder pushes passes [`Self::push`] into the reply's
/// [`Ready`] queue, and each event is [`absorb`](Self::absorb)ed as the
/// consumer takes it, so the fold never runs ahead of what was delivered.
/// An operation whose events need no normalization implements only
/// `absorb` and `finish`.
pub trait Fold<Op: Operation> {
    /// Absorb one event the consumer is about to see. An error fails the
    /// operation.
    fn absorb(&mut self, event: &Op::Event) -> Result<(), ProviderError>;

    /// The folded response, from the events absorbed and what the driver
    /// learned about the reply.
    fn finish(self, reply: Reply) -> Result<Op::Response, ProviderError>;

    /// Accept one item a decoder pushed. The default queues it as is.
    fn push(&mut self, item: Result<Op::Event, ProviderError>, ready: &mut Ready<Op>) {
        ready.push(item);
    }

    /// An unmodeled payload a decoder classified but cannot interpret. The
    /// default skips it.
    fn unknown(&mut self, _payload: crate::streaming::UnknownPayload, _ready: &mut Ready<Op>) {}

    /// A decoder step finished: after every frame, and once more when the
    /// reply ends (`end` is `Some`). The default does nothing.
    fn settle(&mut self, _ready: &mut Ready<Op>, _end: Option<End>) {}
}

/// The items of one reply waiting for the consumer, and what the transport
/// reported about the reply so far.
pub struct Ready<Op: Operation> {
    pub(crate) items: std::collections::VecDeque<Result<Op::Event, ProviderError>>,
    request_id: Option<String>,
}

impl<Op: Operation> Default for Ready<Op> {
    fn default() -> Self {
        Self {
            items: std::collections::VecDeque::new(),
            request_id: None,
        }
    }
}

impl<Op: Operation> Ready<Op> {
    /// Queue one item for the consumer.
    pub fn push(&mut self, item: Result<Op::Event, ProviderError>) {
        self.items.push_back(item);
    }

    /// The provider's transport request id for the reply being read.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub(crate) fn set_request_id(&mut self, request_id: Option<String>) {
        self.request_id = request_id;
    }

    pub(crate) fn pop(&mut self) -> Option<Result<Op::Event, ProviderError>> {
        self.items.pop_front()
    }
}

/// Where a decoder writes one step's items: through the operation's
/// [`Fold`] into the reply's [`Ready`] queue.
pub struct Out<'a, Op: Operation> {
    pub(crate) fold: &'a mut Op::Fold,
    pub(crate) ready: &'a mut Ready<Op>,
    pub(crate) finished: &'a mut bool,
    pub(crate) document: &'a mut Option<serde_json::Value>,
}

impl<Op: Operation> Out<'_, Op> {
    /// The operation's per-reply fold, for the operation's own writer
    /// helpers.
    pub fn fold(&mut self) -> &mut Op::Fold {
        self.fold
    }

    /// Push one event or one in-band error the reply continues past.
    pub fn push(&mut self, item: Result<Op::Event, ProviderError>) {
        self.fold.push(item, self.ready);
    }

    /// The decoder consumed the wire's own in-band terminal failure and
    /// pushed the flush-then-error sequence itself: the reply ends here.
    pub fn end_reply(&mut self) {
        *self.finished = true;
    }

    /// The provider document of a whole reply the decoder reassembled from
    /// its frames, for a reply whose body is not one JSON document.
    pub fn document(&mut self, document: serde_json::Value) {
        *self.document = Some(document);
    }
}

/// What the driver learned about a reply beyond its events: the provider's
/// name, the body as JSON (a completion's `raw`) and the transport request
/// id.
pub struct Reply {
    /// The provider descriptor name, for the response's `provider` field.
    pub provider: String,
    /// The reply body parsed as JSON, `Null` when it is not JSON.
    pub raw: serde_json::Value,
    /// The provider's transport request id from the reply headers.
    pub provider_request_id: Option<String>,
}

/// What a runtime accounts for about a model. Each field matters to the
/// operations that name it and is left at its default by the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    /// What a completion provider composes.
    pub completion: crate::completion::ProviderCapabilities,
    /// The most documents one embedding or reranking request takes.
    pub max_documents: usize,
    /// The dimensionality of the returned vectors. Zero is unknown.
    pub ndims: usize,
    /// The embedding width the caller asked for, when they named one. `None`
    /// and zero disable the check that replies honour it.
    pub declared: Option<usize>,
}

impl Capabilities {
    /// A completion provider's capabilities.
    pub const fn completion(completion: crate::completion::ProviderCapabilities) -> Self {
        Self {
            completion,
            ..Self::embedding(0, 0)
        }
    }

    /// An embedding model's batch limit and width, with no width declared.
    pub const fn embedding(max_documents: usize, ndims: usize) -> Self {
        Self {
            completion: crate::completion::ProviderCapabilities::new(),
            max_documents,
            ndims,
            declared: None,
        }
    }

    /// A reranking model's batch limit.
    pub const fn rerank(max_documents: usize) -> Self {
        Self::embedding(max_documents, 0)
    }

    /// Record the width the caller named, when they named one.
    pub const fn declaring(mut self, declared: Option<usize>) -> Self {
        self.declared = declared;
        self
    }

    /// Returns [`ProviderError::MismatchedDimensions`] for the first width
    /// differing from a positive caller declaration. Otherwise succeeds.
    pub(crate) fn honour_declaration(
        &self,
        provider: &str,
        widths: impl IntoIterator<Item = usize>,
    ) -> Result<(), ProviderError> {
        // Zero is rig's sentinel for an unknown width, never a claim about
        // one: a model absent from every table this build knows resolves to
        // it, and treating that as a declaration would fail every reply.
        let Some(requested) = self.declared.filter(|declared| *declared > 0) else {
            return Ok(());
        };
        let Some(returned) = widths.into_iter().find(|width| *width != requested) else {
            return Ok(());
        };
        Err(ProviderError::MismatchedDimensions {
            provider: provider.to_owned(),
            requested,
            returned,
        })
    }
}

/// What a wire says about itself: plain data, read before every call.
#[derive(Debug, Clone)]
pub struct Descriptor<'a> {
    /// The provider descriptor name (`"anthropic"`), as records and
    /// telemetry name it.
    pub name: &'a str,
    /// The model id the wire addresses, for telemetry. `None` for
    /// operations that address no model.
    pub model: Option<&'a str>,
    /// What a runtime accounts for.
    pub capabilities: Capabilities,
    /// The telemetry operation for a call in each mode, when the endpoint
    /// has a canonical name of its own (Gemini `generate_content`).
    pub telemetry: Option<fn(Mode) -> crate::telemetry::GenAiOperation>,
}

impl<'a> Descriptor<'a> {
    /// A wire named `name`, addressing no model, with default capabilities.
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            model: None,
            capabilities: Capabilities::default(),
            telemetry: None,
        }
    }

    /// The model id the wire addresses, when it addresses one.
    pub fn model(mut self, model: impl Into<Option<&'a str>>) -> Self {
        self.model = model.into();
        self
    }

    /// What a runtime accounts for.
    pub fn capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// The endpoint's own telemetry operation for each mode.
    pub fn telemetry(mut self, telemetry: fn(Mode) -> crate::telemetry::GenAiOperation) -> Self {
        self.telemetry = Some(telemetry);
        self
    }
}

/// One classified wire frame.
#[derive(Debug)]
pub enum WireEvent<T> {
    /// The frame carries a discriminator this client models and its payload
    /// decoded fully.
    Known(T),
    /// Valid JSON not recognized by this classifier.
    /// Drivers log structural metadata only and skip interpretation.
    Unknown {
        /// The unmodeled discriminator value.
        event_type: String,
        /// Full payload for raw passthrough, never warning logs. Debug is redacted.
        value: crate::streaming::UnknownPayload,
    },
    /// Invalid JSON or a recognized frame that failed typed decoding.
    /// Must not be demoted to `Unknown`.
    Corrupt(serde_json::Error),
}

impl<T> WireEvent<T> {
    /// Map the `Known` payload, preserving the classification.
    ///
    /// This is how an adapter layers a pure event-shape mapping on top of a
    /// classifier without restating the triage: `Unknown` and `Corrupt` pass
    /// through untouched, so policy stays with the driver.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> WireEvent<U> {
        match self {
            Self::Known(event) => WireEvent::Known(f(event)),
            Self::Unknown { event_type, value } => WireEvent::Unknown { event_type, value },
            Self::Corrupt(error) => WireEvent::Corrupt(error),
        }
    }
}

/// Triage of one already-deserialized event from a typed-transport wire
/// (an aws-sdk event stream, a prost/tonic gRPC stream, an in-process
/// generation channel), for
/// [`classify_typed_event`](crate::providers::internal::wire::classify_typed_event).
#[derive(Debug)]
pub enum TypedEvent<T> {
    /// A variant this client models.
    Modeled(T),
    /// An unrecognized variant reported by the transport SDK.
    Unrecognized {
        /// Discriminator for the driver's warn log.
        event_type: String,
        /// Frame detail retained for raw passthrough, not warning logs.
        detail: String,
    },
    /// SDK decode failure for a modeled event.
    Malformed(String),
}

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

    /// Map one `Known` event onto the operation's items. Stateful: index
    /// maps, open-block state and wire-quirk quarantine live here.
    fn interpret(&mut self, event: Self::Event, out: &mut Out<'_, Op>);

    /// The reply ended without the provider's terminal record. The default
    /// does nothing.
    fn end(&mut self, _out: &mut Out<'_, Op>, _end: End) {}
}

/// A provider endpoint: plain data that encodes requests and names a decoder
/// for each reply.
///
/// No transport, no future, no type parameter. Implementations are plain
/// data (`Clone + PartialEq + Debug + Serialize + Deserialize`, with
/// credentials held in [`Secret`]), so a host can store one in a scene, a
/// component, or a config file. `Clone` is a supertrait: the driver clones
/// the wire into every call's `'static` stream.
pub trait Wire: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// The operation this wire performs.
    type Op: Operation;
    /// What [`Self::encode`] produces for the transport to send.
    type Payload: WasmCompatSend + 'static;
    /// One unit of a reply, as the transport delivers it.
    type Frame: WasmCompatSend + 'static;
    /// The decoder for one of its replies.
    type Decoder: Decoder<Self::Op, Self::Frame> + WasmCompatSend + 'static;

    /// What the wire says about itself.
    fn describe(&self) -> Descriptor<'_>;

    /// The request to send. Pure: it may read `self`, `request` and `mode`,
    /// and nothing else. A request that cannot be built is an
    /// [`EncodeError`], which always reports as a request failure.
    fn encode(&self, request: Request<Self>, mode: Mode) -> Result<Self::Payload, EncodeError>;

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
}

/// A wire's request type.
pub type Request<W> = <<W as Wire>::Op as Operation>::Request;
/// A wire's response type.
pub type Response<W> = <<W as Wire>::Op as Operation>::Response;
/// A wire's event type.
pub type Event<W> = <<W as Wire>::Op as Operation>::Event;
