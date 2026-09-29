//! Operations, provider wires and their decoders. An [`Operation`] names what
//! goes in, what comes out event by event and what ends a reply, and its
//! [`Fold`] turns one reply's events and end into the response. A [`Wire`]
//! describes a provider endpoint as plain data, encodes requests without
//! transport access, and names a fresh [`Decoder`] for each reply. A decoder
//! is told nothing about how the reply arrives, so buffered and streamed
//! replies go through the same decoder and fold.
//!
//! ```
//! use rig_core::wire::WireFrame;
//!
//! let frame = WireFrame::Bytes(b"response".to_vec());
//! assert_eq!(frame.as_str(), "response");
//! ```

use crate::error::{EncodeError, ProviderError};
use crate::http_client::MultipartForm;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub use crate::http_client::framing::{Framing, WireFrame};
pub use crate::observe::{
    AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, ObservationSink,
};

pub(crate) mod secret;

pub use secret::Secret;

/// The request a wire sends, and how its reply is framed.
///
/// Data only: built by [`Wire::encode`] from the wire and the request, and
/// read by the driver. A wire never touches a socket or a request extension.
///
/// `Debug` shows request methods and URI paths, not schemes, authorities,
/// queries, header values or bodies. Paths are not scrubbed: callers must
/// still avoid placing sensitive data in them.
pub struct Encoded {
    /// The HTTP request.
    pub request: http::Request<Body>,
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
        Self {
            request,
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
        f.debug_struct("Encoded")
            .field(
                "request",
                &(self.request.method(), self.request.uri().path()),
            )
            .field("framing", &self.framing)
            .field("request_id_header", &self.request_id_header)
            .field("relaxed_content_type", &self.relaxed_content_type)
            .field("route", &self.route)
            .field("project", &self.project.is_some())
            .finish()
    }
}

/// Reply mode used by the wire to select request encoding and framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One whole reply ([`Model::call`](crate::driver::Model::call)).
    Unary,
    /// A streamed reply ([`Model::stream`](crate::driver::Model::stream)).
    Streaming,
}

/// An operation: what goes in, what comes out event by event, what ends a
/// reply, and how those fold into one response.
///
/// Implemented once per operation, never per provider, so every provider of
/// an operation is interchangeable behind a [`DynModel`](crate::DynModel).
pub trait Operation: Sized + 'static {
    /// The normalized request this operation accepts.
    type Request: WasmCompatSend + 'static;
    /// One decoded step of a reply.
    type Event: WasmCompatSend + 'static;
    /// What the provider sends when it ends the reply. A fold needs one to
    /// produce a response, so a reply that stopped early has none.
    type End: WasmCompatSend + 'static;
    /// The normalized response the events and the end fold into.
    type Response: WasmCompatSend + 'static;
    /// One reply's state: it absorbs the events the consumer sees, then
    /// folds them with the end into the response.
    type Fold: Fold<Self> + WasmCompatSend + 'static;
    /// Who builds the events: [`Free`] for events a decoder writes itself,
    /// [`Assembled`] for the completion events only the writer's part
    /// handles build.
    type Emit: Emit<Self>;

    /// The state for one reply to `request`, before its first frame. The
    /// call says who answers and in which mode; an operation that records
    /// telemetry opens its span here and hands it to
    /// [`Call::instrument`].
    fn fold(request: &Self::Request, call: &mut Call<'_>) -> Self::Fold;
}

/// Events a decoder builds and writes with [`Out::event`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Free {}

/// Events only the completion writer's part handles build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assembled {}

/// Who builds an operation's events: [`Free`] or [`Assembled`]. Sealed.
pub trait Emit<Op: Operation>: reply::Closing<Op> {}

impl<Op: Operation> reply::Closing<Op> for Free {
    fn close(_shared: &mut Shared<Op>) {}
}

impl<Op: Operation> Emit<Op> for Free {}

/// What the driver knows about a call before its first frame.
pub struct Call<'a> {
    /// What the wire says about itself.
    pub wire: &'a Descriptor<'a>,
    /// How the reply arrives.
    pub mode: Mode,
    pub(crate) span: tracing::Span,
}

impl<'a> Call<'a> {
    /// A call to the wire `wire` describes, in `mode`, under no span.
    pub(crate) fn new(wire: &'a Descriptor<'a>, mode: Mode) -> Self {
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
/// Each event is [`absorb`](Self::absorb)ed as the consumer takes it, so
/// the fold never runs ahead of what was delivered. The reply's end is what
/// [`finish`](Self::finish) needs: without the provider's end there is no
/// response.
pub trait Fold<Op: Operation> {
    /// Absorb one event the consumer is about to see. An error fails the
    /// reply.
    fn absorb(&mut self, event: &Op::Event) -> Result<(), ProviderError>;

    /// The response, from the events absorbed, the provider's end and what
    /// the driver learned about the reply.
    fn finish(self, end: Op::End, reply: Reply) -> Result<Op::Response, ProviderError>;
}

/// What the driver learned about a reply beyond its events.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    /// The provider descriptor name, for the response's `provider` field.
    pub provider: String,
    /// The reply's provider document (a completion's `raw`): the whole body
    /// when it is one JSON document, else what the decoder recorded, else
    /// `Null`.
    pub raw: serde_json::Value,
    /// The provider's transport request id from the reply headers.
    pub provider_request_id: Option<String>,
}

/// Whether a decoder step left the reply open. Only [`Out::end`] makes an
/// [`Ended`].
#[derive(Debug)]
#[must_use]
pub enum Flow {
    /// The reply continues.
    More,
    /// The provider ended the reply; nothing is read after it.
    Ended(Ended),
}

/// Proof that a decoder ended its reply with [`Out::end`].
#[derive(Debug)]
pub struct Ended(());

pub(crate) use reply::Shared;

pub(crate) mod reply {
    use std::collections::VecDeque;

    use super::{Operation, ProviderError};
    use crate::streaming::Item;

    /// One reply's items and end, shared by the driver that writes them and
    /// the stream that takes them.
    pub struct Shared<Op: Operation> {
        pub(crate) fold: Op::Fold,
        pub(crate) items: VecDeque<Result<Item<Op::Event>, ProviderError>>,
        pub(crate) end: Option<Op::End>,
        pub(crate) raw: Option<serde_json::Value>,
        /// The response a relayed reply's origin already folded.
        pub(crate) response: Option<Op::Response>,
        /// What the transport reported: the reply's request id, its whole
        /// document and its request path.
        pub(crate) request_id: Option<String>,
        pub(crate) document: Option<serde_json::Value>,
        pub(crate) route: String,
    }

    impl<Op: Operation> Shared<Op> {
        pub(crate) fn new(fold: Op::Fold) -> Self {
            Self {
                fold,
                items: VecDeque::new(),
                end: None,
                raw: None,
                response: None,
                request_id: None,
                document: None,
                route: String::new(),
            }
        }
    }

    /// What the reply's end closes before it is recorded.
    pub trait Closing<Op: Operation> {
        fn close(shared: &mut Shared<Op>);
    }
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
    /// An event a typed transport (an SDK event stream, a gRPC stream)
    /// reported that this client does not model. The detail is kept for
    /// raw passthrough, never for warning logs.
    pub fn unrecognized(event_type: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Unknown {
            event_type: event_type.into(),
            value: serde_json::Value::String(detail.into()).into(),
        }
    }

    /// A modeled event a typed transport failed to decode.
    pub fn malformed(message: impl std::fmt::Display) -> Self {
        Self::Corrupt(<serde_json::Error as serde::de::Error>::custom(message))
    }

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

/// Where a decoder writes one reply. The `'id` brand ties it, and every
/// part handle it hands out, to that one reply: a handle cannot be used
/// with another reply's writer or kept past its own.
///
/// ```compile_fail
/// use rig_core::operation::Completion;
/// use rig_core::wire::Out;
///
/// // A text part opened on one reply cannot grow on another.
/// fn cross<'a, 'b>(a: &mut Out<'a, Completion>, b: &mut Out<'b, Completion>) {
///     let part = a.text();
///     b.push_text(&part, "x");
/// }
/// ```
///
/// ```compile_fail
/// use rig_core::operation::{Completion, TextPart};
/// use rig_core::wire::Out;
///
/// // A handle cannot outlive its reply to be used on a later one.
/// struct Stash(Option<TextPart<'static>>);
///
/// fn keep<'id>(stash: &mut Stash, out: &mut Out<'id, Completion>) {
///     stash.0 = Some(out.text());
/// }
/// ```
pub struct Out<'id, Op: Operation> {
    shared: &'id std::sync::Mutex<Shared<Op>>,
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id, Op: Operation> Out<'id, Op> {
    pub(crate) fn new(shared: &'id std::sync::Mutex<Shared<Op>>) -> Self {
        Self {
            shared,
            brand: std::marker::PhantomData,
        }
    }

    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'id, Shared<Op>> {
        self.shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// End the reply with what the provider sent at its end. It consumes the
    /// writer: nothing is written after the end.
    pub fn end(self, end: Op::End) -> Flow {
        let mut shared = self.lock();
        <Op::Emit as reply::Closing<Op>>::close(&mut shared);
        shared.end = Some(end);
        Flow::Ended(Ended(()))
    }

    /// Record the reply's provider document, the response's `raw`, for any
    /// operation. A whole JSON body the transport reported outranks it, and
    /// a `raw` the decoder writes onto its response is replaced: a decoder
    /// whose reply is not one JSON document records it here.
    pub fn raw(&mut self, raw: serde_json::Value) {
        self.lock().raw = Some(raw);
    }

    /// A payload the provider sent that this decoder does not model. It
    /// reaches the consumer as [`Item::Unknown`](crate::streaming::Item).
    pub fn unknown(&mut self, payload: crate::streaming::UnknownPayload) {
        self.lock()
            .items
            .push_back(Ok(crate::streaming::Item::Unknown(payload)));
    }
}

impl<Op: Operation<Emit = Free>> Out<'_, Op> {
    /// One event of the reply.
    ///
    /// Only an operation whose decoders build their own events has this; a
    /// completion's events come from its part handles:
    ///
    /// ```compile_fail,E0599
    /// use rig_core::operation::Completion;
    /// use rig_core::streaming::StreamEvent;
    /// use rig_core::wire::Out;
    ///
    /// fn reinject(out: &mut Out<'_, Completion>, seen: &StreamEvent) {
    ///     out.event(seen.clone());
    /// }
    /// ```
    ///
    /// An event rebuilt from its serialized form is no different:
    ///
    /// ```compile_fail,E0599
    /// use rig_core::operation::Completion;
    /// use rig_core::streaming::Transcript;
    /// use rig_core::wire::Out;
    ///
    /// fn rebuild(out: &mut Out<'_, Completion>, recorded: serde_json::Value) {
    ///     let Ok(transcript) = Transcript::parse(recorded) else { return };
    ///     for event in transcript.events() {
    ///         out.event(event.clone());
    ///     }
    /// }
    /// ```
    pub fn event(&mut self, event: Op::Event) {
        self.lock()
            .items
            .push_back(Ok(crate::streaming::Item::Event(event)));
    }
}

/// Synchronous state machine for one reply. It classifies frames and
/// decodes known events into the reply's writer, without transport access
/// and without being told how the reply arrives. HTTP wires read
/// [`WireFrame`]s; other transports name their own frame type.
pub trait Decoder<'id, Op: Operation, Frame = WireFrame> {
    /// The wire's typed event, produced by this decoder's classifier.
    type Event;

    /// Decode and classify one frame. A JSON wire MUST delegate to a
    /// classifier in [`crate::providers::internal::wire`], so the
    /// decode-then-validate policy is not re-derived per provider.
    fn classify(&self, frame: Frame) -> WireEvent<Self::Event>;

    /// Write one `Known` event into the reply. An error ends the reply with
    /// it.
    fn decode(&mut self, event: Self::Event, out: Out<'id, Op>) -> Result<Flow, ProviderError>;

    /// The frames ran out without an end. A decoder that already saw the
    /// provider's end, in a form the provider sends before its last frame,
    /// ends the reply here; otherwise the reply is truncated.
    fn eof(&mut self, out: Out<'id, Op>) -> Result<Flow, ProviderError> {
        let _ = out;
        Err(ProviderError::Truncated)
    }
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
    /// The decoder for one of its replies, branded with that reply.
    type Decoder<'id>: Decoder<'id, Self::Op, Self::Frame> + WasmCompatSend;

    /// What the wire says about itself.
    fn describe(&self) -> Descriptor<'_>;

    /// The request to send. Pure: it may read `self`, `request` and `mode`,
    /// and nothing else. A request that cannot be built is an
    /// [`EncodeError`], which always reports as a request failure.
    fn encode(&self, request: Request<Self>, mode: Mode) -> Result<Self::Payload, EncodeError>;

    /// A fresh decoder for one reply.
    fn decoder<'id>(&self) -> Self::Decoder<'id>;
}

/// The decoder of a reply that is one JSON document: the document is the
/// operation's end, and is kept as the reply's `raw`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Json;

impl<'id, Op> Decoder<'id, Op, WireFrame> for Json
where
    Op: Operation,
    Op::End: serde::de::DeserializeOwned,
{
    type Event = (Op::End, serde_json::Value);

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        // Read from the text, so an end holding raw JSON can borrow it.
        let text = frame.as_str();
        let document = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(document) => document,
            Err(error) => return WireEvent::Corrupt(error),
        };
        match serde_json::from_str::<Op::End>(&text) {
            Ok(end) => WireEvent::Known((end, document)),
            Err(error) => WireEvent::Corrupt(error),
        }
    }

    fn decode(
        &mut self,
        (end, document): Self::Event,
        mut out: Out<'id, Op>,
    ) -> Result<Flow, ProviderError> {
        out.raw(document);
        Ok(out.end(end))
    }
}

/// A wire's request type.
pub type Request<W> = <<W as Wire>::Op as Operation>::Request;
/// A wire's response type.
pub type Response<W> = <<W as Wire>::Op as Operation>::Response;
/// A wire's event type.
pub type Event<W> = <<W as Wire>::Op as Operation>::Event;
