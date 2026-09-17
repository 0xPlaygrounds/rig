//! What a provider is: data, an encoder, and a decoder.
//!
//! A [`Wire`] holds no transport, no future, no `dyn`, and no type
//! parameter. It turns one request into bytes ([`Wire::encode`]) and hands
//! out a fresh [`Decoder`] for one reply. The reply's frames fold into the
//! operation's response, and the only `async` in the provider layer is the
//! two functions in [`crate::driver`] that push bytes between them.
//!
//! Unary and streaming replies go through the *same* decoder: a unary reply
//! is a stream of one frame, and a provider whose unary body has a different
//! shape from its stream events names that shape in `classify` as one more
//! [`WireEvent`] variant. There is no `decode`, so the two paths cannot drift.
//!
//! # Writing a provider
//!
//! A provider is a config struct plus one wire per operation, and a dialect
//! constant per gateway that speaks the same format. This is a complete one,
//! end to end:
//!
//! ```
//! use rig_core::completion::{CompletionError, CompletionRequest};
//! use rig_core::driver::Bound;
//! use rig_core::operation::{Completion, CompletionEvent};
//! use rig_core::streaming::{StreamEvent, StreamFinal};
//! use rig_core::wire::{
//!     Body, Decoder, Encoded, Framing, Mode, Output, Secret, Wire, WireEvent, WireFrame,
//! };
//!
//! /// What differs between gateways speaking this format: data, `const`.
//! /// A dialect serializes as its name and deserializes by looking the
//! /// name up, so a config stays plain data without copying the constant.
//! #[derive(Clone, Debug, PartialEq)]
//! pub struct Dialect {
//!     pub name: &'static str,
//!     pub base_url: &'static str,
//!     pub api_key_env: &'static str,
//! }
//!
//! pub const EXAMPLE: Dialect = Dialect {
//!     name: "example",
//!     base_url: "https://example.invalid/v1",
//!     api_key_env: "EXAMPLE_API_KEY",
//! };
//! const ALL: &[Dialect] = &[EXAMPLE];
//!
//! impl serde::Serialize for Dialect {
//!     fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
//!         serializer.serialize_str(self.name)
//!     }
//! }
//! impl<'de> serde::Deserialize<'de> for Dialect {
//!     fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
//!         let name = String::deserialize(deserializer)?;
//!         ALL.iter()
//!             .find(|dialect| dialect.name == name)
//!             .cloned()
//!             .ok_or_else(|| serde::de::Error::custom(format!("unknown dialect `{name}`")))
//!     }
//! }
//!
//! /// The provider's shared configuration: plain data, key redacted.
//! #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
//! pub struct Example {
//!     pub dialect: Dialect,
//!     pub api_key: Secret,
//!     pub base_url: String,
//! }
//!
//! impl Example {
//!     pub fn new(api_key: impl Into<Secret>) -> Self {
//!         Self::with_dialect(EXAMPLE, api_key)
//!     }
//!     pub fn with_dialect(dialect: Dialect, api_key: impl Into<Secret>) -> Self {
//!         Self { base_url: dialect.base_url.to_owned(), dialect, api_key: api_key.into() }
//!     }
//!     /// The completion wire.
//!     pub fn messages(&self, model: impl Into<String>) -> Messages {
//!         Messages { provider: self.clone(), model: model.into() }
//!     }
//! }
//!
//! /// One operation's wire: the config plus what this endpoint needs.
//! #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
//! pub struct Messages {
//!     pub provider: Example,
//!     pub model: String,
//! }
//!
//! /// One frame. The unary body is a whole `message`; a stream sends
//! /// `delta`s and a `stop`. Both are named here, and nowhere else.
//! #[derive(serde::Deserialize)]
//! #[serde(tag = "type", rename_all = "snake_case")]
//! pub enum Frame {
//!     Message { text: String },
//!     Delta { text: String },
//!     Stop,
//! }
//!
//! #[derive(Default)]
//! pub struct ExampleDecoder;
//!
//! impl Decoder<Completion> for ExampleDecoder {
//!     type Event = Frame;
//!
//!     fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
//!         match serde_json::from_str(&frame.as_str()) {
//!             Ok(frame) => WireEvent::Known(frame),
//!             Err(error) => WireEvent::Corrupt(error),
//!         }
//!     }
//!
//!     fn interpret(&mut self, event: Self::Event, out: &mut Output<Completion>) {
//!         match event {
//!             // The unary shape synthesizes the stream's events; it does
//!             // not carry a second content mapping.
//!             Frame::Message { text } => {
//!                 self.interpret(Frame::Delta { text }, out);
//!                 self.interpret(Frame::Stop, out);
//!             }
//!             Frame::Delta { text } => out.text(text),
//!             Frame::Stop => {
//!                 out.close_active_blocks();
//!                 out.final_record(StreamFinal::new(
//!                     EXAMPLE.name,
//!                     rig_core::completion::Usage::default(),
//!                 ));
//!             }
//!         }
//!     }
//! }
//!
//! impl Wire for Messages {
//!     type Op = Completion;
//!     type Decoder = ExampleDecoder;
//!
//!     fn name(&self) -> &str {
//!         self.provider.dialect.name
//!     }
//!
//!     fn model(&self) -> Option<&str> {
//!         Some(&self.model)
//!     }
//!
//!     fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
//!         // The body says whether to stream; the framing says how the
//!         // reply splits. Both follow from the mode and nothing else.
//!         let streaming = matches!(mode, Mode::Streaming);
//!         let body = serde_json::json!({
//!             "model": self.model,
//!             "messages": request.chat_history,
//!             "stream": streaming,
//!         });
//!         let request = http::Request::post(format!("{}/messages", self.provider.base_url))
//!             .header("authorization", self.provider.api_key.expose())
//!             .body(Body::Bytes(serde_json::to_vec(&body)?))
//!             .map_err(|error| CompletionError::ResponseError(error.to_string()))?;
//!         let framing = if streaming { Framing::Sse } else { Framing::Whole };
//!         Ok(Encoded::new(request, framing))
//!     }
//!
//!     fn decoder(&self, _mode: Mode) -> Self::Decoder {
//!         ExampleDecoder
//!     }
//! }
//!
//! # fn main() {
//! // A wire plus a socket is a `CompletionModel`.
//! let _ = |http: rig_core::http_client::BoxedHttpClient| {
//!     Bound::new(Example::new("k").messages("m"), http)
//! };
//! // The wire is data: it serializes, and the key does not.
//! let json = serde_json::to_string(&Example::new("k").messages("m")).unwrap();
//! assert!(!json.contains("\"k\""));
//!
//! // Decoding is testable from bytes alone, with no socket at all — and the
//! // unary body folds to the same events as the stream that says the same.
//! fn events(frames: &[&str]) -> Vec<CompletionEvent> {
//!     let mut decoder = ExampleDecoder;
//!     let mut out = Output::<Completion>::new();
//!     for frame in frames {
//!         let WireEvent::Known(event) = decoder.classify(WireFrame::Text((*frame).into()))
//!         else {
//!             unreachable!("the fixture is a modeled frame")
//!         };
//!         decoder.interpret(event, &mut out);
//!     }
//!     out.drain().map(|item| item.unwrap()).collect()
//! }
//! let unary = events(&[r#"{"type":"message","text":"hi"}"#]);
//! let streamed = events(&[r#"{"type":"delta","text":"hi"}"#, r#"{"type":"stop"}"#]);
//! assert_eq!(unary, streamed);
//! assert!(matches!(unary.last(), Some(StreamEvent::Final(_))));
//! # }
//! ```

use std::borrow::Cow;

use crate::http_client::MultipartForm;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub use crate::http_client::framing::Framing;
pub use crate::observe::{AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict};
pub use crate::providers::internal::wire::WireEvent;

mod error;
pub(crate) mod secret;

pub use error::WireError;
pub(crate) use error::impl_wire_error;
pub use secret::Secret;

/// One transport frame, after framing but before decoding.
///
/// The transport layer (SSE framer, NDJSON splitter, websocket reader) owns
/// byte splitting and yields these; a decoder never splits bytes.
#[derive(Debug, Clone)]
pub enum WireFrame {
    /// A decoded text payload — an SSE `data:` field or a ws message body.
    Text(String),
    /// A raw byte payload — an NDJSON line or a binary SDK frame.
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
    /// The HTTP requests to send, in order — one for all but the batch
    /// endpoints. A wire whose provider takes one item per request (Cohere
    /// embeds one image per call) returns one request per item through
    /// [`Encoded::batch`], and the driver folds every reply into the one
    /// response.
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
            .finish()
    }
}

/// Which reply a request asks for.
///
/// A wire that asks for a streamed reply sends a *different request* — the
/// chat wires set `stream: true`, Gemini switches endpoint and adds
/// `?alt=sse` — and recorded traffic pins those bytes, so the mode is an
/// input to [`Wire::encode`] rather than something the driver adds after
/// the fact. It is the only thing the two call paths tell a wire, and the
/// *decoder* is the same either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One whole reply ([`crate::driver::call`]).
    Unary,
    /// A streamed reply ([`crate::driver::stream`]).
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
    type Response;
    /// The operation's error enum.
    type Error: WireError;
    /// What a runtime accounts for. `()` for operations with nothing to
    /// declare.
    type Capabilities: Default;
    /// Where a decoder writes the events of one `interpret` step.
    type Output: Sink<Self> + WasmCompatSend;
    /// The fold from events to the response.
    type Fold: Fold<Self>;
    /// The canonical telemetry operation a wire performs. `()` for
    /// operations that open no span.
    type Telemetry: Copy;

    /// The operation's name, as telemetry and records spell it.
    const NAME: &'static str;

    /// Whether this event is the provider's genuine terminal — after it the
    /// driver stops consuming.
    fn is_terminal(event: &Self::Event) -> bool;

    /// The fold for one reply, seeded from the request.
    ///
    /// An embedding response joins the provider's vectors back onto the
    /// request's input texts, and nothing downstream of
    /// [`Wire::encode`] can see the request — so the operations whose
    /// response needs its own input take it here, once, instead of every
    /// wire carrying a copy.
    fn fold(_request: &Self::Request) -> Self::Fold {
        Self::Fold::default()
    }

    /// Stamp the transport request id read off the reply's headers onto a
    /// terminal event. Operations whose events carry no transport id do
    /// nothing.
    fn stamp_request_id(_event: &mut Self::Event, _request_id: &Option<String>) {}

    /// Stamp what the driver learned about a unary reply beyond its events.
    fn stamp_reply(_response: &mut Self::Response, _reply: Reply) {}

    /// The operation's channel for an unmodeled frame's raw payload.
    /// `None` — the default — skips it: only a stream of assistant content
    /// has somewhere to put a frame nothing models.
    fn unknown(_payload: crate::streaming::UnknownPayload) -> Option<Self::Event> {
        None
    }

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

    /// Record what one streamed event says onto the operation's span.
    ///
    /// A streamed reply is never folded by the driver — the consumer owns
    /// the fold — so the terminal event is where a stream's response
    /// metadata and usage come from. Off the same span `record` writes to,
    /// so a unary and a streamed call report the same fields.
    fn record_event(_span: &tracing::Span, _event: &Self::Event) {}
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
    fn push(&mut self, item: Result<Op::Event, Op::Error>);

    /// Take everything pushed since the last drain.
    fn drain(&mut self) -> std::vec::Drain<'_, Result<Op::Event, Op::Error>>;

    /// What this sink holds, without taking it.
    fn items(&self) -> &[Result<Op::Event, Op::Error>];

    /// Check the operation's sequence laws over this batch.
    fn check_laws(&self, _laws: &mut Self::Laws) {}
}

/// The fold from a reply's events to its response.
pub trait Fold<Op: Operation>: Default {
    /// Absorb one event. An error fails the whole operation: a buffered
    /// reply has no stream to carry an in-band defect.
    fn absorb(&mut self, event: Op::Event) -> Result<(), Op::Error>;

    /// The folded response.
    fn finish(self, reply: Reply) -> Result<Op::Response, Op::Error>;
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

/// A sync state machine over one reply's frames.
///
/// Everything a decoder does is pure and synchronous; frame-triage policy
/// is the driver's (see [`crate::driver`]), so a decoder contains no
/// `match WireEvent`.
///
/// `Frame` is [`WireFrame`] for every HTTP wire — bytes the framers split.
/// A typed transport (an AWS event stream, a gRPC stream, an in-process
/// generator) names its SDK's event type instead and inherits the same fold
/// through [`run_wire_stream`](crate::driver::run_wire_stream).
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

    /// The reply as one document, for a wire whose unary reply is not one.
    ///
    /// The driver captures a unary reply's bytes as `raw` by parsing them,
    /// which is the verbatim document for every wire that answers with one.
    /// A wire that answers a *unary* call with an event stream — the
    /// Responses endpoint does, on the dialects that always stream — has no
    /// such document, and its terminal event carries the envelope instead.
    /// Returning it here is what keeps `raw` the reply rather than a
    /// summary of it.
    fn document(&self) -> Option<serde_json::Value> {
        None
    }

    /// A paged operation's next request, if the reply named one.
    fn continuation(&self) -> Option<http::Request<Body>> {
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
    /// The decoder for one of its replies.
    type Decoder: Decoder<Self::Op> + WasmCompatSend + 'static;

    /// The provider descriptor name (`"anthropic"`), as records and
    /// telemetry name it.
    fn name(&self) -> &str;

    /// The request to send. Pure: it may read `self`, `request` and `mode`,
    /// and nothing else.
    fn encode(&self, request: Request<Self>, mode: Mode) -> Result<Encoded, Error<Self>>;

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

    /// The model this wire addresses, for telemetry. `None` for operations
    /// that address no model.
    fn model(&self) -> Option<&str> {
        None
    }

    /// The route template this wire posts to, as the provider declares it:
    /// the endpoint's own path, with no base-URL prefix and no interpolated
    /// values. Observation groups attempts by it, so it must not change when
    /// a caller points the same endpoint at a different base URL — which the
    /// concrete request path does, since a dialect's base URL may carry a
    /// version segment (`https://api.openai.com/v1` + `/responses`).
    ///
    /// `None` is "this wire declares no template apart from the path it
    /// builds", and the driver then reports that path. Override it wherever
    /// the declaration and the request path can differ.
    fn route(&self) -> Option<&str> {
        None
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
/// A wire's error type.
pub type Error<W> = <<W as Wire>::Op as Operation>::Error;
/// A wire's capability type.
pub type Capabilities<W> = <<W as Wire>::Op as Operation>::Capabilities;
/// A wire's telemetry operation type.
pub type Telemetry<W> = <<W as Wire>::Op as Operation>::Telemetry;

/// A provider config that has a completion wire.
///
/// One small trait, implemented by provider config structs, so
/// `Bound<P, H>` can build the provider's completion wire and rig-agent can
/// offer `agent(model)` / `extractor(model)` on it without naming a provider.
pub trait HasCompletion: WasmCompatSend + WasmCompatSync {
    /// The provider's completion wire.
    type Wire: Wire<Op = crate::operation::Completion>;

    /// Build the completion wire for `model`.
    fn completion(&self, model: impl Into<String>) -> Self::Wire;
}
