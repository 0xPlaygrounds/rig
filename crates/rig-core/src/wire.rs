//! The wire system: data, an encoder, and a decoder.
//!
//! A [`Wire`] is what a provider is: plain data (Clone + PartialEq + Debug +
//! Serialize + Deserialize; secrets redacted by [`Secret`]), an encoder, and a
//! decoder. No transport, no future, no type parameter.

use std::collections::VecDeque;

use crate::http_client::framing::Framing;
use crate::operation::Operation;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub use crate::http_client::framing;
pub use crate::providers::internal::adapter::WireFrame;
pub use crate::providers::internal::wire::WireEvent;

/// What a provider is: data (Clone + PartialEq + Debug + Serialize + Deserialize; secrets redacted
/// by `Secret`), an encoder, and a decoder. No transport, no future, no type parameter.
pub trait Wire: WasmCompatSend + WasmCompatSync + 'static {
    type Op: Operation;
    type Decoder: Decoder<Self::Op>;

    /// Provider descriptor name (`"anthropic"`), as records and telemetry name it.
    fn name(&self) -> &str;

    /// The request to send. Pure: it may read `self` and `request`, and nothing else.
    fn encode(
        &self,
        request: <Self::Op as Operation>::Request,
    ) -> Result<Encoded, <Self::Op as Operation>::Error>;

    /// A fresh decoder for one reply.
    fn decoder(&self) -> Self::Decoder;

    /// What a runtime accounts for (today `ProviderCapabilities`, completion only; `()` elsewhere).
    fn capabilities(&self) -> <Self::Op as Operation>::Capabilities;
}

/// The encoded HTTP request and its expected framing.
pub struct Encoded {
    pub request: http::Request<Body>,
    pub framing: Framing,
    pub request_id_header: Option<&'static str>,
    pub route: &'static str,
}

impl Encoded {
    pub fn new(request: http::Request<Body>, framing: Framing, route: &'static str) -> Self {
        Self {
            request,
            framing,
            request_id_header: None,
            route,
        }
    }

    pub fn with_request_id_header(mut self, header: &'static str) -> Self {
        self.request_id_header = Some(header);
        self
    }
}

/// The body of an encoded HTTP request: raw bytes or multipart form.
#[derive(Debug)]
pub enum Body {
    Bytes(Vec<u8>),
    Multipart(crate::http_client::MultipartForm),
}

impl From<Vec<u8>> for Body {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }
}

impl From<crate::http_client::MultipartForm> for Body {
    fn from(form: crate::http_client::MultipartForm) -> Self {
        Self::Multipart(form)
    }
}

impl Body {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Bytes(bytes) => bytes.is_empty(),
            Self::Multipart(_) => false,
        }
    }
}

/// The decoder: a sync state machine over frames. `WireAdapter` generalized to any operation.
pub trait Decoder<Op: Operation>: 'static {
    type Event: 'static;
    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event>;
    fn interpret(&mut self, event: Self::Event, out: &mut Output<Op>);
    fn finish(&mut self, out: &mut Output<Op>);
    fn flush_before_terminal_error(&mut self, out: &mut Output<Op>);

    /// The observation projection: verdicts, usage, ids, error envelopes read off a raw payload
    /// before normalization discards them. Replaces `PayloadObserver`; the decoder already
    /// understands the payload.
    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink);

    /// A paged operation's next request, if the reply named one (model listing).
    fn continuation(&self) -> Option<http::Request<Body>> {
        None
    }
}

/// What one `interpret` or `finish` step emitted: canonical events or in-band errors.
#[derive(Debug)]
pub struct Output<Op: Operation> {
    pub(crate) items: VecDeque<Result<Op::Event, Op::Error>>,
}

impl<Op: Operation> Default for Output<Op> {
    fn default() -> Self {
        Self {
            items: VecDeque::new(),
        }
    }
}

impl<Op: Operation> Output<Op> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit(&mut self, event: Op::Event) {
        self.items.push_back(Ok(event));
    }

    pub fn error(&mut self, error: Op::Error) {
        self.items.push_back(Err(error));
    }

    pub fn push(&mut self, item: Result<Op::Event, Op::Error>) {
        self.items.push_back(item);
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Result<Op::Event, Op::Error>> + '_ {
        self.items.drain(..)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// Target for observation projections read off raw wire payloads.
pub trait ObservationSink {
    fn emit(&mut self, event: crate::observe::AdapterEvent);
    fn provider(&mut self, verdict: crate::observe::AdapterVerdict, response_id: Option<String>);
    fn text(&self, text: &str) -> String {
        text.to_owned()
    }
}

/// A secret value whose Debug and Serialize representations are redacted as `"[redacted]"`.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Secret(String);

impl Secret {
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[redacted]")
    }
}

impl serde::Serialize for Secret {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str("[redacted]")
    }
}

impl<'de> serde::Deserialize<'de> for Secret {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(s))
    }
}

impl From<String> for Secret {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Secret {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl std::ops::Deref for Secret {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for Secret {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests;
