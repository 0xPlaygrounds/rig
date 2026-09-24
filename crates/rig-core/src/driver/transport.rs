//! How a wire's payload travels: the [`Transport`] contract, and its HTTP
//! implementation for every [`HttpClientExt`].

use std::future::Future;

use bytes::Bytes;
use futures::StreamExt;

use crate::error::ProviderError;
use crate::http_client::framing::{Framing, NdjsonFramer, SseFramer};
use crate::http_client::{self, HttpClientExt};
use crate::observe::{AdapterContext, AdapterErrorBoundary, AdapterSlot};
use crate::wasm_compat::{WasmBoxedStream, WasmCompatSend, WasmCompatSync};
use crate::wire::{Body, Encoded, Mode, WireFrame};

/// Sends payloads and delivers their replies as frames.
///
/// A transport knows payloads and frames, not wires or operations. Every
/// [`HttpClientExt`] is the transport for [`Encoded`] requests and
/// [`WireFrame`] replies.
pub trait Transport: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// What one send carries.
    type Payload: WasmCompatSend + 'static;
    /// One unit of a reply, before decoding.
    type Frame: WasmCompatSend + 'static;

    /// Send `payload` and open its reply.
    ///
    /// An error is a payload the transport refused to send in `mode`, before
    /// anything was sent. Every later failure, including one to open the
    /// reply, is the last item of [`Opened::frames`].
    ///
    /// The HTTP transport reads the
    /// [`AdapterContext`](crate::observe::AdapterContext) observing the
    /// attempt from the outgoing request's extensions, where a completion
    /// wire moves its request's extensions.
    fn send(
        &self,
        payload: Self::Payload,
        mode: Mode,
    ) -> impl Future<Output = Result<Opened<Self::Payload, Self::Frame>, ProviderError>> + WasmCompatSend;
}

/// A reply the transport opened: its frames, and the facts the transport owns.
pub struct Opened<P, F> {
    /// The reply's frames in order. A transport failure is the last item.
    pub frames: WasmBoxedStream<'static, Result<F, ProviderError>>,
    /// The provider's transport request id, when the reply carried one.
    pub request_id: Option<String>,
    /// The reply's status, when the transport has one.
    pub status: Option<http::StatusCode>,
    /// The reply's headers, when the transport has them.
    pub headers: Option<http::HeaderMap>,
    /// The concrete request path, when the transport has one.
    pub route: Option<String>,
    /// The whole reply body, when the transport buffered it.
    pub body: Option<Bytes>,
    /// What remains of a payload that carried several requests. The driver
    /// sends it next.
    pub rest: Option<P>,
    /// The attempt's observation, when the payload carried a context. The
    /// driver records the decoder's facts into the same attempt.
    pub(crate) observation: Option<AdapterSlot>,
}

impl<P, F> Opened<P, F> {
    /// A reply of `frames` with no transport facts.
    pub fn new(
        frames: impl futures::Stream<Item = Result<F, ProviderError>> + WasmCompatSend + 'static,
    ) -> Self {
        Self {
            frames: Box::pin(frames),
            request_id: None,
            status: None,
            headers: None,
            route: None,
            body: None,
            rest: None,
            observation: None,
        }
    }

    /// A reply that failed with `error` before any frame.
    pub fn failed(error: ProviderError) -> Self
    where
        F: WasmCompatSend + 'static,
    {
        Self::new(futures::stream::once(async move { Err(error) }))
    }
}

impl<H> Transport for H
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    type Payload = Encoded;
    type Frame = WireFrame;

    async fn send(
        &self,
        payload: Encoded,
        mode: Mode,
    ) -> Result<Opened<Encoded, WireFrame>, ProviderError> {
        let Encoded {
            requests,
            framing,
            request_id_header,
            relaxed_content_type,
            route,
        } = payload;
        // No streamed operation sends a batch: a batch exists for providers
        // that take one item per request, and those are all unary.
        if mode == Mode::Streaming && requests.len() != 1 {
            return Err(ProviderError::Request(
                format!(
                    "a streamed reply takes exactly one request, not {}",
                    requests.len()
                )
                .into(),
            ));
        }
        let mut requests = requests.into_iter();
        let Some(mut request) = requests.next() else {
            return Err(ProviderError::Request(
                "an encoded payload carries no request".into(),
            ));
        };
        let rest: Vec<_> = requests.collect();
        let rest = (!rest.is_empty()).then(|| Encoded {
            requests: rest,
            framing,
            request_id_header,
            relaxed_content_type,
            route,
        });
        accept_header(&mut request, framing);
        let observation = Observation::from(request.extensions());
        let slot = observation
            .as_ref()
            .map(|observation| observation.slot.clone());
        // Errors name the concrete path; observations group attempts by the
        // template the wire declared, when it declared one.
        let path = request.uri().path().to_owned();
        let exchange = Exchange {
            framing,
            request_id_header,
            relaxed_content_type,
            declared: route.map_or_else(|| path.clone(), str::to_owned),
            path,
            observation,
        };
        let mut opened = match mode {
            Mode::Unary => {
                exchange.install(&request);
                exchange.unary(self, request).await
            }
            Mode::Streaming => {
                let request = byte_request(request)?;
                exchange.install(&request);
                exchange.streaming(self, request).await
            }
        };
        opened.rest = rest;
        opened.observation = slot;
        Ok(opened)
    }
}

/// What an observed exchange records into: the context read off the
/// outgoing request's extensions, and the attempt's slot.
struct Observation {
    context: AdapterContext,
    slot: AdapterSlot,
}

impl Observation {
    fn from(extensions: &http::Extensions) -> Option<Self> {
        Some(Self {
            context: extensions.get::<AdapterContext>()?.clone(),
            slot: AdapterSlot::default(),
        })
    }
}

/// What one request's exchange needs besides the client.
struct Exchange {
    framing: Framing,
    request_id_header: Option<&'static str>,
    relaxed_content_type: bool,
    /// The concrete request path, for errors.
    path: String,
    /// The path template observations group this attempt under.
    declared: String,
    observation: Option<Observation>,
}

impl Exchange {
    fn slot(&self) -> Option<&AdapterSlot> {
        self.observation
            .as_ref()
            .map(|observation| &observation.slot)
    }

    fn install<B>(&self, request: &http::Request<B>) {
        if let Some(observation) = &self.observation {
            observation
                .slot
                .install(observation.context.attempt_for(request, &self.declared));
        }
    }

    fn failed(
        self,
        error: ProviderError,
        request_id: Option<String>,
        body: Option<Bytes>,
    ) -> Opened<Encoded, WireFrame> {
        let mut opened = Opened::failed(error);
        opened.request_id = request_id;
        opened.route = Some(self.path);
        opened.body = body;
        opened
    }

    /// Send a buffered request and frame its whole body.
    async fn unary<H: HttpClientExt>(
        self,
        http: &H,
        request: http::Request<Body>,
    ) -> Opened<Encoded, WireFrame> {
        let sent = match send(http, request, self.request_id_header, self.slot()).await {
            Ok(sent) => sent,
            Err(error) => return self.failed(error, None, None),
        };
        // The status said success, but an SSE framer over a body that is not
        // an event stream yields no frames at all, which would fold to a
        // contentless success. The reply the provider actually sent is the
        // error.
        if let Some(rejected) =
            wrong_content_type(&sent.headers, self.framing, self.relaxed_content_type)
        {
            let error = ProviderError::from_transport_error(rejected)
                .with_provider_status(Some(sent.status))
                .with_provider_request_id(sent.provider_request_id.clone())
                .with_response_headers(Some(sent.headers.clone()));
            return self.failed(error, sent.provider_request_id, Some(sent.body));
        }
        // Frame the reply the way a stream is framed: a wire whose unary
        // reply is an event stream reads it frame by frame.
        let mut framer = Framer::new(self.framing);
        let frames: Vec<Result<WireFrame, ProviderError>> = framer
            .push(&sent.body)
            .into_iter()
            .chain(framer.finish())
            .filter_map(|payload| payload.into_frame().map(Ok))
            .collect();
        Opened {
            request_id: sent.provider_request_id,
            status: Some(sent.status),
            headers: Some(sent.headers),
            route: Some(self.path),
            body: Some(sent.body),
            ..Opened::new(futures::stream::iter(frames))
        }
    }

    /// Open a streamed reply and frame its body as it arrives. A chunk's
    /// bytes are recorded before its frames are read.
    async fn streaming<H: HttpClientExt>(
        self,
        http: &H,
        request: http::Request<Vec<u8>>,
    ) -> Opened<Encoded, WireFrame> {
        let response = match http.send_streaming(request).await {
            // Custom transports may return rejected responses directly; preserve
            // their status, headers, and bounded body in the error.
            Ok(response) if response.status() != http::StatusCode::OK => {
                Err(reject_response(response).await)
            }
            Ok(response) => {
                match wrong_content_type(
                    response.headers(),
                    self.framing,
                    self.relaxed_content_type,
                ) {
                    Some(error) => Err(error),
                    None => Ok(response),
                }
            }
            other => other,
        };
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                if let Some(slot) = self.slot() {
                    slot.error_boundary(AdapterErrorBoundary::from_http(&error));
                    if let Some(status) = error.non_success_status() {
                        slot.response_with_headers(status, error.non_success_headers());
                    }
                }
                let request_id = error
                    .non_success_headers()
                    .and_then(|headers| request_id_from(headers, self.request_id_header));
                let error = ProviderError::from_transport_error(error)
                    .with_provider_request_id(request_id.clone());
                return self.failed(error, request_id, None);
            }
        };
        if let Some(slot) = self.slot() {
            slot.response_with_headers(response.status(), Some(response.headers()));
        }
        let request_id = request_id_from(response.headers(), self.request_id_header);
        let status = response.status();
        let headers = response.headers().clone();
        let Self {
            framing,
            path,
            observation,
            ..
        } = self;
        let slot = observation.map(|observation| observation.slot);
        let mut body = response.into_body();
        let frames = async_stream::stream! {
            let mut framer = Framer::new(framing);
            while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        if let Some(slot) = &slot {
                            slot.error_boundary(AdapterErrorBoundary::Transport);
                        }
                        yield Err(ProviderError::from_transport_error(error));
                        return;
                    }
                };
                if let Some(slot) = &slot {
                    slot.bytes(&chunk);
                }
                for payload in framer.push(&chunk) {
                    if let Some(frame) = payload.into_frame() {
                        yield Ok(frame);
                    }
                }
            }
            for payload in framer.finish() {
                if let Some(frame) = payload.into_frame() {
                    yield Ok(frame);
                }
            }
        };
        Opened {
            request_id,
            status: Some(status),
            headers: Some(headers),
            route: Some(path),
            ..Opened::new(frames)
        }
    }
}

/// What one send learned about its reply.
struct Sent {
    status: http::StatusCode,
    headers: http::HeaderMap,
    body: Bytes,
    provider_request_id: Option<String>,
}

/// Sends a buffered request, preserving non-success response details and IDs.
async fn send<H>(
    http: &H,
    request: http::Request<Body>,
    request_id_header: Option<&'static str>,
    observation: Option<&AdapterSlot>,
) -> Result<Sent, ProviderError>
where
    H: HttpClientExt,
{
    let (parts, body) = request.into_parts();
    let response = match body {
        Body::Bytes(bytes) => {
            http.send::<_, Bytes>(http::Request::from_parts(parts, bytes))
                .await
        }
        Body::Multipart(form) => {
            http.send_multipart::<Bytes>(http::Request::from_parts(parts, form))
                .await
        }
    };
    let response = match response {
        Ok(response) => response,
        // A transport that reports the non-success reply as an error: the
        // reply is the provider's, so it funnels to a preserved provider
        // response with the id read off its headers and the headers
        // themselves; a response-less failure stays a transport error.
        Err(error) => {
            if let Some(observation) = observation
                && let Some(status) = error.non_success_status()
            {
                observation.response_with_headers(status, error.non_success_headers());
            }
            let request_id = error
                .non_success_headers()
                .and_then(|headers| request_id_from(headers, request_id_header));
            return Err(
                ProviderError::from_transport_error(error).with_provider_request_id(request_id)
            );
        }
    };

    // Take the reply apart before awaiting the body: the headers are then
    // owned, so preserving them onto an error costs no clone.
    let (parts, body) = response.into_parts();
    let status = parts.status;
    if let Some(observation) = observation {
        observation.response_with_headers(status, Some(&parts.headers));
    }
    let provider_request_id = request_id_from(&parts.headers, request_id_header);
    let body = body.await.map_err(ProviderError::from_transport_error)?;

    if !status.is_success() {
        return Err(
            ProviderError::from_http_response(status, String::from_utf8_lossy(&body))
                .with_provider_request_id(provider_request_id)
                .with_response_headers(Some(parts.headers)),
        );
    }
    Ok(Sent {
        status,
        headers: parts.headers,
        body,
        provider_request_id,
    })
}

/// The provider's transport request id, when it names such a header and the
/// reply carries a non-empty value.
fn request_id_from(headers: &http::HeaderMap, header: Option<&str>) -> Option<String> {
    crate::providers::internal::request_id_from_headers(headers, header)
}

/// Defaults byte-body requests to `application/json`, including bodyless GETs.
/// Preserves an explicitly supplied content type.
fn content_type(request: &mut http::Request<Body>) {
    if matches!(request.body(), Body::Bytes(_)) {
        request
            .headers_mut()
            .entry(http::header::CONTENT_TYPE)
            .or_insert(http::HeaderValue::from_static("application/json"));
    }
}

/// Add `Accept: text/event-stream` to an SSE request, without overriding an
/// `Accept` the wire set itself.
fn accept_header(request: &mut http::Request<Body>, framing: Framing) {
    content_type(request);
    if framing == Framing::Sse {
        request
            .headers_mut()
            .entry("Accept")
            .or_insert(http::HeaderValue::from_static("text/event-stream"));
    }
}

/// Whether a reply's content type is not the event stream an SSE wire asked
/// for: the framer would silently produce no frames, which reads as
/// truncation rather than as the wrong endpoint. A reply that names no
/// content type at all is accepted only by a wire that opted in.
///
/// The one predicate both modes ask, so a unary reply and a streamed one
/// cannot disagree about what the provider sent.
fn wrong_content_type(
    headers: &http::HeaderMap,
    framing: Framing,
    relaxed: bool,
) -> Option<http_client::Error> {
    if framing != Framing::Sse {
        return None;
    }
    let Some(content_type) = headers.get(&http::header::CONTENT_TYPE) else {
        return (!relaxed)
            .then(|| http_client::Error::InvalidContentType(http::HeaderValue::from_static("")));
    };
    let event_stream = content_type
        .to_str()
        .ok()
        .and_then(|value| value.parse::<mime::Mime>().ok())
        .is_some_and(|mime_type| {
            matches!(
                (mime_type.type_(), mime_type.subtype()),
                (mime::TEXT, mime::EVENT_STREAM)
            )
        });
    (!event_stream).then(|| http_client::Error::InvalidContentType(content_type.clone()))
}

/// A request whose body is bytes. Multipart replies are never streamed.
fn byte_request(request: http::Request<Body>) -> Result<http::Request<Vec<u8>>, ProviderError> {
    let (parts, body) = request.into_parts();
    match body {
        Body::Bytes(bytes) => Ok(http::Request::from_parts(parts, bytes)),
        Body::Multipart(_) => Err(ProviderError::Request(
            "a multipart request cannot open a streamed reply".into(),
        )),
    }
}

/// A framed payload. Whitespace-only SSE payloads are heartbeats: framed,
/// but never decoded.
struct Framed {
    payload: Vec<u8>,
    frame: bool,
}

impl Framed {
    fn into_frame(self) -> Option<WireFrame> {
        self.frame.then(|| match String::from_utf8(self.payload) {
            Ok(text) => WireFrame::Text(text),
            Err(error) => WireFrame::Bytes(error.into_bytes()),
        })
    }
}

/// The framer for one reply's bytes.
enum Framer {
    Sse(SseFramer),
    Ndjson(NdjsonFramer),
    Whole(Vec<u8>),
}

impl Framer {
    fn new(framing: Framing) -> Self {
        match framing {
            Framing::Sse => Self::Sse(SseFramer::new()),
            Framing::Ndjson => Self::Ndjson(NdjsonFramer::new()),
            Framing::Whole => Self::Whole(Vec::new()),
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Vec<Framed> {
        match self {
            Self::Sse(framer) => framer
                .push(chunk)
                .map(|event| Framed {
                    frame: !event.data.trim().is_empty(),
                    payload: event.data.into_bytes(),
                })
                .collect(),
            Self::Ndjson(framer) => framer
                .push(chunk)
                .map(|line| Framed {
                    frame: true,
                    payload: line,
                })
                .collect(),
            Self::Whole(buffer) => {
                buffer.extend_from_slice(chunk);
                Vec::new()
            }
        }
    }

    fn finish(&mut self) -> Vec<Framed> {
        match self {
            // The grammar dispatches only on a blank line: an unterminated
            // trailing event is not a frame.
            Self::Sse(_) => Vec::new(),
            Self::Ndjson(framer) => framer
                .finish()
                .map(|line| Framed {
                    frame: true,
                    payload: line,
                })
                .into_iter()
                .collect(),
            Self::Whole(buffer) => {
                let payload = std::mem::take(buffer);
                if payload.is_empty() {
                    Vec::new()
                } else {
                    vec![Framed {
                        frame: true,
                        payload,
                    }]
                }
            }
        }
    }
}

/// Bytes of a rejected reply's body kept on the error; a reply longer than
/// this is cut there.
const REJECTED_BODY_LIMIT: usize = 1 << 20;

/// Chunks read off a rejected reply before giving up on it, so a transport
/// that keeps yielding empty chunks cannot hold the opener.
const REJECTED_CHUNK_LIMIT: usize = 4096;

/// Turn a reply the transport will not stream (any status but 200, a 204
/// included: a status is a status) into the non-success error, reading the
/// body to its end (bounded in bytes and chunks) so the provider's payload
/// and the transport's headers ride on the error.
async fn reject_response(
    response: http::Response<crate::http_client::BoxedStream>,
) -> http_client::Error {
    let status = response.status();
    let headers = response.headers().clone();
    let mut body = response.into_body();
    let mut bytes: Vec<u8> = Vec::new();
    let mut chunks = 0usize;
    while let Some(chunk) = body.next().await {
        chunks += 1;
        if let Ok(chunk) = chunk {
            let room = REJECTED_BODY_LIMIT.saturating_sub(bytes.len());
            bytes.extend_from_slice(chunk.get(..chunk.len().min(room)).unwrap_or_default());
        }
        if bytes.len() >= REJECTED_BODY_LIMIT || chunks >= REJECTED_CHUNK_LIMIT {
            break;
        }
    }
    http_client::Error::InvalidStatusCodeWithDetails {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
        headers,
    }
}
