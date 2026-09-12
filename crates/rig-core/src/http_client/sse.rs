//! An SSE implementation that leverages [`crate::http_client::HttpClientExt`] to allow streaming with automatic retry handling for any implementor of HttpClientExt.
//!
//! Primarily intended for internal usage. However if you also wish to implement generic HTTP streaming for your custom completion model,
//! you may find this helpful.
use crate::{
    http_client::{
        HttpClientExt, Result as StreamResult,
        retry::{DEFAULT_RETRY, ExponentialBackoff, RetryPolicy},
    },
    wasm_compat::{WasmCompatSend, WasmCompatSendStream},
};
use bytes::Bytes;
use eventsource_stream::{Event as MessageEvent, EventStreamError, Eventsource};
use futures::{Stream, StreamExt};

pub(crate) mod tail;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use futures::{future::BoxFuture, stream::BoxStream};
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use futures::{future::LocalBoxFuture, stream::LocalBoxStream};
use futures_timer::Delay;
use http::Response;
use http::{HeaderName, HeaderValue, Request, StatusCode};
use mime_guess::mime;
use pin_project_lite::pin_project;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

pub type BoxedStream = Pin<Box<dyn WasmCompatSendStream<InnerItem = StreamResult<Bytes>>>>;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type ResponseFuture = BoxFuture<'static, Result<Response<BoxedStream>, super::Error>>;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type ResponseFuture = LocalBoxFuture<'static, Result<Response<BoxedStream>, super::Error>>;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type EventStream = BoxStream<'static, Result<MessageEvent, EventStreamError<super::Error>>>;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type EventStream = LocalBoxStream<'static, Result<MessageEvent, EventStreamError<super::Error>>>;

pin_project! {
    /// Internal state variants for the SSE state machine.
    #[project = SourceStateProjection]
    enum SourceState {
        /// A connection attempt in flight, carrying the retry that produced it
        /// — `None` for the initial connect. The history belongs in the state
        /// rather than in a separate `Reconnecting` variant because it is the
        /// only thing a reconnect ever did differently: everything else (the
        /// response check, the request-id capture, the handoff to `Open`) was
        /// identical, so two variants meant two copies of it.
        Connecting {
            #[pin]
            response_future: ResponseFuture,
            last_retry: Option<(usize, Duration)>,
        },
        /// Actively receiving SSE events
        Open {
            #[pin]
            event_stream: EventStream,
        },
        /// Waiting before retry after an error
        WaitingToRetry {
            #[pin]
            retry_delay: Delay,
            current_retry: (usize, Duration),
        },
        /// Terminal state
        Closed,
    }
}

/// Shared slot for the transport request id captured off an SSE connection's
/// response headers. Overwritten on every successful (re)connect — with
/// `None` when that connection's response omits (or garbles) the header — so
/// a reader at stream end sees the id of exactly the connection that
/// delivered the terminal, never a previous connection's.
pub type RequestIdSlot = std::sync::Arc<std::sync::Mutex<Option<String>>>;

pin_project! {
    /// A generic SSE event source that works with any [`HttpClientExt`] implementation.
    #[project = GenericEventSourceProjection]
    pub struct GenericEventSource<HttpClient, RequestBody, Retry = ExponentialBackoff> {
        client: HttpClient,
        req: Request<RequestBody>,
        retry_policy: Retry,
        last_event_id: Option<String>,
        allow_missing_content_type: bool,
        request_id_capture: Option<(String, RequestIdSlot)>,
        observation: Option<crate::observe::AdapterSlot>,
        #[pin]
        state: SourceState,
    }
}

impl<HttpClient, RequestBody> GenericEventSource<HttpClient, RequestBody>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<Bytes> + Clone + WasmCompatSend + 'static,
{
    /// Create a new event source that will connect to the given request.
    pub fn new(client: HttpClient, req: Request<RequestBody>) -> Self {
        let observation = crate::observe::AdapterContext::slot_for_request(&req);
        let response_future =
            Self::create_response_future(&client, &req, None, observation.clone());
        let state = SourceState::Connecting {
            response_future,
            last_retry: None,
        };

        Self {
            client,
            req,
            retry_policy: DEFAULT_RETRY,
            last_event_id: None,
            allow_missing_content_type: false,
            request_id_capture: None,
            observation,
            state,
        }
    }

    pub fn allow_missing_content_type(mut self) -> Self {
        self.allow_missing_content_type = true;
        self
    }

    /// Capture the named response header from each successful (re)connect into
    /// the returned [`RequestIdSlot`]. Each (re)connect *replaces* the slot —
    /// a connection whose response omits the header resets it to `None`, so a
    /// stale id from a previous connection is never attributed to the one
    /// that delivered the terminal.
    pub fn capture_request_id(mut self, header: impl Into<String>) -> (Self, RequestIdSlot) {
        let slot = RequestIdSlot::default();
        self.request_id_capture = Some((header.into(), slot.clone()));
        (self, slot)
    }

    pub(crate) fn observation(&self) -> Option<crate::observe::AdapterSlot> {
        self.observation.clone()
    }

    /// Create a response future for connecting/reconnecting
    fn create_response_future(
        client: &HttpClient,
        req: &Request<RequestBody>,
        last_event_id: Option<&str>,
        observation: Option<crate::observe::AdapterSlot>,
    ) -> ResponseFuture {
        let mut req_clone = req.clone();
        req_clone
            .headers_mut()
            .entry("Accept")
            .or_insert(HeaderValue::from_static("text/event-stream"));

        if let Some(id) = last_event_id
            && let Ok(value) = HeaderValue::from_str(id)
        {
            req_clone
                .headers_mut()
                .insert(HeaderName::from_static("last-event-id"), value);
        }

        let client_clone = client.clone();
        Box::pin(async move {
            if let Some(observation) = &observation {
                observation.start(&req_clone);
            }
            let response = match client_clone.send_streaming(req_clone).await {
                // The bundled transports reject a non-success reply before it
                // gets here; a custom `HttpClientExt` may hand it back as a
                // response. Either way the server answered, and its answer —
                // status, headers, body — is the error, never a bare status.
                Ok(response) if response.status() != StatusCode::OK => {
                    Err(reject_response(response).await)
                }
                other => other,
            };
            if let Some(observation) = &observation {
                match &response {
                    Ok(response) => observation
                        .response_with_headers(response.status(), Some(response.headers())),
                    Err(error) => {
                        observation
                            .error_boundary(crate::observe::AdapterErrorBoundary::from_http(error));
                        if let Some(status) = error.non_success_status() {
                            observation.response_with_headers(status, error.non_success_headers());
                        }
                        if let Some(body) = error.non_success_body() {
                            observation.payload(body.as_bytes());
                        }
                        // The frame driver preserves the owned error and closes the
                        // attempt. Do not clone or consume transport errors here.
                    }
                }
            }
            response
        })
    }

    /// Get the last event id
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// Close the event source, transitioning to the Closed state.
    /// After calling this, the stream will yield `None` on the next poll.
    pub fn close(&mut self) {
        self.state = SourceState::Closed;
    }
}

/// Events created by the [`GenericEventSource`]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Event {
    /// The event fired when the connection is opened
    Open,
    /// The event fired when a [`MessageEvent`] is received
    Message(MessageEvent),
}

impl From<MessageEvent> for Event {
    fn from(event: MessageEvent) -> Self {
        Event::Message(event)
    }
}

impl<HttpClient, RequestBody> Stream for GenericEventSource<HttpClient, RequestBody>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<Bytes> + Clone + WasmCompatSend + 'static,
{
    type Item = Result<Event, super::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.state.as_mut().project() {
                SourceStateProjection::Connecting {
                    response_future,
                    last_retry,
                } => {
                    // Copied out before the poll so the state projection's
                    // borrow ends before the transition writes `this.state`.
                    let last_retry = *last_retry;
                    match response_future.poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Ok(response)) => {
                            match check_response(response, *this.allow_missing_content_type) {
                                Ok(response) => {
                                    // Transition: Connecting -> Open
                                    capture_request_id_header(
                                        this.request_id_capture.as_ref(),
                                        &response,
                                    );
                                    let body = response.into_body();
                                    let body: BoxedStream = match this.observation.clone() {
                                        Some(observation) => Box::pin(body.map(move |item| {
                                            if let Ok(bytes) = &item {
                                                observation.bytes(bytes);
                                            }
                                            item
                                        })),
                                        None => body,
                                    };
                                    let mut event_stream = body.eventsource();
                                    if let Some(id) = &this.last_event_id {
                                        event_stream.set_last_event_id(id.clone());
                                    }
                                    this.state.set(SourceState::Open {
                                        event_stream: Box::pin(event_stream),
                                    });
                                    return Poll::Ready(Some(Ok(Event::Open)));
                                }
                                Err(err) => {
                                    if let Some(observation) = this.observation.as_ref() {
                                        observation.error_boundary(
                                            crate::observe::AdapterErrorBoundary::from_http(&err),
                                        );
                                    }
                                    // Transition: Connecting -> Closed. Only a
                                    // content-type failure reaches here (a non-200
                                    // was rejected in the response future and goes
                                    // to the retry policy like any transport
                                    // rejection); a 200 that is not an event stream
                                    // is terminal.
                                    this.state.set(SourceState::Closed);
                                    return Poll::Ready(Some(Err(err)));
                                }
                            }
                        }
                        Poll::Ready(Err(err)) => {
                            // Transition: Connecting -> WaitingToRetry or Closed,
                            // continuing the retry cycle `last_retry` describes.
                            this.state.set(state_after_transport_error(
                                this.retry_policy,
                                &err,
                                last_retry,
                            ));
                            return Poll::Ready(Some(Err(err)));
                        }
                    }
                }

                SourceStateProjection::Open { event_stream } => {
                    match event_stream.poll_next(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Some(Ok(event))) => {
                            if !event.id.is_empty() {
                                *this.last_event_id = Some(event.id.clone());
                            }
                            if let Some(duration) = event.retry {
                                this.retry_policy.set_reconnection_time(duration);
                            }
                            return Poll::Ready(Some(Ok(Event::Message(event))));
                        }
                        Poll::Ready(Some(Err(EventStreamError::Transport(err)))) => {
                            if let Some(observation) = this.observation.as_ref() {
                                observation.error_boundary(
                                    crate::observe::AdapterErrorBoundary::Transport,
                                );
                            }
                            // Transition: Open -> WaitingToRetry or Closed. A
                            // failure mid-stream starts a *fresh* cycle (history
                            // `None`): this connection had already succeeded, so
                            // the attempts that preceded it no longer apply.
                            this.state.set(state_after_transport_error(
                                this.retry_policy,
                                &err,
                                None,
                            ));
                            return Poll::Ready(Some(Err(err)));
                        }
                        Poll::Ready(Some(Err(EventStreamError::Parser(_)))) => {
                            // Parser errors are recoverable - continue polling
                            continue;
                        }
                        Poll::Ready(Some(Err(EventStreamError::Utf8(_)))) => {
                            // UTF-8 errors are recoverable - continue polling
                            continue;
                        }
                        Poll::Ready(None) => {
                            // Transition: Open -> Closed
                            this.state.set(SourceState::Closed);
                            return Poll::Ready(None);
                        }
                    }
                }

                SourceStateProjection::WaitingToRetry {
                    retry_delay,
                    current_retry,
                } => {
                    // Copy before polling to avoid borrow conflicts
                    let retry_info = *current_retry;
                    match retry_delay.poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(()) => {
                            // Transition: WaitingToRetry -> Connecting
                            let response_future =
                                GenericEventSource::<HttpClient, RequestBody>::create_response_future(
                                    this.client,
                                    this.req,
                                    this.last_event_id.as_deref(),
                                    this.observation.clone(),
                                );
                            this.state.set(SourceState::Connecting {
                                response_future,
                                last_retry: Some(retry_info),
                            });
                            continue;
                        }
                    }
                }

                SourceStateProjection::Closed => {
                    return Poll::Ready(None);
                }
            }
        }
    }
}

/// The state a transport failure moves the machine to: wait out the policy's
/// next delay, or close when it declines to retry.
///
/// `last_retry` is the retry that produced the failed attempt, so the retry
/// number the policy sees and the one recorded for the next attempt advance
/// together — the numbering is stated once instead of per call site.
fn state_after_transport_error(
    retry_policy: &impl RetryPolicy,
    error: &super::Error,
    last_retry: Option<(usize, Duration)>,
) -> SourceState {
    match retry_policy.retry(error, last_retry) {
        Some(delay) => SourceState::WaitingToRetry {
            retry_delay: Delay::new(delay),
            current_retry: (last_retry.map_or(1, |(retry_num, _)| retry_num + 1), delay),
        },
        None => SourceState::Closed,
    }
}

/// Replace the shared slot with this connection's request-id header value —
/// `None` when the response omits the header or its value is empty/invalid.
/// Overwriting (rather than only writing on presence) is what prevents a
/// reconnect from reporting the *previous* connection's id.
fn capture_request_id_header<T>(capture: Option<&(String, RequestIdSlot)>, response: &Response<T>) {
    if let Some((header, slot)) = capture
        && let Ok(mut slot) = slot.lock()
    {
        *slot = response
            .headers()
            .get(header.as_str())
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_string);
    }
}

/// Bytes of a rejected reply's body kept on the error; a reply longer than
/// this is cut there, the way a provider's error payload never is (a cut
/// body is not JSON any more, so `provider_response_json` reports it as
/// malformed rather than absent).
const REJECTED_BODY_LIMIT: usize = 1 << 20;

/// Chunks read off a rejected reply before giving up on it, so a transport
/// that keeps yielding empty chunks cannot hold the opener.
const REJECTED_CHUNK_LIMIT: usize = 4096;

/// Turn a reply the event source will not stream (any status but 200,
/// a 204 included: a status is a status) into the non-success error,
/// reading the body to its end (bounded in bytes and chunks; the transport's
/// own timeouts bound the time) so the provider's payload and the
/// transport's headers ride on the error.
async fn reject_response(response: Response<BoxedStream>) -> super::Error {
    let (parts, mut body) = response.into_parts();
    let mut collected = Vec::new();
    let mut chunks = 0;
    while let Some(chunk) = body.next().await {
        chunks += 1;
        let room = REJECTED_BODY_LIMIT.saturating_sub(collected.len());
        match chunk {
            Ok(bytes) if room > 0 && chunks <= REJECTED_CHUNK_LIMIT => {
                collected.extend(bytes.iter().take(room));
            }
            _ => break,
        }
    }
    super::Error::non_success_with_details(
        parts.status,
        parts.headers,
        String::from_utf8_lossy(&collected).into_owned(),
    )
}

fn check_response<T>(
    response: Response<T>,
    allow_missing_content_type: bool,
) -> Result<Response<T>, super::Error> {
    let Some(content_type) = response.headers().get(&http::header::CONTENT_TYPE) else {
        if allow_missing_content_type {
            return Ok(response);
        }
        return Err(super::Error::InvalidContentType(HeaderValue::from_static(
            "",
        )));
    };

    if content_type
        .to_str()
        .map_err(|_| ())
        .and_then(|s| s.parse::<mime::Mime>().map_err(|_| ()))
        .map(|mime_type| {
            matches!(
                (mime_type.type_(), mime_type.subtype()),
                (mime::TEXT, mime::EVENT_STREAM)
            )
        })
        .unwrap_or(false)
    {
        Ok(response)
    } else {
        Err(super::Error::InvalidContentType(content_type.clone()))
    }
}

#[cfg(all(test, not(all(target_arch = "wasm32", target_os = "unknown"))))]
mod tests;
