//! The SSE transport edge.
//!
//! An event source over any `Backend`, with
//! automatic retry handling. Providers never see it directly: they consume the
//! boxed [`BoxedEventSource`], which is where backend genericity ends.
use crate::{
    http_client::{
        Backend, Result as StreamResult,
        retry::{DEFAULT_RETRY, ExponentialBackoff},
    },
    wasm_compat::WasmCompatSendStream,
};
use bytes::Bytes;
use eventsource_stream::{Event as MessageEvent, EventStreamError, Eventsource};
use futures::Stream;
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

/// A type-erased SSE event stream — the transport edge for the sans-IO
/// provider stream parsers.
///
/// Provider stream state machines consume this concrete type instead of being
/// generic over the crate-internal `Backend`: genericity ends at the boxing site (see
/// `boxed_event_source` and `HttpRuntime::sse_events`).
pub type BoxedEventSource = Pin<Box<dyn WasmCompatEventStream>>;

/// Helper supertrait so [`BoxedEventSource`] can be a trait object:
/// `WasmCompatSend` is not an auto trait, so it cannot be an additional bound
/// on `dyn Stream`. `Send`-bounded on native targets, unbounded on browser wasm.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub trait WasmCompatEventStream: Stream<Item = Result<Event, super::Error>> + Send {}
/// Helper supertrait so [`BoxedEventSource`] can be a trait object.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait WasmCompatEventStream: Stream<Item = Result<Event, super::Error>> {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T> WasmCompatEventStream for T where T: Stream<Item = Result<Event, super::Error>> + Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> WasmCompatEventStream for T where T: Stream<Item = Result<Event, super::Error>> {}

/// Box an event source over `client` into the transport-edge
/// [`BoxedEventSource`], ending the `Backend` genericity.
pub(crate) fn boxed_event_source<HttpClient>(
    client: HttpClient,
    req: Request<Vec<u8>>,
    allow_missing_content_type: bool,
) -> BoxedEventSource
where
    HttpClient: Backend + Clone + 'static,
{
    let source = GenericEventSource::new(client, req);
    let source = if allow_missing_content_type {
        source.allow_missing_content_type()
    } else {
        source
    };
    Box::pin(source)
}

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
        /// Initial connection attempt (no retry history yet)
        Connecting {
            #[pin]
            response_future: ResponseFuture,
        },
        /// Reconnection attempt after a retry delay (always has retry history)
        Reconnecting {
            #[pin]
            response_future: ResponseFuture,
            last_retry: (usize, Duration),
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

pin_project! {
    /// An SSE event source over any [`Backend`].
    #[project = GenericEventSourceProjection]
    pub(crate) struct GenericEventSource<HttpClient> {
        client: HttpClient,
        req: Request<Vec<u8>>,
        backoff: ExponentialBackoff,
        last_event_id: Option<String>,
        allow_missing_content_type: bool,
        #[pin]
        state: SourceState,
    }
}

impl<HttpClient> GenericEventSource<HttpClient>
where
    HttpClient: Backend + Clone + 'static,
{
    /// Create a new event source that will connect to the given request.
    pub fn new(client: HttpClient, req: Request<Vec<u8>>) -> Self {
        let response_future = Self::create_response_future(&client, &req, None);
        let state = SourceState::Connecting { response_future };

        Self {
            client,
            req,
            backoff: DEFAULT_RETRY,
            last_event_id: None,
            allow_missing_content_type: false,
            state,
        }
    }

    pub fn allow_missing_content_type(mut self) -> Self {
        self.allow_missing_content_type = true;
        self
    }

    /// Create a response future for connecting/reconnecting
    fn create_response_future(
        client: &HttpClient,
        req: &Request<Vec<u8>>,
        last_event_id: Option<&str>,
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
        Box::pin(async move { client_clone.send_streaming(req_clone).await })
    }
}

/// Events created by the SSE event source.
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

impl<HttpClient> Stream for GenericEventSource<HttpClient>
where
    HttpClient: Backend + Clone + 'static,
{
    type Item = Result<Event, super::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.state.as_mut().project() {
                SourceStateProjection::Connecting { response_future } => {
                    match response_future.poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Ok(response)) => {
                            match check_response(response, *this.allow_missing_content_type) {
                                Ok(response) => {
                                    // Transition: Connecting -> Open
                                    let mut event_stream = response.into_body().eventsource();
                                    if let Some(id) = &this.last_event_id {
                                        event_stream.set_last_event_id(id.clone());
                                    }
                                    this.state.set(SourceState::Open {
                                        event_stream: Box::pin(event_stream),
                                    });
                                    return Poll::Ready(Some(Ok(Event::Open)));
                                }
                                Err(err) => {
                                    // Transition: Connecting -> Closed (non-retryable error)
                                    this.state.set(SourceState::Closed);
                                    return Poll::Ready(Some(Err(err)));
                                }
                            }
                        }
                        Poll::Ready(Err(err)) => {
                            // First connection attempt failed - start retry cycle
                            if let Some(delay_duration) = this.backoff.retry(&err, None) {
                                // Transition: Connecting -> WaitingToRetry
                                this.state.set(SourceState::WaitingToRetry {
                                    retry_delay: Delay::new(delay_duration),
                                    current_retry: (1, delay_duration),
                                });
                                return Poll::Ready(Some(Err(err)));
                            } else {
                                // Transition: Connecting -> Closed
                                this.state.set(SourceState::Closed);
                                return Poll::Ready(Some(Err(err)));
                            }
                        }
                    }
                }

                SourceStateProjection::Reconnecting {
                    response_future,
                    last_retry,
                } => {
                    match response_future.poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Ok(response)) => {
                            match check_response(response, *this.allow_missing_content_type) {
                                Ok(response) => {
                                    // Transition: Reconnecting -> Open (retry cycle complete)
                                    let mut event_stream = response.into_body().eventsource();
                                    if let Some(id) = &this.last_event_id {
                                        event_stream.set_last_event_id(id.clone());
                                    }
                                    this.state.set(SourceState::Open {
                                        event_stream: Box::pin(event_stream),
                                    });
                                    return Poll::Ready(Some(Ok(Event::Open)));
                                }
                                Err(err) => {
                                    // Transition: Reconnecting -> Closed (non-retryable error)
                                    this.state.set(SourceState::Closed);
                                    return Poll::Ready(Some(Err(err)));
                                }
                            }
                        }
                        Poll::Ready(Err(err)) => {
                            // Reconnection attempt failed - continue retry cycle
                            if let Some(delay_duration) =
                                this.backoff.retry(&err, Some(*last_retry))
                            {
                                let (retry_num, _) = *last_retry;
                                // Transition: Reconnecting -> WaitingToRetry
                                this.state.set(SourceState::WaitingToRetry {
                                    retry_delay: Delay::new(delay_duration),
                                    current_retry: (retry_num + 1, delay_duration),
                                });
                                return Poll::Ready(Some(Err(err)));
                            } else {
                                // Transition: Reconnecting -> Closed (max retries exceeded)
                                this.state.set(SourceState::Closed);
                                return Poll::Ready(Some(Err(err)));
                            }
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
                                this.backoff.set_reconnection_time(duration);
                            }
                            return Poll::Ready(Some(Ok(Event::Message(event))));
                        }
                        Poll::Ready(Some(Err(EventStreamError::Transport(err)))) => {
                            // Connection error while open - start fresh retry cycle
                            if let Some(delay_duration) = this.backoff.retry(&err, None) {
                                // Transition: Open -> WaitingToRetry
                                this.state.set(SourceState::WaitingToRetry {
                                    retry_delay: Delay::new(delay_duration),
                                    current_retry: (1, delay_duration),
                                });
                                return Poll::Ready(Some(Err(err)));
                            } else {
                                // Transition: Open -> Closed
                                this.state.set(SourceState::Closed);
                                return Poll::Ready(Some(Err(err)));
                            }
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
                            // Transition: WaitingToRetry -> Reconnecting
                            let response_future =
                                GenericEventSource::<HttpClient>::create_response_future(
                                    this.client,
                                    this.req,
                                    this.last_event_id.as_deref(),
                                );
                            this.state.set(SourceState::Reconnecting {
                                response_future,
                                last_retry: retry_info,
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

fn check_response<T>(
    response: Response<T>,
    allow_missing_content_type: bool,
) -> Result<Response<T>, super::Error> {
    let StatusCode::OK = response.status() else {
        return Err(super::Error::InvalidStatusCode(response.status()));
    };

    let content_type =
        if let Some(content_type) = response.headers().get(&reqwest::header::CONTENT_TYPE) {
            content_type
        } else if allow_missing_content_type {
            return Ok(response);
        } else {
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
