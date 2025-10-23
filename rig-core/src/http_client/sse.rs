//! An SSE implementation that leverages [`crate::http_client::HttpClientExt`] to allow streaming with automatic retry handling for any implementor of HttpClientExt.
//!
//! Primarily intended for internal usage. However if you also wish to implement generic HTTP streaming for your custom completion model,
//! you may find this helpful.

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use bytes::Bytes;
use eventsource_stream::{Event as MessageEvent, EventStreamError, Eventsource};
use futures::Stream;
#[cfg(not(target_arch = "wasm32"))]
use futures::{future::BoxFuture, stream::BoxStream};
#[cfg(target_arch = "wasm32")]
use futures::{future::LocalBoxFuture, stream::LocalBoxStream};
use futures_timer::Delay;
use http::Response;
use http::{HeaderName, HeaderValue, Request, StatusCode};
use mime_guess::mime;
use pin_project_lite::pin_project;

use crate::{
    http_client::{
        HttpClientExt, Result as StreamResult, instance_error,
        retry::{DEFAULT_RETRY, RetryPolicy},
    },
    wasm_compat::{WasmCompatSend, WasmCompatSendStream},
};

pub type BoxedStream = Pin<Box<dyn WasmCompatSendStream<InnerItem = StreamResult<Bytes>>>>;

#[cfg(not(target_arch = "wasm32"))]
type ResponseFuture<T> = BoxFuture<'static, Result<Response<T>, super::Error>>;
#[cfg(target_arch = "wasm32")]
type ResponseFuture<T> = LocalBoxFuture<'static, Result<Response<T>, super::Error>>;

#[cfg(not(target_arch = "wasm32"))]
type EventStream = BoxStream<'static, Result<MessageEvent, EventStreamError<super::Error>>>;
#[cfg(target_arch = "wasm32")]
type EventStream = LocalBoxStream<'static, Result<MessageEvent, EventStreamError<super::Error>>>;
type BoxedRetry = Box<dyn RetryPolicy + Send + Unpin + 'static>;

/// The ready state of a [`GenericEventSource`]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum ReadyState {
    /// The EventSource is waiting on a response from the endpoint
    Connecting = 0,
    /// The EventSource is connected
    Open = 1,
    /// The EventSource is closed and no longer emitting Events
    Closed = 2,
}

pin_project! {
    /// A generic event source that can use any HTTP client.
    /// Modeled heavily on the `reqwest-eventsource` implementation.
    #[project = GenericEventSourceProjection]
    pub struct GenericEventSource<HttpClient, RequestBody, ResponseBody>
    where
        HttpClient: HttpClientExt,
    {
        client: HttpClient,
        req: Request<RequestBody>,
        #[pin]
        next_response: Option<ResponseFuture<ResponseBody>>,
        #[pin]
        cur_stream: Option<EventStream>,
        #[pin]
        delay: Option<Delay>,
        is_closed: bool,
        retry_policy: BoxedRetry,
        last_event_id: String,
        last_retry: Option<(usize, Duration)>,
    }
}

impl<HttpClient, RequestBody>
    GenericEventSource<
        HttpClient,
        RequestBody,
        Pin<Box<dyn WasmCompatSendStream<InnerItem = StreamResult<Bytes>>>>,
    >
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<Bytes> + Clone + Send + 'static,
{
    pub fn new(client: HttpClient, req: Request<RequestBody>) -> Self {
        let client_clone = client.clone();
        let mut req_clone = req.clone();
        req_clone
            .headers_mut()
            .insert("Accept", HeaderValue::from_static("text/event-stream"));
        let res_fut = Box::pin(async move { client_clone.clone().send_streaming(req_clone).await });
        Self {
            client,
            next_response: Some(res_fut),
            cur_stream: None,
            req,
            delay: None,
            is_closed: false,
            retry_policy: Box::new(DEFAULT_RETRY),
            last_event_id: String::new(),
            last_retry: None,
        }
    }

    /// Close the EventSource stream and stop trying to reconnect
    pub fn close(&mut self) {
        self.is_closed = true;
    }

    /// Get the last event id
    pub fn last_event_id(&self) -> &str {
        &self.last_event_id
    }

    /// Get the current ready state
    pub fn ready_state(&self) -> ReadyState {
        if self.is_closed {
            ReadyState::Closed
        } else if self.delay.is_some() || self.next_response.is_some() {
            ReadyState::Connecting
        } else {
            ReadyState::Open
        }
    }
}

impl<'a, HttpClient, RequestBody>
    GenericEventSourceProjection<'a, HttpClient, RequestBody, BoxedStream>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<Bytes> + Clone + WasmCompatSend + 'static,
{
    fn clear_fetch(&mut self) {
        self.next_response.take();
        self.cur_stream.take();
    }

    fn retry_fetch(&mut self) -> Result<(), super::Error> {
        self.cur_stream.take();
        let mut req = self.req.clone();
        req.headers_mut().insert(
            HeaderName::from_static("last-event-id"),
            HeaderValue::from_str(self.last_event_id).map_err(instance_error)?,
        );
        let client = self.client.clone();
        let res_future = Box::pin(async move { client.send_streaming(req).await });
        self.next_response.replace(res_future);
        Ok(())
    }

    fn handle_response<T>(&mut self, res: Response<T>)
    where
        T: Stream<Item = StreamResult<Bytes>> + WasmCompatSend + 'static,
    {
        self.last_retry.take();
        let mut stream = res.into_body().eventsource();
        stream.set_last_event_id(self.last_event_id.clone());
        self.cur_stream.replace(Box::pin(stream));
    }

    fn handle_event(&mut self, event: &eventsource_stream::Event) {
        *self.last_event_id = event.id.clone();
        if let Some(duration) = event.retry {
            self.retry_policy.set_reconnection_time(duration)
        }
    }

    fn handle_error(&mut self, error: &super::Error) {
        self.clear_fetch();
        if let Some(retry_delay) = self.retry_policy.retry(error, *self.last_retry) {
            let retry_num = self
                .last_retry
                .map(|retry| retry.0.saturating_add(1))
                .unwrap_or(1);
            *self.last_retry = Some((retry_num, retry_delay));
            self.delay.replace(Delay::new(retry_delay));
        } else {
            *self.is_closed = true;
        }
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

impl<HttpClient, RequestBody> Stream for GenericEventSource<HttpClient, RequestBody, BoxedStream>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<Bytes> + Clone + WasmCompatSend + 'static,
{
    type Item = Result<Event, super::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.is_closed {
            return Poll::Ready(None);
        }

        if let Some(delay) = this.delay.as_mut().as_pin_mut() {
            match delay.poll(cx) {
                Poll::Ready(_) => {
                    this.delay.take();
                    if let Err(err) = this.retry_fetch() {
                        *this.is_closed = true;
                        return Poll::Ready(Some(Err(err)));
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        if let Some(response_future) = this.next_response.as_mut().as_pin_mut() {
            match response_future.poll(cx) {
                Poll::Ready(Ok(res)) => {
                    this.clear_fetch();
                    match check_response(res) {
                        Ok(res) => {
                            this.handle_response(res);
                            return Poll::Ready(Some(Ok(Event::Open)));
                        }
                        Err(err) => {
                            *this.is_closed = true;
                            return Poll::Ready(Some(Err(err)));
                        }
                    }
                }
                Poll::Ready(Err(err)) => {
                    this.handle_error(&err);
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }

        match this
            .cur_stream
            .as_mut()
            .as_pin_mut()
            .unwrap()
            .as_mut()
            .poll_next(cx)
        {
            Poll::Ready(Some(Err(err))) => {
                let EventStreamError::Transport(err) = err else {
                    panic!("u");
                };
                this.handle_error(&err);
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(event))) => {
                this.handle_event(&event);
                Poll::Ready(Some(Ok(event.into())))
            }
            Poll::Ready(None) => {
                let err = super::Error::StreamEnded;
                this.handle_error(&err);
                Poll::Ready(Some(Err(err)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn check_response<T>(response: Response<T>) -> Result<Response<T>, super::Error> {
    match response.status() {
        StatusCode::OK => {}
        status => {
            return Err(super::Error::InvalidStatusCode(status));
        }
    }
    let content_type =
        if let Some(content_type) = response.headers().get(&reqwest::header::CONTENT_TYPE) {
            content_type
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
