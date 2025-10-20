//! A generic SSE implementation.

use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use bytes::Bytes;
#[cfg(not(target_arch = "wasm32"))]
use eventsource_stream::EventStreamError;
#[cfg(not(target_arch = "wasm32"))]
use futures::{future::BoxFuture, stream::BoxStream};
#[cfg(target_arch = "wasm32")]
use futures::{future::LocalBoxFuture, stream::LocalBoxStream};
use futures_timer::Delay;
#[cfg(not(target_arch = "wasm32"))]
use http::Response;
use http::{HeaderName, HeaderValue, Request};
use pin_project_lite::pin_project;

#[cfg(not(target_arch = "wasm32"))]
use crate::http_client::LazyBody;
use crate::{
    http_client::{
        HttpClientExt, instance_error,
        retry::{DEFAULT_RETRY, RetryPolicy},
    },
    wasm_compat::WasmCompatSend,
};

#[cfg(not(target_arch = "wasm32"))]
type ResponseFuture<T> = BoxFuture<'static, Result<Response<LazyBody<T>>, super::Error>>;
#[cfg(target_arch = "wasm32")]
type ResponseFuture<T> = LocalBoxFuture<'static, Result<Response<LazyBody<T>>, ReqwestError>>;

#[cfg(not(target_arch = "wasm32"))]
type EventStream =
    BoxStream<'static, Result<eventsource_stream::Event, EventStreamError<super::Error>>>;
#[cfg(target_arch = "wasm32")]
type EventStream = LocalBoxStream<'static, Result<MessageEvent, EventStreamError<ReqwestError>>>;
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
        ResponseBody: From<Bytes>
    {
        client: HttpClient,
        #[pin]
        next_response: Option<ResponseFuture<ResponseBody>>,
        #[pin]
        cur_stream: Option<EventStream>,
        #[pin]
        req: Request<RequestBody>,
        delay: Option<Delay>,
        is_closed: bool,
        retry_policy: BoxedRetry,
        last_event_id: String,
        last_retry: Option<(usize, Duration)>,
    }
}

impl<HttpClient, RequestBody, ResponseBody>
    GenericEventSource<HttpClient, RequestBody, ResponseBody>
where
    HttpClient: HttpClientExt,
    RequestBody: Into<Bytes>,
    ResponseBody: From<Bytes>,
{
    pub fn new(client: HttpClient, req: Request<RequestBody>) -> Self {
        Self {
            client,
            next_response: None,
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
impl<'a, HttpClient, RequestBody, ResponseBody>
    GenericEventSourceProjection<'a, HttpClient, RequestBody, ResponseBody>
where
    HttpClient: HttpClientExt,
    RequestBody: Into<Bytes> + Clone + WasmCompatSend,
    ResponseBody: From<Bytes> + Clone + WasmCompatSend + 'static,
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
        let res_future = Box::pin(self.client.send(req));
        self.next_response.replace(res_future);
        Ok(())
    }

    fn handle_response(&mut self, res: Response<ResponseBody>) {
        self.last_retry.take();
        let mut stream = Bytes::from(res.into_body());

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
            let retry_num = self.last_retry.map(|retry| retry.0).unwrap_or(1);
            *self.last_retry = Some((retry_num, retry_delay));
            self.delay.replace(Delay::new(retry_delay));
        } else {
            *self.is_closed = true;
        }
    }
}
