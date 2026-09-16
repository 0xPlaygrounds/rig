//! HTTP client doubles for provider tests.

use std::{
    collections::VecDeque,
    future::{self, Future},
    sync::{Arc, Mutex, MutexGuard},
};

use bytes::Bytes;

use crate::{
    http_client::{
        self, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
    },
    wasm_compat::WasmCompatSend,
};

/// The reply a double gives on a surface it does not script: the request
/// reached a transport that has no answer for it.
fn not_implemented() -> http_client::Error {
    http_client::Error::non_success_with_details(
        http::StatusCode::NOT_IMPLEMENTED,
        http::HeaderMap::new(),
        String::new(),
    )
}

/// The unary half of a streaming-only double.
///
/// Four doubles below script `send_streaming` and nothing else; that they
/// have no unary surface is one fact about streaming-only doubles, so it is
/// written here once and each impl states it by name.
macro_rules! no_unary_surface {
    () => {
        fn send<T, U>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + WasmCompatSend
        + 'static
        where
            T: Into<Bytes> + WasmCompatSend,
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(not_implemented()))
        }

        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + WasmCompatSend
        + 'static
        where
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(not_implemented()))
        }
    };
}

/// Request data captured by [`RecordingHttpClient`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedHttpRequest {
    /// Request URI.
    pub uri: String,
    /// Request headers.
    pub headers: http::HeaderMap,
    /// Request body bytes.
    pub body: Bytes,
}

/// Response scripted for [`RecordingHttpClient`].
#[derive(Clone, Debug)]
pub enum MockHttpResponse {
    /// Return this body with a successful HTTP status and no headers.
    Success(Bytes),
    /// Return this body with a successful HTTP status and these headers —
    /// the reply whose `Content-Type` decides how it may be framed.
    SuccessWithHeaders(Bytes, http::HeaderMap),
    /// Return an HTTP response with the given (typically non-success) status
    /// and body, instead of a transport-level error.
    ErrorResponse(http::StatusCode, Bytes),
    /// Return a status-code error that preserved the failed response's
    /// headers, exactly as the bundled reqwest transport does (rig#2210).
    ErrorWithHeaders(http::StatusCode, String, http::HeaderMap),
    /// Return an HTTP response with the given (typically non-success) status,
    /// body, and response headers, instead of a transport-level error.
    ErrorResponseWithHeaders(http::StatusCode, Bytes, http::HeaderMap),
}

impl MockHttpResponse {
    /// Create a successful response from bytes.
    pub fn success(body: impl Into<Bytes>) -> Self {
        Self::Success(body.into())
    }

    /// Create a successful response carrying `content_type`.
    pub fn success_typed(body: impl Into<Bytes>, content_type: &'static str) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static(content_type),
        );
        Self::SuccessWithHeaders(body.into(), headers)
    }

    /// Create a transport-level status error whose reply carried no headers
    /// of interest: the same shape as [`Self::error_with_headers`] with an
    /// empty map, since a rejection always comes with its headers.
    pub fn error(status: http::StatusCode, message: impl Into<String>) -> Self {
        Self::error_with_headers(status, message, http::HeaderMap::new())
    }

    /// Create a transport-level status error that preserved the failed
    /// response's headers, as the bundled reqwest transport does.
    pub fn error_with_headers(
        status: http::StatusCode,
        message: impl Into<String>,
        headers: http::HeaderMap,
    ) -> Self {
        Self::ErrorWithHeaders(status, message.into(), headers)
    }
}

impl Default for MockHttpResponse {
    fn default() -> Self {
        Self::Success(Bytes::new())
    }
}

/// An [`HttpClientExt`] implementation that records unary requests and returns
/// a fixed response.
#[derive(Clone, Debug, Default)]
pub struct RecordingHttpClient {
    requests: Arc<Mutex<Vec<CapturedHttpRequest>>>,
    response: Arc<Mutex<MockHttpResponse>>,
}

impl RecordingHttpClient {
    /// Create a client that returns `response_body` for unary requests.
    pub fn new(response_body: impl Into<Bytes>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: Arc::new(Mutex::new(MockHttpResponse::success(response_body))),
        }
    }

    /// Create a client that returns an HTTP status error for unary requests.
    pub fn with_error(status: http::StatusCode, message: impl Into<String>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: Arc::new(Mutex::new(MockHttpResponse::error(status, message))),
        }
    }

    /// Create a client that returns a non-success HTTP response (status and body)
    /// for unary requests, instead of a transport-level error.
    pub fn with_error_response(status: http::StatusCode, body: impl Into<Bytes>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: Arc::new(Mutex::new(MockHttpResponse::ErrorResponse(
                status,
                body.into(),
            ))),
        }
    }

    /// Create a client whose transport reports a non-success status *and*
    /// preserves the failed response's headers, as the bundled reqwest client
    /// does (rig#2210).
    pub fn with_error_headers(
        status: http::StatusCode,
        message: impl Into<String>,
        headers: http::HeaderMap,
    ) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: Arc::new(Mutex::new(MockHttpResponse::error_with_headers(
                status, message, headers,
            ))),
        }
    }

    /// Create a client that hands back a non-success HTTP response carrying
    /// `headers`, instead of a transport-level error.
    pub fn with_error_response_headers(
        status: http::StatusCode,
        body: impl Into<Bytes>,
        headers: http::HeaderMap,
    ) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: Arc::new(Mutex::new(MockHttpResponse::ErrorResponseWithHeaders(
                status,
                body.into(),
                headers,
            ))),
        }
    }

    /// Return the requests captured so far.
    pub fn requests(&self) -> Vec<CapturedHttpRequest> {
        self.requests_guard().clone()
    }

    /// Replace the scripted unary response.
    pub fn set_response(&self, response: MockHttpResponse) {
        *self.response_guard() = response;
    }

    fn requests_guard(&self) -> MutexGuard<'_, Vec<CapturedHttpRequest>> {
        match self.requests.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn response_guard(&self) -> MutexGuard<'_, MockHttpResponse> {
        match self.response.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn record_request(&self, uri: String, headers: http::HeaderMap, body: Bytes) {
        self.requests_guard()
            .push(CapturedHttpRequest { uri, headers, body });
    }

    fn build_unary_response<U>(
        response: MockHttpResponse,
    ) -> http_client::Result<Response<LazyBody<U>>>
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let (status, response_body, response_headers) = match response {
            MockHttpResponse::Success(response_body) => (http::StatusCode::OK, response_body, None),
            MockHttpResponse::SuccessWithHeaders(response_body, headers) => {
                (http::StatusCode::OK, response_body, Some(headers))
            }
            MockHttpResponse::ErrorWithHeaders(status, body, headers) => {
                return Err(http_client::Error::InvalidStatusCodeWithDetails {
                    status,
                    body,
                    headers,
                });
            }
            MockHttpResponse::ErrorResponse(status, response_body) => (status, response_body, None),
            MockHttpResponse::ErrorResponseWithHeaders(status, response_body, headers) => {
                (status, response_body, Some(headers))
            }
        };
        let body: LazyBody<U> = Box::pin(async move { Ok(U::from(response_body)) });
        let mut builder = Response::builder().status(status);
        if let Some(headers) = response_headers
            && let Some(slot) = builder.headers_mut()
        {
            *slot = headers;
        }
        builder.body(body).map_err(http_client::Error::Protocol)
    }
}

impl HttpClientExt for RecordingHttpClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let response = self.response_guard().clone();
        let (parts, body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, body.into());

        async move { Self::build_unary_response(response) }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let response = self.response_guard().clone();
        let (parts, body) = req.into_parts();
        let (_, body) = body.boundary("recording-http-client").encode();
        self.record_request(parts.uri.to_string(), parts.headers, body);

        async move { Self::build_unary_response(response) }
    }

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        future::ready(Err(not_implemented()))
    }
}

/// An [`HttpClientExt`] implementation that records unary requests and returns
/// one scripted response per request.
///
/// This is useful for testing retry and recovery paths through real provider
/// request/response conversion without live credentials.
#[derive(Clone, Debug, Default)]
pub struct SequencedHttpClient {
    requests: Arc<Mutex<Vec<CapturedHttpRequest>>>,
    responses: Arc<Mutex<VecDeque<MockHttpResponse>>>,
}

impl SequencedHttpClient {
    /// Create a client that returns the supplied responses in order.
    pub fn new(responses: impl IntoIterator<Item = MockHttpResponse>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
        }
    }

    /// Return the requests captured so far.
    pub fn requests(&self) -> Vec<CapturedHttpRequest> {
        match self.requests.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Return the number of scripted responses that have not been consumed.
    pub fn remaining_responses(&self) -> usize {
        match self.responses.lock() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    fn record_request(&self, uri: String, headers: http::HeaderMap, body: Bytes) {
        let request = CapturedHttpRequest { uri, headers, body };
        match self.requests.lock() {
            Ok(mut guard) => guard.push(request),
            Err(poisoned) => poisoned.into_inner().push(request),
        }
    }

    fn next_response(&self) -> Option<MockHttpResponse> {
        match self.responses.lock() {
            Ok(mut guard) => guard.pop_front(),
            Err(poisoned) => poisoned.into_inner().pop_front(),
        }
    }
}

impl HttpClientExt for SequencedHttpClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let response = self.next_response();
        let (parts, body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, body.into());

        async move {
            match response {
                Some(response) => RecordingHttpClient::build_unary_response(response),
                None => Err(not_implemented()),
            }
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let response = self.next_response();
        let (parts, _body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, Bytes::new());

        async move {
            match response {
                Some(response) => RecordingHttpClient::build_unary_response(response),
                None => Err(not_implemented()),
            }
        }
    }

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        future::ready(Err(not_implemented()))
    }
}

/// A mock HTTP client that returns pre-built SSE bytes from `send_streaming`.
///
/// `send` and `send_multipart` always return `NOT_IMPLEMENTED`.
#[derive(Clone, Debug, Default)]
pub struct MockStreamingClient {
    /// Bytes returned as a single streaming response chunk.
    pub sse_bytes: Bytes,
}

impl HttpClientExt for MockStreamingClient {
    no_unary_surface!();

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let sse_bytes = self.sse_bytes.clone();
        async move {
            let byte_stream =
                futures::stream::iter(vec![Ok::<Bytes, http_client::Error>(sse_bytes)]);
            let boxed_stream: http_client::BoxedStream = Box::pin(byte_stream);

            Response::builder()
                .status(http::StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "text/event-stream")
                .body(boxed_stream)
                .map_err(http_client::Error::Protocol)
        }
    }
}

/// An [`HttpClientExt`] implementation whose `send_streaming` fails immediately
/// with a non-success HTTP status and response body.
#[derive(Debug, Clone)]
pub struct HttpErrorStreamingClient {
    pub status: http::StatusCode,
    pub body: String,
}

impl HttpErrorStreamingClient {
    /// Create a streaming client that fails `send_streaming` with the given status and body.
    pub fn new(status: http::StatusCode, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
        }
    }
}

impl Default for HttpErrorStreamingClient {
    /// The completion-model client bound requires `H: Default`; this lets the
    /// streaming error client back a real model in tests.
    fn default() -> Self {
        Self::new(http::StatusCode::INTERNAL_SERVER_ERROR, String::new())
    }
}

impl HttpClientExt for HttpErrorStreamingClient {
    no_unary_surface!();

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let status = self.status;
        let body = self.body.clone();
        async move {
            Err(http_client::Error::non_success_with_details(
                status,
                http::HeaderMap::new(),
                body,
            ))
        }
    }
}

/// An [`HttpClientExt`] whose `send_streaming` hands a non-success reply back
/// as an `Ok` response — the fourth non-success cell: a custom transport that
/// does not reject on status, so the driver must.
#[derive(Debug, Clone)]
pub struct NonSuccessStreamingClient {
    pub status: http::StatusCode,
    pub headers: http::HeaderMap,
    pub body: Bytes,
}

impl HttpClientExt for NonSuccessStreamingClient {
    no_unary_surface!();

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let status = self.status;
        let headers = self.headers.clone();
        let body = self.body.clone();
        async move {
            let byte_stream = futures::stream::iter(vec![Ok::<Bytes, http_client::Error>(body)]);
            let boxed_stream: http_client::BoxedStream = Box::pin(byte_stream);
            let mut response = Response::builder()
                .status(status)
                .body(boxed_stream)
                .map_err(http_client::Error::Protocol)?;
            *response.headers_mut() = headers;
            Ok(response)
        }
    }
}

/// An [`HttpClientExt`] implementation that returns one scripted stream of byte
/// chunks from `send_streaming`.
#[derive(Debug, Clone, Default)]
pub struct SequencedStreamingHttpClient {
    chunks: Arc<Mutex<Option<Vec<http_client::Result<Bytes>>>>>,
}

impl SequencedStreamingHttpClient {
    /// Create a streaming client from the chunks it should yield.
    pub fn new(chunks: Vec<http_client::Result<Bytes>>) -> Self {
        Self {
            chunks: Arc::new(Mutex::new(Some(chunks))),
        }
    }
}

impl HttpClientExt for SequencedStreamingHttpClient {
    no_unary_surface!();

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let chunks = match self.chunks.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };

        async move {
            let Some(chunks) = chunks else {
                return Err(http_client::Error::non_success_with_details(
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    http::HeaderMap::new(),
                    "streaming chunks should only be consumed once".to_string(),
                ));
            };

            let byte_stream = futures::stream::iter(chunks);
            let boxed_stream: http_client::BoxedStream = Box::pin(byte_stream);

            Response::builder()
                .status(http::StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "text/event-stream")
                .body(boxed_stream)
                .map_err(http_client::Error::Protocol)
        }
    }
}
