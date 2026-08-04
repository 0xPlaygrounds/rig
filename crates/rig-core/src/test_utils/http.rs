//! HTTP client doubles for provider tests.

use std::{
    collections::VecDeque,
    future::{self, Future},
    sync::{Arc, Mutex, MutexGuard},
};

use bytes::Bytes;

use crate::{
    http_client::{self, Backend, MultipartForm, Request, Response, StreamingResponse},
    wasm_compat::WasmCompatSend,
};

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
    /// Return this body with a successful HTTP status.
    Success(Bytes),
    /// Return a status-code error with the given body text.
    Error(http::StatusCode, String),
    /// Return an HTTP response with the given (typically non-success) status
    /// and body, instead of a transport-level error.
    ErrorResponse(http::StatusCode, Bytes),
}

impl MockHttpResponse {
    /// Create a successful response from bytes.
    pub fn success(body: impl Into<Bytes>) -> Self {
        Self::Success(body.into())
    }

    /// Create an error response with a status code and message.
    pub fn error(status: http::StatusCode, message: impl Into<String>) -> Self {
        Self::Error(status, message.into())
    }
}

impl Default for MockHttpResponse {
    fn default() -> Self {
        Self::Success(Bytes::new())
    }
}

/// A `Backend` implementation that records unary requests and returns
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

    fn build_unary_response(response: MockHttpResponse) -> http_client::Result<Response<Bytes>> {
        let (status, response_body) = match response {
            MockHttpResponse::Success(response_body) => (http::StatusCode::OK, response_body),
            MockHttpResponse::Error(status, message) => {
                return Err(http_client::Error::InvalidStatusCodeWithMessage(
                    status, message,
                ));
            }
            MockHttpResponse::ErrorResponse(status, response_body) => (status, response_body),
        };
        Response::builder()
            .status(status)
            .body(response_body)
            .map_err(http_client::Error::Protocol)
    }
}

impl Backend for RecordingHttpClient {
    fn send(
        &self,
        req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        let response = self.response_guard().clone();
        let (parts, body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, Bytes::from(body));

        async move { Self::build_unary_response(response) }
    }

    fn send_multipart(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        let response = self.response_guard().clone();
        let (parts, _body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, Bytes::new());

        async move { Self::build_unary_response(response) }
    }

    fn send_streaming(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }
}

/// A `Backend` implementation that records unary requests and returns
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

impl Backend for SequencedHttpClient {
    fn send(
        &self,
        req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        let response = self.next_response();
        let (parts, body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, Bytes::from(body));

        async move {
            match response {
                Some(response) => RecordingHttpClient::build_unary_response(response),
                None => Err(http_client::Error::InvalidStatusCode(
                    http::StatusCode::NOT_IMPLEMENTED,
                )),
            }
        }
    }

    fn send_multipart(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        let response = self.next_response();
        let (parts, _body) = req.into_parts();
        self.record_request(parts.uri.to_string(), parts.headers, Bytes::new());

        async move {
            match response {
                Some(response) => RecordingHttpClient::build_unary_response(response),
                None => Err(http_client::Error::InvalidStatusCode(
                    http::StatusCode::NOT_IMPLEMENTED,
                )),
            }
        }
    }

    fn send_streaming(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
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

impl Backend for MockStreamingClient {
    fn send(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_multipart(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_streaming(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        let sse_bytes = self.sse_bytes.clone();
        async move {
            let byte_stream =
                futures::stream::iter(vec![Ok::<Bytes, http_client::Error>(sse_bytes)]);
            let boxed_stream: http_client::sse::BoxedStream = Box::pin(byte_stream);

            Response::builder()
                .status(http::StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "text/event-stream")
                .body(boxed_stream)
                .map_err(http_client::Error::Protocol)
        }
    }
}

/// A `Backend` implementation whose `send_streaming` fails immediately
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

impl Backend for HttpErrorStreamingClient {
    fn send(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_multipart(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_streaming(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        let status = self.status;
        let body = self.body.clone();
        async move {
            Err(http_client::Error::InvalidStatusCodeWithMessage(
                status, body,
            ))
        }
    }
}

/// A `Backend` implementation that returns one scripted stream of byte
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

impl Backend for SequencedStreamingHttpClient {
    fn send(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_multipart(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<Bytes>>> + WasmCompatSend + 'static {
        future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_streaming(
        &self,
        _req: Request<Vec<u8>>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend {
        let chunks = match self.chunks.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };

        async move {
            let Some(chunks) = chunks else {
                return Err(http_client::Error::InvalidStatusCodeWithMessage(
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "streaming chunks should only be consumed once".to_string(),
                ));
            };

            let byte_stream = futures::stream::iter(chunks);
            let boxed_stream: http_client::sse::BoxedStream = Box::pin(byte_stream);

            Response::builder()
                .status(http::StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "text/event-stream")
                .body(boxed_stream)
                .map_err(http_client::Error::Protocol)
        }
    }
}
