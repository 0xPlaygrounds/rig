//! Public, transport-neutral HTTP input protocol.

use std::{fmt, sync::Arc};

use futures::StreamExt;

use super::{Backend, Bytes, Error, MultipartForm, Request, Response, Result, StreamingResponse};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend};

/// A caller-supplied HTTP transport for [`HttpRuntime`](crate::http_runtime::HttpRuntime).
///
/// Implementations receive transport-neutral request bodies and may return any
/// HTTP status as an `Ok(Response)`. Rig's adapter converts non-success
/// responses into [`Error::InvalidStatusCodeWithMessage`] while preserving the
/// body, so provider parsers remain responsible for provider-specific error
/// shaping. `Err` is reserved for I/O, protocol, response-body, or explicit
/// transport failures.
///
/// The returned futures are `'static`: clone any shared state needed by an
/// invocation before constructing the future rather than borrowing `self`
/// across an `await`.
///
/// # Example
///
/// ```no_run
/// use futures::stream;
/// use rig_core::{
///     http_client::{
///         BoxedStream, Bytes, HttpTransport, Request, Response, Result,
///         StreamingResponse,
///     },
///     http_runtime::HttpRuntime,
///     providers::openai,
///     wasm_compat::WasmBoxedFuture,
/// };
///
/// #[derive(Clone)]
/// struct CannedTransport;
///
/// impl HttpTransport for CannedTransport {
///     fn send(
///         &self,
///         _request: Request<Vec<u8>>,
///     ) -> WasmBoxedFuture<'static, Result<Response<Bytes>>> {
///         Box::pin(async {
///             Response::builder()
///                 .status(200)
///                 .body(Bytes::from_static(br#"{"ok":true}"#))
///                 .map_err(Into::into)
///         })
///     }
///
///     fn send_streaming(
///         &self,
///         _request: Request<Vec<u8>>,
///     ) -> WasmBoxedFuture<'static, Result<StreamingResponse>> {
///         Box::pin(async {
///             let body: BoxedStream = Box::pin(stream::empty());
///             Response::builder()
///                 .status(200)
///                 .header("content-type", "text/event-stream")
///                 .body(body)
///                 .map_err(Into::into)
///         })
///     }
/// }
///
/// # fn build_client() -> std::result::Result<(), Box<dyn std::error::Error>> {
/// let runtime = HttpRuntime::from_transport(CannedTransport);
/// let _client = openai::Client::builder()
///     .api_key("test-key")
///     .http_runtime(runtime)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub trait HttpTransport: Send + Sync + 'static {
    /// Send a request and return its complete response body.
    ///
    /// Any HTTP status may be returned as `Ok`; Rig normalizes non-success
    /// statuses before passing the response to provider code.
    fn send(&self, request: Request<Vec<u8>>) -> WasmBoxedFuture<'static, Result<Response<Bytes>>>;

    /// Send a request and return its response body as a byte stream.
    ///
    /// Any HTTP status may be returned as `Ok`. For a non-success status, Rig
    /// collects the stream and preserves it in the normalized status error.
    /// For SSE endpoints, return the raw response bytes: Rig's existing event
    /// parser owns framing, retry timing, last-event IDs, and reconnection.
    fn send_streaming(
        &self,
        request: Request<Vec<u8>>,
    ) -> WasmBoxedFuture<'static, Result<StreamingResponse>>;

    /// Send a multipart request and return its complete response body.
    ///
    /// Transports that do not serve multipart endpoints may keep this default,
    /// which reports [`Error::UnsupportedMultipart`] without performing I/O.
    /// As with [`send`](Self::send), any HTTP status may otherwise be returned
    /// as `Ok` and is normalized by Rig.
    fn send_multipart(
        &self,
        _request: Request<MultipartForm>,
    ) -> WasmBoxedFuture<'static, Result<Response<Bytes>>> {
        Box::pin(async { Err(Error::UnsupportedMultipart) })
    }
}

/// A cloneable, type-erased caller-supplied HTTP transport.
///
/// Clones share one retained transport instance. Its value is deliberately
/// omitted from [`Debug`](fmt::Debug), because it may retain credentials,
/// default headers, or other secrets.
#[derive(Clone)]
pub struct CustomTransport {
    inner: Arc<dyn HttpTransport>,
}

impl CustomTransport {
    /// Erase `transport` into the concrete record held by `HttpRuntime`.
    pub fn new(transport: impl HttpTransport) -> Self {
        Self {
            inner: Arc::new(transport),
        }
    }
}

impl fmt::Debug for CustomTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CustomTransport")
            .finish_non_exhaustive()
    }
}

impl Backend for CustomTransport {
    fn send(
        &self,
        request: Request<Vec<u8>>,
    ) -> impl Future<Output = Result<Response<Bytes>>> + WasmCompatSend + 'static {
        let inner = Arc::clone(&self.inner);
        async move { normalize_buffered(inner.send(request).await) }
    }

    fn send_multipart(
        &self,
        request: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<Bytes>>> + WasmCompatSend + 'static {
        let inner = Arc::clone(&self.inner);
        async move { normalize_buffered(inner.send_multipart(request).await) }
    }

    fn send_streaming(
        &self,
        request: Request<Vec<u8>>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend {
        let inner = Arc::clone(&self.inner);
        async move { normalize_streaming(inner.send_streaming(request).await).await }
    }
}

fn normalize_buffered(response: Result<Response<Bytes>>) -> Result<Response<Bytes>> {
    let response = response?;
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = String::from_utf8_lossy(response.body()).into_owned();
    Err(Error::InvalidStatusCodeWithMessage(status, body))
}

async fn normalize_streaming(response: Result<StreamingResponse>) -> Result<StreamingResponse> {
    let response = response?;
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let mut stream = response.into_body();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk?);
    }
    Err(Error::InvalidStatusCodeWithMessage(
        status,
        String::from_utf8_lossy(&body).into_owned(),
    ))
}
