#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! The bundled reqwest HTTP transport and default transport constructors for Rig.
//!
//! Native requests and bodies enter the captured Tokio context on each poll,
//! using a lazy fallback when no runtime is current. Callers retain ownership;
//! dropping an operation cancels local work, not accepted remote work. Hosts
//! must keep their runtime driven with I/O and timers enabled until operations
//! finish. Missing drivers can panic; a stopped runtime causes I/O failure.
//!
//! ```no_run
//! let transport = rig_reqwest::client::bundled()?;
//! # Ok::<(), rig_core::client::ProviderClientError>(())
//! ```

pub use reqwest;

/// A reqwest client implementing [`HttpClientExt`].
///
/// Use [`AsRef`] to borrow the client or [`into_inner`](Self::into_inner) to
/// recover ownership.
#[derive(Clone, Debug, Default)]
pub struct ReqwestClient(reqwest::Client);

impl ReqwestClient {
    /// Wrap a configured reqwest client, retaining its connection pool.
    #[must_use]
    pub fn new(client: reqwest::Client) -> Self {
        Self(client)
    }

    /// Take the inner client back.
    #[must_use]
    pub fn into_inner(self) -> reqwest::Client {
        self.0
    }

    /// Erase this transport behind [`BoxedHttpClient`], for hosts that hold
    /// one transport for many providers without naming it in their types.
    #[must_use]
    pub fn boxed(self) -> BoxedHttpClient {
        BoxedHttpClient::new(self)
    }
}

impl From<reqwest::Client> for ReqwestClient {
    fn from(client: reqwest::Client) -> Self {
        Self(client)
    }
}

impl From<ReqwestClient> for BoxedHttpClient {
    fn from(client: ReqwestClient) -> Self {
        client.boxed()
    }
}

impl AsRef<reqwest::Client> for ReqwestClient {
    fn as_ref(&self) -> &reqwest::Client {
        &self.0
    }
}

/// A configured middleware client implementing [`HttpClientExt`].
///
/// Construct the inner client with `reqwest_middleware::ClientBuilder`.
#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(
        feature = "reqwest-middleware-rustls",
        feature = "reqwest-middleware-native-tls"
    )))
)]
#[derive(Clone, Debug)]
pub struct ReqwestMiddlewareClient(reqwest_middleware::ClientWithMiddleware);

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
impl ReqwestMiddlewareClient {
    /// Wrap a built `ClientWithMiddleware`.
    #[must_use]
    pub fn new(client: reqwest_middleware::ClientWithMiddleware) -> Self {
        Self(client)
    }

    /// Take the inner client back.
    #[must_use]
    pub fn into_inner(self) -> reqwest_middleware::ClientWithMiddleware {
        self.0
    }

    /// Erase this transport behind [`BoxedHttpClient`].
    #[must_use]
    pub fn boxed(self) -> BoxedHttpClient {
        BoxedHttpClient::new(self)
    }
}

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
impl From<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn from(client: reqwest_middleware::ClientWithMiddleware) -> Self {
        Self(client)
    }
}

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
impl From<ReqwestMiddlewareClient> for BoxedHttpClient {
    fn from(client: ReqwestMiddlewareClient) -> Self {
        client.boxed()
    }
}

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
impl AsRef<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn as_ref(&self) -> &reqwest_middleware::ClientWithMiddleware {
        &self.0
    }
}

pub mod client;
#[cfg(not(target_family = "wasm"))]
mod runtime;

/// Bring the construction traits into scope.
pub mod prelude {
    pub use crate::client::DefaultTransport;
}

use bytes::Bytes;
use rig_core::http_client::{
    BoxedHttpClient, Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result,
    StreamingResponse, multipart::PartContent,
};
use rig_core::wasm_compat::*;
use std::pin::Pin;

/// Wrap a reqwest transport error as [`Error::Instance`], retaining its source.
///
/// HTTP status failures are handled separately to preserve response headers
/// and bodies.
pub fn from_reqwest(err: reqwest::Error) -> Error {
    Error::instance(err)
}

/// Read the status, headers and body off a failed `reqwest::Response` and
/// build a non-success error that preserves the headers.
async fn non_success_status_error(response: reqwest::Response) -> Error {
    let status = response.status();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
    Error::non_success_with_details(status, headers, body)
}

/// Keep successful bodies lazy and owned by the response on every executor.
async fn into_response<U>(response: reqwest::Response) -> Result<Response<LazyBody<U>>>
where
    U: From<Bytes>,
    U: WasmCompatSend + 'static,
{
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    let mut res = Response::builder().status(response.status());
    if let Some(headers) = res.headers_mut() {
        *headers = response.headers().clone();
    }

    let body = async {
        let bytes = response.bytes().await.map_err(Error::instance)?;
        Ok(U::from(bytes))
    };
    #[cfg(not(target_family = "wasm"))]
    let body = runtime::bind(body)?;
    let body: LazyBody<U> = Box::pin(body);
    res.body(body).map_err(Error::Protocol)
}

fn streaming_head(response: &reqwest::Response) -> http::response::Builder {
    #[cfg(not(target_family = "wasm"))]
    let mut res = Response::builder()
        .status(response.status())
        .version(response.version());

    #[cfg(target_family = "wasm")]
    let mut res = Response::builder().status(response.status());

    if let Some(hs) = res.headers_mut() {
        *hs = response.headers().clone();
    }
    res
}

/// Convert an already-sent streaming response into the transport-agnostic
/// [`StreamingResponse`], rejecting non-success statuses with the
/// headers-preserving error. The body retains the request's reactor context.
async fn into_streaming_response(response: reqwest::Response) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }
    let res = streaming_head(&response);

    use futures::StreamExt;
    let stream = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(Error::instance));
    #[cfg(not(target_family = "wasm"))]
    let stream = runtime::bind_stream(stream)?;
    let stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(stream);
    res.body(stream).map_err(Error::Protocol)
}

/// A part's content type was not a MIME type reqwest would accept.
#[derive(Debug, thiserror::Error)]
#[error("multipart part {part:?} has an unusable content type {content_type:?}: {source}")]
struct InvalidPartContentType {
    part: String,
    content_type: String,
    source: reqwest::Error,
}

/// Render a [`MultipartForm`] as a `reqwest::multipart::Form`.
///
/// Returns an error when a binary part has a content type reqwest rejects.
pub fn multipart_form(value: MultipartForm) -> Result<reqwest::multipart::Form> {
    let mut form = reqwest::multipart::Form::new();

    for part in value.into_parts() {
        let (name, content, filename, content_type) = part.into_pieces();
        match content {
            PartContent::Text(text) => {
                form = form.text(name, text);
            }
            PartContent::Binary(bytes) => {
                let mut req_part = reqwest::multipart::Part::bytes(bytes.to_vec());
                if let Some(content_type) = content_type.as_ref() {
                    req_part = req_part.mime_str(content_type.as_ref()).map_err(|source| {
                        Error::instance(InvalidPartContentType {
                            part: name.clone(),
                            content_type: content_type.as_ref().to_string(),
                            source,
                        })
                    })?;
                }

                if let Some(filename) = filename {
                    req_part = req_part.file_name(filename);
                }

                form = form.part(name, req_part);
            }
        }
    }

    Ok(form)
}

/// Creates request builders for plain and middleware reqwest clients.
trait ReqwestLike: Clone + WasmCompatSend + WasmCompatSync + 'static {
    type Builder: RequestBuilderLike;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder;
}

trait RequestBuilderLike: Sized + WasmCompatSend + 'static {
    fn with_headers(self, headers: http::HeaderMap) -> Self;
    fn with_body(self, body: reqwest::Body) -> Self;
    fn with_multipart(self, form: reqwest::multipart::Form) -> Self;
    fn send_request(self) -> impl Future<Output = Result<reqwest::Response>> + WasmCompatSend;
}

impl ReqwestLike for ReqwestClient {
    type Builder = reqwest::RequestBuilder;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder {
        self.0.request(method, url)
    }
}

impl RequestBuilderLike for reqwest::RequestBuilder {
    fn with_headers(self, headers: http::HeaderMap) -> Self {
        self.headers(headers)
    }
    fn with_body(self, body: reqwest::Body) -> Self {
        self.body(body)
    }
    fn with_multipart(self, form: reqwest::multipart::Form) -> Self {
        self.multipart(form)
    }
    async fn send_request(self) -> Result<reqwest::Response> {
        self.send().await.map_err(Error::instance)
    }
}

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
impl ReqwestLike for ReqwestMiddlewareClient {
    type Builder = reqwest_middleware::RequestBuilder;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder {
        self.0.request(method, url)
    }
}

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
impl RequestBuilderLike for reqwest_middleware::RequestBuilder {
    fn with_headers(self, headers: http::HeaderMap) -> Self {
        self.headers(headers)
    }
    fn with_body(self, body: reqwest::Body) -> Self {
        self.body(body)
    }
    fn with_multipart(self, form: reqwest::multipart::Form) -> Self {
        self.multipart(form)
    }
    async fn send_request(self) -> Result<reqwest::Response> {
        self.send().await.map_err(Error::instance)
    }
}

/// Select a reactor on first poll and retain it through response conversion.
async fn drive<B, T, Convert, F>(request: B, convert: Convert) -> Result<T>
where
    B: RequestBuilderLike,
    Convert: FnOnce(reqwest::Response) -> F + WasmCompatSend,
    F: Future<Output = Result<T>> + WasmCompatSend,
{
    let operation = async move { convert(request.send_request().await?).await };
    #[cfg(not(target_family = "wasm"))]
    let operation = runtime::bind(operation)?;
    operation.await
}

fn send_via<C, T, U>(
    client: &C,
    req: Request<T>,
) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
where
    C: ReqwestLike,
    T: Into<Bytes>,
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let (parts, body) = req.into_parts();
    let req = client
        .request_builder(parts.method, parts.uri.to_string())
        .with_headers(parts.headers)
        .with_body(body.into().into());

    drive(req, into_response::<U>)
}

fn send_multipart_via<C, U>(
    client: &C,
    req: Request<MultipartForm>,
) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
where
    C: ReqwestLike,
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let (parts, body) = req.into_parts();
    // Reject invalid MIME types locally rather than sending incomplete metadata.
    let form = multipart_form(body);
    let req = form.map(|form| {
        client
            .request_builder(parts.method, parts.uri.to_string())
            .with_headers(parts.headers)
            .with_multipart(form)
    });

    async move { drive(req?, into_response::<U>).await }
}

fn send_streaming_via<C, T>(
    client: &C,
    req: Request<T>,
) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
where
    C: ReqwestLike,
    T: Into<Bytes> + WasmCompatSend,
{
    let (parts, body) = req.into_parts();
    let req = client
        .request_builder(parts.method, parts.uri.to_string())
        .with_headers(parts.headers)
        .with_body(body.into().into());

    drive(req, into_streaming_response)
}

macro_rules! impl_http_client_ext_via {
    ($(#[$attribute:meta])* $client:ty) => {
        $(#[$attribute])*
        impl HttpClientExt for $client {
            fn send<T, U>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                T: Into<Bytes>,
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                send_via(self, req)
            }

            fn send_multipart<U>(
                &self,
                req: Request<MultipartForm>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                send_multipart_via(self, req)
            }

            fn send_streaming<T>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
            where
                T: Into<Bytes> + WasmCompatSend,
            {
                send_streaming_via(self, req)
            }
        }
    };
}

impl_http_client_ext_via!(ReqwestClient);

impl_http_client_ext_via!(
    #[cfg(any(
        feature = "reqwest-middleware-rustls",
        feature = "reqwest-middleware-native-tls"
    ))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(
            feature = "reqwest-middleware-rustls",
            feature = "reqwest-middleware-native-tls"
        )))
    )]
    ReqwestMiddlewareClient
);

// Compile-time thread-safety contract: the transport handle is shared across
// threads by every host runtime.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<ReqwestClient>();
};

#[cfg(test)]
mod tests;
