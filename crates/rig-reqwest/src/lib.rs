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
//! The bundled reqwest HTTP transport for Rig.
//!
//! [`ReqwestClient::default()`] is the process-wide client: built once on
//! first use and shared by every clone, so every provider client made with
//! it shares one connection pool. Construction never fails. When reqwest
//! cannot build its client (a host with no CA store, say), every send
//! reports that build failure in-band.
//!
//! Native requests and bodies enter the captured Tokio context on each poll,
//! using a lazy fallback when no runtime is current. Callers retain ownership;
//! dropping an operation cancels local work, not accepted remote work. Hosts
//! must keep their runtime driven with I/O and timers enabled until operations
//! finish. Missing drivers can panic; a stopped runtime causes I/O failure.
//!
//! ```no_run
//! use rig_reqwest::ReqwestClient;
//!
//! let shared = ReqwestClient::default();
//! let custom = ReqwestClient::from(rig_reqwest::reqwest::Client::new());
//! # let _ = (shared, custom);
//! ```

pub use reqwest;

/// A reqwest client implementing [`HttpClientExt`].
///
/// [`Default`] is the process-wide client, built once and shared by every
/// clone; `From<reqwest::Client>` wraps a configured client and keeps its
/// connection pool.
#[derive(Clone, Debug)]
pub struct ReqwestClient(Built);

/// The process-wide client, erased behind [`DynHttpClient`].
pub fn shared() -> DynHttpClient {
    DynHttpClient::new(ReqwestClient::default())
}

/// What building the client produced: the client, or why reqwest refused.
#[derive(Clone, Debug)]
enum Built {
    Client(Arc<reqwest::Client>),
    /// The shared client could not be built: every send reports this, so
    /// the failure surfaces where the first request is made rather than
    /// where the client was named.
    Failed(Arc<reqwest::Error>),
}

impl Default for ReqwestClient {
    /// The process-wide client, built on first use. It never panics: a
    /// client reqwest cannot build reports the build error on every send.
    fn default() -> Self {
        fn build() -> ReqwestClient {
            ReqwestClient(match reqwest::Client::builder().build() {
                Ok(client) => Built::Client(Arc::new(client)),
                Err(error) => Built::Failed(Arc::new(error)),
            })
        }
        #[cfg(not(target_family = "wasm"))]
        {
            static SHARED: std::sync::LazyLock<ReqwestClient> = std::sync::LazyLock::new(build);
            SHARED.clone()
        }
        #[cfg(target_family = "wasm")]
        {
            thread_local! {
                static SHARED: ReqwestClient = build();
            }
            SHARED.with(Clone::clone)
        }
    }
}

impl ReqwestClient {
    /// The reqwest client, or `None` for the shared client when reqwest
    /// could not build it.
    pub fn inner(&self) -> Option<&reqwest::Client> {
        match &self.0 {
            Built::Client(client) => Some(client),
            Built::Failed(_) => None,
        }
    }

    /// Whether `self` and `other` are clones of one client.
    #[cfg(test)]
    fn same(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Built::Client(a), Built::Client(b)) => Arc::ptr_eq(a, b),
            (Built::Failed(a), Built::Failed(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// The error every send on an unbuilt client reports.
fn unbuilt(error: &Arc<reqwest::Error>) -> Error {
    Error::instance(TransportBuildError(Arc::clone(error)))
}

impl From<reqwest::Client> for ReqwestClient {
    fn from(client: reqwest::Client) -> Self {
        Self(Built::Client(Arc::new(client)))
    }
}

/// A configured middleware client implementing [`HttpClientExt`].
///
/// Build the inner client with `reqwest_middleware::ClientBuilder`, then wrap
/// it with `From<ClientWithMiddleware>`.
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
    /// Take the inner client back.
    #[must_use]
    pub fn into_inner(self) -> reqwest_middleware::ClientWithMiddleware {
        self.0
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
impl AsRef<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn as_ref(&self) -> &reqwest_middleware::ClientWithMiddleware {
        &self.0
    }
}

#[cfg(not(target_family = "wasm"))]
mod runtime;

use bytes::Bytes;
use futures::future::Either;
use rig_http::http_client::{
    DynHttpClient, Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result,
    StreamingResponse, multipart::PartContent,
};
use rig_http::wasm_compat::*;
use std::pin::Pin;
use std::sync::Arc;

/// A transport build failure that displays the source chain and retains the
/// original reqwest error as its source.
#[derive(Debug)]
struct TransportBuildError(Arc<reqwest::Error>);

impl std::fmt::Display for TransportBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not build the bundled reqwest transport: {}",
            self.0
        )?;
        let mut source = std::error::Error::source(&*self.0);
        while let Some(cause) = source {
            write!(f, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

impl std::error::Error for TransportBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

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

impl ReqwestLike for Arc<reqwest::Client> {
    type Builder = reqwest::RequestBuilder;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder {
        self.request(method, url)
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

impl HttpClientExt for ReqwestClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        match &self.0 {
            Built::Client(client) => Either::Left(send_via(client, req)),
            Built::Failed(error) => Either::Right(std::future::ready(Err(unbuilt(error)))),
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        match &self.0 {
            Built::Client(client) => Either::Left(send_multipart_via(client, req)),
            Built::Failed(error) => Either::Right(std::future::ready(Err(unbuilt(error)))),
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        match &self.0 {
            Built::Client(client) => Either::Left(send_streaming_via(client, req)),
            Built::Failed(error) => Either::Right(std::future::ready(Err(unbuilt(error)))),
        }
    }
}

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
