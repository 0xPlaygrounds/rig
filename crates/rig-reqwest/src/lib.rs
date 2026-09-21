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
//! The bundled `reqwest` transport for Rig.
//!
//! `rig-core` is transport-agnostic: a provider's wire says what to send and
//! how to read the reply, and [`Bound`](rig_core::driver::Bound) pairs it with
//! an `H: HttpClientExt` — defaulting to the erased
//! [`BoxedHttpClient`] — to make a model. rig-core itself depends on neither
//! reqwest nor tokio. This crate supplies:
//!
//! - [`ReqwestClient`], a newtype over [`reqwest::Client`] implementing
//!   [`HttpClientExt`] (and, behind the `reqwest-middleware` feature,
//!   [`ReqwestMiddlewareClient`] over `reqwest_middleware::ClientWithMiddleware`).
//! - The construction convenience the `rig` facade re-exports:
//!   [`client::DefaultTransport`], whose `bound()` builds the erased default
//!   over a `ReqwestClient`, so
//!   `openai::wire::OpenAI::from_env()?.bound()?` yields a usable provider
//!   without naming a transport. Name one explicitly with
//!   [`Bind::bind`](rig_core::driver::Bind::bind) instead and the concrete
//!   type stays in the signature.
//!
//! # Running without a tokio runtime
//!
//! Async reqwest needs a tokio reactor on native targets. Inside a tokio
//! runtime this transport captures its handle. Outside one — Bevy task pools,
//! smol, `futures::executor::block_on` — it uses a lazily started single-worker
//! fallback reactor. Requests and bodies enter that context on each poll;
//! they remain owned by the caller, without detached forwarding tasks. Dropping
//! them cancels their local operation, not remote work already accepted.
//!
//! A host-supplied runtime must enable I/O and timers and remain driven until
//! requests and bodies finish. A runtime handle alone does not keep it alive.
//! Stop admitting work, cancel/drop operations, then release clients and shut
//! down the host runtime. Context detection cannot verify an enabled reactor:
//! polling under a runtime without I/O/timers can panic in Tokio. A stopped
//! runtime cannot be restarted by retaining its handle; in-flight I/O fails.

pub use reqwest;

/// The bundled transport: a thin newtype over [`reqwest::Client`] that
/// implements [`HttpClientExt`].
///
/// A newtype rather than `reqwest::Client` itself because the orphan rule
/// forbids implementing rig-core's trait for reqwest's type from this crate.
/// That is the whole of its job, so its surface is deliberately small: convert
/// in with [`From`] or [`Default`], borrow the inner client with [`AsRef`],
/// take it back with [`into_inner`](Self::into_inner).
///
/// It used to expose the inner client three ways at once — a public field,
/// `Deref`, and `From`. `Deref` on a type that is not a smart pointer makes
/// reqwest's whole inherent API look like this type's own, which it is not;
/// the two explicit accessors say the same thing without the illusion.
#[derive(Clone, Debug, Default)]
pub struct ReqwestClient(reqwest::Client);

impl ReqwestClient {
    /// Wrap an already-configured `reqwest::Client` — timeouts, proxies, a
    /// connection pool shared with the rest of the host.
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

/// [`HttpClientExt`] for a `reqwest_middleware::ClientWithMiddleware`.
///
/// The same shape as [`ReqwestClient`], minus `Default`: a middleware client
/// with no middleware is just a `reqwest::Client` with extra indirection, so
/// there is no default worth having — build one with
/// `reqwest_middleware::ClientBuilder` and convert it in.
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

    /// Erase this transport behind [`BoxedHttpClient`], as
    /// [`ReqwestClient::boxed`] does — a host that erases its transport should
    /// not lose the option by having chosen middleware.
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

/// Map a transport-level `reqwest::Error` onto the transport-agnostic
/// [`Error`].
///
/// This is the response-less side (connect, decode, timeout): a reply the
/// server made is read off the `reqwest::Response` with its body and headers
/// by the send path instead, never reduced to a bare status.
/// Rig never calls `error_for_status`, so a `reqwest::Error` here carries no
/// reply to preserve.
pub fn from_reqwest(err: reqwest::Error) -> Error {
    Error::instance(err)
}

/// Read the status, headers and body off a failed `reqwest::Response` and
/// build the headers-preserving non-success error (rig#2314).
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
/// Fails when a part names a content type reqwest rejects. That used to be
/// swallowed — the part was rebuilt without its content type and the request
/// went out anyway — so a typo'd MIME reached the provider as a *missing* one
/// and came back as an opaque provider error about the payload. A content type
/// the caller asked for and did not get is a caller bug worth reporting at the
/// call site.
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

/// The one request-driving routine both reqwest-flavoured clients share:
/// `reqwest::Client` and `ClientWithMiddleware` expose the same
/// `request(..) -> RequestBuilder` / `send()` surface but are unrelated types,
/// so the shared code is written once against a tiny private trait.
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
    // The form is rendered before the request is driven, so an unusable
    // content type fails here rather than reaching the provider as a silently
    // missing one.
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
