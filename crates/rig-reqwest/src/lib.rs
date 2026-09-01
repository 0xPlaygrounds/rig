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
//! `rig-core` is transport-agnostic: every provider client is generic over an
//! `H: HttpClientExt`, and rig-core itself names no default transport and
//! depends on neither reqwest nor tokio. This crate supplies:
//!
//! - [`ReqwestClient`], a newtype over [`reqwest::Client`] implementing
//!   [`HttpClientExt`] (and, behind the `reqwest-middleware` feature,
//!   [`ReqwestMiddlewareClient`] over `reqwest_middleware::ClientWithMiddleware`).
//! - The default-transport conveniences the `rig` facade re-exports:
//!   [`client::DefaultTransportClient`] / [`client::DefaultTransportBuilder`]
//!   (so `Client::new(key)`, `Client::from_env()`, `builder().…build()` work
//!   without naming a transport) and the [`providers`] alias tree (so
//!   `openai::CompletionModel` means `…<reqwest::Client>` in type position).
//! # Running without a tokio runtime
//!
//! Async reqwest needs a tokio reactor on native targets. Inside a tokio
//! runtime this transport awaits reqwest futures directly. Outside one —
//! Bevy task pools, smol, `futures::executor::block_on` — it drives them on a
//! lazily started single-worker fallback runtime and hands the caller plain
//! runtime-agnostic futures (a tokio `JoinHandle`, a `futures` channel receiver
//! for streamed bodies), so nothing is ever `block_on`'d and no thread parks.

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
#[cfg(feature = "reqwest-middleware")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
#[derive(Clone, Debug)]
pub struct ReqwestMiddlewareClient(reqwest_middleware::ClientWithMiddleware);

#[cfg(feature = "reqwest-middleware")]
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

#[cfg(feature = "reqwest-middleware")]
impl From<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn from(client: reqwest_middleware::ClientWithMiddleware) -> Self {
        Self(client)
    }
}

#[cfg(feature = "reqwest-middleware")]
impl From<ReqwestMiddlewareClient> for BoxedHttpClient {
    fn from(client: ReqwestMiddlewareClient) -> Self {
        client.boxed()
    }
}

#[cfg(feature = "reqwest-middleware")]
impl AsRef<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn as_ref(&self) -> &reqwest_middleware::ClientWithMiddleware {
        &self.0
    }
}

pub mod client;
pub mod providers;
#[cfg(not(target_family = "wasm"))]
mod runtime;

/// Bring the default-transport traits into scope.
pub mod prelude {
    pub use crate::client::{DefaultTransportBuilder, DefaultTransportClient};
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
/// A failure that carries a status (an `error_for_status` rejection) keeps
/// it as [`Error::InvalidStatusCode`] so provider retry and error-inspection
/// paths can still read the code; a response-less failure (connect, decode,
/// timeout) becomes [`Error::Instance`].
pub fn from_reqwest(err: reqwest::Error) -> Error {
    err.status()
        .map_or_else(|| Error::instance(err), Error::InvalidStatusCode)
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

/// When the body is read.
///
/// The two moments look alike and are not interchangeable: the caller decides
/// which by whether it can still poll reqwest when it awaits the body.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BodyTiming {
    /// Hand back a body future the caller awaits later. Only valid where the
    /// caller can poll reqwest futures — inside a tokio runtime, or on wasm.
    Lazy,
    /// Read the body now and hand back a ready future. This is the off-runtime
    /// path: the future it returns may be polled by an executor that cannot
    /// drive reqwest, so nothing reqwest-shaped may survive into it.
    Eager,
}

/// Build the transport-agnostic response, reading the body at `timing`.
async fn into_response<U>(
    response: reqwest::Response,
    timing: BodyTiming,
) -> Result<Response<LazyBody<U>>>
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

    let body: LazyBody<U> = match timing {
        BodyTiming::Lazy => Box::pin(async {
            let bytes = response.bytes().await.map_err(Error::instance)?;
            Ok(U::from(bytes))
        }),
        BodyTiming::Eager => {
            let bytes = response.bytes().await.map_err(Error::instance)?;
            Box::pin(std::future::ready(Ok(U::from(bytes))))
        }
    };

    res.body(body).map_err(Error::Protocol)
}

/// The timing the off-runtime side of [`drive`] needs.
///
/// On wasm there is no fallback runtime and no off-runtime path, so the body
/// stays lazy; natively the off-runtime side must not hand reqwest futures to
/// an executor that cannot poll them.
const OFF_RUNTIME_TIMING: BodyTiming = if cfg!(target_family = "wasm") {
    BodyTiming::Lazy
} else {
    BodyTiming::Eager
};

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
/// headers-preserving error. The byte stream is reqwest's own, so this is
/// only valid where the caller can poll reqwest futures.
async fn into_streaming_response(response: reqwest::Response) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }
    let res = streaming_head(&response);

    use futures::StreamExt;
    let mapped_stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(
        response
            .bytes_stream()
            .map(|chunk| chunk.map_err(Error::instance)),
    );

    res.body(mapped_stream).map_err(Error::Protocol)
}

/// Off-runtime streaming: the body is *driven* on the fallback runtime and
/// forwarded through a bounded channel; the caller's executor polls only the
/// receiver, which is a plain `futures` stream.
#[cfg(not(target_family = "wasm"))]
async fn into_forwarded_streaming_response(
    response: reqwest::Response,
) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }
    let res = streaming_head(&response);

    use futures::{SinkExt, StreamExt};
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Bytes>>(16);
    runtime::spawn_off_runtime(async move {
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            if tx.send(chunk.map_err(Error::instance)).await.is_err() {
                // Receiver dropped: the consumer stopped reading.
                break;
            }
        }
    })?;
    let stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(rx);
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

#[cfg(feature = "reqwest-middleware")]
impl ReqwestLike for ReqwestMiddlewareClient {
    type Builder = reqwest_middleware::RequestBuilder;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder {
        self.0.request(method, url)
    }
}

#[cfg(feature = "reqwest-middleware")]
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

/// Drive `request` and convert its response on whichever side can poll
/// reqwest: directly when inside tokio (or on wasm), on the fallback runtime
/// otherwise — in which case `off_runtime` must produce a response whose
/// body no longer needs reqwest to be polled.
async fn drive<B, T, OnRt, OffRt, FutOn, FutOff>(
    request: B,
    on_runtime: OnRt,
    off_runtime: OffRt,
) -> Result<T>
where
    B: RequestBuilderLike,
    T: WasmCompatSend + 'static,
    OnRt: FnOnce(reqwest::Response) -> FutOn + WasmCompatSend + 'static,
    FutOn: Future<Output = Result<T>> + WasmCompatSend,
    OffRt: FnOnce(reqwest::Response) -> FutOff + WasmCompatSend + 'static,
    FutOff: Future<Output = Result<T>> + WasmCompatSend,
{
    #[cfg(not(target_family = "wasm"))]
    if !runtime::in_tokio() {
        return runtime::run_off_runtime(async move {
            let response = request.send_request().await?;
            off_runtime(response).await
        })
        .await?;
    }
    #[cfg(target_family = "wasm")]
    let _ = &off_runtime;
    let response = request.send_request().await?;
    on_runtime(response).await
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

    drive(
        req,
        |response| into_response::<U>(response, BodyTiming::Lazy),
        |response| into_response::<U>(response, OFF_RUNTIME_TIMING),
    )
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

    async move {
        drive(
            req?,
            |response| into_response::<U>(response, BodyTiming::Lazy),
            |response| into_response::<U>(response, OFF_RUNTIME_TIMING),
        )
        .await
    }
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

    #[cfg(not(target_family = "wasm"))]
    let off = into_forwarded_streaming_response;
    #[cfg(target_family = "wasm")]
    let off = into_streaming_response;
    drive(req, into_streaming_response, off)
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
    #[cfg(feature = "reqwest-middleware")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
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
mod tests {
    use super::*;
    use http::StatusCode;

    /// rig#2210: the bundled transport's own error constructor is where the
    /// headers are captured, so drive it with a real `reqwest::Response`.
    #[tokio::test]
    async fn non_success_status_error_preserves_response_headers() {
        let response = http::Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("retry-after", "20")
            .header("x-ratelimit-remaining", "0")
            .body(r#"{"error":{"message":"rate limited"}}"#)
            .expect("valid response");

        let error = non_success_status_error(reqwest::Response::from(response)).await;

        assert!(matches!(
            &error,
            Error::InvalidStatusCodeWithDetails { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS
        ));
        let headers = error
            .non_success_headers()
            .expect("headers captured at error construction");
        assert_eq!(
            headers.get("retry-after").and_then(|v| v.to_str().ok()),
            Some("20")
        );
        assert_eq!(
            headers
                .get("x-ratelimit-remaining")
                .and_then(|v| v.to_str().ok()),
            Some("0")
        );
    }
}
