use crate::http_client::{sse::BoxedStream, util::Escape};
use bytes::Bytes;
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use http::{StatusCode, request::Parts};
use reqwest::{Body, multipart::Form};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::Level;

pub mod retry;
pub mod sse;
pub mod util;

use std::pin::Pin;

use crate::wasm_compat::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Http error: {0}")]
    Protocol(#[from] http::Error),
    #[error("Invalid status code: {0}")]
    InvalidStatusCode(StatusCode),
    #[error("Invalid status code {0} with message: {1}")]
    InvalidStatusCodeWithMessage(StatusCode, String),
    #[error("Stream ended")]
    StreamEnded,
    #[error("Invalid content type was returned: {0:?}")]
    InvalidContentType(HeaderValue),
    #[cfg(not(target_family = "wasm"))]
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + 'static>),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(not(target_family = "wasm"))]
pub(crate) fn instance_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

#[cfg(target_family = "wasm")]
fn instance_error<E: std::error::Error + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

pub type LazyBytes = WasmBoxedFuture<'static, Result<Bytes>>;
pub type LazyBody<T> = WasmBoxedFuture<'static, Result<T>>;

pub type StreamingResponse<T> = Response<T>;

pub struct NoBody;

impl From<NoBody> for Bytes {
    fn from(_: NoBody) -> Self {
        Bytes::new()
    }
}

impl From<NoBody> for Body {
    fn from(_: NoBody) -> Self {
        reqwest::Body::default()
    }
}

pub async fn text(response: Response<LazyBody<Vec<u8>>>) -> Result<String> {
    let text = response.into_body().await?;
    Ok(String::from(String::from_utf8_lossy(&text)))
}

pub fn with_bearer_auth(req: Builder, auth: &str) -> Result<Builder> {
    let auth_header =
        HeaderValue::from_str(&format!("Bearer {}", auth)).map_err(http::Error::from)?;

    Ok(req.header("Authorization", auth_header))
}

static LOG_HTTP_BODY_MAX: AtomicUsize = AtomicUsize::new(8 * 1024);

/// Extension trait for client builders to configure HTTP logging.
///
/// # Example
/// ```
/// use rig::prelude::*;
/// use rig::providers::openai;
///
/// let client = openai::Client::builder("api-key")
///     .max_log_body_preview(8192)
///     .build();
/// ```
pub trait HttpLogConfigExt {
    /// Set the maximum number of bytes to preview from the body when logging in the `TRACE` level.
    /// Defaults to 8192 bytes. Set to 0 to disable body preview logging.
    ///
    /// This method can be called on any client builder to configure HTTP logging before building the client.
    fn max_log_body_preview(self, max_preview_bytes: usize) -> Self;
}

impl<T> HttpLogConfigExt for T
where
    T: Sized,
{
    fn max_log_body_preview(self, max_preview_bytes: usize) -> Self {
        LOG_HTTP_BODY_MAX.store(max_preview_bytes, Ordering::Relaxed);
        self
    }
}

/// Set the maximum number of bytes to preview from the body when logging in the `TRACE` level. Defaults to 8192 bytes. Set to 0 to disable body preview logging.
pub fn set_max_log_body_preview(max_preview_bytes: usize) {
    LOG_HTTP_BODY_MAX.store(max_preview_bytes, Ordering::Relaxed);
}

fn body_preview_len() -> usize {
    LOG_HTTP_BODY_MAX.load(Ordering::Relaxed)
}

/// A helper trait to make generic requests (both regular and SSE) possible.
pub trait HttpClientExt: WasmCompatSend + WasmCompatSync {
    /// Send a HTTP request, get a response back (as bytes). Response must be able to be turned back into Bytes.
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request with a multipart body, get a response back (as bytes). Response must be able to be turned back into Bytes (although usually for the response, you will probably want to specify Bytes anyway).
    fn send_multipart<U>(
        &self,
        req: Request<Form>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request, get a streamed response back (as a stream of [`bytes::Bytes`].)
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse<BoxedStream>>> + WasmCompatSend
    where
        T: Into<Bytes>;
}

impl HttpClientExt for reqwest::Client {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        U: From<Bytes> + WasmCompatSend,
    {
        let (parts, body) = req.into_parts();

        let body_bytes: Bytes = body.into();
        log_request(&parts, &body_bytes);

        let req = self
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body_bytes);

        async move {
            let response = req.send().await.map_err(instance_error)?;
            if !response.status().is_success() {
                return Err(Error::InvalidStatusCodeWithMessage(
                    response.status(),
                    response.text().await.unwrap(),
                ));
            }

            let mut res = Response::builder().status(response.status());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let body: LazyBody<U> = Box::pin(async {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| Error::Instance(e.into()))?;

                let body = U::from(bytes);
                Ok(body)
            });

            res.body(body).map_err(Error::Protocol)
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<Form>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let (parts, body) = req.into_parts();

        log_headers(&parts);

        let req = self
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .multipart(body);

        async move {
            let response = req.send().await.map_err(instance_error)?;
            if !response.status().is_success() {
                return Err(Error::InvalidStatusCodeWithMessage(
                    response.status(),
                    response.text().await.unwrap(),
                ));
            }

            let mut res = Response::builder().status(response.status());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let body: LazyBody<U> = Box::pin(async {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| Error::Instance(e.into()))?;

                let body = U::from(bytes);
                Ok(body)
            });

            res.body(body).map_err(Error::Protocol)
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse<BoxedStream>>> + WasmCompatSend
    where
        T: Into<Bytes>,
    {
        let (parts, body) = req.into_parts();

        let body_bytes: Bytes = body.into();
        log_request(&parts, &body_bytes);

        let req = self
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body_bytes)
            .build()
            .map_err(|x| Error::Instance(x.into()))
            .unwrap();

        let client = self.clone();

        async move {
            let response: reqwest::Response = client.execute(req).await.map_err(instance_error)?;
            if !response.status().is_success() {
                return Err(Error::InvalidStatusCodeWithMessage(
                    response.status(),
                    response.text().await.unwrap(),
                ));
            }

            #[cfg(not(target_family = "wasm"))]
            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            #[cfg(target_family = "wasm")]
            let mut res = Response::builder().status(response.status());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            use futures::StreamExt;

            let mapped_stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> =
                Box::pin(
                    response
                        .bytes_stream()
                        .map(|chunk| chunk.map_err(|e| Error::Instance(Box::new(e)))),
                );

            res.body(mapped_stream).map_err(Error::Protocol)
        }
    }
}

fn log_request(parts: &Parts, body: &Bytes) {
    if tracing::enabled!(Level::TRACE) {
        // Redact sensitive headers (e.g., Authorization) for logging
        let mut redacted_headers = parts.headers.clone();
        redacted_headers.remove("Authorization");
        let preview_len = body_preview_len();

        if preview_len > 0 {
            let shown = std::cmp::min(preview_len, body.len());
            let preview = Escape::new(&body[..shown]);
            tracing::trace!(
                target: "rig::http",
                method = %parts.method,
                uri = %parts.uri,
                body_len = body.len(),
                body_preview_len = shown,
                headers = ?redacted_headers,
                body_preview = ?preview,
                "sending HTTP request"
            );
        } else {
            tracing::trace!(
                target: "rig::http",
                method = %parts.method,
                uri = %parts.uri,
                body_len = body.len(),
                headers = ?redacted_headers,
                "sending HTTP request"
            );
        }
    }
}

fn log_headers(parts: &Parts) {
    if tracing::enabled!(Level::TRACE) {
        // Redact sensitive headers (e.g., Authorization) for logging
        let mut redacted_headers = parts.headers.clone();
        redacted_headers.remove("Authorization");
        tracing::trace!(
            target: "rig::http",
            method = %parts.method,
            uri = %parts.uri,
            headers = ?redacted_headers,
            "sending HTTP request"
        );
    }
}
