use crate::http_client::{escape::Escape, sse::BoxedStream};
use bytes::Bytes;
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use http::{StatusCode, request::Parts};
use reqwest::{Body, multipart::Form};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

mod escape;
pub mod retry;
pub mod sse;

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

#[derive(Clone, Debug)]
pub struct HttpLogSettings {
    max_body_preview: Arc<AtomicUsize>,
    headers_enabled: Arc<AtomicBool>,
}

impl Default for HttpLogSettings {
    fn default() -> Self {
        Self {
            max_body_preview: Arc::new(AtomicUsize::new(8 * 1024)),
            headers_enabled: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl HttpLogSettings {
    pub fn new(max_preview_bytes: usize) -> Self {
        Self {
            max_body_preview: Arc::new(AtomicUsize::new(max_preview_bytes)),
            headers_enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn max_body_preview(&self) -> usize {
        self.max_body_preview.load(Ordering::Relaxed)
    }

    pub fn set_max_body_preview(&self, max_preview_bytes: usize) {
        self.max_body_preview
            .store(max_preview_bytes, Ordering::Relaxed);
    }

    pub fn log_headers_enabled(&self) -> bool {
        self.headers_enabled.load(Ordering::Relaxed)
    }

    pub fn set_log_headers_enabled(&self, enabled: bool) {
        self.headers_enabled.store(enabled, Ordering::Relaxed);
    }
}

fn http_log_settings() -> &'static HttpLogSettings {
    static SETTINGS: OnceLock<HttpLogSettings> = OnceLock::new();
    SETTINGS.get_or_init(HttpLogSettings::default)
}

/// Set the maximum number of bytes to preview from the body when logging at the `TRACE` level.
/// Defaults to 8192 bytes. Set to 0 to disable body preview logging.
pub fn set_max_log_body_preview(max_preview_bytes: usize) {
    http_log_settings().set_max_body_preview(max_preview_bytes);
}

/// Get the current maximum number of bytes previewed from the body when logging at the `TRACE` level.
fn max_log_body_preview() -> usize {
    http_log_settings().max_body_preview()
}

/// Enable or disable header logging when tracing HTTP requests/responses.
pub fn set_log_headers_enabled(enabled: bool) {
    http_log_settings().set_log_headers_enabled(enabled);
}

/// Returns whether header logging is currently enabled.
fn log_headers_enabled() -> bool {
    http_log_settings().log_headers_enabled()
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
    if !log_headers_enabled() {
        return;
    }

    let redacted_headers = redact_sensitive_headers(&parts.headers);
    let preview_len = max_log_body_preview();

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

fn log_headers(parts: &Parts) {
    if !log_headers_enabled() {
        return;
    }

    let redacted_headers = redact_sensitive_headers(&parts.headers);
    tracing::trace!(
        target: "rig::http",
        method = %parts.method,
        uri = %parts.uri,
        headers = ?redacted_headers,
        "sending HTTP request"
    );
}

/// Redact sensitive headers (e.g., Authorization) for logging
fn redact_sensitive_headers(headers: &HeaderMap) -> HeaderMap {
    let is_sensitive_header = |name: &str| {
        let trimmed = name.trim();
        match trimmed.len() {
            13 => trimmed.eq_ignore_ascii_case("authorization"),
            len if len > 4 && trimmed[len - 4..].eq_ignore_ascii_case("-key") => true,
            _ => false,
        }
    };
    let mut filtered = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers.iter() {
        // avoid the closure allocation, inline check, avoid clones if possible
        if !is_sensitive_header(name.as_str()) {
            filtered.append(name, value.clone());
        }
    }
    filtered
}
