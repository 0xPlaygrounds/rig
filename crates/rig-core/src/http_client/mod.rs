use crate::http_client::sse::BoxedStream;
use bytes::Bytes;
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use http::{HeaderName, StatusCode};
pub mod multipart;
pub mod retry;
pub mod sse;
use crate::wasm_compat::*;
pub use multipart::MultipartForm;
pub use reqwest::Client as ReqwestClient;
use std::pin::Pin;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Http error: {0}")]
    Protocol(#[from] http::Error),
    #[error("Invalid status code: {0}")]
    InvalidStatusCode(StatusCode),
    #[error("Invalid status code {0} with message: {1}")]
    InvalidStatusCodeWithMessage(StatusCode, String),
    #[error("Header value outside of legal range: {0}")]
    InvalidHeaderValue(#[from] http::header::InvalidHeaderValue),
    #[error("Request in error state, cannot access headers")]
    NoHeaders,
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

impl Error {
    pub(crate) fn non_success_status(&self) -> Option<StatusCode> {
        match self {
            Self::InvalidStatusCode(status) | Self::InvalidStatusCodeWithMessage(status, _) => {
                Some(*status)
            }
            _ => None,
        }
    }

    pub(crate) fn non_success_body(&self) -> Option<&str> {
        match self {
            Self::InvalidStatusCodeWithMessage(_, body) => Some(body.as_str()),
            _ => None,
        }
    }
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

async fn non_success_status_error(response: reqwest::Response) -> Error {
    let status = response.status();
    let message = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
    Error::InvalidStatusCodeWithMessage(status, message)
}

pub type StreamingResponse = Response<BoxedStream>;

#[derive(Debug, Clone, Copy)]
pub struct NoBody;

impl From<NoBody> for Bytes {
    fn from(_: NoBody) -> Self {
        Bytes::new()
    }
}

pub fn make_auth_header(key: impl AsRef<str>) -> Result<(HeaderName, HeaderValue)> {
    Ok((
        http::header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", key.as_ref()))?,
    ))
}

pub fn bearer_auth_header(headers: &mut HeaderMap, key: impl AsRef<str>) -> Result<()> {
    let (k, v) = make_auth_header(key)?;

    headers.insert(k, v);

    Ok(())
}

pub fn with_bearer_auth(mut req: Builder, auth: &str) -> Result<Builder> {
    bearer_auth_header(req.headers_mut().ok_or(Error::NoHeaders)?, auth)?;

    Ok(req)
}

/// The concrete HTTP backends [`Transport`](crate::http_runtime::Transport)
/// dispatches to.
///
/// This is deliberately *not* a public extension point: it exists so
/// [`HttpRuntime`](crate::http_runtime::HttpRuntime) can hold `reqwest` and the
/// test transports behind one enum. It replaced the public, body-generic
/// `HttpClientExt` trait that used to be threaded through every provider as an
/// `H` type parameter — providers now take an `HttpRuntime`, so no HTTP
/// genericity reaches them. Bodies are `Vec<u8>` in and [`Bytes`] out; the only
/// remaining type-erased edge is the streaming one, where the response body is
/// a boxed byte stream.
pub(crate) trait Backend: WasmCompatSend + WasmCompatSync {
    /// Send a request and read the full response body.
    ///
    /// Non-success statuses are reported as
    /// [`Error::InvalidStatusCodeWithMessage`] with the body preserved, so
    /// callers that want them as values can recover them.
    fn send(
        &self,
        req: Request<Vec<u8>>,
    ) -> impl Future<Output = Result<Response<Bytes>>> + WasmCompatSend + 'static;

    /// Send a multipart request and read the full response body. Same
    /// status-code contract as [`send`](Self::send).
    fn send_multipart(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<Bytes>>> + WasmCompatSend + 'static;

    /// Send a request and return the response as a stream of byte chunks.
    fn send_streaming(
        &self,
        req: Request<Vec<u8>>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend;
}

async fn into_response(response: reqwest::Response) -> Result<Response<Bytes>> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    let mut res = Response::builder().status(response.status());

    if let Some(headers) = res.headers_mut() {
        *headers = response.headers().clone();
    }

    let bytes = response.bytes().await.map_err(instance_error)?;

    res.body(bytes).map_err(Error::Protocol)
}

macro_rules! impl_http_backend {
    ($(#[$attribute:meta])* $client:ty) => {
        $(#[$attribute])*
        impl Backend for $client {
            fn send(
                &self,
                req: Request<Vec<u8>>,
            ) -> impl Future<Output = Result<Response<Bytes>>> + WasmCompatSend + 'static {
                let (parts, body) = req.into_parts();
                let req = self
                    .request(parts.method, parts.uri.to_string())
                    .headers(parts.headers)
                    .body(body);

                async move {
                    let response = req.send().await.map_err(instance_error)?;
                    into_response(response).await
                }
            }

            fn send_multipart(
                &self,
                req: Request<MultipartForm>,
            ) -> impl Future<Output = Result<Response<Bytes>>> + WasmCompatSend + 'static {
                let (parts, body) = req.into_parts();
                let body = reqwest::multipart::Form::from(body);

                let req = self
                    .request(parts.method, parts.uri.to_string())
                    .headers(parts.headers)
                    .multipart(body);

                async move {
                    let response = req.send().await.map_err(instance_error)?;
                    into_response(response).await
                }
            }

            fn send_streaming(
                &self,
                req: Request<Vec<u8>>,
            ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend {
                let (parts, body) = req.into_parts();

                let client = self.clone();

                async move {
                    let req = self
                        .request(parts.method, parts.uri.to_string())
                        .headers(parts.headers)
                        .body(body)
                        .build()
                        .map_err(|error| Error::Instance(error.into()))?;
                    let response: reqwest::Response =
                        client.execute(req).await.map_err(instance_error)?;
                    if !response.status().is_success() {
                        return Err(non_success_status_error(response).await);
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

                    let mapped_stream: Pin<
                        Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>,
                    > = Box::pin(
                        response
                            .bytes_stream()
                            .map(|chunk| chunk.map_err(|e| Error::Instance(Box::new(e)))),
                    );

                    res.body(mapped_stream).map_err(Error::Protocol)
                }
            }
        }
    };
}

impl_http_backend!(reqwest::Client);

impl_http_backend!(
    #[cfg(feature = "reqwest-middleware")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
    reqwest_middleware::ClientWithMiddleware
);
