use crate::http_client::sse::BoxedStream;
use bytes::Bytes;
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use http::{HeaderName, StatusCode};
use reqwest::Body;
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
    /// A non-success HTTP response whose headers were preserved alongside the
    /// body, so provider layers can read transport metadata — e.g. their
    /// request-id contract — off the failed response (rig#2314). Displays
    /// identically to [`Self::InvalidStatusCodeWithMessage`].
    #[error("Invalid status code {status} with message: {body}")]
    InvalidStatusCodeWithDetails {
        /// The non-success status.
        status: StatusCode,
        /// The raw response body.
        body: String,
        /// The failed response's headers, verbatim.
        headers: Box<http::HeaderMap>,
    },
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
            Self::InvalidStatusCodeWithDetails { status, .. } => Some(*status),
            _ => None,
        }
    }

    pub(crate) fn non_success_body(&self) -> Option<&str> {
        match self {
            Self::InvalidStatusCodeWithMessage(_, body) => Some(body.as_str()),
            Self::InvalidStatusCodeWithDetails { body, .. } => Some(body.as_str()),
            _ => None,
        }
    }

    /// Returns the failed response's headers, when this error preserved them.
    ///
    /// Rig's bundled HTTP clients capture the full [`HeaderMap`] whenever a
    /// non-success status error is built from a live response, so rate-limit
    /// metadata such as `Retry-After` or `x-ratelimit-*` stays readable
    /// (rig#2210). This is the accessor a [`retry::RetryPolicy`] uses to honor
    /// a server-supplied backoff, since it is handed this error directly:
    ///
    /// ```
    /// # use rig_core::http_client::{Error, retry::RetryPolicy};
    /// # use std::time::Duration;
    /// fn retry_after(error: &Error) -> Option<Duration> {
    ///     let seconds = error
    ///         .non_success_headers()?
    ///         .get(http::header::RETRY_AFTER)?
    ///         .to_str()
    ///         .ok()?
    ///         .parse()
    ///         .ok()?;
    ///     Some(Duration::from_secs(seconds))
    /// }
    /// ```
    ///
    /// Returns `None` when the error carries no captured headers: transports
    /// that report a non-success status without them, and errors built from
    /// only a status and body.
    pub fn non_success_headers(&self) -> Option<&HeaderMap> {
        match self {
            Self::InvalidStatusCodeWithDetails { headers, .. } => Some(headers),
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
    // Preserve the failed response's headers: provider layers read their
    // request-id contract off them (rig#2314). The Display is identical to
    // the header-less variant, so surfaced error text is unchanged.
    let headers = Box::new(response.headers().clone());
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
    Error::InvalidStatusCodeWithDetails {
        status,
        body,
        headers,
    }
}

pub type LazyBytes = WasmBoxedFuture<'static, Result<Bytes>>;
pub type LazyBody<T> = WasmBoxedFuture<'static, Result<T>>;

pub type StreamingResponse = Response<BoxedStream>;

#[derive(Debug, Clone, Copy)]
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
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request, get a streamed response back (as a stream of [`bytes::Bytes`].)
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend;
}

async fn into_lazy_response<U>(response: reqwest::Response) -> Result<Response<LazyBody<U>>>
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

    let body: LazyBody<U> = Box::pin(async {
        let bytes = response.bytes().await.map_err(instance_error)?;
        Ok(U::from(bytes))
    });

    res.body(body).map_err(Error::Protocol)
}

macro_rules! impl_http_client_ext {
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
                let (parts, body) = req.into_parts();
                let req = self
                    .request(parts.method, parts.uri.to_string())
                    .headers(parts.headers)
                    .body(body.into());

                async move {
                    let response = req.send().await.map_err(instance_error)?;
                    into_lazy_response(response).await
                }
            }

            fn send_multipart<U>(
                &self,
                req: Request<MultipartForm>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                U: From<Bytes>,
                U: WasmCompatSend + 'static,
            {
                let (parts, body) = req.into_parts();
                let body = reqwest::multipart::Form::from(body);

                let req = self
                    .request(parts.method, parts.uri.to_string())
                    .headers(parts.headers)
                    .multipart(body);

                async move {
                    let response = req.send().await.map_err(instance_error)?;
                    into_lazy_response(response).await
                }
            }

            fn send_streaming<T>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
            where
                T: Into<Bytes> + WasmCompatSend,
            {
                let (parts, body) = req.into_parts();

                let client = self.clone();

                async move {
                    let req = self
                        .request(parts.method, parts.uri.to_string())
                        .headers(parts.headers)
                        .body(body.into())
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

impl_http_client_ext!(reqwest::Client);

impl_http_client_ext!(
    #[cfg(feature = "reqwest-middleware")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
    reqwest_middleware::ClientWithMiddleware
);

#[cfg(test)]
mod non_success_header_tests {
    use super::*;

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

        assert_eq!(
            error.non_success_status(),
            Some(StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(
            error.non_success_body(),
            Some(r#"{"error":{"message":"rate limited"}}"#)
        );
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

    /// `None` means "not captured" and must not be confused with an empty map:
    /// every other shape of this error reports it.
    #[test]
    fn non_success_headers_absent_when_not_captured() {
        for error in [
            Error::InvalidStatusCodeWithMessage(
                StatusCode::TOO_MANY_REQUESTS,
                "rate limited".to_string(),
            ),
            Error::InvalidStatusCode(StatusCode::TOO_MANY_REQUESTS),
            Error::StreamEnded,
        ] {
            assert!(error.non_success_headers().is_none());
        }

        // A captured-but-empty map is `Some`, not `None`.
        let error = Error::InvalidStatusCodeWithDetails {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: "rate limited".to_string(),
            headers: Box::new(HeaderMap::new()),
        };
        assert!(error.non_success_headers().is_some_and(HeaderMap::is_empty));
    }
}
