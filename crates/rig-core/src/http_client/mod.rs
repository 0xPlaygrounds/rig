use crate::http_client::sse::BoxedStream;
use bytes::Bytes;
use http::HeaderName;
pub use http::{
    HeaderMap, HeaderValue, Method, Request, Response, StatusCode, Uri, request::Builder,
};
mod erased;
pub mod middleware;
pub mod multipart;
pub mod retry;
#[cfg(all(feature = "reqwest", not(target_family = "wasm")))]
mod runtime;
pub mod sse;

#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
mod reqwest_transport;

/// The bundled `reqwest` transport, re-exported so callers need not depend on
/// it separately.
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub use ::reqwest;
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub use reqwest_transport::{from_reqwest, multipart_form};

/// The transport a provider type falls back to when its `H` is not named.
///
/// With the `reqwest` feature this is the bundled transport, which is what
/// makes `openai::CompletionModel` mean `…<reqwest::Client>` in type position.
/// Without it there is no transport to default to, so the slot resolves to the
/// same [`Missing`](crate::markers::Missing) marker the client builder uses for
/// an unfilled transport — every constructor still takes an explicit
/// `H: HttpClientExt`.
#[cfg(feature = "reqwest")]
pub type DefaultHttp = ::reqwest::Client;
/// See [`DefaultHttp`].
#[cfg(not(feature = "reqwest"))]
pub type DefaultHttp = crate::markers::Missing;
use crate::wasm_compat::*;
pub use erased::BoxedHttpClient;
pub use middleware::HttpMiddleware;
pub use multipart::MultipartForm;

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

    /// Build the headers-preserving non-success error from a failed
    /// response's parts. Transports call this with the status, headers and
    /// body they read off the wire, so provider layers can recover transport
    /// metadata — request ids, rate-limit headers — from the error (rig#2314).
    pub fn non_success_with_details(status: StatusCode, headers: HeaderMap, body: String) -> Self {
        Self::InvalidStatusCodeWithDetails {
            status,
            body,
            headers: Box::new(headers),
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

impl Error {
    /// Wrap a transport's native error as [`Error::Instance`]. Transports use
    /// this for response-less failures (connect, decode, timeout); non-success
    /// responses go through [`Error::non_success_with_details`] instead so the
    /// status stays inspectable.
    #[cfg(not(target_family = "wasm"))]
    pub fn instance<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        Self::Instance(error.into())
    }

    /// Wrap a transport's native error as [`Error::Instance`].
    #[cfg(target_family = "wasm")]
    pub fn instance<E: std::error::Error + 'static>(error: E) -> Self {
        Self::Instance(error.into())
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

    /// Erase this transport behind [`BoxedHttpClient`], for hosts that hold one
    /// transport for many providers without naming it in their types.
    ///
    /// Not to be confused with `Client::boxed`, which erases a built provider
    /// client's transport and returns a `Client`. A `Client` also implements
    /// this trait, so its inherent method wins at a call site — but a generic
    /// `fn erase<H: HttpClientExt>(h: H)` handed a `Client` reaches *this* one
    /// and boxes the whole client as a transport, which is almost never what
    /// the caller meant.
    fn boxed(self) -> BoxedHttpClient
    where
        Self: Sized + 'static,
    {
        BoxedHttpClient::new(self)
    }
}

#[cfg(test)]
mod non_success_header_tests {
    use super::*;

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
