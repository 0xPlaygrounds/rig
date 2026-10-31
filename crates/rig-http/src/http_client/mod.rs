//! Transport-independent HTTP requests, lazy response bodies, and errors.
//!
//! ```
//! use rig_http::http_client::{NoBody, Request};
//!
//! let request = Request::get("https://example.com/v1/models").body(NoBody)?;
//! assert_eq!(request.uri().path(), "/v1/models");
//! # Ok::<(), rig_http::http_client::Error>(())
//! ```

pub use bytes::Bytes;
pub use http::{
    HeaderMap, HeaderValue, Method, Request, Response, StatusCode, Uri, request::Builder,
};
mod erased;
pub mod framing;
pub mod middleware;
pub mod multipart;
use crate::wasm_compat::*;
pub use erased::DynHttpClient;
pub use middleware::{Exchange, HttpMiddleware};
pub use multipart::MultipartForm;

/// A boxed error: `Send + Sync` except on wasm.
#[cfg(not(target_family = "wasm"))]
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// A boxed error: `Send + Sync` except on wasm.
#[cfg(target_family = "wasm")]
pub type BoxError = Box<dyn std::error::Error + 'static>;

/// Why an HTTP request failed: a protocol or header error, a non-success
/// reply, or the transport's own error, which says whether sending the same
/// request again may work.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The request could not be built.
    #[error("Http error: {0}")]
    Protocol(#[from] http::Error),
    /// Failed HTTP response preserving its status, body, and headers.
    /// Transport failures without a response use [`Self::Transient`],
    /// [`Self::Permanent`] or a protocol variant.
    #[error("Invalid status code {status} with message: {body}")]
    InvalidStatusCodeWithDetails {
        /// The non-success status.
        status: StatusCode,
        /// The raw response body.
        body: String,
        /// The failed response's headers, verbatim.
        headers: http::HeaderMap,
    },
    /// A header value with characters a header cannot carry.
    #[error("Header value outside of legal range: {0}")]
    InvalidHeaderValue(#[from] http::header::InvalidHeaderValue),
    /// The request builder failed earlier, so its headers cannot be read.
    #[error("Request in error state, cannot access headers")]
    NoHeaders,
    /// The response stream ended before it was done.
    #[error("Stream ended")]
    StreamEnded,
    /// The response's content type was not the one expected.
    #[error("Invalid content type was returned: {0:?}")]
    InvalidContentType(HeaderValue),
    /// The transport failed without a reply, and sending the same request
    /// again may work: a refused or reset connection, a timeout, a body cut
    /// short.
    #[error("Http client error: {0}")]
    Transient(#[source] BoxError),
    /// The transport failed without a reply, and sending the same request
    /// again fails the same way: a request it cannot send as it is, missing
    /// credentials, a client that could not be built.
    #[error("Http client error: {0}")]
    Permanent(#[source] BoxError),
}

impl Error {
    /// Preserved HTTP failure status, or `None` for other error variants.
    pub fn non_success_status(&self) -> Option<StatusCode> {
        match self {
            Self::InvalidStatusCodeWithDetails { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// The response body this error preserved, when it has one. Companion to
    /// [`Self::non_success_status`] and [`Self::non_success_headers`].
    pub fn non_success_body(&self) -> Option<&str> {
        match self {
            Self::InvalidStatusCodeWithDetails { body, .. } => Some(body.as_str()),
            _ => None,
        }
    }

    /// Constructs a failure from its HTTP status, headers, and body without
    /// validating that the status is non-successful.
    pub fn non_success_with_details(status: StatusCode, headers: HeaderMap, body: String) -> Self {
        Self::InvalidStatusCodeWithDetails {
            status,
            body,
            headers,
        }
    }

    /// Returns the failed response's headers, when this error preserved them.
    ///
    /// The following example reads the seconds form of `Retry-After`:
    ///
    /// ```
    /// # use rig_http::http_client::Error;
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
    /// Returns `None` for a response-less failure: every non-success error
    /// carries its headers.
    pub fn non_success_headers(&self) -> Option<&HeaderMap> {
        match self {
            Self::InvalidStatusCodeWithDetails { headers, .. } => Some(headers),
            _ => None,
        }
    }
}

/// A result with an HTTP [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Whether sending the same request again may pass: after `StreamEnded`
    /// and what the transport called [`Error::Transient`] it may, a status
    /// decides through [`retryable_status`], and after anything else
    /// (protocol, header and content-type errors, [`Error::Permanent`]) it
    /// does not. A retry may repeat a request the server already processed.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::StreamEnded | Self::Transient(_) => true,
            Self::InvalidStatusCodeWithDetails { status, .. } => {
                retryable_status(Some(status.as_u16()))
            }
            _ => false,
        }
    }

    /// A transport's own error after which sending the same request again
    /// may work, such as a reset connection or a timeout. A non-success
    /// reply goes through [`Error::non_success_with_details`] instead, so its
    /// status decides.
    pub fn transient(error: impl Into<BoxError>) -> Self {
        Self::Transient(error.into())
    }

    /// A transport's own error that sending the same request again meets
    /// again, such as a request it cannot send or a client it could not
    /// build.
    pub fn permanent(error: impl Into<BoxError>) -> Self {
        Self::Permanent(error.into())
    }
}

/// The one status → retryable table.
///
/// Request Timeout (408), Too Early (425), Too Many Requests (429) and every
/// server error (5xx) are retryable; every other status is not. A missing
/// status decides nothing here: a failure with no status is classified by
/// what its transport called it ([`Error::Transient`] or
/// [`Error::Permanent`]), not by what it lacks.
pub const fn retryable_status(status: Option<u16>) -> bool {
    match status {
        None => false,
        Some(408 | 425 | 429) => true,
        Some(s) => s >= 500 && s <= 599,
    }
}

/// A response body that is read when awaited.
pub type LazyBody<T> = WasmBoxedFuture<'static, Result<T>>;

/// The body of a streaming response: the transport's own chunks, boxed.
pub type BoxedStream = std::pin::Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>>;

/// A response whose body streams.
pub type StreamingResponse = Response<BoxedStream>;

/// A request with an empty body.
#[derive(Debug, Clone, Copy)]
pub struct NoBody;

impl From<NoBody> for Bytes {
    fn from(_: NoBody) -> Self {
        Bytes::new()
    }
}

/// Reads `response`'s body as text, replacing invalid UTF-8.
pub async fn text(response: Response<LazyBody<Vec<u8>>>) -> Result<String> {
    let text = response.into_body().await?;
    Ok(String::from(String::from_utf8_lossy(&text)))
}

/// An HTTP client Rig sends through: unary, multipart and streaming requests.
///
/// `rig-reqwest`'s `ReqwestClient` is the bundled implementation; a provider
/// configuration takes any implementation with `connect(client)`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an HTTP client Rig can send through",
    label = "not an `HttpClientExt`",
    note = "use `rig_reqwest::ReqwestClient` (wrap a configured `reqwest::Client` with `ReqwestClient::from(client)`), erase a client with `DynHttpClient::new(client)`, or implement `HttpClientExt`; a provider configuration takes it with `.connect(client)`"
)]
pub trait HttpClientExt: WasmCompatSend + WasmCompatSync {
    /// Sends a request and returns headers with a lazy body converted from bytes to `U`.
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Sends a multipart request and returns headers with a lazy body converted to `U`.
    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Sends a request and returns headers with a stream of transport byte chunks.
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend;
}

#[cfg(test)]
mod non_success_header_tests;
