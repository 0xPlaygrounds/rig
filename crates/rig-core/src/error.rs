//! The provider error type, serializable error reports, and shared retry
//! classifications for the effect protocol. Reports preserve classifications,
//! available provider metadata, and textual source chains.
//!
//! ```
//! use rig_core::error::{ErrorKind, ProviderError};
//!
//! let error = ProviderError::from_http_response(http::StatusCode::TOO_MANY_REQUESTS, "slow down");
//! assert!(error.is_retryable());
//! assert_eq!(error.report().kind, ErrorKind::ProviderResponse);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    http_client,
    memory::MemoryError,
    observe::AdapterErrorBoundary,
    provider_response::ProviderResponseError,
    tool::{ToolErrorKind, ToolExecutionError},
    vector_store::VectorStoreError,
};

/// Normalized classification of an [`ErrorReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// A transport failure that produced no provider reply: a reset
    /// connection, a timeout, a protocol error, an unreadable response. It
    /// never carries a status; a reply the server made, whatever its status,
    /// is [`Self::ProviderResponse`].
    Http,
    /// JSON serialization or deserialization failed.
    Json,
    /// A URL could not be parsed.
    Url,
    /// The request could not be built.
    Request,
    /// The response could not be parsed.
    Response,
    /// The provider reported a failure without a preserved raw response.
    Provider,
    /// The provider replied and its raw response was preserved: a non-2xx
    /// status with a body, a 2xx error envelope, or a non-HTTP transport's
    /// error payload. Status, body, request id and headers are on the report.
    ProviderResponse,
    /// A tool failed; the inner kind is the tool's own classification.
    Tool(ToolErrorKind),
    /// A conversation-memory backend failed.
    MemoryBackend,
    /// A conversation-memory policy or filter rejected the history.
    MemoryPolicy,
    /// An internal invariant was violated.
    Internal,
    /// The operation was cancelled.
    Cancelled,
    /// The operation exceeded its deadline.
    Timeout,
    /// The effect bus's driver is gone: the owner dropped it, so nothing can
    /// serve a dispatch. A lifecycle event; never retryable on the same bus.
    BusClosed,
    /// No handler serves the requested key and effect family on the live bus.
    /// The message identifies the key.
    HandlerUnavailable,
    /// Replay requested a different effect or exhausted the log.
    /// The run fails without retry rather than inventing a recorded answer.
    Divergence,
    /// Policy denied dispatch before handler execution. Not retryable under
    /// the same policy; distinct from program cancellation.
    Denied,
    /// Anything else.
    Other,
}

impl ErrorKind {
    /// Stable snake-case classification code, excluding variant payloads.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Json => "json",
            Self::Url => "url",
            Self::Request => "request",
            Self::Response => "response",
            Self::Provider => "provider",
            Self::ProviderResponse => "provider_response",
            Self::Tool(_) => "tool",
            Self::MemoryBackend => "memory_backend",
            Self::MemoryPolicy => "memory_policy",
            Self::Internal => "internal",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
            Self::BusClosed => "bus_closed",
            Self::HandlerUnavailable => "handler_unavailable",
            Self::Divergence => "divergence",
            Self::Denied => "denied",
            Self::Other => "other",
        }
    }
}

/// A serde-able error crossing a wire boundary.
///
/// Field semantics:
/// - `retryable` is the one policy signal; it is decided at conversion time
///   from the source's own classification (see [`retryable_status`]).
/// - `message` is the source's `Display`; `source_chain` is the `Display` of
///   each `source()` link, outermost first, excluding `message` itself.
/// - `code`, `http_status`, `refusal` are copied when the source had them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Normalized classification.
    pub kind: ErrorKind,
    /// Whether the same operation may reasonably be retried.
    pub retryable: bool,
    /// Human-readable description (the source's `Display`).
    pub message: String,
    /// A provider- or tool-specific machine code, when one was reported:
    /// for a provider's reply, the transport's own code when it gave one
    /// apart from the body, else the code the body names
    /// (`provider_response::body_code`).
    pub code: Option<String>,
    /// The HTTP status, when the failure had one.
    pub http_status: Option<u16>,
    /// The failure was an intentional refusal rather than a fault.
    pub refusal: bool,
    /// `Display` of each `source()` link, outermost first.
    pub source_chain: Vec<String>,
    /// The provider's request id, when the failure had a response that
    /// carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Preserved provider failure response, including available status, body,
    /// headers, and request ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_response: Option<crate::provider_response::ProviderResponseError>,
    /// Structured diagnostic for failures a consumer may want to *route*
    /// rather than only display. Absent for the common case; see
    /// [`ErrorDetail`] for what each variant carries and why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ErrorDetail>,
}

/// Optional structured recovery diagnostic supplementing a report's kind and message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "detail", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorDetail {
    /// A tool block declared complete contained invalid JSON, rather than
    /// truncated input. Carries exact accumulated arguments and the parser
    /// error for recovery or replay.
    MalformedToolInput(MalformedToolInput),
}

/// The payload of [`ErrorDetail::MalformedToolInput`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MalformedToolInput {
    /// The tool the model named.
    pub name: String,
    /// Durable correlation ID retained from the call for recovery actions,
    /// including tool results and rollback.
    pub id: crate::message::ToolCallId,
    /// The provider's own call id(s), when the wire supplied any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<crate::message::ProviderCallId>,
    /// The raw argument text, byte-for-byte as accumulated.
    pub raw: String,
    /// The JSON parser's description of what was wrong.
    pub error: String,
}

impl ErrorReport {
    /// Build a report of `kind` with `message` and no other metadata.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            retryable: false,
            message: message.into(),
            code: None,
            http_status: None,
            refusal: false,
            source_chain: Vec::new(),
            request_id: None,
            provider_response: None,
            detail: None,
        }
    }

    /// Set `retryable`.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    /// Set the machine code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Set the HTTP status.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    /// Mark the report as an intentional refusal.
    pub fn refused(mut self) -> Self {
        self.refusal = true;
        self
    }

    /// Attach the provider's request id.
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Attach a typed diagnostic.
    pub fn with_detail(mut self, detail: ErrorDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Whether the same operation may reasonably be retried.
    pub const fn is_retryable(&self) -> bool {
        self.retryable
    }
}

impl ErrorReport {
    /// The provider response body preserved on this report, if any.
    pub fn provider_response_body(&self) -> Option<&str> {
        self.provider_response
            .as_ref()
            .map(|response| response.body.as_str())
    }

    /// The preserved provider response body parsed as JSON, when present.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        crate::provider_response::json(self.provider_response_body())
    }

    /// The preserved provider response headers, if any.
    pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
        self.provider_response
            .as_ref()
            .and_then(|response| response.headers.as_ref())
    }

    /// The HTTP status this report carries, as a status code.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        self.http_status
            .and_then(|status| http::StatusCode::from_u16(status).ok())
    }

    /// The provider's transport request id, if the failure carried one.
    pub fn provider_request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ErrorReport {}

/// The one status → retryable table.
///
/// Request Timeout (408), Too Early (425), Too Many Requests (429) and every
/// server error (5xx) are retryable; every other status is not. A missing
/// status decides nothing here: a failure with no status is classified by
/// what it is ([`transient_transport`]), not by what it lacks.
pub const fn retryable_status(status: Option<u16>) -> bool {
    match status {
        None => false,
        Some(408 | 425 | 429) => true,
        Some(s) => s >= 500 && s <= 599,
    }
}

/// Classifies `StreamEnded` and backend `Instance` errors as retryable.
/// Protocol, header, and content-type errors are not retryable. Errors carrying
/// an HTTP status use [`retryable_status`]. This does not guarantee the server
/// has not processed the request.
pub fn transient_transport(error: &http_client::Error) -> bool {
    match error {
        http_client::Error::StreamEnded | http_client::Error::Instance(_) => true,
        http_client::Error::Protocol(_)
        | http_client::Error::InvalidHeaderValue(_)
        | http_client::Error::NoHeaders
        | http_client::Error::InvalidContentType(_) => false,
        http_client::Error::InvalidStatusCodeWithDetails { status, .. } => {
            retryable_status(Some(status.as_u16()))
        }
    }
}

/// Collect the `Display` of each `source()` link below `error`, outermost
/// first.
pub(crate) fn source_chain(error: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = error.source();
    while let Some(source) = current {
        chain.push(source.to_string());
        current = source.source();
    }
    chain
}

/// A boxed request-building failure.
#[cfg(not(target_family = "wasm"))]
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// A boxed request-building failure.
#[cfg(target_family = "wasm")]
pub type BoxError = Box<dyn std::error::Error + 'static>;

/// A failed provider operation: completion, embedding, reranking,
/// transcription, image or audio generation, verification, model listing, or
/// context caching.
///
/// The variant is the classification, and each variant maps to one
/// [`ErrorKind`]. A provider's reply is preserved as a
/// [`ProviderResponseError`] with its status, body, headers, and request ID;
/// read it through [`Self::provider_response`] or the accessors below.
///
/// ```
/// use rig_core::error::ProviderError;
///
/// fn log(error: &ProviderError) {
///     if let Some(status) = error.provider_response_status() {
///         // Error envelopes can arrive with successful HTTP statuses.
///         eprintln!("provider returned HTTP {status}");
///     }
///     match error.provider_response_json() {
///         Ok(Some(json)) => eprintln!("provider error payload: {json}"),
///         Ok(None) => eprintln!("no provider response body: {error}"),
///         Err(_) => eprintln!("non-JSON body: {:?}", error.provider_response_body()),
///     }
/// }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// A transport failure that produced no provider reply: a reset
    /// connection, a timeout, an unreadable response. A reply the server
    /// made is [`Self::ProviderResponse`].
    #[error("HttpError: {0}")]
    Http(http_client::Error),
    /// JSON serialization or deserialization failed.
    #[error("JsonError: {0}")]
    Json(#[from] serde_json::Error),
    /// A URL could not be parsed.
    #[error("UrlError: {0}")]
    Url(#[from] url::ParseError),
    /// The request could not be built.
    #[error("RequestError: {0}")]
    Request(#[from] BoxError),
    /// The reply decoded but does not answer the request.
    #[error("ResponseError: {0}")]
    Response(String),
    /// The provider reported a failure without a preserved reply.
    #[error("ProviderError: {0}")]
    Provider(String),
    /// The provider's reply, preserved: a non-success status with its body, a
    /// 2xx error envelope, or a non-HTTP transport's error payload.
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(ProviderResponseError),
    /// The provider rejected the configured credentials with 401 or 403.
    #[error("invalid authentication: {0}")]
    InvalidAuthentication(ProviderResponseError),
    /// A request for an existing context-cache handle answered 403 or 404.
    /// The reply's body is the provider's explanation, which can name a
    /// cause other than expiry, such as a credential or quota failure.
    #[error("cached content `{name}` is expired or was deleted: {}", response.body)]
    CacheExpired {
        /// The cache handle the request named.
        name: String,
        /// The provider's reply.
        response: ProviderResponseError,
    },
    /// The provider returned vectors of a width other than the one the caller
    /// declared through
    /// [`embedding`](crate::driver::HasEmbedding::embedding)'s `ndims`
    /// argument. Raised only when the width was set explicitly.
    #[error(
        "{provider} embedding response returned {returned}-dimension vectors, but the model was \
         created with {requested} dimensions; this provider does not resize embeddings"
    )]
    MismatchedDimensions {
        /// Provider whose response disagreed with the declared width.
        provider: String,
        /// Width the caller declared.
        requested: usize,
        /// Width the provider actually returned.
        returned: usize,
    },
}

impl ProviderError {
    /// Preserves the status and verbatim body as [`Self::ProviderResponse`],
    /// including error envelopes returned with 2xx statuses.
    pub fn from_http_response(status: http::StatusCode, body: impl Into<String>) -> Self {
        Self::ProviderResponse(ProviderResponseError::new(status, body))
    }

    /// Preserves a verbatim provider error body with no HTTP status as
    /// [`Self::ProviderResponse`].
    pub fn from_provider_body(body: impl Into<String>) -> Self {
        Self::ProviderResponse(ProviderResponseError::without_status(body))
    }

    /// Converts a non-success reply the transport reported as an error to
    /// [`Self::ProviderResponse`], keeping its status, body, and headers.
    /// Other transport errors become [`Self::Http`].
    pub fn from_transport_error(error: http_client::Error) -> Self {
        match error {
            http_client::Error::InvalidStatusCodeWithDetails {
                status,
                body,
                headers,
            } => Self::from_http_response(status, body).with_response_headers(Some(headers)),
            other => Self::Http(other),
        }
    }

    /// The classification this error reports as.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Http(_) => ErrorKind::Http,
            Self::Json(_) => ErrorKind::Json,
            Self::Url(_) => ErrorKind::Url,
            Self::Request(_) => ErrorKind::Request,
            Self::Response(_) | Self::MismatchedDimensions { .. } => ErrorKind::Response,
            Self::Provider(_) => ErrorKind::Provider,
            Self::ProviderResponse(_)
            | Self::InvalidAuthentication(_)
            | Self::CacheExpired { .. } => ErrorKind::ProviderResponse,
        }
    }

    /// Classifies transport failures with [`transient_transport`] and
    /// preserved replies with [`ProviderResponseError::is_retryable`]. Every
    /// other failure, rejected credentials and expired caches included, is not
    /// retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => transient_transport(error),
            Self::ProviderResponse(response) => response.is_retryable(),
            _ => false,
        }
    }

    /// The provider's preserved reply, when this error carries one.
    pub fn provider_response(&self) -> Option<&ProviderResponseError> {
        match self {
            Self::ProviderResponse(response)
            | Self::InvalidAuthentication(response)
            | Self::CacheExpired { response, .. } => Some(response),
            _ => None,
        }
    }

    /// The preserved reply's body. An empty body returns `Some("")`, while
    /// [`Self::provider_response_json`] maps it to `Ok(None)`.
    pub fn provider_response_body(&self) -> Option<&str> {
        self.provider_response()
            .map(|response| response.body.as_str())
    }

    /// Parses the preserved reply's body as JSON: `Ok(None)` when there is no
    /// body or it is empty, `Err` when it is not valid JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        crate::provider_response::json(self.provider_response_body())
    }

    /// The preserved reply's HTTP status. It may be 2xx for an error envelope.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        self.provider_response()
            .and_then(|response| response.status)
    }

    /// The provider's transport request ID, when the reply carried one.
    pub fn provider_request_id(&self) -> Option<&str> {
        self.provider_response()
            .and_then(|response| response.provider_request_id.as_deref())
    }

    /// The preserved reply's headers. `None` means not captured, as for
    /// non-HTTP transports and replies built from only a status and body.
    /// This example reads the seconds form of `Retry-After`:
    ///
    /// ```no_run
    /// # use rig_core::error::ProviderError;
    /// # use std::time::Duration;
    /// fn backoff(error: &ProviderError) -> Option<Duration> {
    ///     let seconds = error
    ///         .provider_response_headers()?
    ///         .get(http::header::RETRY_AFTER)?
    ///         .to_str()
    ///         .ok()?
    ///         .parse()
    ///         .ok()?;
    ///     Some(Duration::from_secs(seconds))
    /// }
    /// ```
    pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
        self.provider_response()
            .and_then(|response| response.headers.as_ref())
    }

    /// Fills an absent request ID on the preserved reply, ignoring empty
    /// strings.
    pub fn with_provider_request_id(self, request_id: Option<String>) -> Self {
        self.map_response(|response| match response.provider_request_id {
            Some(_) => response,
            None => response.with_provider_request_id(request_id),
        })
    }

    /// Fills absent headers on the preserved reply.
    pub fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self {
        self.map_response(|response| match (&response.headers, headers) {
            (None, Some(headers)) => response.with_headers(Some(headers)),
            _ => response,
        })
    }

    /// Attaches the HTTP status a transport reported beside a reply preserved
    /// without one, so it classifies by status. A captured status is kept.
    pub fn with_provider_status(self, status: Option<http::StatusCode>) -> Self {
        self.map_response(|response| response.with_status(status))
    }

    /// Attaches the provider's machine-readable code for the failure, such as
    /// a gRPC status name or an AWS exception type.
    pub fn with_provider_code(self, code: Option<String>) -> Self {
        self.map_response(|response| response.with_code(code))
    }

    /// Replaces the preserved reply's transport retry verdict, used when its
    /// status is absent or successful.
    pub fn with_transient(self, transient: Option<bool>) -> Self {
        self.map_response(|response| response.with_transient(transient))
    }

    /// The wire form of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }

    /// Which boundary produced this error, by its classification.
    pub(crate) fn boundary(&self) -> AdapterErrorBoundary {
        match (self, self.kind()) {
            (Self::Http(error), _) => AdapterErrorBoundary::from_http(error),
            (_, ErrorKind::Json | ErrorKind::Response) => AdapterErrorBoundary::Decode,
            (_, ErrorKind::Provider | ErrorKind::ProviderResponse) => {
                AdapterErrorBoundary::ProviderResponse
            }
            _ => AdapterErrorBoundary::Request,
        }
    }

    fn map_response(
        self,
        map: impl FnOnce(ProviderResponseError) -> ProviderResponseError,
    ) -> Self {
        match self {
            Self::ProviderResponse(response) => Self::ProviderResponse(map(response)),
            Self::InvalidAuthentication(response) => Self::InvalidAuthentication(map(response)),
            Self::CacheExpired { name, response } => Self::CacheExpired {
                name,
                response: map(response),
            },
            other => other,
        }
    }
}

impl From<http_client::Error> for ProviderError {
    fn from(error: http_client::Error) -> Self {
        Self::from_transport_error(error)
    }
}

/// A failure to build a provider request, returned by
/// [`Wire::encode`](crate::wire::Wire::encode). It converts only into
/// [`ProviderError::Request`], so every provider classifies its encode
/// failures alike.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct EncodeError(ProviderError);

impl EncodeError {
    /// A request that could not be built, for the given reason.
    pub fn request(reason: impl Into<BoxError>) -> Self {
        Self(ProviderError::Request(reason.into()))
    }
}

impl From<EncodeError> for ProviderError {
    fn from(error: EncodeError) -> Self {
        debug_assert_eq!(error.0.kind(), ErrorKind::Request);
        error.0
    }
}

impl From<http::Error> for EncodeError {
    fn from(error: http::Error) -> Self {
        Self::request(error)
    }
}

impl From<serde_json::Error> for EncodeError {
    fn from(error: serde_json::Error) -> Self {
        Self::request(error)
    }
}

impl From<crate::message::MessageError> for EncodeError {
    fn from(error: crate::message::MessageError) -> Self {
        Self::request(error)
    }
}

impl From<BoxError> for EncodeError {
    fn from(error: BoxError) -> Self {
        Self(ProviderError::Request(error))
    }
}

impl From<http::Error> for ProviderError {
    fn from(error: http::Error) -> Self {
        Self::Request(Box::new(error))
    }
}

/// A client that could not be built: transport-configuration failures keep
/// their HTTP identity, anything else (a missing key, an unreadable
/// environment variable) is reported as a provider error.
impl From<crate::client::ProviderClientError> for ProviderError {
    fn from(error: crate::client::ProviderClientError) -> Self {
        match error {
            crate::client::ProviderClientError::Http(error) => Self::Http(error),
            other => Self::Provider(other.to_string()),
        }
    }
}

impl From<&ProviderError> for ErrorReport {
    fn from(error: &ProviderError) -> Self {
        let response = error.provider_response();
        ErrorReport {
            kind: error.kind(),
            retryable: error.is_retryable(),
            message: error.to_string(),
            code: response.and_then(ProviderResponseError::machine_code),
            http_status: response
                .and_then(|response| response.status)
                .map(|status| status.as_u16()),
            refusal: response.is_some_and(|response| response.refusal),
            source_chain: source_chain(error),
            request_id: response.and_then(|response| response.provider_request_id.clone()),
            provider_response: response.cloned(),
            detail: None,
        }
    }
}

impl From<ProviderError> for ErrorReport {
    fn from(error: ProviderError) -> Self {
        Self::from(&error)
    }
}

impl ToolExecutionError {
    /// The wire form of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }
}

impl ToolExecutionError {
    /// Whether the tool may reasonably be re-run: the explicit override when
    /// one was set, else the kind's own default
    /// ([`ToolErrorKind::default_retryable`]); a kind that leaves it to the
    /// tool (`None`) is not retryable on the wire.
    pub fn is_retryable(&self) -> bool {
        self.retryable()
            .or_else(|| self.kind().default_retryable())
            .unwrap_or(false)
    }
}

impl From<&ToolExecutionError> for ErrorReport {
    fn from(error: &ToolExecutionError) -> Self {
        ErrorReport {
            kind: ErrorKind::Tool(error.kind()),
            retryable: error.is_retryable(),
            message: error.message().to_string(),
            code: error.code().map(str::to_string),
            http_status: error.http_status(),
            refusal: error.is_refusal(),
            source_chain: source_chain(error),
            request_id: None,
            provider_response: None,
            detail: None,
        }
    }
}

impl From<ToolExecutionError> for ErrorReport {
    fn from(error: ToolExecutionError) -> Self {
        Self::from(&error)
    }
}

impl MemoryError {
    /// The wire form of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }
}

impl From<&MemoryError> for ErrorReport {
    fn from(error: &MemoryError) -> Self {
        let kind = match error {
            MemoryError::Backend(_) => ErrorKind::MemoryBackend,
            MemoryError::Policy(_) => ErrorKind::MemoryPolicy,
            MemoryError::Internal(_) => ErrorKind::Internal,
        };
        ErrorReport {
            kind,
            retryable: false,
            message: error.to_string(),
            code: None,
            http_status: None,
            refusal: false,
            source_chain: source_chain(error),
            request_id: None,
            provider_response: None,
            detail: None,
        }
    }
}

impl From<MemoryError> for ErrorReport {
    fn from(error: MemoryError) -> Self {
        Self::from(&error)
    }
}

impl From<&VectorStoreError> for ErrorReport {
    fn from(error: &VectorStoreError) -> Self {
        let (kind, provider_response) = match error {
            VectorStoreError::EmbeddingError(inner) => {
                return Self {
                    message: error.to_string(),
                    source_chain: source_chain(error),
                    ..Self::from(inner)
                };
            }
            VectorStoreError::JsonError(_) => (ErrorKind::Json, None),
            VectorStoreError::DatastoreError(_) => (ErrorKind::Provider, None),
            VectorStoreError::FilterError(_) | VectorStoreError::BuilderError(_) => {
                (ErrorKind::Request, None)
            }
            VectorStoreError::MissingIdError(_) => (ErrorKind::Response, None),
            VectorStoreError::Http(crate::http_client::Error::InvalidStatusCodeWithDetails {
                status,
                body,
                headers,
            }) => (
                ErrorKind::ProviderResponse,
                Some(
                    crate::provider_response::ProviderResponseError::new(*status, body.clone())
                        .with_headers(Some(headers.clone())),
                ),
            ),
            VectorStoreError::Http(_) => (ErrorKind::Http, None),
            // A store's own non-2xx reply: the server answered, so the status
            // classifies it and its retryability, like any provider reply.
            VectorStoreError::ExternalAPIError(status, body) => (
                ErrorKind::ProviderResponse,
                Some(crate::provider_response::ProviderResponseError::new(
                    *status,
                    body.clone(),
                )),
            ),
        };
        let http_status = provider_response
            .as_ref()
            .and_then(|response| response.status.map(|status| status.as_u16()));
        let retryable = match error {
            VectorStoreError::Http(inner) => transient_transport(inner),
            VectorStoreError::ExternalAPIError(..) => retryable_status(http_status),
            _ => false,
        };
        let code = provider_response
            .as_ref()
            .and_then(|response| response.machine_code());
        ErrorReport {
            kind,
            retryable,
            message: error.to_string(),
            code,
            http_status,
            refusal: provider_response.as_ref().is_some_and(|r| r.refusal),
            source_chain: source_chain(error),
            request_id: None,
            provider_response,
            detail: None,
        }
    }
}

impl From<VectorStoreError> for ErrorReport {
    fn from(error: VectorStoreError) -> Self {
        Self::from(&error)
    }
}

// The report is the wire error of the effect protocol: it must cross threads
// and serialize on every target, browser wasm included.
const _: fn() = || {
    fn assert_wire<T: Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<ErrorReport>();
    assert_wire::<ErrorKind>();
};

#[cfg(test)]
mod encode_tests;
#[cfg(test)]
mod tests;
