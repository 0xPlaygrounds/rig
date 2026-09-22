//! Serializable error reports and shared retry classifications for the effect protocol.
//! Reports preserve classifications, available provider metadata, and textual source chains.
//!
//! ```
//! use rig_core::error::{ErrorKind, ErrorReport};
//!
//! let report = ErrorReport::new(ErrorKind::Cancelled, "cancelled");
//! assert!(!report.is_retryable());
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    completion::CompletionError,
    embeddings::EmbeddingError,
    memory::MemoryError,
    rerank::RerankError,
    tool::{ToolErrorKind, ToolExecutionError},
    transcription::TranscriptionError,
    vector_store::VectorStoreError,
};

#[cfg(feature = "audio")]
use crate::audio_generation::AudioGenerationError;
#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationError;

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
pub fn transient_transport(error: &crate::http_client::Error) -> bool {
    match error {
        crate::http_client::Error::StreamEnded | crate::http_client::Error::Instance(_) => true,
        crate::http_client::Error::Protocol(_)
        | crate::http_client::Error::InvalidHeaderValue(_)
        | crate::http_client::Error::NoHeaders
        | crate::http_client::Error::InvalidContentType(_) => false,
        crate::http_client::Error::InvalidStatusCodeWithDetails { status, .. } => {
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

/// Implements report conversions for provider errors. Additional variants
/// default to request faults unless an explicit pattern and kind override them.
macro_rules! impl_report_for_provider_error {
    ($error:ident $(, $extra:pat => $extra_kind:expr)* $(,)?) => {
        impl From<&$error> for $crate::error::ErrorReport {
            fn from(error: &$error) -> Self {
                use $crate::error::ErrorKind;
                let (kind, provider_response) = match error {
                    $error::HttpError(_) => (ErrorKind::Http, None),
                    $error::JsonError(_) => (ErrorKind::Json, None),
                    $error::ResponseError(_) => (ErrorKind::Response, None),
                    $error::ProviderError(_) => (ErrorKind::Provider, None),
                    $error::ProviderResponse(response) => {
                        (ErrorKind::ProviderResponse, Some(response.clone()))
                    }
                    $( $extra => ($extra_kind, None), )*
                    _ => (ErrorKind::Request, None),
                };
                $crate::error::ErrorReport {
                    kind,
                    retryable: error.is_retryable(),
                    message: error.to_string(),
                    code: provider_response
                        .as_ref()
                        .and_then(|response| response.machine_code()),
                    http_status: provider_response
                        .as_ref()
                        .and_then(|response| response.status.map(|status| status.as_u16())),
                    refusal: provider_response
                        .as_ref()
                        .is_some_and(|response| response.refusal),
                    source_chain: $crate::error::source_chain(error),
                    request_id: provider_response
                        .as_ref()
                        .and_then(|response| response.provider_request_id.clone()),
                    provider_response,
                    detail: None,
                }
            }
        }

        impl From<$error> for $crate::error::ErrorReport {
            fn from(error: $error) -> Self {
                Self::from(&error)
            }
        }
    };
}

pub(crate) use impl_report_for_provider_error;

impl_report_for_provider_error!(TranscriptionError);
#[cfg(feature = "image")]
impl_report_for_provider_error!(ImageGenerationError);
#[cfg(feature = "audio")]
impl_report_for_provider_error!(AudioGenerationError);

impl CompletionError {
    /// The wire form of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }
}

impl From<&CompletionError> for ErrorReport {
    fn from(error: &CompletionError) -> Self {
        let (kind, http_status) = match error {
            CompletionError::HttpError(_) => (ErrorKind::Http, None),
            CompletionError::JsonError(_) => (ErrorKind::Json, None),
            CompletionError::UrlError(_) => (ErrorKind::Url, None),
            CompletionError::RequestError(_) => (ErrorKind::Request, None),
            CompletionError::ResponseError(_) => (ErrorKind::Response, None),
            CompletionError::ProviderError(_) => (ErrorKind::Provider, None),
            CompletionError::ProviderResponse(response) => {
                let status = response.status.map(|s| s.as_u16());
                (ErrorKind::ProviderResponse, status)
            }
        };
        let request_id = match error {
            CompletionError::ProviderResponse(response) => response.provider_request_id.clone(),
            CompletionError::HttpError(_)
            | CompletionError::JsonError(_)
            | CompletionError::UrlError(_)
            | CompletionError::RequestError(_)
            | CompletionError::ResponseError(_)
            | CompletionError::ProviderError(_) => None,
        };
        let provider_response = match error {
            CompletionError::ProviderResponse(response) => Some(response.clone()),
            CompletionError::HttpError(_)
            | CompletionError::JsonError(_)
            | CompletionError::UrlError(_)
            | CompletionError::RequestError(_)
            | CompletionError::ResponseError(_)
            | CompletionError::ProviderError(_) => None,
        };
        let code = match error {
            CompletionError::ProviderResponse(response) => response.machine_code(),
            _ => None,
        };
        let refusal = match error {
            CompletionError::ProviderResponse(response) => response.refusal,
            _ => false,
        };
        ErrorReport {
            kind,
            retryable: error.is_retryable(),
            message: error.to_string(),
            code,
            http_status,
            refusal,
            source_chain: source_chain(error),
            request_id,
            provider_response,
            detail: None,
        }
    }
}

impl From<CompletionError> for ErrorReport {
    fn from(error: CompletionError) -> Self {
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

impl From<&EmbeddingError> for ErrorReport {
    fn from(error: &EmbeddingError) -> Self {
        let (kind, http_status) = match error {
            EmbeddingError::HttpError(_) => (ErrorKind::Http, None),
            EmbeddingError::JsonError(_) => (ErrorKind::Json, None),
            EmbeddingError::UrlError(_) => (ErrorKind::Url, None),
            EmbeddingError::DocumentError(_) => (ErrorKind::Request, None),
            EmbeddingError::ResponseError(_) => (ErrorKind::Response, None),
            EmbeddingError::UnsupportedParameter { .. }
            | EmbeddingError::InvalidParameterValue { .. } => (ErrorKind::Request, None),
            EmbeddingError::UnsupportedResponseEncoding { .. }
            | EmbeddingError::MissingUsage { .. }
            | EmbeddingError::MismatchedDimensions { .. } => (ErrorKind::Response, None),
            EmbeddingError::ProviderError(_) => (ErrorKind::Provider, None),
            EmbeddingError::ProviderResponse(response) => {
                let status = response.status.map(|s| s.as_u16());
                (ErrorKind::ProviderResponse, status)
            }
        };
        // One rule per error type: the retry verdict is the error's own,
        // never a copy of its table kept beside the report.
        let retryable = error.is_retryable();
        let provider_response = match error {
            EmbeddingError::ProviderResponse(response) => Some(response.clone()),
            _ => None,
        };
        let request_id = provider_response
            .as_ref()
            .and_then(|response| response.provider_request_id.clone());
        let code = provider_response
            .as_ref()
            .and_then(|response| response.machine_code());
        let refusal = provider_response
            .as_ref()
            .is_some_and(|response| response.refusal);
        ErrorReport {
            kind,
            retryable,
            message: error.to_string(),
            code,
            http_status,
            refusal,
            source_chain: source_chain(error),
            request_id,
            provider_response,
            detail: None,
        }
    }
}

impl From<EmbeddingError> for ErrorReport {
    fn from(error: EmbeddingError) -> Self {
        Self::from(&error)
    }
}

impl From<&RerankError> for ErrorReport {
    fn from(error: &RerankError) -> Self {
        let (kind, http_status) = match error {
            RerankError::HttpError(_) => (ErrorKind::Http, None),
            RerankError::JsonError(_) => (ErrorKind::Json, None),
            RerankError::UrlError(_) => (ErrorKind::Url, None),
            RerankError::ResponseError(_) => (ErrorKind::Response, None),
            RerankError::ProviderError(_) => (ErrorKind::Provider, None),
            RerankError::ProviderResponse(response) => {
                let status = response.status.map(|s| s.as_u16());
                (ErrorKind::ProviderResponse, status)
            }
        };
        // One rule per error type: the retry verdict is the error's own,
        // never a copy of its table kept beside the report.
        let retryable = error.is_retryable();
        let provider_response = match error {
            RerankError::ProviderResponse(response) => Some(response.clone()),
            _ => None,
        };
        let request_id = provider_response
            .as_ref()
            .and_then(|response| response.provider_request_id.clone());
        let code = provider_response
            .as_ref()
            .and_then(|response| response.machine_code());
        let refusal = provider_response
            .as_ref()
            .is_some_and(|response| response.refusal);
        ErrorReport {
            kind,
            retryable,
            message: error.to_string(),
            code,
            http_status,
            refusal,
            source_chain: source_chain(error),
            request_id,
            provider_response,
            detail: None,
        }
    }
}

impl From<RerankError> for ErrorReport {
    fn from(error: RerankError) -> Self {
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
mod tests;
