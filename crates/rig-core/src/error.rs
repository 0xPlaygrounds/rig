//! The wire form of an error: a serde-able, classified report that crosses
//! task, channel, and process boundaries where the concrete error types
//! (`CompletionError`, `ToolExecutionError`, `MemoryError`) cannot.
//!
//! `ErrorReport` is the *only* error shape the effect protocol speaks. Every
//! concrete error converts into it losslessly enough for policy — kind,
//! retryability, HTTP status, provider code, refusal — and carries the
//! original `Display` chain as text for diagnostics.
//!
//! Retryability has one home: [`retryable_status`] is the single place a
//! status code is classified, and both [`CompletionError::is_retryable`] and
//! the `From` impls here consult it.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    completion::CompletionError,
    embeddings::EmbeddingError,
    memory::MemoryError,
    tool::{ToolErrorKind, ToolExecutionError},
    vector_store::VectorStoreError,
};

/// Normalized classification of an [`ErrorReport`].
///
/// Each arm mirrors one variant family of the concrete error types it is
/// transcribed from; a report never invents a classification the source did
/// not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// A non-success HTTP response, or a transport failure without a status.
    Http {
        /// The status, when the failure had a response.
        status: Option<u16>,
    },
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
    /// The provider's raw error response was preserved.
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
    /// The bus is alive but no handler serves the requested key — it was
    /// never registered, was deregistered, or is of another family than the
    /// typed view asked for. A wiring or liveness event; the key is in the
    /// message.
    HandlerUnavailable,
    /// Anything else.
    Other,
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
    /// A provider- or tool-specific machine code, when one was reported.
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

    /// Whether the same operation may reasonably be retried.
    pub const fn is_retryable(&self) -> bool {
        self.retryable
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
/// status (a response-less transport failure: connect, decode, timeout) is
/// retryable — the request never reached a decision.
pub const fn retryable_status(status: Option<u16>) -> bool {
    match status {
        None => true,
        Some(408 | 425 | 429) => true,
        Some(s) => s >= 500 && s <= 599,
    }
}

/// Collect the `Display` of each `source()` link below `error`, outermost
/// first.
fn source_chain(error: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = error.source();
    while let Some(source) = current {
        chain.push(source.to_string());
        current = source.source();
    }
    chain
}

impl CompletionError {
    /// Whether the same request may reasonably be retried, per
    /// [`retryable_status`]: HTTP failures classify by status, everything
    /// else is a fault in the request or the response and is not retried.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::HttpError(error) => {
                retryable_status(error.non_success_status().map(|s| s.as_u16()))
            }
            Self::ProviderResponse(response) => {
                retryable_status(response.status.map(|s| s.as_u16()))
            }
            Self::Report(report) => report.retryable,
            Self::JsonError(_)
            | Self::UrlError(_)
            | Self::RequestError(_)
            | Self::ResponseError(_)
            | Self::ProviderError(_) => false,
        }
    }

    /// The wire form of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }
}

impl From<&CompletionError> for ErrorReport {
    fn from(error: &CompletionError) -> Self {
        let (kind, http_status) = match error {
            CompletionError::HttpError(inner) => {
                let status = inner.non_success_status().map(|s| s.as_u16());
                (ErrorKind::Http { status }, status)
            }
            CompletionError::JsonError(_) => (ErrorKind::Json, None),
            CompletionError::UrlError(_) => (ErrorKind::Url, None),
            CompletionError::RequestError(_) => (ErrorKind::Request, None),
            CompletionError::ResponseError(_) => (ErrorKind::Response, None),
            CompletionError::ProviderError(_) => (ErrorKind::Provider, None),
            CompletionError::ProviderResponse(response) => {
                let status = response.status.map(|s| s.as_u16());
                (ErrorKind::ProviderResponse, status)
            }
            CompletionError::Report(report) => return report.clone(),
        };
        let request_id = match error {
            CompletionError::ProviderResponse(response) => response.provider_request_id.clone(),
            CompletionError::HttpError(_)
            | CompletionError::JsonError(_)
            | CompletionError::UrlError(_)
            | CompletionError::RequestError(_)
            | CompletionError::ResponseError(_)
            | CompletionError::ProviderError(_)
            | CompletionError::Report(_) => None,
        };
        ErrorReport {
            kind,
            retryable: error.is_retryable(),
            message: error.to_string(),
            code: None,
            http_status,
            refusal: false,
            source_chain: source_chain(error),
            request_id,
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
    /// one was set, else the kind's own default.
    pub fn is_retryable(&self) -> bool {
        self.retryable().unwrap_or_else(|| {
            matches!(
                self.kind(),
                ToolErrorKind::Timeout
                    | ToolErrorKind::RateLimited
                    | ToolErrorKind::Network
                    | ToolErrorKind::Provider
            )
        })
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
            EmbeddingError::HttpError(inner) => {
                let status = inner.non_success_status().map(|s| s.as_u16());
                (ErrorKind::Http { status }, status)
            }
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
        let retryable = match kind {
            ErrorKind::Http { status } => retryable_status(status),
            ErrorKind::ProviderResponse => retryable_status(http_status),
            ErrorKind::Json
            | ErrorKind::Url
            | ErrorKind::Request
            | ErrorKind::Response
            | ErrorKind::Provider
            | ErrorKind::Tool(_)
            | ErrorKind::MemoryBackend
            | ErrorKind::MemoryPolicy
            | ErrorKind::Internal
            | ErrorKind::Cancelled
            | ErrorKind::Timeout
            | ErrorKind::BusClosed
            | ErrorKind::HandlerUnavailable
            | ErrorKind::Other => false,
        };
        ErrorReport {
            kind,
            retryable,
            message: error.to_string(),
            code: None,
            http_status,
            refusal: false,
            source_chain: source_chain(error),
            request_id: None,
        }
    }
}

impl From<EmbeddingError> for ErrorReport {
    fn from(error: EmbeddingError) -> Self {
        Self::from(&error)
    }
}

impl From<&VectorStoreError> for ErrorReport {
    fn from(error: &VectorStoreError) -> Self {
        let (kind, http_status) = match error {
            VectorStoreError::EmbeddingError(inner) => {
                let report = ErrorReport::from(inner);
                (report.kind, report.http_status)
            }
            VectorStoreError::JsonError(_) => (ErrorKind::Json, None),
            VectorStoreError::DatastoreError(_) => (ErrorKind::Provider, None),
            VectorStoreError::FilterError(_) | VectorStoreError::BuilderError(_) => {
                (ErrorKind::Request, None)
            }
            VectorStoreError::MissingIdError(_) => (ErrorKind::Response, None),
            VectorStoreError::Http(inner) => {
                let status = inner.non_success_status().map(|s| s.as_u16());
                (ErrorKind::Http { status }, status)
            }
            VectorStoreError::ExternalAPIError(status, _) => {
                let status = Some(status.as_u16());
                (ErrorKind::Http { status }, status)
            }
        };
        let retryable = matches!(kind, ErrorKind::Http { .. }) && retryable_status(http_status);
        ErrorReport {
            kind,
            retryable,
            message: error.to_string(),
            code: None,
            http_status,
            refusal: false,
            source_chain: source_chain(error),
            request_id: None,
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
