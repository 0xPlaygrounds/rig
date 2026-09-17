//! Failures constructing, sending, or decoding an evaluation.

use rig_core::ProviderResponseError;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::http_client;
use rig_core::observe::AdapterErrorBoundary;
use rig_core::wire::WireError;

/// An evaluation failed before or after reaching Typesafe.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The question or state is not a valid request.
    #[error("invalid evaluation request: {0}")]
    InvalidRequest(String),
    /// The reply does not satisfy the question's contract.
    #[error("invalid evaluation response: {0}")]
    InvalidResponse(String),
    /// JSON could not be serialized or decoded.
    #[error("evaluation JSON error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// A transport or HTTP protocol failure.
    #[error("evaluation HTTP error: {0}")]
    Http(#[source] http_client::Error),
    /// The provider's preserved response, including transport metadata.
    #[error("evaluation provider error: {0}")]
    ProviderResponse(#[from] ProviderResponseError),
}

impl From<http_client::Error> for Error {
    fn from(error: http_client::Error) -> Self {
        Self::transport(error)
    }
}

impl WireError for Error {
    fn transport(error: http_client::Error) -> Self {
        match error {
            http_client::Error::InvalidStatusCodeWithDetails {
                status,
                body,
                headers,
            } => Self::ProviderResponse(
                ProviderResponseError::new(status, body).with_headers(Some(headers)),
            ),
            error => Self::Http(error),
        }
    }

    fn http_response(status: http::StatusCode, body: &str) -> Self {
        Self::ProviderResponse(ProviderResponseError::new(status, body))
    }

    fn json(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
    fn decode(message: String) -> Self {
        Self::InvalidResponse(message)
    }
    fn provider_body(body: &str) -> Self {
        Self::ProviderResponse(ProviderResponseError::without_status(body))
    }

    fn with_provider_status(self, status: Option<http::StatusCode>) -> Self {
        match self {
            Self::ProviderResponse(response) => {
                Self::ProviderResponse(response.with_status(status))
            }
            error => error,
        }
    }

    fn with_provider_request_id(self, request_id: Option<String>) -> Self {
        match self {
            Self::ProviderResponse(response) => {
                Self::ProviderResponse(response.with_provider_request_id(request_id))
            }
            error => error,
        }
    }

    fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self {
        match self {
            Self::ProviderResponse(response) => {
                Self::ProviderResponse(response.with_headers(headers))
            }
            error => error,
        }
    }

    fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::ProviderResponse(response) => response.status,
            _ => None,
        }
    }

    fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::ProviderResponse(response) => Some(&response.body),
            _ => None,
        }
    }

    fn report(&self) -> ErrorReport {
        let kind = match self {
            Self::InvalidRequest(_) => ErrorKind::Request,
            Self::InvalidResponse(_) => ErrorKind::Response,
            Self::Serialization(_) => ErrorKind::Json,
            Self::Http(_) => ErrorKind::Http,
            Self::ProviderResponse(_) => ErrorKind::ProviderResponse,
        };
        let mut report = ErrorReport::new(kind, self.to_string());
        match self {
            Self::Http(error) => report.retryable = rig_core::error::transient_transport(error),
            Self::ProviderResponse(response) => {
                report.retryable = response.is_retryable();
                report.code = response.machine_code();
                report.http_status = response.status.map(|status| status.as_u16());
                report.refusal = response.refusal;
                report.request_id = response.provider_request_id.clone();
                report.provider_response = Some(response.clone());
            }
            _ => {}
        }
        report
    }

    fn boundary(&self) -> AdapterErrorBoundary {
        use AdapterErrorBoundary as B;
        match self {
            Self::InvalidRequest(_) => B::Request,
            Self::Serialization(_) | Self::InvalidResponse(_) => B::Decode,
            Self::ProviderResponse(_) => B::ProviderResponse,
            Self::Http(error) => match error {
                http_client::Error::Protocol(_)
                | http_client::Error::InvalidHeaderValue(_)
                | http_client::Error::NoHeaders => B::Request,
                http_client::Error::InvalidContentType(_) => B::Decode,
                http_client::Error::StreamEnded => B::Transport,
                http_client::Error::InvalidStatusCodeWithDetails { .. } => B::ProviderResponse,
                http_client::Error::Instance(_) => B::Unknown,
            },
        }
    }
}
