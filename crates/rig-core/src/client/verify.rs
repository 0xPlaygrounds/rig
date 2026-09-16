use crate::error::{ErrorKind, ErrorReport};
use crate::observe::AdapterErrorBoundary;
use crate::wire::WireError;
use crate::{http_client, provider_response};
use thiserror::Error;

/// Errors from the `Verify` operation: asking a provider whether it accepts
/// the configured credentials.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
///
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("invalid authentication")]
    InvalidAuthentication,
    #[error("provider error: {0}")]
    ProviderError(String),
    /// Raw error response preserved from the provider
    #[error("provider response error: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
    /// A transport failure that produced no provider reply; a rejected
    /// verification with a body is [`Self::ProviderResponse`].
    #[error("http error: {0}")]
    HttpError(#[source] http_client::Error),
}

crate::provider_response::impl_provider_response_helpers!(VerifyError);

impl From<http_client::Error> for VerifyError {
    fn from(error: http_client::Error) -> Self {
        Self::from_transport_error(error)
    }
}

/// Verification is the one operation whose *status* is the answer, so the
/// 401/403 classification lives here rather than in every wire: the driver
/// funnels a rejected reply through [`WireError::http_response`] and this
/// is where it becomes [`VerifyError::InvalidAuthentication`].
impl WireError for VerifyError {
    fn transport(error: http_client::Error) -> Self {
        match error.non_success_status() {
            Some(http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN) => {
                Self::InvalidAuthentication
            }
            _ => Self::from_transport_error(error),
        }
    }

    fn http_response(status: http::StatusCode, body: &str) -> Self {
        match status {
            http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN => {
                Self::InvalidAuthentication
            }
            status => Self::from_http_response(status, body),
        }
    }

    fn json(error: serde_json::Error) -> Self {
        Self::ProviderError(error.to_string())
    }

    fn decode(message: String) -> Self {
        Self::ProviderError(message)
    }

    fn provider_body(body: &str) -> Self {
        Self::from_provider_body(body)
    }

    fn with_provider_status(self, status: Option<http::StatusCode>) -> Self {
        Self::with_provider_status(self, status)
    }

    fn with_provider_request_id(self, request_id: Option<String>) -> Self {
        Self::with_provider_request_id(self, request_id)
    }

    fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self {
        Self::with_response_headers(self, headers)
    }

    fn provider_response_status(&self) -> Option<http::StatusCode> {
        Self::provider_response_status(self)
    }

    fn provider_response_body(&self) -> Option<&str> {
        Self::provider_response_body(self)
    }

    fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }

    fn boundary(&self) -> AdapterErrorBoundary {
        match self {
            Self::HttpError(error) => AdapterErrorBoundary::from_http(error),
            Self::InvalidAuthentication | Self::ProviderError(_) | Self::ProviderResponse(_) => {
                AdapterErrorBoundary::ProviderResponse
            }
        }
    }
}

impl From<&VerifyError> for ErrorReport {
    fn from(error: &VerifyError) -> Self {
        let provider_response = match error {
            VerifyError::ProviderResponse(response) => Some(response.clone()),
            _ => None,
        };
        let kind = match error {
            VerifyError::HttpError(_) => ErrorKind::Http,
            // An invalid credential is the provider's verdict on the
            // request, even when the transport hid the reply.
            VerifyError::InvalidAuthentication | VerifyError::ProviderError(_) => {
                ErrorKind::Provider
            }
            VerifyError::ProviderResponse(_) => ErrorKind::ProviderResponse,
        };
        let mut report =
            ErrorReport::new(kind, error.to_string()).with_retryable(error.is_retryable());
        report.code = provider_response
            .as_ref()
            .and_then(|response| response.machine_code());
        report.http_status = provider_response
            .as_ref()
            .and_then(|response| response.status.map(|status| status.as_u16()));
        report.request_id = provider_response
            .as_ref()
            .and_then(|response| response.provider_request_id.clone());
        report.provider_response = provider_response;
        report
    }
}

#[cfg(test)]
mod provider_response_tests;
