use crate::{http_client, provider_response, wasm_compat::WasmCompatSend};
use thiserror::Error;

/// Errors from provider client verification.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("invalid authentication")]
    InvalidAuthentication,
    #[error("provider error: {0}")]
    ProviderError(String),
    /// Raw error response preserved from the provider
    #[error("provider response error: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        http_client::Error,
    ),
}

impl VerifyError {
    /// Returns the raw provider response body when available.
    ///
    /// This is available for:
    /// - [`VerifyError::ProviderResponse`] using its preserved body.
    /// - [`VerifyError::HttpError`] when it wraps an HTTP non-success response that carries a body.
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::ProviderResponse(response) => Some(response.body.as_str()),
            Self::HttpError(error) => error.non_success_body(),
            _ => None,
        }
    }

    /// Parses the provider response body as JSON.
    ///
    /// Returns:
    /// - `Ok(Some(value))` when a body is present and valid JSON.
    /// - `Ok(None)` when no provider response body is available.
    /// - `Err(error)` when a body is present but isn't valid JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        provider_response::json(self.provider_response_body())
    }

    /// Returns the HTTP status code when this error preserves one, either from a
    /// non-success HTTP response or from a preserved provider response.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::ProviderResponse(response) => response.status,
            Self::HttpError(error) => error.non_success_status(),
            _ => None,
        }
    }
}

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn verify_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"rate limited"}}"#;
        let error = VerifyError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: body.to_string(),
        });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "rate limited" } }))
        );
    }

    #[test]
    fn verify_error_provider_response_helpers_with_http_non_success() {
        let body = r#"{"error":{"message":"bad request"}}"#;
        let error = VerifyError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
            StatusCode::BAD_REQUEST,
            body.to_string(),
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "bad request" } }))
        );
    }

    #[test]
    fn verify_error_provider_response_helpers_with_preserved_plain_text_body() {
        let error = VerifyError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: "not json".to_string(),
        });

        assert_eq!(error.provider_response_body(), Some("not json"));
        assert!(error.provider_response_json().is_err());
    }

    #[test]
    fn verify_error_provider_error_is_not_a_provider_response() {
        let error = VerifyError::ProviderError("internal diagnostic".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[test]
    fn verify_error_provider_response_helpers_with_unrelated_variant() {
        let error = VerifyError::InvalidAuthentication;

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }
}
