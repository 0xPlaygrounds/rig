//! Shared logic for inspecting provider error response bodies across capability errors.
use http::StatusCode;

/// A raw error response preserved from a provider.
///
/// Capability errors store this in their `ProviderResponse` variants when Rig
/// has the provider's response body in hand. Unlike `ProviderError(String)`,
/// which may carry Rig-generated diagnostics, this type always represents the
/// payload the provider actually returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponseError {
    /// HTTP status of the provider response, when it was captured alongside the body.
    pub status: Option<StatusCode>,
    /// Raw response body as returned by the provider.
    pub body: String,
}

impl ProviderResponseError {
    pub(crate) fn without_status(body: impl Into<String>) -> Self {
        Self {
            status: None,
            body: body.into(),
        }
    }
}

impl std::fmt::Display for ProviderResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(f, "status {status}: {}", self.body),
            None => write!(f, "{}", self.body),
        }
    }
}

impl std::error::Error for ProviderResponseError {}

/// Parses an optional response body as JSON.
///
/// Returns:
/// - `Ok(Some(value))` when a body is present and valid JSON.
/// - `Ok(None)` when no body is present.
/// - `Err(error)` when a body is present but isn't valid JSON.
pub(crate) fn json(body: Option<&str>) -> Result<Option<serde_json::Value>, serde_json::Error> {
    body.filter(|body| !body.is_empty())
        .map(serde_json::from_str)
        .transpose()
}

pub(crate) fn completion_error_from_body(
    body: impl Into<String>,
) -> crate::completion::CompletionError {
    crate::completion::CompletionError::ProviderResponse(ProviderResponseError::without_status(
        body,
    ))
}

/// Implements the `provider_response_*` inspection helpers on a capability error
/// enum.
///
/// The enum must have a `ProviderResponse(`[`ProviderResponseError`]`)` variant
/// and an `HttpError(`[`http_client::Error`](crate::http_client::Error)`)`
/// variant; the generated helpers read from those two sources only, since they
/// are the only ones that genuinely represent a provider's response.
macro_rules! impl_provider_response_helpers {
    ($error:ty) => {
        impl $error {
            /// Returns the raw provider response body when available.
            ///
            /// This is available for:
            /// - `Self::ProviderResponse` using its preserved body.
            /// - `Self::HttpError` when it wraps an HTTP non-success response that
            ///   carries a body.
            ///
            /// Availability depends on the provider path preserving the response
            /// body; not every provider is covered yet, so this can return `None`
            /// for failures from providers that haven't been wired up.
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
            pub fn provider_response_json(
                &self,
            ) -> Result<Option<serde_json::Value>, serde_json::Error> {
                $crate::provider_response::json(self.provider_response_body())
            }

            /// Returns the HTTP status code when this error preserves one, either
            /// from a non-success HTTP response, from a preserved provider
            /// response, or from a 2xx error envelope.
            pub fn provider_response_status(&self) -> Option<http::StatusCode> {
                match self {
                    Self::ProviderResponse(response) => response.status,
                    Self::HttpError(error) => error.non_success_status(),
                    _ => None,
                }
            }
        }
    };
}

pub(crate) use impl_provider_response_helpers;
