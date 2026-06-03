//! Shared logic for inspecting provider error response bodies across capability errors.
use crate::http_client;
use http::StatusCode;

/// Returns the best available provider response body for an error.
///
/// Prefers an explicitly supplied provider body, and falls back to the
/// non-success body captured on the HTTP client error.
pub(crate) fn body<'a>(
    provider_body: Option<&'a str>,
    http_error: Option<&'a http_client::Error>,
) -> Option<&'a str> {
    if let Some(body) = provider_body {
        return Some(body);
    }

    if let Some(error) = http_error {
        return error.non_success_body();
    }

    None
}

/// Parses an optional response body as JSON.
///
/// Returns:
/// - `Ok(Some(value))` when a body is present and valid JSON.
/// - `Ok(None)` when no body is present.
/// - `Err(error)` when a body is present but isn't valid JSON.
pub(crate) fn json(body: Option<&str>) -> Result<Option<serde_json::Value>, serde_json::Error> {
    body.map(serde_json::from_str).transpose()
}

/// Returns the non-success HTTP status from an HTTP client error, when present.
pub(crate) fn status(http_error: Option<&http_client::Error>) -> Option<StatusCode> {
    if let Some(error) = http_error {
        return error.non_success_status();
    }
    None
}
