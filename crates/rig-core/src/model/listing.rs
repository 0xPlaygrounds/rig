//! Provider model metadata, listing interfaces, and errors.
//!
//! ```
//! use rig_core::model::{Model, ModelList};
//!
//! let models = ModelList::new(vec![Model::new("example", "Example model")]);
//! assert_eq!(models.len(), 1);
//! ```

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Provider-advertised model identifier and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Model {
    /// The unique identifier for the model (required)
    pub id: String,

    /// A human-readable name for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// A detailed description of the model's capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The type of model (e.g., "chat", "completion", "embedding")
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub r#type: Option<String>,

    /// Timestamp when the model was created (Unix epoch)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,

    /// The organization or entity that owns the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,

    /// The maximum context window size for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    /// Provider-reported output-token ceiling, or `None` when unreported.
    /// Distinct from context length and not automatically applied to requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

impl Model {
    /// Creates a model with an ID and display name; other metadata is absent.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: Some(name.into()),
            description: None,
            r#type: None,
            created_at: None,
            owned_by: None,
            context_length: None,
            max_output_tokens: None,
        }
    }

    /// Creates a model with an ID and no optional metadata.
    pub fn from_id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            description: None,
            r#type: None,
            created_at: None,
            owned_by: None,
            context_length: None,
            max_output_tokens: None,
        }
    }

    /// Returns the name when present, otherwise the ID.
    pub fn display_name(&self) -> &str {
        self.name.as_ref().unwrap_or(&self.id)
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Ordered provider model entries. May represent one page or an aggregated listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    /// Model entries in returned order.
    pub data: Vec<Model>,
}

impl ModelList {
    /// Wraps model entries without sorting or deduplicating them.
    pub fn new(data: Vec<Model>) -> Self {
        Self { data }
    }

    /// Returns whether the list has no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Borrows entries in list order.
    pub fn iter(&self) -> std::slice::Iter<'_, Model> {
        self.data.iter()
    }
}

impl IntoIterator for ModelList {
    type Item = Model;
    type IntoIter = std::vec::IntoIter<Model>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a ModelList {
    type Item = &'a Model;
    type IntoIter = std::slice::Iter<'a, Model>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Retrieves provider model metadata. Wire-backed implementations follow
/// pagination within the driver's repeated-cursor and page-count limits.
pub trait ModelLister: WasmCompatSend + WasmCompatSync {
    /// Every model the provider offers.
    fn list_all(
        &self,
    ) -> impl Future<Output = Result<ModelList, ModelListingError>> + WasmCompatSend;
}

/// Model-listing request, authentication, provider, or parsing failure.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum ModelListingError {
    /// The provider returned an error response with a status code
    #[error("API error (status {status_code}): {message}")]
    ApiError {
        /// HTTP status code
        status_code: u16,
        /// Error message from the provider
        message: String,
    },

    /// Failed to send the request to the provider
    #[error("Request error: {message}")]
    RequestError {
        /// Description of the request error
        message: String,
    },

    /// Failed to parse the provider's response
    #[error("Parse error: {message}")]
    ParseError {
        /// Description of the parsing error
        message: String,
    },

    /// Authentication failed (invalid API key, etc.)
    #[error("Authentication error: {message}")]
    AuthError {
        /// Authentication error details
        message: String,
    },
}

const RESPONSE_BODY_PREVIEW_LIMIT: usize = 2048;

fn format_response_body_preview(body: &[u8]) -> String {
    let preview_len = body.len().min(RESPONSE_BODY_PREVIEW_LIMIT);
    let preview_bytes = body.get(..preview_len).unwrap_or(body);
    let mut preview = String::from_utf8_lossy(preview_bytes).into_owned();

    if body.len() > RESPONSE_BODY_PREVIEW_LIMIT {
        preview.push_str(&format!(
            "\n...<truncated {} bytes>",
            body.len() - RESPONSE_BODY_PREVIEW_LIMIT
        ));
    }

    preview
}

fn format_response_context(
    provider: &str,
    path: &str,
    details: impl fmt::Display,
    body: &[u8],
) -> String {
    format!(
        "provider={provider}\npath={path}\n{details}\nbody_bytes={}\nresponse_body_preview:\n{}",
        body.len(),
        format_response_body_preview(body)
    )
}

impl ModelListingError {
    /// Creates a new ApiError with the given status code and message.
    pub fn api_error(status_code: u16, message: impl Into<String>) -> Self {
        Self::ApiError {
            status_code,
            message: message.into(),
        }
    }

    /// Creates a new RequestError with the given message.
    pub fn request_error(message: impl Into<String>) -> Self {
        Self::RequestError {
            message: message.into(),
        }
    }

    /// Creates a new ParseError with the given message.
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
        }
    }

    pub(crate) fn parse_error_with_context(
        provider: &str,
        path: &str,
        error: &serde_json::Error,
        body: &[u8],
    ) -> Self {
        let message =
            format_response_context(provider, path, format_args!("parse_error={error}"), body);
        Self::parse_error(message)
    }

    pub(crate) fn parse_error_with_details(
        provider: &str,
        path: &str,
        details: impl fmt::Display,
        body: &[u8],
    ) -> Self {
        let message = format_response_context(provider, path, details, body);
        Self::parse_error(message)
    }
}

impl From<crate::http_client::Error> for ModelListingError {
    fn from(e: crate::http_client::Error) -> Self {
        Self::request_error(e.to_string())
    }
}

impl From<http::Error> for ModelListingError {
    fn from(e: http::Error) -> Self {
        Self::request_error(e.to_string())
    }
}

impl From<serde_json::Error> for ModelListingError {
    fn from(e: serde_json::Error) -> Self {
        Self::parse_error(e.to_string())
    }
}

/// The listing wire reports the same four shapes every other operation
/// does; the mapping onto this enum's own vocabulary lives here so no wire
/// restates it.
impl crate::wire::WireError for ModelListingError {
    fn transport(error: crate::http_client::Error) -> Self {
        match error.non_success_status() {
            Some(status) => Self::api_error(
                status.as_u16(),
                error.non_success_body().unwrap_or_default().to_owned(),
            ),
            None => Self::request_error(error.to_string()),
        }
    }

    fn http_response(status: http::StatusCode, body: &str) -> Self {
        Self::api_error(status.as_u16(), body)
    }

    fn json(error: serde_json::Error) -> Self {
        Self::parse_error(error.to_string())
    }

    fn decode(message: String) -> Self {
        Self::parse_error(message)
    }

    fn provider_body(body: &str) -> Self {
        Self::parse_error(body.to_owned())
    }

    /// A listing error carries its status in its own `ApiError` variant, so
    /// there is no status-less reply to stamp.
    fn with_provider_status(self, _status: Option<http::StatusCode>) -> Self {
        self
    }

    /// Returns the error unchanged; transport request IDs are not retained.
    fn with_provider_request_id(self, _request_id: Option<String>) -> Self {
        self
    }

    fn with_response_headers(self, _headers: Option<http::HeaderMap>) -> Self {
        self
    }

    fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::ApiError { status_code, .. } => http::StatusCode::from_u16(*status_code).ok(),
            _ => None,
        }
    }

    /// Adds provider and path context plus a bounded body preview to API and
    /// parse errors. Other variants remain unchanged.
    fn with_route(self, provider: &str, path: &str) -> Self {
        match self {
            Self::ApiError {
                status_code,
                message,
            } => Self::api_error(
                status_code,
                format_response_context(
                    provider,
                    path,
                    format_args!("status={status_code}"),
                    message.as_bytes(),
                ),
            ),
            Self::ParseError { message } => Self::parse_error(format_response_context(
                provider,
                path,
                format_args!("parse_error"),
                message.as_bytes(),
            )),
            other => other,
        }
    }

    /// The listing error's `ApiError` message *is* the reply's body (with
    /// its request context), which is what a projector would read.
    fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::ApiError { message, .. } => Some(message),
            _ => None,
        }
    }

    fn report(&self) -> crate::error::ErrorReport {
        crate::error::ErrorReport::from(self)
    }

    fn boundary(&self) -> crate::observe::AdapterErrorBoundary {
        use crate::observe::AdapterErrorBoundary as B;
        match self {
            Self::ApiError { .. } | Self::AuthError { .. } => B::ProviderResponse,
            Self::ParseError { .. } => B::Decode,
            Self::RequestError { .. } => B::Request,
        }
    }
}

impl From<&ModelListingError> for crate::error::ErrorReport {
    fn from(error: &ModelListingError) -> Self {
        use crate::error::{ErrorKind, retryable_status};
        let status = match error {
            ModelListingError::ApiError { status_code, .. } => Some(*status_code),
            _ => None,
        };
        let kind = match error {
            ModelListingError::ApiError { .. } | ModelListingError::AuthError { .. } => {
                ErrorKind::ProviderResponse
            }
            ModelListingError::ParseError { .. } => ErrorKind::Response,
            ModelListingError::RequestError { .. } => ErrorKind::Http,
        };
        let mut report = crate::error::ErrorReport::new(kind, error.to_string())
            .with_retryable(retryable_status(status));
        report.http_status = status;
        report
    }
}

impl From<ModelListingError> for crate::error::ErrorReport {
    fn from(error: ModelListingError) -> Self {
        Self::from(&error)
    }
}

#[cfg(test)]
mod tests;
