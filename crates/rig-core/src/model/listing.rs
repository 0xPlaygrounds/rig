//! Model listing types and error handling.
//!
//! This module provides types for representing available models from providers.
//! All models are returned in a single list; providers with pagination
//! handle fetching all pages internally.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a single model available from a provider.
///
/// This struct is designed to be flexible enough to accommodate the varying
/// responses from different LLM providers while providing a common interface.
///
/// # Fields
///
/// - `id`: The unique identifier for the model (required)
/// - `name`: A human-readable name for the model
/// - `description`: A detailed description of the model's capabilities
/// - `r#type`: The type of model (e.g., "chat", "completion", "embedding")
/// - `created_at`: Timestamp when the model was created
/// - `owned_by`: The organization or entity that owns the model
/// - `context_length`: The maximum context window size for the model
/// - `max_output_tokens`: The maximum tokens the model may generate per response
///
/// # Example
///
/// ```rust
/// use rig_core::model::Model;
///
/// // Create a model with just an ID
/// let model = Model::from_id("gpt-4");
///
/// // Create a model with ID and name
/// let model = Model::new("gpt-4", "GPT-4");
///
/// // Create a model with all fields
/// let model = Model {
///     id: "gpt-4".to_string(),
///     name: Some("GPT-4".to_string()),
///     description: Some("A large language model...".to_string()),
///     r#type: Some("chat".to_string()),
///     created_at: Some(1677610600),
///     owned_by: Some("openai".to_string()),
///     context_length: Some(8192),
///     max_output_tokens: Some(4096),
/// };
/// ```
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

    /// The maximum number of tokens the model may generate in one response.
    ///
    /// Distinct from [`Self::context_length`]: that is the input window, this
    /// is the output ceiling, and for most models the output ceiling is far
    /// smaller (Gemini 2.5 Flash: 1,048,576 in, 65,536 out).
    ///
    /// `None` means the provider's listing does not report one — never a
    /// default rig invented. Rig does **not** send this value on requests:
    /// omitting an output limit lets the provider apply its own per-model
    /// default, and populating it from here would reintroduce a rig-chosen cap
    /// by another route (rig#2322). It is for callers and diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

impl Model {
    /// Creates a new Model with the given ID and name.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier for the model
    /// * `name` - A human-readable name for the model
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::Model;
    ///
    /// let model = Model::new("gpt-4", "GPT-4");
    /// assert_eq!(model.id, "gpt-4");
    /// assert_eq!(model.name, Some("GPT-4".to_string()));
    /// ```
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

    /// Creates a new Model with only the required ID field.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier for the model
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::Model;
    ///
    /// let model = Model::from_id("gpt-4");
    /// assert_eq!(model.id, "gpt-4");
    /// assert_eq!(model.name, None);
    /// ```
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

    /// Returns a reference to the model's name, or the ID if no name is set.
    ///
    /// This is useful for display purposes when you want to show the most
    /// human-readable identifier available.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::Model;
    ///
    /// let model_with_name = Model::new("gpt-4", "GPT-4");
    /// assert_eq!(model_with_name.display_name(), "GPT-4");
    ///
    /// let model_without_name = Model::from_id("gpt-4");
    /// assert_eq!(model_without_name.display_name(), "gpt-4");
    /// ```
    pub fn display_name(&self) -> &str {
        self.name.as_ref().unwrap_or(&self.id)
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Represents a complete list of models from a provider.
///
/// This struct contains all available models from a provider. Providers that
/// support pagination internally handle fetching all pages before returning results.
///
/// # Fields
///
/// - `data`: The complete list of available models
///
/// # Example
///
/// ```rust
/// use rig_core::model::{Model, ModelList};
///
/// let list = ModelList::new(vec![
///     Model::from_id("gpt-4"),
///     Model::from_id("gpt-3.5-turbo"),
/// ]);
///
/// println!("Found {} models", list.len());
/// for model in list.iter() {
///     println!("- {}", model.display_name());
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    /// The complete list of available models
    pub data: Vec<Model>,
}

impl ModelList {
    /// Creates a new ModelList with the given models.
    ///
    /// # Arguments
    ///
    /// * `data` - The list of models
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::{Model, ModelList};
    ///
    /// let list = ModelList::new(vec![
    ///     Model::from_id("gpt-4"),
    ///     Model::from_id("gpt-3.5-turbo"),
    /// ]);
    /// assert_eq!(list.len(), 2);
    /// ```
    pub fn new(data: Vec<Model>) -> Self {
        Self { data }
    }

    /// Returns true if the list is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::ModelList;
    ///
    /// let empty = ModelList::new(vec![]);
    /// assert!(empty.is_empty());
    ///
    /// let non_empty = ModelList::new(vec![rig_core::model::Model::from_id("gpt-4")]);
    /// assert!(!non_empty.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of models in this list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::{Model, ModelList};
    ///
    /// let list = ModelList::new(vec![
    ///     Model::from_id("gpt-4"),
    ///     Model::from_id("gpt-3.5-turbo"),
    /// ]);
    /// assert_eq!(list.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns an iterator over the models in this list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::model::{Model, ModelList};
    ///
    /// let list = ModelList::new(vec![
    ///     Model::from_id("gpt-4"),
    ///     Model::from_id("gpt-3.5-turbo"),
    /// ]);
    ///
    /// for model in list.iter() {
    ///     println!("Model: {}", model.display_name());
    /// }
    /// ```
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

/// Errors that can occur when listing models from a provider.
///
/// This enum represents the various error conditions that may arise when
/// attempting to retrieve the list of available models from an LLM provider.
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

    pub(crate) fn api_error_with_context(
        provider: &str,
        path: &str,
        status_code: u16,
        body: &[u8],
    ) -> Self {
        let message =
            format_response_context(provider, path, format_args!("status={status_code}"), body);
        Self::api_error(status_code, message)
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

#[cfg(test)]
mod tests;
