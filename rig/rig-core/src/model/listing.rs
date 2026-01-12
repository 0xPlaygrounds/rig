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
///
/// # Example
///
/// ```rust
/// use rig::model::Model;
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
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
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
    /// use rig::model::Model;
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
    /// use rig::model::Model;
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
    /// use rig::model::Model;
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
/// - `total`: Total number of models
///
/// # Example
///
/// ```rust
/// use rig::model::{Model, ModelList};
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
#[serde(rename_all = "camelCase")]
pub struct ModelList {
    /// The complete list of available models
    pub data: Vec<Model>,

    /// Total number of models available
    pub total: usize,
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
    /// use rig::model::{Model, ModelList};
    ///
    /// let list = ModelList::new(vec![
    ///     Model::from_id("gpt-4"),
    ///     Model::from_id("gpt-3.5-turbo"),
    /// ]);
    /// assert_eq!(list.len(), 2);
    /// assert_eq!(list.total, 2);
    /// ```
    pub fn new(data: Vec<Model>) -> Self {
        let total = data.len();
        Self { data, total }
    }

    /// Returns true if the list is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::model::ModelList;
    ///
    /// let empty = ModelList::new(vec![]);
    /// assert!(empty.is_empty());
    ///
    /// let non_empty = ModelList::new(vec![rig::model::Model::from_id("gpt-4")]);
    /// assert!(!non_empty.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of models in this page.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::model::{Model, ModelList};
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
    /// use rig::model::{Model, ModelList};
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
///
/// # Variants
///
/// - `ApiError`: The provider returned an error response with a status code
/// - `RequestError`: Failed to send the request to the provider
/// - `ParseError`: Failed to parse the provider's response
/// - `AuthError`: Authentication failed (invalid API key, etc.)
/// - `RateLimitError`: Rate limit was exceeded
/// - `ServiceUnavailable`: The provider service is temporarily unavailable
/// - `UnknownError`: An unexpected error occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelListingError {
    /// The provider returned an error response with a status code
    ApiError {
        /// HTTP status code
        status_code: u16,
        /// Error message from the provider
        message: String,
    },

    /// Failed to send the request to the provider
    RequestError {
        /// Description of the request error
        message: String,
    },

    /// Failed to parse the provider's response
    ParseError {
        /// Description of the parsing error
        message: String,
    },

    /// Authentication failed (invalid API key, etc.)
    AuthError {
        /// Authentication error details
        message: String,
    },

    /// Rate limit was exceeded
    RateLimitError {
        /// Rate limit error details
        message: String,
    },

    /// The provider service is temporarily unavailable
    ServiceUnavailable {
        /// Unavailable error details
        message: String,
    },

    /// An unexpected error occurred
    UnknownError {
        /// Details of the unknown error
        message: String,
    },
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

    /// Creates a new AuthError with the given message.
    pub fn auth_error(message: impl Into<String>) -> Self {
        Self::AuthError {
            message: message.into(),
        }
    }

    /// Creates a new RateLimitError with the given message.
    pub fn rate_limit_error(message: impl Into<String>) -> Self {
        Self::RateLimitError {
            message: message.into(),
        }
    }

    /// Creates a new ServiceUnavailable error with the given message.
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::ServiceUnavailable {
            message: message.into(),
        }
    }

    /// Creates a new UnknownError with the given message.
    pub fn unknown_error(message: impl Into<String>) -> Self {
        Self::UnknownError {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModelListingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiError {
                status_code,
                message,
            } => write!(f, "API error (status {}): {}", status_code, message),
            Self::RequestError { message } => write!(f, "Request error: {}", message),
            Self::ParseError { message } => write!(f, "Parse error: {}", message),
            Self::AuthError { message } => write!(f, "Authentication error: {}", message),
            Self::RateLimitError { message } => write!(f, "Rate limit error: {}", message),
            Self::ServiceUnavailable { message } => write!(f, "Service unavailable: {}", message),
            Self::UnknownError { message } => write!(f, "Unknown error: {}", message),
        }
    }
}

impl std::error::Error for ModelListingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_from_id() {
        let model = Model::from_id("gpt-4");
        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.name, None);
        assert_eq!(model.description, None);
        assert_eq!(model.r#type, None);
        assert_eq!(model.created_at, None);
        assert_eq!(model.owned_by, None);
        assert_eq!(model.context_length, None);
    }

    #[test]
    fn test_model_new() {
        let model = Model::new("gpt-4", "GPT-4");
        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.name, Some("GPT-4".to_string()));
    }

    #[test]
    fn test_model_display_name() {
        let model_with_name = Model::new("gpt-4", "GPT-4");
        assert_eq!(model_with_name.display_name(), "GPT-4");

        let model_without_name = Model::from_id("gpt-4");
        assert_eq!(model_without_name.display_name(), "gpt-4");
    }

    #[test]
    fn test_model_display() {
        let model = Model::new("gpt-4", "GPT-4");
        assert_eq!(format!("{}", model), "GPT-4");
    }

    #[test]
    fn test_model_list_new() {
        let list = ModelList::new(vec![Model::from_id("gpt-4")]);
        assert_eq!(list.len(), 1);
        assert_eq!(list.total, 1);
    }

    #[test]
    fn test_model_list_empty() {
        let list = ModelList::new(vec![]);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_model_list_iter() {
        let list = ModelList::new(vec![
            Model::from_id("gpt-4"),
            Model::from_id("gpt-3.5-turbo"),
        ]);
        let models: Vec<_> = list.iter().collect();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_model_list_into_iter() {
        let list = ModelList::new(vec![
            Model::from_id("gpt-4"),
            Model::from_id("gpt-3.5-turbo"),
        ]);
        let models: Vec<_> = list.into_iter().collect();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_model_listing_error_display() {
        let error = ModelListingError::api_error(404, "Not found");
        assert_eq!(error.to_string(), "API error (status 404): Not found");

        let error = ModelListingError::request_error("Connection failed");
        assert_eq!(error.to_string(), "Request error: Connection failed");

        let error = ModelListingError::parse_error("Invalid JSON");
        assert_eq!(error.to_string(), "Parse error: Invalid JSON");

        let error = ModelListingError::auth_error("Invalid API key");
        assert_eq!(error.to_string(), "Authentication error: Invalid API key");

        let error = ModelListingError::rate_limit_error("Too many requests");
        assert_eq!(error.to_string(), "Rate limit error: Too many requests");

        let error = ModelListingError::service_unavailable("Maintenance mode");
        assert_eq!(error.to_string(), "Service unavailable: Maintenance mode");

        let error = ModelListingError::unknown_error("Something went wrong");
        assert_eq!(error.to_string(), "Unknown error: Something went wrong");
    }

    #[test]
    fn test_model_serde() {
        let model = Model {
            id: "gpt-4".to_string(),
            name: Some("GPT-4".to_string()),
            description: None,
            r#type: Some("chat".to_string()),
            created_at: Some(1677610600),
            owned_by: Some("openai".to_string()),
            context_length: Some(8192),
        };

        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("gpt-4"));
        assert!(json.contains("GPT-4"));

        let deserialized: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "gpt-4");
        assert_eq!(deserialized.name, Some("GPT-4".to_string()));
    }

    #[test]
    fn test_model_list_serde() {
        let list = ModelList {
            data: vec![Model::from_id("gpt-4")],
            total: 100,
        };

        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("gpt-4"));
        assert!(json.contains("total"));

        let deserialized: ModelList = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized.total, 100);
    }

    #[test]
    fn test_model_listing_error_serde() {
        let error = ModelListingError::api_error(404, "Not found");

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("ApiError"));

        let deserialized: ModelListingError = serde_json::from_str(&json).unwrap();
        match deserialized {
            ModelListingError::ApiError {
                status_code,
                message,
            } => {
                assert_eq!(status_code, 404);
                assert_eq!(message, "Not found");
            }
            _ => panic!("Expected ApiError"),
        }
    }
}
