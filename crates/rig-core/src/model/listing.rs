//! Provider model metadata and listing interfaces.
//!
//! ```
//! use rig_core::model::{Model, ModelList};
//!
//! let models = ModelList::new(vec![Model::new("example", "Example model")]);
//! assert_eq!(models.len(), 1);
//! ```

use crate::error::ProviderError;
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
    fn list_all(&self) -> impl Future<Output = Result<ModelList, ProviderError>> + WasmCompatSend;
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

/// A listing page that did not parse, with the request context and a bounded
/// preview of the body.
pub(crate) fn parse_error(
    provider: &str,
    path: &str,
    details: impl fmt::Display,
    body: &[u8],
) -> ProviderError {
    ProviderError::Response(format_response_context(provider, path, details, body))
}

/// Adds the provider and request path to a failed listing: a preserved reply
/// records them as its route, and a decode failure names them in its message.
pub(crate) fn with_route(error: ProviderError, provider: &str, path: &str) -> ProviderError {
    match error {
        ProviderError::ProviderResponse(mut response) => {
            response.route = Some(format!("provider={provider} path={path}"));
            ProviderError::ProviderResponse(response)
        }
        ProviderError::Json(error) => parse_error(
            provider,
            path,
            format_args!("parse_error"),
            error.to_string().as_bytes(),
        ),
        ProviderError::Response(message) => parse_error(
            provider,
            path,
            format_args!("parse_error"),
            message.as_bytes(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests;
