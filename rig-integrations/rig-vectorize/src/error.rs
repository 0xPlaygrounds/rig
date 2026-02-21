//! Error types for the Vectorize integration.

use rig::vector_store::VectorStoreError;

/// Errors that can occur when interacting with Cloudflare Vectorize.
#[derive(Debug, thiserror::Error)]
pub enum VectorizeError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Vectorize API error (code: {code}): {message}")]
    ApiError { code: u32, message: String },

    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Unsupported filter operation: {0}")]
    UnsupportedFilterOperation(String),
}

impl From<VectorizeError> for VectorStoreError {
    fn from(err: VectorizeError) -> Self {
        VectorStoreError::DatastoreError(Box::new(err))
    }
}
