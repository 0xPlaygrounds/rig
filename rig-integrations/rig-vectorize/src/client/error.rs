//! Error types for the Vectorize client.

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
