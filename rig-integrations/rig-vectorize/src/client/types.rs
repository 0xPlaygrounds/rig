//! API request and response types for Cloudflare Vectorize.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request body for the Vectorize query endpoint.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryRequest {
    /// The query vector.
    pub vector: Vec<f64>,

    /// Maximum number of results to return (max: 20 with metadata, 100 without).
    pub top_k: u64,

    /// Whether to return the vector values in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_values: Option<bool>,

    /// What metadata to return: "none", "indexed", or "all".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_metadata: Option<ReturnMetadata>,

    /// Optional filter for metadata fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
}

/// Options for what metadata to return in query results.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReturnMetadata {
    /// Return no metadata.
    #[default]
    None,
    /// Return only indexed metadata fields.
    Indexed,
    /// Return all metadata.
    All,
}

/// Response from the Vectorize API (wrapped in Cloudflare API envelope).
#[derive(Debug, Clone, Deserialize)]
pub struct ApiResponse<T> {
    /// Whether the request was successful.
    pub success: bool,

    /// The result payload.
    pub result: Option<T>,

    /// Error messages if the request failed.
    pub errors: Vec<ApiErrorDetail>,

    /// Informational messages.
    #[allow(dead_code)]
    pub messages: Vec<ApiMessage>,
}

/// Error detail from the Cloudflare API.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiErrorDetail {
    /// Error code.
    pub code: u32,

    /// Error message.
    pub message: String,
}

/// Informational message from the Cloudflare API.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ApiMessage {
    /// Message code.
    pub code: Option<u32>,

    /// Message text.
    pub message: String,
}

/// Result payload from a query request.
#[derive(Debug, Clone, Deserialize)]
pub struct QueryResult {
    /// Number of matches returned.
    #[allow(dead_code)]
    pub count: u64,

    /// The matching vectors.
    pub matches: Vec<VectorMatch>,
}

/// A single matching vector from a query.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorMatch {
    /// The vector ID.
    pub id: String,

    /// Similarity score (higher is more similar for cosine/dot product).
    pub score: f64,

    /// The vector values (only present if `returnValues: true`).
    #[serde(default)]
    #[allow(dead_code)]
    pub values: Option<Vec<f64>>,

    /// Metadata associated with the vector.
    #[serde(default)]
    pub metadata: Option<Value>,

    /// The namespace this vector belongs to.
    #[serde(default)]
    #[allow(dead_code)]
    pub namespace: Option<String>,
}

// ============================================================================
// Upsert Types
// ============================================================================

/// A single vector to be inserted or upserted.
#[derive(Debug, Clone, Serialize)]
pub struct VectorInput {
    /// Unique identifier for this vector (max 64 bytes).
    pub id: String,

    /// The embedding vector values.
    pub values: Vec<f64>,

    /// Optional metadata to store with the vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    /// Optional namespace for the vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Request body for the Vectorize upsert endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct UpsertRequest {
    /// The vectors to upsert.
    pub vectors: Vec<VectorInput>,
}

/// Result payload from an upsert request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct UpsertResult {
    /// Mutation identifier for tracking async processing.
    pub mutation_id: String,
}

// ============================================================================
// Delete Types
// ============================================================================

/// Request body for the Vectorize delete_by_ids endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct DeleteByIdsRequest {
    /// The vector IDs to delete.
    pub ids: Vec<String>,
}

/// Result payload from a delete_by_ids request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct DeleteResult {
    /// Mutation identifier for tracking async processing.
    pub mutation_id: String,
}

// ============================================================================
// List Vectors Types
// ============================================================================

/// Result payload from a list_vectors request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListVectorsResult {
    /// Number of vectors returned in this page.
    pub count: u64,

    /// Whether there are more vectors to fetch.
    pub is_truncated: bool,

    /// Total number of vectors in the index.
    pub total_count: u64,

    /// The vector IDs in this page.
    pub vectors: Vec<VectorIdEntry>,

    /// Cursor for fetching the next page.
    pub next_cursor: Option<String>,
}

/// A vector ID entry from the list_vectors response.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorIdEntry {
    /// The vector ID.
    pub id: String,
}
