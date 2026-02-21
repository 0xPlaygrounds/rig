//! Cloudflare Vectorize integration for the Rig framework.
//!
//! This crate provides a vector store implementation using Cloudflare Vectorize,
//! a globally distributed vector database built for AI applications.
//!
//! # Example
//!
//! ```ignore
//! use rig::providers::openai;
//! use rig_vectorize::VectorizeVectorStore;
//!
//! let openai = openai::Client::from_env();
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);
//!
//! let vector_store = VectorizeVectorStore::new(
//!     embedding_model,
//!     "your-account-id",
//!     "your-index-name",
//!     std::env::var("CLOUDFLARE_API_TOKEN").unwrap(),
//! );
//! ```

mod client;
mod error;
mod filter;
mod types;

pub use client::VectorizeClient;
pub use error::VectorizeError;
pub use filter::VectorizeFilter;
pub use types::{QueryRequest, QueryResult, ReturnMetadata, VectorMatch};

use rig::embeddings::EmbeddingModel;
use rig::vector_store::request::VectorSearchRequest;
use rig::vector_store::{VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use types::QueryRequest as ApiQueryRequest;

/// A vector store backed by Cloudflare Vectorize.
///
/// This struct implements [`VectorStoreIndex`] to provide vector similarity search
/// using Cloudflare's globally distributed Vectorize service.
#[derive(Debug, Clone)]
pub struct VectorizeVectorStore<M> {
    /// The embedding model used to generate query embeddings.
    model: M,
    /// The HTTP client for Vectorize API.
    client: VectorizeClient,
}

impl<M> VectorizeVectorStore<M> {
    /// Creates a new Vectorize vector store.
    ///
    /// # Arguments
    /// * `model` - The embedding model to use for query embedding
    /// * `account_id` - Cloudflare account ID
    /// * `index_name` - Name of the Vectorize index
    /// * `api_token` - Cloudflare API token with Vectorize read permissions
    pub fn new(
        model: M,
        account_id: impl Into<String>,
        index_name: impl Into<String>,
        api_token: impl Into<String>,
    ) -> Self {
        Self {
            model,
            client: VectorizeClient::new(account_id, index_name, api_token),
        }
    }

    /// Creates a new Vectorize vector store with a custom base URL.
    ///
    /// Useful for testing or enterprise deployments.
    pub fn with_base_url(
        model: M,
        account_id: impl Into<String>,
        index_name: impl Into<String>,
        api_token: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model,
            client: VectorizeClient::with_base_url(account_id, index_name, api_token, base_url),
        }
    }

    /// Returns a reference to the underlying Vectorize client.
    pub fn client(&self) -> &VectorizeClient {
        &self.client
    }
}

impl<M> VectorStoreIndex for VectorizeVectorStore<M>
where
    M: EmbeddingModel + Sync + Send,
{
    type Filter = VectorizeFilter;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        // Validate filter if present
        if let Some(filter) = req.filter() {
            filter.validate().map_err(VectorizeError::from)?;
        }

        // Generate embedding for the query
        let embedding = self.model.embed_text(req.query()).await?;

        // Build the query request
        let query_request = ApiQueryRequest {
            vector: embedding.vec,
            top_k: req.samples(),
            return_values: Some(false),
            return_metadata: Some(ReturnMetadata::All),
            filter: req.filter().as_ref().map(|f| f.clone().into_inner()),
        };

        // Execute the query
        let result = self.client.query(query_request).await?;

        // Convert results to the expected format
        let results = result
            .matches
            .into_iter()
            .filter(|m| req.threshold().is_none_or(|t| m.score >= t))
            .map(|m| {
                let metadata = m.metadata.unwrap_or(serde_json::Value::Null);
                let doc: T = serde_json::from_value(metadata)?;
                Ok((m.score, m.id, doc))
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()?;

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        // Validate filter if present
        if let Some(filter) = req.filter() {
            filter.validate().map_err(VectorizeError::from)?;
        }

        // Generate embedding for the query
        let embedding = self.model.embed_text(req.query()).await?;

        // Build the query request - no metadata needed for IDs only
        let query_request = ApiQueryRequest {
            vector: embedding.vec,
            top_k: req.samples(),
            return_values: Some(false),
            return_metadata: Some(ReturnMetadata::None),
            filter: req.filter().as_ref().map(|f| f.clone().into_inner()),
        };

        // Execute the query
        let result = self.client.query(query_request).await?;

        // Convert results to (score, id) tuples
        let results = result
            .matches
            .into_iter()
            .filter(|m| req.threshold().is_none_or(|t| m.score >= t))
            .map(|m| (m.score, m.id))
            .collect();

        Ok(results)
    }
}
