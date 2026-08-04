//! Cloudflare Vectorize integration for the Rig framework.
//!
//! This crate provides a vector store implementation using Cloudflare Vectorize,
//! a globally distributed vector database built for AI applications.
//!
//! Queries arrive pre-embedded via [`VectorSearchRequest`]; the store never
//! embeds text itself.
//!
//! # Example
//!
//! ```ignore
//! use rig_vectorize::VectorizeVectorStore;
//!
//! let vector_store = VectorizeVectorStore::new(
//!     "your-account-id",
//!     "your-index-name",
//!     std::env::var("CLOUDFLARE_API_TOKEN").unwrap(),
//! );
//! ```

mod client;

// Re-export client types
pub use client::{
    DeleteByIdsRequest, DeleteResult, ListVectorsResult, QueryRequest, QueryResult, ReturnMetadata,
    UpsertRequest, UpsertResult, VectorIdEntry, VectorInput, VectorMatch, VectorizeClient,
    VectorizeError, VectorizeFilter,
};

use client::{QueryRequest as ApiQueryRequest, VectorInput as ApiVectorInput};
use rig_core::vector_store::request::VectorSearchRequest;
use rig_core::vector_store::{SearchHit, StoreRecord, VectorStoreError};
use rig_core::{OneOrMany, embeddings::Embedding};
use serde::{Serialize, de::DeserializeOwned};

impl From<VectorizeError> for VectorStoreError {
    fn from(err: VectorizeError) -> Self {
        VectorStoreError::DatastoreError(Box::new(err))
    }
}

/// A vector store backed by Cloudflare Vectorize.
///
/// Provides vector similarity search over pre-embedded queries using
/// Cloudflare's globally distributed Vectorize service. The store holds no
/// embedding model: queries and records arrive pre-embedded.
#[derive(Debug, Clone)]
pub struct VectorizeVectorStore {
    /// The HTTP client for Vectorize API.
    client: VectorizeClient,
}

impl VectorizeVectorStore {
    /// Creates a new Vectorize vector store.
    ///
    /// # Arguments
    /// * `account_id` - Cloudflare account ID
    /// * `index_name` - Name of the Vectorize index
    /// * `api_token` - Cloudflare API token with Vectorize read permissions
    pub fn new(
        account_id: impl Into<String>,
        index_name: impl Into<String>,
        api_token: impl Into<String>,
    ) -> Self {
        Self {
            client: VectorizeClient::new(account_id, index_name, api_token),
        }
    }

    /// Insert precomputed records into the Vectorize index.
    ///
    /// Each embedding of a record becomes one stored vector whose metadata is
    /// the record payload. A record with a single embedding is keyed by
    /// [`StoreRecord::id`]; additional embeddings get `"{id}#{n}"` ids so they
    /// don't overwrite each other.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let mut vectors: Vec<ApiVectorInput> = Vec::new();

        for record in records {
            let single = record.embeddings.len() == 1;

            for (n, embedding) in record.embeddings.into_iter().enumerate() {
                let id = if single {
                    record.id.clone()
                } else {
                    format!("{id}#{n}", id = record.id)
                };

                vectors.push(ApiVectorInput {
                    id,
                    values: embedding.vec,
                    metadata: Some(record.payload.clone()),
                    namespace: None,
                });
            }
        }

        if vectors.is_empty() {
            return Ok(());
        }

        tracing::debug!("Upserting {} vectors to Vectorize", vectors.len());

        const BATCH_SIZE: usize = 1000;

        for batch in vectors.chunks(BATCH_SIZE) {
            let request = UpsertRequest {
                vectors: batch.to_vec(),
            };

            self.client.upsert(request).await?;
        }

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(String, T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }

    /// Returns the top N most similar documents for a pre-embedded query.
    ///
    /// The [`SearchHit::payload`] is the stored vector metadata.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<VectorizeFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        if let Some(filter) = req.filter() {
            filter.validate()?;
        }

        let query_request = ApiQueryRequest {
            vector: req.query().first().vec,
            top_k: req.samples(),
            return_values: Some(false),
            return_metadata: Some(ReturnMetadata::All),
            filter: req.filter().as_ref().map(|f| f.clone().into_inner()),
        };

        let result = self.client.query(query_request).await?;

        // Convert results to the expected format
        let results = result
            .matches
            .into_iter()
            .filter(|m| req.threshold().is_none_or(|t| m.score >= t))
            .map(|m| SearchHit {
                score: m.score,
                id: m.id,
                payload: m.metadata.unwrap_or(serde_json::Value::Null),
            })
            .collect();

        Ok(results)
    }

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<VectorizeFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        if let Some(filter) = req.filter() {
            filter.validate()?;
        }

        let query_request = ApiQueryRequest {
            vector: req.query().first().vec,
            top_k: req.samples(),
            return_values: Some(false),
            return_metadata: Some(ReturnMetadata::None),
            filter: req.filter().as_ref().map(|f| f.clone().into_inner()),
        };

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

    /// Returns the top N most similar documents deserialized into `T` as
    /// `(score, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<VectorizeFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc = serde_json::from_value(hit.payload)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }
}
