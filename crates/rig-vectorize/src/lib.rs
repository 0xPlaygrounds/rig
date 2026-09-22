//! Cloudflare Vectorize vector store for Rig.
//!
//! [`VectorizeVectorStore`] queries a Vectorize index over Cloudflare's HTTP API,
//! narrowed by a [`VectorizeFilter`] metadata filter.
//!
//! # Example
//!
//! ```no_run
//! use rig_reqwest::prelude::*;
//! use rig_core::providers::openai;
//! use rig_vectorize::VectorizeVectorStore;
//!
//! # fn example() -> anyhow::Result<()> {
//! let openai = openai::wire::OpenAI::from_env()?.bound()?;
//! let embedding_model = openai.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
//!
//! let vector_store = VectorizeVectorStore::new(
//!     embedding_model,
//!     "your-account-id",
//!     "your-index-name",
//!     std::env::var("CLOUDFLARE_API_TOKEN")?,
//! );
//! # let _ = vector_store;
//! # Ok(())
//! # }
//! ```

mod client;

pub use client::{
    DeleteByIdsRequest, DeleteResult, ListVectorsResult, QueryRequest, QueryResult, ReturnMetadata,
    UpsertRequest, UpsertResult, VectorIdEntry, VectorInput, VectorMatch, VectorizeClient,
    VectorizeError, VectorizeFilter,
};

use client::{QueryRequest as ApiQueryRequest, VectorInput as ApiVectorInput};
use rig_core::embeddings::EmbeddingModel;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_core::vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex};
use rig_core::wasm_compat::WasmCompatSend;
use rig_core::{Embed, embeddings::Embedding};
use serde::{Serialize, de::DeserializeOwned};
use uuid::Uuid;

impl From<VectorizeError> for VectorStoreError {
    fn from(err: VectorizeError) -> Self {
        VectorStoreError::datastore(err)
    }
}

/// Vector store backed by a Cloudflare Vectorize index.
///
/// Queries are embedded with the same model `M` that populated the index, so
/// results are meaningless under another model.
#[derive(Debug, Clone)]
pub struct VectorizeVectorStore<M> {
    model: M,
    client: VectorizeClient,
}

impl<M: EmbeddingModel> VectorizeVectorStore<M> {
    /// Creates a store over the named index, authenticating with a Cloudflare API
    /// token. Inserting documents additionally requires write permission.
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
}

impl<M: EmbeddingModel> VectorizeVectorStore<M> {
    /// Embeds the query and returns matches at or above any request threshold.
    /// Errors before querying when the filter uses an unsupported operation.
    async fn query_matches(
        &self,
        req: &VectorSearchRequest<VectorizeFilter>,
        return_metadata: ReturnMetadata,
    ) -> Result<Vec<VectorMatch>, VectorStoreError> {
        if let Some(filter) = req.filter() {
            filter.validate()?;
        }

        let embedding = self.model.embed_text(req.query()).await?;

        let query_request = ApiQueryRequest {
            vector: embedding.vec,
            top_k: req.samples(),
            return_values: Some(false),
            return_metadata: Some(return_metadata),
            filter: req.filter().as_ref().map(|f| f.clone().into_inner()),
        };

        let result = self.client.query(query_request).await?;

        Ok(result
            .matches
            .into_iter()
            .filter(|m| req.threshold().is_none_or(|t| m.score >= t))
            .collect())
    }
}

impl<M: EmbeddingModel> VectorStoreIndex for VectorizeVectorStore<M> {
    type Filter = VectorizeFilter;

    /// Returns matches as `(score, vector id, metadata)`. A match without
    /// metadata deserializes from JSON null, which fails for most `T`.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let matches = self.query_matches(&req, ReturnMetadata::All).await?;

        let results = matches
            .into_iter()
            .map(|m| {
                let metadata = m.metadata.unwrap_or(serde_json::Value::Null);
                let doc: T = serde_json::from_value(metadata)?;
                Ok((m.score, m.id, doc))
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()?;

        Ok(results)
    }

    /// Like `top_n` but returns `(score, vector id)` without requesting metadata.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let matches = self.query_matches(&req, ReturnMetadata::None).await?;

        Ok(matches.into_iter().map(|m| (m.score, m.id)).collect())
    }
}

impl<M: EmbeddingModel> InsertDocuments for VectorizeVectorStore<M> {
    /// Upserts one vector per embedding, storing the document as metadata under a
    /// fresh identifier, in batches of a thousand vectors.
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let vectors =
            rig_core::vector_store::flatten_embedded(documents, |metadata, embedding| {
                Ok(ApiVectorInput {
                    id: Uuid::new_v4().to_string(),
                    values: embedding.vec,
                    metadata: Some(metadata.clone()),
                    namespace: None,
                })
            })?;

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
}
