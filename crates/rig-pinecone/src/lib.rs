//! Pinecone vector store integration for Rig.
//!
//! This crate provides [`PineconeVectorStore`], a Rig vector store index backed
//! by a Pinecone index. It supports dense vector search, namespace isolation,
//! and metadata filtering.
//!
//! The root `rig` facade re-exports this crate as `rig::pinecone` when the
//! `pinecone` feature is enabled.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use rig_core::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{Filter, VectorSearchRequest},
    },
};

/// Errors returned by the Pinecone client.
#[derive(Debug, thiserror::Error)]
pub enum PineconeError {
    /// Error communicating with the server.
    #[error("error communicating with server: {0}")]
    ReqwestError(#[from] reqwest::Error),

    /// Serialization/deserialization error.
    #[error("serialization/deserialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    /// Pinecone API returned a non-success response.
    #[error("got error from server: {status} - {message}")]
    RemoteError {
        status: reqwest::StatusCode,
        message: String,
    },
}

#[cfg(not(target_family = "wasm"))]
fn datastore_error<E>(error: E) -> VectorStoreError
where
    E: std::error::Error + Send + Sync + 'static,
{
    VectorStoreError::DatastoreError(Box::new(error))
}

#[cfg(target_family = "wasm")]
fn datastore_error<E>(error: E) -> VectorStoreError
where
    E: std::error::Error + 'static,
{
    VectorStoreError::DatastoreError(Box::new(error))
}

/// A minimal Pinecone REST API client.
#[derive(Clone, Debug)]
pub struct PineconeClient {
    client: reqwest::Client,
    api_key: String,
    host: String,
}

impl PineconeClient {
    /// Creates a new Pinecone client.
    ///
    /// # Arguments
    /// * `api_key` - Pinecone API Key
    /// * `host` - Host URL of the index (e.g. `https://my-index-123.svc.us-east1-gcp.pinecone.io`)
    pub fn new(api_key: &str, host: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            host: host.trim_end_matches('/').to_string(),
        }
    }
}

#[derive(Serialize)]
struct UpsertRequest {
    vectors: Vec<PineconeVector>,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,
}

#[derive(Serialize)]
struct PineconeVector {
    id: String,
    values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct QueryRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,
    vector: Vec<f32>,
    top_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<serde_json::Value>,
    include_values: bool,
    include_metadata: bool,
}

#[derive(Deserialize)]
struct QueryResponse {
    matches: Vec<PineconeMatch>,
}

#[derive(Deserialize)]
struct PineconeMatch {
    id: String,
    score: f32,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

impl PineconeClient {
    async fn upsert(&self, req: UpsertRequest) -> Result<(), PineconeError> {
        let url = format!("{}/vectors/upsert", self.host);
        let resp = self.client.post(&url)
            .header("Api-Key", &self.api_key)
            .header("X-Pinecone-Api-Version", "2025-01")
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let message = resp.text().await.unwrap_or_default();
            return Err(PineconeError::RemoteError { status, message });
        }

        Ok(())
    }

    async fn query(&self, req: QueryRequest) -> Result<QueryResponse, PineconeError> {
        let url = format!("{}/query", self.host);
        let resp = self.client.post(&url)
            .header("Api-Key", &self.api_key)
            .header("X-Pinecone-Api-Version", "2025-01")
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let message = resp.text().await.unwrap_or_default();
            return Err(PineconeError::RemoteError { status, message });
        }

        let query_response: QueryResponse = resp.json().await?;
        Ok(query_response)
    }
}

/// A Pinecone vector store implementation.
pub struct PineconeVectorStore<M: EmbeddingModel> {
    client: PineconeClient,
    model: M,
    namespace: Option<String>,
}

impl<M: EmbeddingModel> PineconeVectorStore<M> {
    /// Creates a new Pinecone vector store.
    pub fn new(client: PineconeClient, model: M, namespace: Option<String>) -> Self {
        Self {
            client,
            model,
            namespace,
        }
    }
}

impl<M> InsertDocuments for PineconeVectorStore<M>
where
    M: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let mut vectors = Vec::new();

        for (document, embeddings) in documents {
            let json_document = serde_json::to_value(&document)?;
            let metadata = serde_json::json!({
                "document": json_document
            });

            for embedding in embeddings {
                let values: Vec<f32> = embedding.vec.into_iter().map(|x| x as f32).collect();
                vectors.push(PineconeVector {
                    id: Uuid::new_v4().to_string(),
                    values,
                    metadata: Some(metadata.clone()),
                });
            }
        }

        if !vectors.is_empty() {
            let req = UpsertRequest {
                vectors,
                namespace: self.namespace.clone(),
            };
            self.client.upsert(req).await.map_err(datastore_error)?;
        }

        Ok(())
    }
}

fn translate_filter(filter: &Filter<serde_json::Value>) -> serde_json::Value {
    match filter {
        Filter::Eq(key, val) => serde_json::json!({ key: { "$eq": val } }),
        Filter::Gt(key, val) => serde_json::json!({ key: { "$gt": val } }),
        Filter::Lt(key, val) => serde_json::json!({ key: { "$lt": val } }),
        Filter::And(lhs, rhs) => serde_json::json!({
            "$and": [translate_filter(lhs), translate_filter(rhs)]
        }),
        Filter::Or(lhs, rhs) => serde_json::json!({
            "$or": [translate_filter(lhs), translate_filter(rhs)]
        }),
    }
}

impl<M> VectorStoreIndex for PineconeVectorStore<M>
where
    M: EmbeddingModel + Send + Sync,
{
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let query_vector = self.model.embed_text(req.query()).await?.vec;
        let query_vector_f32: Vec<f32> = query_vector.into_iter().map(|x| x as f32).collect();

        let filter_val = req.filter().as_ref().map(translate_filter);

        let query_req = QueryRequest {
            namespace: self.namespace.clone(),
            vector: query_vector_f32,
            top_k: req.samples() as usize,
            filter: filter_val,
            include_values: false,
            include_metadata: true,
        };

        let response = self.client.query(query_req).await.map_err(datastore_error)?;

        let mut results = Vec::new();
        for m in response.matches {
            let score = m.score as f64;
            if let Some(threshold) = req.threshold() {
                if score < threshold {
                    continue;
                }
            }

            let metadata = m.metadata.ok_or_else(|| {
                VectorStoreError::DatastoreError("Missing metadata in Pinecone query result".into())
            })?;

            let doc_val = metadata.get("document").ok_or_else(|| {
                VectorStoreError::DatastoreError("Missing 'document' field in Pinecone metadata".into())
            })?;

            let doc: T = serde_json::from_value(doc_val.clone())?;
            results.push((score, m.id, doc));
        }

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query_vector = self.model.embed_text(req.query()).await?.vec;
        let query_vector_f32: Vec<f32> = query_vector.into_iter().map(|x| x as f32).collect();

        let filter_val = req.filter().as_ref().map(translate_filter);

        let query_req = QueryRequest {
            namespace: self.namespace.clone(),
            vector: query_vector_f32,
            top_k: req.samples() as usize,
            filter: filter_val,
            include_values: false,
            include_metadata: false,
        };

        let response = self.client.query(query_req).await.map_err(datastore_error)?;

        let mut results = Vec::new();
        for m in response.matches {
            let score = m.score as f64;
            if let Some(threshold) = req.threshold() {
                if score < threshold {
                    continue;
                }
            }
            results.push((score, m.id));
        }

        Ok(results)
    }
}
