//! Milvus vector store integration for Rig.
//!
//! This crate provides [`MilvusVectorStore`], a Rig vector store implementation
//! that talks to Milvus over its HTTP API.
//!
//! The root `rig` facade re-exports this crate as `rig::milvus` when the
//! `milvus` feature is enabled.

mod filter;

use reqwest::StatusCode;
use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};

use crate::filter::Filter;

/// Represents a vector store implementation using Milvus - <https://milvus.io/> as the backend.
pub struct MilvusVectorStore<M> {
    /// Model used to generate embeddings for the vector store
    model: M,
    base_url: String,
    client: reqwest::Client,
    database_name: String,
    collection_name: String,
    token: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: String,
    embedded_text: String,
    embedding: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InsertRequest<'a> {
    data: Vec<CreateRecord>,
    collection_name: &'a str,
    db_name: &'a str,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest<'a> {
    collection_name: &'a str,
    db_name: &'a str,
    data: Vec<f64>,
    #[serde(skip_serializing_if = "String::is_empty")]
    filter: String,
    anns_field: &'a str,
    limit: usize,
    output_fields: Vec<&'a str>,
}

/// Milvus search response envelope, generic over the row shape.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResult<Row> {
    code: i64,
    data: Vec<Row>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultData<T> {
    id: i64,
    distance: f64,
    document: T,
}

/// Row shape for the id-only search path.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultDataOnlyId {
    id: i64,
    distance: f64,
}

impl<M> MilvusVectorStore<M>
where
    M: EmbeddingModel,
{
    /// Creates a new instance of `MilvusVectorStore`.
    ///
    /// # Arguments
    /// * `model` - Embedding model instance
    /// * `base_url` - The URL of where your Milvus instance is located. Alternatively if you're using the Milvus offering provided by Zilliz, your cluster endpoint.
    /// * `database_name` - The name of your database
    /// * `collection_name` - The name of your collection
    pub fn new(model: M, base_url: String, database_name: String, collection_name: String) -> Self {
        Self {
            model,
            base_url,
            client: reqwest::Client::new(),
            database_name,
            collection_name,
            token: None,
        }
    }

    /// Forms the auth token for Milvus from your username and password. Required if using a Milvus instance that requires authentication.
    pub fn auth(mut self, username: String, password: String) -> Self {
        let str = format!("{username}:{password}");
        self.token = Some(str);

        self
    }

    /// Creates a Milvus insertion request.
    fn create_insert_request(&self, data: Vec<CreateRecord>) -> InsertRequest<'_> {
        InsertRequest {
            data,
            collection_name: &self.collection_name,
            db_name: &self.database_name,
        }
    }

    /// Creates a Milvus semantic search request.
    fn create_search_request(
        &self,
        data: Vec<f64>,
        req: &VectorSearchRequest<Filter>,
        id_only: bool,
    ) -> SearchRequest<'_> {
        const OUTPUT_FIELDS: [&str; 4] = ["id", "distance", "document", "embeddedText"];
        const OUTPUT_FIELDS_ID_ONLY: [&str; 2] = ["id", "distance"];

        let output_fields = if id_only {
            OUTPUT_FIELDS_ID_ONLY.to_vec()
        } else {
            OUTPUT_FIELDS.to_vec()
        };

        let threshold = req
            .threshold()
            .map(|thresh| Filter::gte("distance".into(), thresh.into()));

        let filter = match (threshold, req.filter()) {
            (Some(thresh), Some(filter)) => thresh.and(filter.clone()).into_inner(),
            (Some(thresh), _) => thresh.into_inner(),
            (_, Some(filter)) => filter.clone().into_inner(),
            _ => String::new(),
        };

        SearchRequest {
            collection_name: &self.collection_name,
            db_name: &self.database_name,
            data,
            filter,
            anns_field: "embedding",
            limit: req.samples() as usize,
            output_fields,
        }
    }

    /// Embeds the query, runs the Milvus search endpoint, and parses the response.
    async fn search<T: for<'a> Deserialize<'a>>(
        &self,
        req: &VectorSearchRequest<Filter>,
        id_only: bool,
    ) -> Result<T, VectorStoreError> {
        let embedding = self.model.embed_text(req.query()).await?;
        let url = format!(
            "{base_url}/v2/vectordb/entities/search",
            base_url = self.base_url
        );

        let body = self.create_search_request(embedding.vec, req, id_only);

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authorization", format!("Bearer {token}"));
        }

        let body = serde_json::to_string(&body)?;

        let res = client.body(body).send().await?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(res.json().await?)
    }
}

impl<Model> InsertDocuments for MilvusVectorStore<Model>
where
    Model: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let url = format!(
            "{base_url}/v2/vectordb/entities/insert",
            base_url = self.base_url
        );

        let data =
            rig_core::vector_store::flatten_embedded(documents, |json_document, embedding| {
                Ok(CreateRecord {
                    document: serde_json::to_string(json_document)?,
                    embedded_text: embedding.document,
                    embedding: embedding.vec,
                })
            })?;

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authorization", format!("Bearer {token}"));
        }

        let insert_request = self.create_insert_request(data);

        let body = serde_json::to_string(&insert_request)?;

        let res = client.body(body).send().await?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(())
    }
}

impl<M> VectorStoreIndex for MilvusVectorStore<M>
where
    M: EmbeddingModel,
{
    type Filter = Filter;

    /// Search for the top `n` nearest neighbors to the given query within the Milvus vector store.
    /// Returns a vector of tuples containing the score, ID, and payload of the nearest neighbors.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let json: SearchResult<SearchResultData<T>> = self.search(&req, false).await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string(), x.document))
            .collect();

        Ok(res)
    }

    /// Search for the top `n` nearest neighbors to the given query within the Milvus vector store.
    /// Returns a vector of tuples containing the score and ID of the nearest neighbors.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let json: SearchResult<SearchResultDataOnlyId> = self.search(&req, true).await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string()))
            .collect();

        Ok(res)
    }
}
