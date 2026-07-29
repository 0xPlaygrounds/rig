//! Milvus vector store integration for Rig.
//!
//! This crate provides [`MilvusVectorStore`], a Rig vector store backend
//! that talks to Milvus over its HTTP API. Queries arrive pre-embedded via
//! [`VectorSearchRequest`]; the store never embeds text itself.
//!
//! The root `rig` facade re-exports this crate as `rig::milvus` when the
//! `milvus` feature is enabled.

mod filter;

use reqwest::StatusCode;
use rig_core::{
    OneOrMany,
    embeddings::Embedding,
    vector_store::{
        SearchHit, StoreRecord, VectorStoreError,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::filter::Filter;

/// Represents a vector store implementation using Milvus - <https://milvus.io/> as the backend.
///
/// Queries and records arrive pre-embedded, so the store holds no embedding
/// model.
pub struct MilvusVectorStore {
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResult {
    code: i64,
    data: Vec<SearchResultData>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultData {
    id: i64,
    distance: f64,
    document: serde_json::Value,
    embedded_text: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultOnlyId {
    code: i64,
    data: Vec<SearchResultDataOnlyId>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultDataOnlyId {
    id: i64,
    distance: f64,
}

impl MilvusVectorStore {
    /// Creates a new instance of `MilvusVectorStore`.
    ///
    /// # Arguments
    /// * `base_url` - The URL of where your Milvus instance is located. Alternatively if you're using the Milvus offering provided by Zilliz, your cluster endpoint.
    /// * `database_name` - The name of your database
    /// * `collection_name` - The name of your collection
    pub fn new(base_url: String, database_name: String, collection_name: String) -> Self {
        Self {
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

    /// Sends a POST request to a Milvus endpoint, attaching auth when configured.
    async fn post(&self, path: &str, body: String) -> Result<reqwest::Response, VectorStoreError> {
        let url = format!("{base_url}{path}", base_url = self.base_url);

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authentication", format!("Bearer {token}"));
        }

        let res = client.body(body).send().await?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(res)
    }

    /// Insert precomputed records into the Milvus collection.
    ///
    /// Each embedding of a record becomes one Milvus row carrying the record's
    /// serialized payload as its `document` field. Milvus assigns row ids
    /// itself (`autoId`), so [`StoreRecord::id`] is not persisted.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let data = records
            .into_iter()
            .map(|record| {
                let json_document_as_string = serde_json::to_string(&record.payload)?;

                let rows = record
                    .embeddings
                    .into_iter()
                    .map(|embedding| CreateRecord {
                        document: json_document_as_string.clone(),
                        embedded_text: embedding.document,
                        embedding: embedding.vec,
                    })
                    .collect::<Vec<CreateRecord>>();
                Ok(rows)
            })
            .collect::<Result<Vec<Vec<CreateRecord>>, VectorStoreError>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<CreateRecord>>();

        let insert_request = self.create_insert_request(data);
        let body = serde_json::to_string(&insert_request)?;

        self.post("/v2/vectordb/entities/insert", body).await?;

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

    /// Search for the top `n` nearest neighbors to the pre-embedded query within the Milvus vector store.
    ///
    /// Results are returned as [`SearchHit`]s whose payload is the stored JSON
    /// document.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let embedding = req.query().first();
        let body = self.create_search_request(embedding.vec, &req, false);
        let body = serde_json::to_string(&body)?;

        let res = self.post("/v2/vectordb/entities/search", body).await?;
        let json: SearchResult = res.json().await?;

        let res = json
            .data
            .into_iter()
            .map(|x| SearchHit {
                id: x.id.to_string(),
                score: x.distance,
                // The document is stored as a JSON string in a VARCHAR field;
                // decode it back into structured JSON when possible.
                payload: match x.document {
                    serde_json::Value::String(s) => {
                        serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
                    }
                    other => other,
                },
            })
            .collect();

        Ok(res)
    }

    /// Search for the top `n` nearest neighbors to the pre-embedded query within the Milvus vector store.
    /// Returns a vector of tuples containing the score and ID of the nearest neighbors.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = req.query().first();
        let body = self.create_search_request(embedding.vec, &req, true);
        let body = serde_json::to_string(&body)?;

        let res = self.post("/v2/vectordb/entities/search", body).await?;
        let json: SearchResultOnlyId = res.json().await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string()))
            .collect();

        Ok(res)
    }

    /// Returns the top `n` nearest neighbors deserialized into `T` as
    /// `(score, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<Filter>,
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
