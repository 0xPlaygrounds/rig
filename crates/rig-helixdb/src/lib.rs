//! HelixDB vector store integration for Rig.
//!
//! This crate provides a small HTTP client for HelixDB query endpoints and a
//! [`HelixDBVectorStore`] vector store backend over Rig's vector store data
//! vocabulary. Queries arrive pre-embedded via
//! [`rig_core::vector_store::request::VectorSearchRequest`]; the store never
//! embeds text itself.
//!
//! The root `rig` facade re-exports this crate as `rig::helixdb` when the
//! `helixdb` feature is enabled.

use std::future::Future;

use reqwest::{Client, StatusCode};
use rig_core::{
    OneOrMany,
    embeddings::Embedding,
    vector_store::{
        SearchHit, StoreRecord, VectorStoreError,
        request::{Filter, VectorSearchRequest},
    },
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A minimal HelixDB HTTP client for running generated Helix queries.
#[derive(Debug, Clone)]
pub struct HelixDB {
    port: Option<u16>,
    client: Client,
    endpoint: String,
    api_key: Option<String>,
}

impl HelixDB {
    /// Creates a HelixDB client using the default reqwest client.
    pub fn new(endpoint: Option<&str>, port: Option<u16>, api_key: Option<&str>) -> Self {
        Self::with_client(endpoint, port, api_key, Client::new())
    }

    /// Creates a HelixDB client using a caller-provided reqwest client.
    pub fn with_client(
        endpoint: Option<&str>,
        port: Option<u16>,
        api_key: Option<&str>,
        client: Client,
    ) -> Self {
        Self {
            port,
            client,
            endpoint: endpoint.unwrap_or("http://localhost").to_string(),
            api_key: api_key.map(ToString::to_string),
        }
    }
}

/// Errors returned by the HelixDB HTTP client.
#[derive(Debug, thiserror::Error)]
pub enum HelixError {
    /// A request to HelixDB failed before a response body could be decoded.
    #[error("error communicating with server: {0}")]
    ReqwestError(#[from] reqwest::Error),

    /// HelixDB returned a non-200 response.
    #[error("got error from server: {details}")]
    RemoteError {
        /// Response body or status reason returned by HelixDB.
        details: String,
    },
}

/// Client interface used by [`HelixDBVectorStore`] to execute HelixDB queries.
pub trait HelixDBClient {
    /// Error type returned by this client.
    type Err: std::error::Error;

    /// Sends a query payload to a HelixDB endpoint and decodes the response body.
    fn query<T, R>(
        &self,
        endpoint: &str,
        data: &T,
    ) -> impl Future<Output = Result<R, Self::Err>> + WasmCompatSend
    where
        T: Serialize + WasmCompatSync,
        R: for<'de> Deserialize<'de>;
}

impl HelixDBClient for HelixDB {
    type Err = HelixError;

    async fn query<T, R>(&self, endpoint: &str, data: &T) -> Result<R, HelixError>
    where
        T: Serialize + WasmCompatSync,
        R: for<'de> Deserialize<'de>,
    {
        let port = self.port.map(|port| format!(":{port}")).unwrap_or_default();
        let url = format!("{}{}/{}", self.endpoint, port, endpoint);

        let mut request = self.client.post(&url).json(data);
        if let Some(api_key) = &self.api_key {
            request = request.header("x-api-key", api_key);
        }

        let response = request.send().await?;

        match response.status() {
            StatusCode::OK => response.json().await.map_err(Into::into),
            code => match response.text().await {
                Ok(details) => Err(HelixError::RemoteError { details }),
                Err(_) => Err(HelixError::RemoteError {
                    details: code
                        .canonical_reason()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| format!("unknown error with code: {code}")),
                }),
            },
        }
    }
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

/// A client for easily carrying out Rig-related vector store operations.
///
/// If you are unsure what type to use for the client, [`HelixDB`] is the typical default.
///
/// Usage:
/// ```no_run
/// use rig_helixdb::{HelixDB, HelixDBVectorStore};
///
/// let helixdb_client = HelixDB::new(None, Some(6969), None);
/// let vector_store = HelixDBVectorStore::new(helixdb_client);
/// # let _ = vector_store;
/// ```
///
/// Queries and records arrive pre-embedded, so the store holds no embedding
/// model.
pub struct HelixDBVectorStore<C> {
    client: C,
}

pub type HelixDBFilter = Filter;

/// The result of a query. Only used internally as this is a representative type required for the relevant HelixDB query (`VectorSearch`).
#[derive(Deserialize, Serialize, Clone, Debug)]
struct QueryResult {
    id: String,
    score: f64,
    doc: String,
    json_payload: String,
}

/// An input query. Only used internally as this is a representative type required for the relevant HelixDB query (`VectorSearch`).
#[derive(Deserialize, Serialize, Clone, Debug)]
struct QueryInput {
    vector: Vec<f64>,
    limit: u64,
    threshold: f64,
}

impl QueryInput {
    /// Makes a new instance of `QueryInput`.
    pub(crate) fn new(vector: Vec<f64>, limit: u64, threshold: f64) -> Self {
        Self {
            vector,
            limit,
            threshold,
        }
    }
}

impl<C> HelixDBVectorStore<C>
where
    C: HelixDBClient + WasmCompatSend + WasmCompatSync,
    C::Err: std::error::Error + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Creates a new HelixDB vector store.
    pub fn new(client: C) -> Self {
        Self { client }
    }

    /// Returns the underlying HelixDB client.
    pub fn client(&self) -> &C {
        &self.client
    }
}

impl<C> HelixDBVectorStore<C>
where
    C: HelixDBClient + WasmCompatSend + WasmCompatSync,
    C::Err: std::error::Error + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Insert precomputed records into HelixDB.
    ///
    /// Each embedding of a record becomes one HelixDB vector carrying the
    /// record's serialized payload as `json_payload`. HelixDB assigns vector
    /// ids itself, so [`StoreRecord::id`] is not persisted.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        #[derive(Serialize, Deserialize, Clone, Debug, Default)]
        struct QueryInput {
            vector: Vec<f64>,
            doc: String,
            json_payload: String,
        }

        #[derive(Serialize, Deserialize, Clone, Debug, Default)]
        struct QueryOutput {
            doc: String,
        }

        for record in records {
            let json_document_as_string = serde_json::to_string(&record.payload)?;

            for embedding in record.embeddings {
                let embedded_text = embedding.document;
                let vector: Vec<f64> = embedding.vec;

                let query = QueryInput {
                    vector,
                    doc: embedded_text,
                    json_payload: json_document_as_string.clone(),
                };

                self.client
                    .query::<QueryInput, QueryOutput>("InsertVector", &query)
                    .await
                    .map_err(datastore_error)?;
            }
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
    /// The [`SearchHit::payload`] is the stored JSON document; scores are
    /// cosine similarities.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let vector = req.query().first().vec;

        let query_input =
            QueryInput::new(vector, req.samples(), req.threshold().unwrap_or_default());

        #[derive(Serialize, Deserialize, Debug)]
        struct VecResult {
            vec_docs: Vec<QueryResult>,
        }

        let result: VecResult = self
            .client
            .query::<QueryInput, VecResult>("VectorSearch", &query_input)
            .await
            .map_err(datastore_error)?;

        let docs = result
            .vec_docs
            .into_iter()
            .filter(|x| {
                let is_threshold = req
                    .threshold()
                    .map(|t| -(x.score - 1.) >= t)
                    .unwrap_or(true);

                is_threshold
                    && req
                        .filter()
                        .clone()
                        .zip(serde_json::from_str(&x.json_payload).ok())
                        .map(|(filter, payload): (Filter, serde_json::Value)| {
                            filter.satisfies(&payload)
                        })
                        .unwrap_or(true)
            })
            .map(|x| {
                let payload: serde_json::Value = serde_json::from_str(&x.json_payload)?;

                // HelixDB gives us the cosine distance, so we need to use `-(cosine_dist - 1)` to get the cosine similarity score.
                Ok(SearchHit {
                    id: x.id,
                    score: -(x.score - 1.),
                    payload,
                })
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?;

        Ok(docs)
    }

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let vector = req.query().first().vec;

        let query_input =
            QueryInput::new(vector, req.samples(), req.threshold().unwrap_or_default());

        #[derive(Serialize, Deserialize, Debug)]
        struct VecResult {
            vec_docs: Vec<QueryResult>,
        }

        let result: VecResult = self
            .client
            .query::<QueryInput, VecResult>("VectorSearch", &query_input)
            .await
            .map_err(datastore_error)?;

        // HelixDB gives us the cosine distance, so we need to use `-(cosine_dist - 1)` to get the cosine similarity score.
        let docs = result
            .vec_docs
            .into_iter()
            .filter(|x| -(x.score - 1.) >= req.threshold().unwrap_or_default())
            .map(|x| Ok((-(x.score - 1.), x.id)))
            .collect::<Result<Vec<_>, VectorStoreError>>()?;

        Ok(docs)
    }

    /// Returns the top N most similar documents deserialized into `T` as
    /// `(score, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<HelixDBFilter>,
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
